"""Scanner de paires Hyperliquid pour identifier les meilleures opportunités de market making."""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Forcer UTF-8 sur Windows pour les caractères spéciaux
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from tabulate import tabulate

from scanner import api
from scanner.config import (
    MIN_VOLUME_24H,
    MAX_VOLUME_24H,
    MIN_SPREAD_BPS,
    BOOK_DEPTH_LEVELS,
    REQUEST_DELAY,
)
from scanner.scoring import compute_composite_score


def parse_book(book_data: dict) -> dict:
    """Parse un orderbook L2 et en extrait les métriques.

    Returns:
        dict avec best_bid, best_ask, spread_bps, depth_usd, num_levels
    """
    levels = book_data.get("levels", [[], []])
    bids = levels[0] if len(levels) > 0 else []
    asks = levels[1] if len(levels) > 1 else []

    if not bids or not asks:
        return None

    best_bid = float(bids[0]["px"])
    best_ask = float(asks[0]["px"])
    mid_price = (best_bid + best_ask) / 2

    if mid_price <= 0:
        return None

    spread_bps = ((best_ask - best_bid) / mid_price) * 10_000

    # Profondeur sur les N premiers niveaux (bid + ask) en USD
    depth_usd = 0.0
    for level in bids[:BOOK_DEPTH_LEVELS]:
        depth_usd += float(level["px"]) * float(level["sz"])
    for level in asks[:BOOK_DEPTH_LEVELS]:
        depth_usd += float(level["px"]) * float(level["sz"])

    # Nombre total de niveaux de prix
    num_levels = len(bids) + len(asks)

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread_bps": round(spread_bps, 2),
        "depth_usd": round(depth_usd, 2),
        "num_levels": num_levels,
    }


def format_usd(value: float) -> str:
    """Formate un montant en USD lisible ($1.2M, $450k, etc.)."""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:.0f}"


def scan_all(min_vol: float, max_vol: float) -> list[dict]:
    """Scanne toutes les paires et retourne les résultats triés par score."""
    print("Récupération des métadonnées...")
    meta_and_ctxs = api.get_meta_and_asset_ctxs()
    meta = meta_and_ctxs[0]
    asset_ctxs = meta_and_ctxs[1]
    universe = meta["universe"]

    print(f"{len(universe)} paires trouvées. Analyse en cours...\n")

    results = []

    for i, asset in enumerate(universe):
        coin = asset["name"]
        ctx = asset_ctxs[i]

        # Extraire volume 24h et funding rate depuis le contexte
        volume_24h = float(ctx.get("dayNtlVlm", 0))
        funding_rate = float(ctx.get("funding", 0))
        open_interest = float(ctx.get("openInterest", 0))
        mark_px = float(ctx.get("markPx", 0))
        oi_usd = open_interest * mark_px if mark_px > 0 else 0

        # Filtre volume
        if volume_24h < min_vol or volume_24h > max_vol:
            continue

        # Récupérer l'orderbook
        try:
            book_data = api.get_l2_book(coin)
        except Exception as e:
            print(f"  [!] Erreur book {coin}: {e}")
            continue

        book_metrics = parse_book(book_data)
        if book_metrics is None:
            continue

        # Filtre spread
        if book_metrics["spread_bps"] < MIN_SPREAD_BPS:
            continue

        # Construire les métriques complètes
        metrics = {
            "coin": coin,
            "spread_bps": book_metrics["spread_bps"],
            "volume_24h": volume_24h,
            "depth_usd": book_metrics["depth_usd"],
            "num_levels": book_metrics["num_levels"],
            "funding_rate": funding_rate,
            "open_interest_usd": oi_usd,
            "mid_price": book_metrics["mid_price"],
            "best_bid": book_metrics["best_bid"],
            "best_ask": book_metrics["best_ask"],
        }

        # Calculer le score
        scoring = compute_composite_score(metrics)
        metrics["score"] = scoring["composite"]
        metrics["sub_scores"] = scoring["sub_scores"]

        results.append(metrics)

        sys.stdout.write(f"\r  Analysé: {i + 1}/{len(universe)} — Retenu: {len(results)}")
        sys.stdout.flush()

        time.sleep(REQUEST_DELAY)

    print(f"\n\n{len(results)} paires retenues après filtrage.\n")

    # Trier par score décroissant
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_results_table(results: list[dict]):
    """Affiche les résultats sous forme de tableau."""
    if not results:
        print("Aucune paire ne correspond aux critères.")
        return

    table = []
    for rank, r in enumerate(results, 1):
        table.append([
            rank,
            f"{r['coin']}-PERP",
            f"{r['spread_bps']:.1f} bps",
            format_usd(r["volume_24h"]),
            format_usd(r["depth_usd"]),
            format_usd(r["open_interest_usd"]),
            f"{r['funding_rate'] * 100:.4f}%",
            r["num_levels"],
            f"{r['score']:.0f}",
        ])

    headers = ["Rank", "Paire", "Spread", "Vol 24h", "Depth (5 lvl)", "OI", "Funding", "Levels", "Score"]
    print(tabulate(table, headers=headers, tablefmt="double_outline"))


def export_json(results: list[dict], output_dir: str = "results"):
    """Exporte les résultats en JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = path / f"scan_{timestamp}.json"

    # Rendre sérialisable (les sub_scores sont déjà des dicts)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nExport JSON → {filepath}")


def print_detail(coin: str):
    """Affiche le détail d'une paire spécifique."""
    coin = coin.upper()
    print(f"\n{'='*60}")
    print(f"  Détail: {coin}-PERP")
    print(f"{'='*60}\n")

    # Meta + contexte
    meta_and_ctxs = api.get_meta_and_asset_ctxs()
    meta = meta_and_ctxs[0]
    asset_ctxs = meta_and_ctxs[1]
    universe = meta["universe"]

    coin_idx = None
    for i, asset in enumerate(universe):
        if asset["name"] == coin:
            coin_idx = i
            break

    if coin_idx is None:
        print(f"Paire {coin} introuvable. Paires disponibles:")
        names = [a["name"] for a in universe]
        print(", ".join(names[:30]) + "...")
        return

    ctx = asset_ctxs[coin_idx]
    asset_info = universe[coin_idx]

    print(f"  Nom:            {coin}")
    print(f"  szDecimals:     {asset_info.get('szDecimals', '?')}")
    print(f"  Mark Price:     ${float(ctx.get('markPx', 0)):,.4f}")
    print(f"  Volume 24h:     {format_usd(float(ctx.get('dayNtlVlm', 0)))}")
    print(f"  Open Interest:  {format_usd(float(ctx.get('openInterest', 0)) * float(ctx.get('markPx', 0)))}")
    print(f"  Funding Rate:   {float(ctx.get('funding', 0)) * 100:.4f}%")
    print(f"  Premium:        {float(ctx.get('premium', 0)) * 100:.4f}%")

    # Orderbook
    book = api.get_l2_book(coin)
    levels = book.get("levels", [[], []])
    bids = levels[0]
    asks = levels[1]

    print(f"\n  --- Orderbook ({len(bids)} bids / {len(asks)} asks) ---\n")

    # Afficher les 10 premiers niveaux
    ask_rows = []
    for level in asks[:10][::-1]:
        px = float(level["px"])
        sz = float(level["sz"])
        ask_rows.append(["", "", f"${px:,.4f}", f"{sz:.4f}", format_usd(px * sz)])

    bid_rows = []
    for level in bids[:10]:
        px = float(level["px"])
        sz = float(level["sz"])
        bid_rows.append([f"${px:,.4f}", f"{sz:.4f}", format_usd(px * sz), "", ""])

    print("  ASKS (vente):")
    ask_headers = ["", "", "Price", "Size", "USD"]
    print(tabulate(ask_rows, headers=ask_headers, tablefmt="simple"))

    mid = (float(bids[0]["px"]) + float(asks[0]["px"])) / 2 if bids and asks else 0
    spread_bps = ((float(asks[0]["px"]) - float(bids[0]["px"])) / mid * 10_000) if mid > 0 else 0
    print(f"\n  --- Mid: ${mid:,.4f} | Spread: {spread_bps:.1f} bps ---\n")

    print("  BIDS (achat):")
    bid_headers = ["Price", "Size", "USD", "", ""]
    print(tabulate(bid_rows, headers=bid_headers, tablefmt="simple"))

    # Score
    book_metrics = parse_book(book)
    if book_metrics:
        metrics = {
            "spread_bps": book_metrics["spread_bps"],
            "volume_24h": float(ctx.get("dayNtlVlm", 0)),
            "depth_usd": book_metrics["depth_usd"],
            "num_levels": book_metrics["num_levels"],
            "funding_rate": float(ctx.get("funding", 0)),
        }
        scoring = compute_composite_score(metrics)
        print(f"\n  --- Score Composite: {scoring['composite']:.0f}/100 ---\n")
        weight_labels = {"spread": "30%", "volume": "25%", "depth": "20%", "levels": "15%", "funding": "10%"}
        for k, v in scoring["sub_scores"].items():
            label = weight_labels[k]
            print(f"    {k:<10} {v:5.1f}/100  (poids: {label})")


def main():
    parser = argparse.ArgumentParser(
        description="Scanner de paires Hyperliquid pour market making"
    )
    parser.add_argument("--export", action="store_true", help="Exporter les résultats en JSON")
    parser.add_argument("--detail", type=str, help="Afficher le détail d'une paire (ex: DOGE)")
    parser.add_argument("--min-vol", type=float, default=MIN_VOLUME_24H, help=f"Volume 24h minimum en USD (défaut: {MIN_VOLUME_24H})")
    parser.add_argument("--max-vol", type=float, default=MAX_VOLUME_24H, help=f"Volume 24h maximum en USD (défaut: {MAX_VOLUME_24H})")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Hyperliquid Market Making Scanner               ║")
    print(f"║  {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<49}║")
    print("╚══════════════════════════════════════════════════╝\n")

    if args.detail:
        print_detail(args.detail)
        return

    results = scan_all(args.min_vol, args.max_vol)
    print_results_table(results)

    if args.export:
        export_json(results)


if __name__ == "__main__":
    main()
