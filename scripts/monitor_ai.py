"""AI-powered trading monitor.

Reads from the bot's SQLite DB every N minutes, computes metrics,
sends a structured summary to Claude Sonnet for analysis, and saves reports.

Usage:
    python3 scripts/monitor_ai.py --interval 600 --pair HYPE
    python3 scripts/monitor_ai.py --interval 300 --pair HYPE --db data/trading_log.db
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_fills(db: sqlite3.Connection, pair: str, since: str) -> list[dict]:
    cur = db.execute(
        "SELECT timestamp, side, price, size, size_usd, fee, "
        "mid_price_at_fill, spread_captured_bps, inventory_after "
        "FROM fills WHERE pair = ? AND timestamp >= ? ORDER BY timestamp",
        (pair, since),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def collect_snapshots(db: sqlite3.Connection, pair: str, since: str) -> list[dict]:
    cur = db.execute(
        "SELECT timestamp, mid_price, best_bid, best_ask, spread_bps, "
        "inventory, inventory_usd, unrealized_pnl, realized_pnl, total_pnl "
        "FROM snapshots WHERE pair = ? AND timestamp >= ? ORDER BY timestamp",
        (pair, since),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def collect_orders(db: sqlite3.Connection, pair: str, since: str) -> dict:
    cur = db.execute(
        "SELECT status, COUNT(*) FROM orders WHERE pair = ? AND timestamp >= ? "
        "GROUP BY status",
        (pair, since),
    )
    counts = dict(cur.fetchall())
    return {
        "total": sum(counts.values()),
        "resting": counts.get("resting", 0),
        "error": counts.get("error", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(fills: list[dict], snapshots: list[dict],
                    orders: dict, interval_sec: int) -> dict:
    """Compute all metrics from raw DB data."""

    # ── Fills metrics ──
    n_fills = len(fills)
    buys = [f for f in fills if f["side"] == "buy"]
    sells = [f for f in fills if f["side"] == "sell"]
    total_fees = sum(f["fee"] for f in fills)
    total_volume = sum(f["size_usd"] for f in fills)
    avg_spread_captured = (
        sum(f["spread_captured_bps"] for f in fills) / n_fills
        if n_fills else 0
    )

    # Win rate based on spread captured (positive = good fill)
    winning = sum(1 for f in fills if f["spread_captured_bps"] > 0)
    win_rate = winning / n_fills if n_fills else 0

    # Position flips: sign of inventory changes
    flips = 0
    for i in range(1, len(fills)):
        inv_prev = fills[i - 1]["inventory_after"]
        inv_curr = fills[i]["inventory_after"]
        if inv_prev != 0 and inv_curr != 0:
            if (inv_prev > 0) != (inv_curr > 0):
                flips += 1

    # Time between fills
    avg_time_between_fills = 0
    if n_fills >= 2:
        try:
            ts = [datetime.fromisoformat(f["timestamp"]) for f in fills]
            deltas = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, len(ts))]
            avg_time_between_fills = sum(deltas) / len(deltas) if deltas else 0
        except Exception:
            pass

    # ── Adverse selection ──
    # For each fill, find the mid price ~30s after and measure if the market
    # moved against us (bought and price went down, or sold and price went up)
    adverse_buys_bps = []
    adverse_sells_bps = []
    if snapshots and fills:
        snap_times = []
        for s in snapshots:
            try:
                snap_times.append((datetime.fromisoformat(s["timestamp"]), s["mid_price"]))
            except Exception:
                continue

        for f in fills:
            try:
                fill_time = datetime.fromisoformat(f["timestamp"])
            except Exception:
                continue
            target_time = fill_time + timedelta(seconds=30)
            fill_mid = f["mid_price_at_fill"]
            if fill_mid <= 0:
                continue

            # Find closest snapshot after target_time
            future_mid = None
            for st, sp in snap_times:
                if st >= target_time:
                    future_mid = sp
                    break

            if future_mid is None:
                continue

            move_bps = (future_mid - fill_mid) / fill_mid * 10_000

            if f["side"] == "buy":
                # Bought: if price dropped after, that's adverse selection
                adverse_buys_bps.append(-move_bps)  # negative move = adverse
            else:
                # Sold: if price rose after, that's adverse selection
                adverse_sells_bps.append(move_bps)

    avg_adverse_buy = sum(adverse_buys_bps) / len(adverse_buys_bps) if adverse_buys_bps else 0
    avg_adverse_sell = sum(adverse_sells_bps) / len(adverse_sells_bps) if adverse_sells_bps else 0
    avg_adverse_total = 0
    all_adverse = adverse_buys_bps + adverse_sells_bps
    if all_adverse:
        avg_adverse_total = sum(all_adverse) / len(all_adverse)

    # ── Snapshot metrics ──
    first_mid = snapshots[0]["mid_price"] if snapshots else 0
    last_mid = snapshots[-1]["mid_price"] if snapshots else 0
    mid_change_bps = 0
    if first_mid > 0 and last_mid > 0:
        mid_change_bps = (last_mid - first_mid) / first_mid * 10_000

    current_inventory = snapshots[-1]["inventory"] if snapshots else 0
    current_inventory_usd = snapshots[-1]["inventory_usd"] if snapshots else 0
    max_inventory_usd = max((abs(s["inventory_usd"]) for s in snapshots), default=0)
    avg_spread_market = (
        sum(s["spread_bps"] for s in snapshots) / len(snapshots)
        if snapshots else 0
    )

    total_pnl = snapshots[-1]["total_pnl"] if snapshots else 0
    realized_pnl = snapshots[-1]["realized_pnl"] if snapshots else 0
    unrealized_pnl = snapshots[-1]["unrealized_pnl"] if snapshots else 0

    # PnL per hour extrapolated
    hours = interval_sec / 3600
    pnl_per_hour = total_pnl / hours if hours > 0 else 0

    return {
        "period_minutes": round(interval_sec / 60, 1),
        "fills": {
            "total": n_fills,
            "buys": len(buys),
            "sells": len(sells),
            "volume_usd": round(total_volume, 2),
            "total_fees": round(total_fees, 4),
            "avg_spread_captured_bps": round(avg_spread_captured, 2),
            "win_rate": round(win_rate, 3),
            "position_flips": flips,
            "avg_time_between_fills_sec": round(avg_time_between_fills, 1),
        },
        "adverse_selection": {
            "avg_adverse_buy_bps": round(avg_adverse_buy, 2),
            "avg_adverse_sell_bps": round(avg_adverse_sell, 2),
            "avg_adverse_total_bps": round(avg_adverse_total, 2),
            "samples_buy": len(adverse_buys_bps),
            "samples_sell": len(adverse_sells_bps),
        },
        "market": {
            "first_mid": round(first_mid, 4),
            "last_mid": round(last_mid, 4),
            "mid_change_bps": round(mid_change_bps, 1),
            "avg_spread_bps": round(avg_spread_market, 1),
        },
        "inventory": {
            "current": round(current_inventory, 6),
            "current_usd": round(current_inventory_usd, 2),
            "max_abs_usd": round(max_inventory_usd, 2),
        },
        "pnl": {
            "realized": round(realized_pnl, 4),
            "unrealized": round(unrealized_pnl, 4),
            "total": round(total_pnl, 4),
            "pnl_per_hour": round(pnl_per_hour, 4),
        },
        "orders": orders,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI analysis
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Tu es un analyste quantitatif qui monitore un bot de market making crypto "
    "sur Hyperliquid. Le bot fait du market making avec la stratégie "
    "Avellaneda-Stoikov. Les fees maker sont de +1.5 bps (pas de rebate). "
    "Analyse les données des dernières minutes et donne:\n"
    "1. Un verdict: le bot gagne ou perd de l'argent et pourquoi\n"
    "2. Les patterns problématiques (flips, adverse selection, accumulation d'inventaire)\n"
    "3. Des suggestions concrètes de paramètres à ajuster (spread, gamma, sizing, etc)\n"
    "4. Un score de santé de 1 à 10\n"
    "Sois concis et actionable, pas de blabla. Réponds en français."
)


def call_claude(client: anthropic.Anthropic, metrics: dict, pair: str) -> str:
    """Send metrics to Claude and return analysis."""
    user_msg = (
        f"Données de monitoring pour {pair} "
        f"(dernières {metrics['period_minutes']} minutes):\n\n"
        f"{json.dumps(metrics, indent=2, ensure_ascii=False)}"
    )

    for attempt in range(2):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text
        except Exception as e:
            if attempt == 0:
                print(f"  API error (retrying): {e}")
                time.sleep(5)
            else:
                print(f"  API error (skipping): {e}")
                return f"[API ERROR] {e}"

    return "[API ERROR] unreachable"


def extract_health_score(analysis: str) -> int:
    """Try to extract the health score from Claude's response."""
    import re
    # Look for patterns like "Score: 7", "santé: 7/10", "7/10", etc.
    patterns = [
        r"score[:\s]+(\d+)\s*/?\s*10",
        r"(\d+)\s*/\s*10",
        r"score[:\s]+(\d+)",
        r"santé[:\s]+(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, analysis, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                return val
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Report saving
# ─────────────────────────────────────────────────────────────────────────────

def save_report(metrics: dict, analysis: str, health_score: int,
                reports_dir: str) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    filename = now.strftime("%Y-%m-%d_%H-%M") + ".json"
    filepath = os.path.join(reports_dir, filename)

    report = {
        "timestamp": now.isoformat(),
        "data_summary": metrics,
        "ai_analysis": analysis,
        "health_score": health_score,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_cycle(db_path: str, pair: str, interval_sec: int,
              client: anthropic.Anthropic, reports_dir: str) -> None:
    """Run one monitoring cycle."""
    now_utc = datetime.now(timezone.utc)
    since = (now_utc - timedelta(seconds=interval_sec)).isoformat()

    print(f"\n{'='*60}")
    print(f"  Monitor cycle @ {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Pair: {pair} | Window: {interval_sec // 60} min")
    print(f"{'='*60}")

    if not os.path.exists(db_path):
        print("  DB not found — bot not started yet? Skipping.")
        return

    db = sqlite3.connect(db_path, timeout=5)
    db.row_factory = sqlite3.Row

    try:
        fills = collect_fills(db, pair, since)
        snapshots = collect_snapshots(db, pair, since)
        orders = collect_orders(db, pair, since)
    finally:
        db.close()

    if not fills and not snapshots:
        print("  No activity in this window. Skipping API call.")
        return

    print(f"  Data: {len(fills)} fills, {len(snapshots)} snapshots")

    metrics = compute_metrics(fills, snapshots, orders, interval_sec)

    # Print quick summary
    f = metrics["fills"]
    p = metrics["pnl"]
    inv = metrics["inventory"]
    adv = metrics["adverse_selection"]
    print(f"  Fills: {f['total']} ({f['buys']}B/{f['sells']}S) | "
          f"Vol: ${f['volume_usd']}")
    print(f"  PnL: ${p['total']} (${p['pnl_per_hour']}/h) | "
          f"Fees: ${f['total_fees']}")
    print(f"  Inventory: ${inv['current_usd']} | "
          f"Max: ${inv['max_abs_usd']}")
    print(f"  Spread captured: {f['avg_spread_captured_bps']} bps | "
          f"Flips: {f['position_flips']}")
    print(f"  Adverse selection: {adv['avg_adverse_total_bps']} bps")

    # Call Claude
    print("\n  Calling Claude for analysis...")
    analysis = call_claude(client, metrics, pair)
    health_score = extract_health_score(analysis)

    print(f"\n{'─'*60}")
    print(f"  AI ANALYSIS (Health: {health_score}/10)")
    print(f"{'─'*60}")
    print(analysis)
    print(f"{'─'*60}")

    # Save report
    filepath = save_report(metrics, analysis, health_score, reports_dir)
    print(f"\n  Report saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="AI Trading Monitor")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair (e.g. HYPE)")
    parser.add_argument("--interval", type=int, default=600, help="Interval in seconds (default 600)")
    parser.add_argument("--db", type=str, default="data/trading_log.db", help="Path to SQLite DB")
    parser.add_argument("--reports-dir", type=str, default="data/ai_reports", help="Reports output dir")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"AI Monitor started | Pair: {args.pair} | Interval: {args.interval}s")
    print(f"DB: {args.db} | Reports: {args.reports_dir}")

    if args.once:
        run_cycle(args.db, args.pair.upper(), args.interval, client, args.reports_dir)
        return

    # Wait for first interval before first analysis
    print(f"Waiting {args.interval}s for first data window...")
    try:
        time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
        return

    while True:
        try:
            run_cycle(args.db, args.pair.upper(), args.interval, client, args.reports_dir)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"  Cycle error: {e}")

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
