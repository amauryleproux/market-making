#!/usr/bin/env python3
"""
Analyse post-session du market making.

Lit la base SQLite et génère des rapports par date/paire.

Usage :
    python scripts/analyze.py                          # Rapport dernière session
    python scripts/analyze.py --date 2026-02-08        # Par date
    python scripts/analyze.py --pair MEGA              # Par paire
    python scripts/analyze.py --recommendations        # Suggestions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import sqlite3
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _pnl_color(val: float) -> str:
    if val > 0:
        return f"[green]+${val:,.4f}[/]"
    elif val < 0:
        return f"[red]-${abs(val):,.4f}[/]"
    return "$0.00"


def report_pnl_summary(conn: sqlite3.Connection, pair: str = None, date: str = None):
    """Affiche le résumé PnL par session."""
    sql = "SELECT * FROM pnl_summary WHERE 1=1"
    params = []
    if pair:
        sql += " AND pair = ?"
        params.append(pair.upper())
    if date:
        sql += " AND session_start >= ?"
        params.append(date)
    sql += " ORDER BY session_start DESC LIMIT 20"

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        console.print("[yellow]Aucune session trouvée.[/]")
        return

    table = Table(title="Résumé PnL par session", expand=True)
    table.add_column("Session", width=20)
    table.add_column("Paire", style="cyan", width=8)
    table.add_column("Fills", justify="right", width=6)
    table.add_column("Volume", justify="right", width=12)
    table.add_column("Gross PnL", justify="right", width=12)
    table.add_column("Fees", justify="right", width=10)
    table.add_column("Net PnL", justify="right", width=12)
    table.add_column("Avg Spread", justify="right", width=10)
    table.add_column("Max Inv", justify="right", width=10)
    table.add_column("Max DD", justify="right", width=10)

    for r in rows:
        start = r["session_start"][:16].replace("T", " ")
        table.add_row(
            start,
            r["pair"],
            str(r["fills_count"]),
            f"${r['volume_usd']:,.0f}",
            _pnl_color(r["gross_pnl"]),
            f"${r['total_fees']:,.4f}",
            _pnl_color(r["net_pnl"]),
            f"{r['avg_spread_captured_bps']:.1f} bps",
            f"${r['max_inventory_usd']:,.0f}",
            f"${r['max_drawdown_usd']:,.2f}",
        )

    console.print(table)


def report_fills(conn: sqlite3.Connection, pair: str = None, date: str = None):
    """Analyse détaillée des fills."""
    sql = "SELECT * FROM fills WHERE 1=1"
    params = []
    if pair:
        sql += " AND pair = ?"
        params.append(pair.upper())
    if date:
        sql += " AND timestamp >= ?"
        params.append(date)
    sql += " ORDER BY timestamp"

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        console.print("[yellow]Aucun fill trouvé.[/]")
        return

    # Statistiques
    total = len(rows)
    buys = sum(1 for r in rows if r["side"] == "buy")
    sells = total - buys
    volume = sum(r["size_usd"] for r in rows)
    fees = sum(r["fee"] for r in rows)
    spreads = [r["spread_captured_bps"] for r in rows]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0

    # Adverse selection : fills où le spread capturé est négatif
    # (le prix a bougé contre nous immédiatement)
    adverse = sum(1 for r in rows if r["spread_captured_bps"] < 1.0)
    adverse_pct = (adverse / total * 100) if total > 0 else 0

    # Paires uniques
    pairs = set(r["pair"] for r in rows)

    console.print(Panel(
        f"Fills: [bold]{total}[/] ({buys} buy / {sells} sell)\n"
        f"Paires: [cyan]{', '.join(sorted(pairs))}[/]\n"
        f"Volume total: [yellow]${volume:,.0f}[/]\n"
        f"Fees total: [yellow]${fees:,.4f}[/]\n"
        f"Spread moyen capturé: [green]{avg_spread:.1f} bps[/]\n"
        f"Adverse selection: [{'red' if adverse_pct > 30 else 'green'}]{adverse_pct:.1f}%[/] "
        f"({adverse}/{total} fills < 1 bps)",
        title="Analyse des Fills",
    ))

    # Par paire
    if len(pairs) > 1:
        table = Table(title="Fills par paire", expand=True)
        table.add_column("Paire", style="cyan")
        table.add_column("Fills", justify="right")
        table.add_column("Buys", justify="right")
        table.add_column("Sells", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("Avg Spread", justify="right")

        for p in sorted(pairs):
            p_rows = [r for r in rows if r["pair"] == p]
            p_buys = sum(1 for r in p_rows if r["side"] == "buy")
            p_vol = sum(r["size_usd"] for r in p_rows)
            p_spreads = [r["spread_captured_bps"] for r in p_rows]
            p_avg = sum(p_spreads) / len(p_spreads) if p_spreads else 0
            table.add_row(
                p, str(len(p_rows)), str(p_buys), str(len(p_rows) - p_buys),
                f"${p_vol:,.0f}", f"{p_avg:.1f} bps",
            )
        console.print(table)


def report_inventory(conn: sqlite3.Connection, pair: str = None, date: str = None):
    """Distribution de l'inventaire au fil du temps."""
    sql = "SELECT pair, inventory_usd, total_pnl FROM snapshots WHERE 1=1"
    params = []
    if pair:
        sql += " AND pair = ?"
        params.append(pair.upper())
    if date:
        sql += " AND timestamp >= ?"
        params.append(date)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return

    # Par paire
    pairs_data: dict[str, list] = {}
    for r in rows:
        pairs_data.setdefault(r["pair"], []).append(r)

    table = Table(title="Distribution Inventaire", expand=True)
    table.add_column("Paire", style="cyan")
    table.add_column("Snapshots", justify="right")
    table.add_column("Inv Moyen $", justify="right")
    table.add_column("Inv Max $", justify="right")
    table.add_column("PnL Min", justify="right")
    table.add_column("PnL Max", justify="right")

    for p in sorted(pairs_data.keys()):
        data = pairs_data[p]
        invs = [d["inventory_usd"] for d in data]
        pnls = [d["total_pnl"] for d in data]
        table.add_row(
            p,
            str(len(data)),
            f"${sum(abs(i) for i in invs) / len(invs):,.2f}",
            f"${max(abs(i) for i in invs):,.2f}",
            _pnl_color(min(pnls)),
            _pnl_color(max(pnls)),
        )

    console.print(table)


def report_recommendations(conn: sqlite3.Connection, pair: str = None):
    """Recommandations basées sur les données."""
    # Dernière session
    sql = "SELECT * FROM pnl_summary WHERE 1=1"
    params = []
    if pair:
        sql += " AND pair = ?"
        params.append(pair.upper())
    sql += " ORDER BY session_start DESC"

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        console.print("[yellow]Pas assez de données pour des recommandations.[/]")
        return

    console.print(Panel("[bold]Recommandations[/]", title="Analyse"))

    for r in rows:
        coin = r["pair"]
        recs = []

        # Fills insuffisants
        if r["fills_count"] < 10:
            recs.append(f"  Peu de fills ({r['fills_count']}). Envisager de réduire spread_bps ou augmenter order_size_usd.")

        # Adverse selection élevée
        fills = conn.execute(
            "SELECT spread_captured_bps FROM fills WHERE pair = ? AND timestamp >= ?",
            (coin, r["session_start"]),
        ).fetchall()
        if fills:
            adverse = sum(1 for f in fills if f["spread_captured_bps"] < 1.0)
            pct = adverse / len(fills) * 100
            if pct > 30:
                recs.append(f"  Adverse selection élevée ({pct:.0f}%). Augmenter spread_bps de ~20%.")

        # Drawdown élevé
        if r["max_drawdown_usd"] > 0.5 * 200:  # 50% du kill switch par défaut
            recs.append(f"  Max drawdown élevé (${r['max_drawdown_usd']:.2f}). Réduire max_inventory_usd.")

        # Inventory trop élevé
        if r["max_inventory_usd"] > 0:
            # Lire la config pour comparer
            recs.append(f"  Max inventory atteint: ${r['max_inventory_usd']:,.0f}. Ajuster inventory_skew_factor si trop fréquent.")

        # Net PnL négatif
        if r["net_pnl"] < 0:
            recs.append(f"  PnL net négatif (${r['net_pnl']:,.4f}). Revoir les paramètres de cette paire.")

        if recs:
            console.print(f"\n[cyan bold]{coin}[/]:")
            for rec in recs:
                console.print(rec)
        else:
            console.print(f"\n[cyan bold]{coin}[/]: [green]Paramètres OK[/]")


@click.command()
@click.option("--db", default="data/trading_log.db", help="Chemin base SQLite")
@click.option("--date", default=None, help="Filtrer par date (YYYY-MM-DD)")
@click.option("--pair", default=None, help="Filtrer par paire")
@click.option("--recommendations", is_flag=True, help="Afficher recommandations")
def main(db, date, pair, recommendations):
    """Analyse post-session du market making."""
    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]Base de données introuvable : {db}[/]")
        console.print("Lancez d'abord le bot pour créer la base.")
        return

    conn = _connect(str(db_path))

    console.print(Panel(
        f"Base: [cyan]{db}[/]"
        + (f"\nDate: [yellow]{date}[/]" if date else "")
        + (f"\nPaire: [yellow]{pair.upper()}[/]" if pair else ""),
        title="Analyse Market Making",
    ))

    report_pnl_summary(conn, pair, date)
    console.print()
    report_fills(conn, pair, date)
    console.print()
    report_inventory(conn, pair, date)

    if recommendations:
        console.print()
        report_recommendations(conn, pair)

    conn.close()


if __name__ == "__main__":
    main()
