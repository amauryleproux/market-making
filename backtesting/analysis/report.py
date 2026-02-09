"""Génération de rapports : plots matplotlib + résumé texte."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtesting.engine.backtest_engine import BacktestResult, TickEvent, BacktestConfig
from backtesting.engine.matching import SimulatedFill
from backtesting.analysis.metrics import BacktestMetrics
from src.utils.logger import get_logger

log = get_logger("bt.report")


def generate_report(
    result: BacktestResult,
    metrics: BacktestMetrics,
    output_dir: str = "backtesting/results",
    show_plots: bool = False,
) -> str:
    """Génère le rapport complet avec plots et résumé.

    Crée :
    - {coin}_{date}_summary.txt
    - {coin}_{date}_pnl.png
    - {coin}_{date}_inventory.png
    - {coin}_{date}_fills.csv
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    coin = result.config.coin
    date_tag = result.config.start_date
    prefix = f"{coin}_{date_tag}"

    events = result.tick_events
    fills = result.all_fills

    if not events:
        log.warning("no_events_for_report")
        return ""

    # Résumé texte
    summary_path = str(out / f"{prefix}_summary.txt")
    _write_summary(metrics, result.config, summary_path)

    # Plots
    times = [_ts_to_dt(e.timestamp_ms) for e in events]

    _plot_pnl(events, times, str(out / f"{prefix}_pnl.png"), show_plots)
    _plot_inventory(events, times, str(out / f"{prefix}_inventory.png"), show_plots)

    if len(events) > 10:
        _plot_spread_and_drawdown(events, times, str(out / f"{prefix}_analysis.png"), show_plots)

    # Export fills CSV
    if fills:
        _export_fills_csv(fills, str(out / f"{prefix}_fills.csv"))

    log.info("report_generated", output_dir=output_dir, prefix=prefix)
    return summary_path


def _ts_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _write_summary(
    metrics: BacktestMetrics, config: BacktestConfig, path: str
) -> None:
    """Écrit le résumé texte."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Backtest Report: {config.coin}\n")
        f.write(f"Period: {config.start_date} to {config.end_date}\n")
        f.write(f"Parameters:\n")
        f.write(f"  spread_bps={config.spread_bps}, size=${config.order_size_usd}, "
                f"levels={config.num_levels}, max_inv=${config.max_inventory_usd}\n")
        f.write(f"  skew={config.inventory_skew_factor}, latency={config.latency_ms}ms, "
                f"fee={config.maker_fee_bps}bps\n")
        f.write(metrics.summary())
        f.write("\n")


def _plot_pnl(
    events: list[TickEvent],
    times: list[datetime],
    path: str,
    show: bool,
) -> None:
    """PnL cumulé au fil du temps."""
    fig, ax = plt.subplots(figsize=(12, 5))

    realized = [e.realized_pnl for e in events]
    total = [e.total_pnl for e in events]

    ax.plot(times, total, label="Total PnL", color="blue", linewidth=1)
    ax.plot(times, realized, label="Realized PnL", color="green", linewidth=0.8, alpha=0.7)
    ax.fill_between(times, realized, total, alpha=0.15, color="blue", label="Unrealized")

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_title("PnL Over Time")
    ax.set_ylabel("USD")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_inventory(
    events: list[TickEvent],
    times: list[datetime],
    path: str,
    show: bool,
) -> None:
    """Inventory (USD) au fil du temps."""
    fig, ax = plt.subplots(figsize=(12, 4))

    inv_usd = [e.inventory_usd for e in events]

    ax.fill_between(
        times, 0, inv_usd,
        where=[v >= 0 for v in inv_usd],
        color="green", alpha=0.3, label="Long",
    )
    ax.fill_between(
        times, 0, inv_usd,
        where=[v < 0 for v in inv_usd],
        color="red", alpha=0.3, label="Short",
    )
    ax.plot(times, inv_usd, color="black", linewidth=0.5)

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_title("Inventory Over Time")
    ax.set_ylabel("USD")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_spread_and_drawdown(
    events: list[TickEvent],
    times: list[datetime],
    path: str,
    show: bool,
) -> None:
    """2 subplots : spread marché + drawdown."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Spread
    spreads = [e.spread_bps for e in events]
    ax1.plot(times, spreads, color="orange", linewidth=0.5)
    ax1.set_title("Market Spread")
    ax1.set_ylabel("bps")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    peak = 0.0
    dd = []
    for e in events:
        peak = max(peak, e.total_pnl)
        dd.append(peak - e.total_pnl)

    ax2.fill_between(times, 0, dd, color="red", alpha=0.3)
    ax2.plot(times, dd, color="red", linewidth=0.5)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("USD")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _export_fills_csv(fills: list[SimulatedFill], path: str) -> None:
    """Export des fills simulés en CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ms", "coin", "side", "price", "size",
            "notional", "fee", "mid_at_fill",
        ])
        for fill in fills:
            writer.writerow([
                fill.timestamp_ms, fill.coin, fill.side,
                fill.price, round(fill.size, 6),
                round(fill.price * fill.size, 4),
                round(fill.fee, 6), fill.mid_at_fill,
            ])
