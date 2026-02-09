"""Calcul des métriques de performance depuis un BacktestResult."""

import bisect
from dataclasses import dataclass
from typing import Optional

from backtesting.engine.backtest_engine import BacktestResult, TickEvent
from backtesting.engine.matching import SimulatedFill


@dataclass
class BacktestMetrics:
    """Métriques complètes d'un backtest."""

    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    pnl_per_hour: float = 0.0
    max_drawdown_usd: float = 0.0

    # Trading
    total_fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0
    total_volume_usd: float = 0.0
    avg_fill_size_usd: float = 0.0
    profit_factor: float = 0.0

    # Adverse selection
    avg_pnl_1min_bps: float = 0.0
    avg_pnl_5min_bps: float = 0.0
    avg_pnl_15min_bps: float = 0.0

    # Inventory
    avg_inventory_usd: float = 0.0
    max_inventory_usd: float = 0.0
    pct_time_flat: float = 0.0
    pct_time_long: float = 0.0
    pct_time_short: float = 0.0

    # Spread
    avg_spread_captured_bps: float = 0.0
    avg_market_spread_bps: float = 0.0
    spread_capture_ratio: float = 0.0

    # Execution
    total_orders_placed: int = 0
    total_orders_rejected: int = 0
    fill_rate: float = 0.0

    # Timing
    backtest_duration_sec: float = 0.0
    total_ticks: int = 0
    ticks_per_second: float = 0.0

    def to_dict(self) -> dict:
        """Convertit en dict plat pour export CSV."""
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        """Résumé lisible."""
        lines = [
            "",
            "=" * 60,
            "  BACKTEST RESULTS",
            "=" * 60,
            f"  Total PnL:          ${self.total_pnl:+.4f}",
            f"    Realized:         ${self.realized_pnl:+.4f}",
            f"    Unrealized:       ${self.unrealized_pnl:+.4f}",
            f"    Fees (rebates):   ${self.total_fees:+.4f}",
            f"  PnL/hour:           ${self.pnl_per_hour:+.4f}",
            f"  Max drawdown:       ${self.max_drawdown_usd:.4f}",
            "",
            f"  Fills:              {self.total_fills} ({self.buy_fills}B / {self.sell_fills}S)",
            f"  Volume:             ${self.total_volume_usd:,.2f}",
            f"  Fill rate:          {self.fill_rate:.1%}",
            f"  Profit factor:      {self.profit_factor:.2f}",
            "",
            f"  Adverse selection (bps post-fill):",
            f"    1 min:            {self.avg_pnl_1min_bps:+.2f}",
            f"    5 min:            {self.avg_pnl_5min_bps:+.2f}",
            f"    15 min:           {self.avg_pnl_15min_bps:+.2f}",
            "",
            f"  Inventory avg:      ${self.avg_inventory_usd:.2f}",
            f"  Inventory max:      ${self.max_inventory_usd:.2f}",
            f"  Time flat:          {self.pct_time_flat:.1%}",
            "",
            f"  Spread captured:    {self.avg_spread_captured_bps:.1f} bps",
            f"  Market spread:      {self.avg_market_spread_bps:.1f} bps",
            f"  Capture ratio:      {self.spread_capture_ratio:.1%}",
            "",
            f"  Duration:           {self.backtest_duration_sec:.1f}s ({self.total_ticks} ticks, {self.ticks_per_second:.0f} t/s)",
            "=" * 60,
        ]
        return "\n".join(lines)


def compute_metrics(result: BacktestResult) -> BacktestMetrics:
    """Calcule toutes les métriques depuis un BacktestResult."""
    m = BacktestMetrics()

    events = result.tick_events
    fills = result.all_fills
    tracker = result.pnl_tracker

    if not events:
        return m

    # --- Timing ---
    m.backtest_duration_sec = result.elapsed_seconds
    m.total_ticks = result.total_ticks
    m.ticks_per_second = (
        result.total_ticks / result.elapsed_seconds
        if result.elapsed_seconds > 0
        else 0
    )

    # --- PnL ---
    last_event = events[-1]
    m.realized_pnl = last_event.realized_pnl
    m.unrealized_pnl = last_event.unrealized_pnl
    m.total_pnl = last_event.total_pnl
    m.total_fees = tracker.state.total_fees

    duration_hours = (events[-1].timestamp_ms - events[0].timestamp_ms) / 3_600_000
    m.pnl_per_hour = m.total_pnl / duration_hours if duration_hours > 0 else 0

    # Max drawdown
    peak = 0.0
    max_dd = 0.0
    for e in events:
        peak = max(peak, e.total_pnl)
        dd = peak - e.total_pnl
        max_dd = max(max_dd, dd)
    m.max_drawdown_usd = max_dd

    # --- Trading ---
    m.total_fills = len(fills)
    m.buy_fills = sum(1 for f in fills if f.side == "buy")
    m.sell_fills = sum(1 for f in fills if f.side == "sell")
    m.total_volume_usd = sum(f.price * f.size for f in fills)
    m.avg_fill_size_usd = m.total_volume_usd / m.total_fills if m.total_fills > 0 else 0

    # Profit factor
    gross_profit = sum(f.fee for f in fills if f.fee < 0)  # rebates are negative fees
    gross_loss = sum(f.fee for f in fills if f.fee > 0)
    # Also consider realized PnL from round trips
    m.profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    # --- Execution ---
    m.total_orders_placed = result.total_orders_placed
    m.total_orders_rejected = result.total_orders_rejected
    m.fill_rate = m.total_fills / m.total_orders_placed if m.total_orders_placed > 0 else 0

    # --- Adverse selection ---
    _compute_adverse_selection(fills, events, m)

    # --- Inventory ---
    _compute_inventory_metrics(events, m)

    # --- Spread ---
    _compute_spread_metrics(fills, events, m)

    return m


def _compute_adverse_selection(
    fills: list[SimulatedFill],
    events: list[TickEvent],
    m: BacktestMetrics,
) -> None:
    """PnL mark-to-market à 1/5/15 min après chaque fill."""
    if not fills or not events:
        return

    timestamps = [e.timestamp_ms for e in events]
    mids = [e.mid_price for e in events]

    horizons_ms = [60_000, 300_000, 900_000]
    results = {h: [] for h in horizons_ms}

    for fill in fills:
        for horizon_ms in horizons_ms:
            target_ts = fill.timestamp_ms + horizon_ms
            idx = bisect.bisect_left(timestamps, target_ts)
            if idx >= len(timestamps):
                continue

            future_mid = mids[idx]
            if fill.mid_at_fill <= 0:
                continue

            # PnL en bps depuis le fill
            if fill.side == "buy":
                pnl_bps = ((future_mid - fill.price) / fill.price) * 10_000
            else:
                pnl_bps = ((fill.price - future_mid) / fill.price) * 10_000

            results[horizon_ms].append(pnl_bps)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    m.avg_pnl_1min_bps = _avg(results[60_000])
    m.avg_pnl_5min_bps = _avg(results[300_000])
    m.avg_pnl_15min_bps = _avg(results[900_000])


def _compute_inventory_metrics(events: list[TickEvent], m: BacktestMetrics) -> None:
    """Métriques d'inventaire."""
    if not events:
        return

    inv_usd_values = [abs(e.inventory_usd) for e in events]
    m.avg_inventory_usd = sum(inv_usd_values) / len(inv_usd_values)
    m.max_inventory_usd = max(inv_usd_values)

    flat_threshold = 1.0  # USD
    n_flat = sum(1 for e in events if abs(e.inventory_usd) < flat_threshold)
    n_long = sum(1 for e in events if e.inventory_usd >= flat_threshold)
    n_short = sum(1 for e in events if e.inventory_usd <= -flat_threshold)
    total = len(events)

    m.pct_time_flat = n_flat / total
    m.pct_time_long = n_long / total
    m.pct_time_short = n_short / total


def _compute_spread_metrics(
    fills: list[SimulatedFill],
    events: list[TickEvent],
    m: BacktestMetrics,
) -> None:
    """Analyse du spread capturé vs marché."""
    if not events:
        return

    # Spread marché moyen
    spreads = [e.spread_bps for e in events if e.spread_bps > 0]
    m.avg_market_spread_bps = sum(spreads) / len(spreads) if spreads else 0

    # Spread capturé par fill
    if fills:
        captured = []
        for f in fills:
            if f.mid_at_fill > 0:
                c_bps = abs(f.price - f.mid_at_fill) / f.mid_at_fill * 10_000
                captured.append(c_bps)
        m.avg_spread_captured_bps = sum(captured) / len(captured) if captured else 0

    m.spread_capture_ratio = (
        m.avg_spread_captured_bps / m.avg_market_spread_bps
        if m.avg_market_spread_bps > 0
        else 0
    )
