"""Dashboard en temps réel avec Rich Live.

Affiche un tableau par paire avec spread, inventaire, PnL,
et une ligne TOTAL en bas.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.strategies.pair_market_maker import PairState
from src.monitoring.pnl_tracker import PairPnLTracker


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def _pnl_style(value: float) -> str:
    if value > 0:
        return f"[green]+${value:,.4f}[/]"
    elif value < 0:
        return f"[red]-${abs(value):,.4f}[/]"
    return "$0.00"


class Dashboard:
    """Composant dashboard pour le moteur MM."""

    def __init__(self, console: Optional[Console] = None):
        self._console = console or Console()
        self._live: Optional[Live] = None

    def start(self) -> None:
        self._live = Live(
            self._build_table({}, {}),
            console=self._console,
            refresh_per_second=1,
        )
        self._live.start()

    def update(
        self,
        pair_states: dict[str, PairState],
        pnl_trackers: dict[str, PairPnLTracker],
        global_info: Optional[dict] = None,
    ) -> None:
        if self._live:
            self._live.update(self._build_table(pair_states, pnl_trackers, global_info))

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    def _build_table(
        self,
        pair_states: dict[str, PairState],
        pnl_trackers: dict[str, PairPnLTracker],
        global_info: Optional[dict] = None,
    ) -> Table:
        info = global_info or {}
        mode = "[yellow]DRY RUN[/]" if info.get("dry_run") else "[red bold]LIVE[/]"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        uptime = _format_duration(info.get("uptime_sec", 0))
        ks = info.get("kill_switch", "OK")
        ks_style = "[green]OK[/]" if ks == "OK" else "[red bold]TRIGGERED[/]"

        title = (
            f"MARKET MAKING DASHBOARD  {mode}  |  {now}  |  "
            f"Uptime: {uptime}  |  Kill switch: {ks_style}"
        )

        table = Table(title=title, expand=True, show_lines=True)
        table.add_column("Paire", style="cyan bold", width=8)
        table.add_column("Mid", justify="right", width=12)
        table.add_column("Spread", justify="right", width=10)
        table.add_column("B/A", justify="center", width=6)
        table.add_column("Inventaire", justify="right", width=14)
        table.add_column("Realized", justify="right", width=12)
        table.add_column("Unrealized", justify="right", width=12)
        table.add_column("Total PnL", justify="right", width=12)
        table.add_column("Fills", justify="right", width=6)
        table.add_column("Status", justify="center", width=8)

        total_realized = 0.0
        total_unrealized = 0.0
        total_inventory_abs = 0.0
        total_fills = 0

        for coin in sorted(pair_states.keys()):
            state = pair_states[coin]
            tracker = pnl_trackers.get(coin)

            realized = tracker.state.realized_pnl if tracker else 0
            unrealized = tracker.get_unrealized_pnl(state.mid_price) if tracker else 0
            fees = tracker.state.total_fees if tracker else 0
            total = realized + unrealized - fees

            total_realized += realized
            total_unrealized += unrealized
            total_inventory_abs += abs(state.inventory_usd)
            total_fills += state.fills_count

            # Couleurs inventaire
            if state.inventory_usd > 0:
                inv_str = f"[green]+${state.inventory_usd:,.2f}[/]"
            elif state.inventory_usd < 0:
                inv_str = f"[red]-${abs(state.inventory_usd):,.2f}[/]"
            else:
                inv_str = "$0"

            # Status
            if not state.is_running:
                status = f"[red]{state.error or 'STOP'}[/]"
            elif state.error:
                status = f"[yellow]ERR[/]"
            else:
                status = "[green]OK[/]"

            table.add_row(
                coin,
                f"${state.mid_price:,.4f}" if state.mid_price > 0 else "—",
                f"{state.spread_bps:.1f} bp" if state.spread_bps > 0 else "—",
                f"{state.active_bids}/{state.active_asks}",
                inv_str,
                _pnl_style(realized),
                _pnl_style(unrealized),
                _pnl_style(total),
                str(state.fills_count),
                status,
            )

        # Ligne TOTAL
        total_total = total_realized + total_unrealized
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/]",
            "",
            "",
            "",
            f"[bold]${total_inventory_abs:,.2f}[/]",
            f"[bold]{_pnl_style(total_realized)}[/]",
            f"[bold]{_pnl_style(total_unrealized)}[/]",
            f"[bold]{_pnl_style(total_total)}[/]",
            f"[bold]{total_fills}[/]",
            "",
        )

        return table
