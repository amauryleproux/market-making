#!/usr/bin/env python3
"""
🤖 Hyperliquid Market Maker Bot

Place des ordres ALO (post-only) bid/ask pour capturer les maker rebates.

Usage :
    python scripts/run_mm.py --dry-run                    # Test sans trader
    python scripts/run_mm.py --coins ETH,BTC              # Market-make ETH et BTC
    python scripts/run_mm.py --size 100 --spread 0.08     # $100/ordre, 0.08% spread
    python scripts/run_mm.py                              # Mode live (attention!)
"""

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from config import config
from src.client.hyperliquid import HyperliquidClient
from src.strategies.market_maker import MarketMaker
from src.utils.logger import setup_logging, get_logger

console = Console()
log = get_logger("main")

# Graceful shutdown
_running = True


def signal_handler(sig, frame):
    global _running
    console.print("\n[yellow]⏹ Arrêt en cours...[/]")
    _running = False


signal.signal(signal.SIGINT, signal_handler)


def make_status_table(mm: MarketMaker, account_equity: float) -> Table:
    """Crée le tableau de statut Rich."""
    table = Table(title="📊 Hyperliquid Market Maker", expand=True)
    table.add_column("Coin", style="cyan", width=8)
    table.add_column("Mid", justify="right", width=12)
    table.add_column("Bids", justify="center", width=6)
    table.add_column("Asks", justify="center", width=6)
    table.add_column("Position $", justify="right", width=12)
    table.add_column("PnL", justify="right", width=10)

    for coin, state in mm.states.items():
        pos_color = "green" if state.position_usd > 0 else "red" if state.position_usd < 0 else "white"
        table.add_row(
            coin,
            f"${state.mid_price:,.2f}",
            str(state.active_bids),
            str(state.active_asks),
            f"[{pos_color}]${state.position_usd:,.2f}[/]",
            "—",
        )

    table.add_section()
    status = mm.get_status()
    table.add_row(
        "[bold]TOTAL[/]",
        "",
        "",
        "",
        f"[bold]${status['total_position_usd']:,.2f}[/]",
        f"Equity: ${account_equity:,.2f}",
    )

    return table


@click.command()
@click.option("--coins", default=None, help="Coins à MM (ex: ETH,BTC)")
@click.option("--size", default=None, type=float, help="Taille d'ordre en USD")
@click.option("--spread", default=None, type=float, help="Spread minimum en %")
@click.option("--levels", default=None, type=int, help="Nombre de niveaux")
@click.option("--max-pos", default=None, type=float, help="Position max par coin (USD)")
@click.option("--interval", default=None, type=float, help="Intervalle entre updates (s)")
@click.option("--dry-run", is_flag=True, default=False, help="Mode simulation")
@click.option("--testnet", is_flag=True, default=False, help="Utiliser le testnet")
def main(coins, size, spread, levels, max_pos, interval, dry_run, testnet):
    """Lance le market maker Hyperliquid."""

    # Override config
    if coins:
        config.coins = coins
    if size:
        config.order_size_usd = size
    if spread:
        config.min_spread_pct = spread
    if levels:
        config.num_levels = levels
    if max_pos:
        config.max_position_usd = max_pos
    if interval:
        config.update_interval = interval
    if dry_run:
        config.dry_run = True
    if testnet:
        config.mainnet = False

    setup_logging(config.log_level)

    # Banner
    mode = "🧪 DRY RUN" if config.dry_run else "🔴 LIVE TRADING"
    network = "testnet" if not config.mainnet else "mainnet"
    console.print(Panel(
        f"[bold]{mode}[/bold] on [cyan]{network}[/]\n"
        f"Coins: [green]{config.coins}[/]\n"
        f"Order size: [yellow]${config.order_size_usd}[/] | "
        f"Spread: [yellow]{config.min_spread_pct}%[/] | "
        f"Levels: [yellow]{config.num_levels}[/]\n"
        f"Max position: [yellow]${config.max_position_usd}[/] per coin | "
        f"Interval: [yellow]{config.update_interval}s[/]",
        title="🤖 Hyperliquid Market Maker",
    ))

    # Vérifications
    if not config.dry_run:
        if not config.secret_key:
            console.print("[red]❌ HL_SECRET_KEY manquant dans .env[/]")
            return
        if not config.account_address:
            console.print("[red]❌ HL_ACCOUNT_ADDRESS manquant dans .env[/]")
            return

    # Initialiser le client
    try:
        client = HyperliquidClient(
            secret_key=config.secret_key,
            account_address=config.account_address,
            mainnet=config.mainnet,
        )
    except Exception as e:
        console.print(f"[red]❌ Erreur de connexion: {e}[/]")
        return

    # Vérifier le compte
    if not config.dry_run:
        try:
            account = client.get_account_state()
            console.print(f"[green]✅ Connecté — Equity: ${account.equity:,.2f} | Available: ${account.available_balance:,.2f}[/]")
            if account.positions:
                for pos in account.positions:
                    console.print(f"   Position: {pos.coin} {pos.size:+.4f} (${pos.size * client.get_mid(pos.coin):,.2f}) PnL: ${pos.unrealized_pnl:,.2f}")
        except Exception as e:
            console.print(f"[red]❌ Erreur d'accès au compte: {e}[/]")
            return
    else:
        console.print("[yellow]⚡ Mode dry-run — pas de connexion nécessaire[/]")
        # Quand même charger les prix
        if config.account_address:
            try:
                client = HyperliquidClient(
                    secret_key="",
                    account_address=config.account_address,
                    mainnet=config.mainnet,
                )
            except Exception:
                pass

    # Vérifier que les coins existent
    try:
        meta = client.get_meta()
        valid_coins = []
        for coin in config.coin_list:
            try:
                client.get_asset_index(coin)
                valid_coins.append(coin)
            except ValueError:
                console.print(f"[yellow]⚠ Coin '{coin}' non trouvé, ignoré[/]")

        if not valid_coins:
            console.print("[red]❌ Aucun coin valide[/]")
            return

        config.coins = ",".join(valid_coins)
        console.print(f"[green]✅ Coins validés: {valid_coins}[/]")
    except Exception as e:
        console.print(f"[red]❌ Erreur API: {e}[/]")
        return

    # Initialiser le market maker
    mm = MarketMaker(
        client=client,
        coins=valid_coins,
        order_size_usd=config.order_size_usd,
        min_spread_pct=config.min_spread_pct,
        num_levels=config.num_levels,
        level_spacing_pct=config.level_spacing_pct,
        max_position_usd=config.max_position_usd,
        inventory_skew=config.inventory_skew,
        dry_run=config.dry_run,
    )

    # Boucle principale
    console.print(f"\n[green]🚀 Bot démarré — Ctrl+C pour arrêter[/]\n")

    iteration = 0
    while _running:
        try:
            mm.update_quotes()
            iteration += 1

            # Afficher le statut
            status = mm.get_status()
            equity = 0
            if not config.dry_run:
                try:
                    equity = client.get_account_state().equity
                except Exception:
                    pass

            log.info(
                "mm_status",
                iteration=iteration,
                coins=len(valid_coins),
                total_orders=status["total_orders"],
                position_usd=status["total_position_usd"],
                equity=round(equity, 2),
            )

            # Attendre
            for _ in range(int(config.update_interval * 10)):
                if not _running:
                    break
                time.sleep(0.1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error("loop_error", error=str(e))
            time.sleep(5)

    # Cleanup
    if not config.dry_run:
        console.print("[yellow]🧹 Annulation de tous les ordres...[/]")
        try:
            for coin in valid_coins:
                client.cancel_coin_orders(coin)
            console.print("[green]✅ Tous les ordres annulés[/]")
        except Exception as e:
            console.print(f"[red]❌ Erreur d'annulation: {e}[/]")

    console.print("[bold]👋 Bot arrêté.[/]")


if __name__ == "__main__":
    main()
