#!/usr/bin/env python3
"""
Bot multi-paires Hyperliquid Market Maker.

Charge la config depuis pairs_config.json, lance une tâche
asyncio par paire, avec dashboard en temps réel.

Usage :
    python scripts/bot.py --dry-run                # Simulation
    python scripts/bot.py                          # Mode live
    python scripts/bot.py --pair MEGA              # Une seule paire
    python scripts/bot.py --config custom.json     # Config custom
    python scripts/bot.py --no-dashboard           # Sans dashboard
"""

import asyncio
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.panel import Panel

from config import config
from src.engine.mm_engine import MMEngine
from src.utils.logger import setup_logging

console = Console()


@click.command()
@click.option("--config", "config_path", default="pairs_config.json", help="Chemin config JSON")
@click.option("--dry-run", is_flag=True, default=False, help="Mode simulation")
@click.option("--testnet", is_flag=True, default=False, help="Utiliser le testnet")
@click.option("--no-dashboard", is_flag=True, default=False, help="Désactiver le dashboard")
@click.option("--pair", default=None, help="Lancer sur une seule paire (ex: MEGA)")
def main(config_path, dry_run, testnet, no_dashboard, pair):
    """Lance le market maker multi-paires."""
    setup_logging(config.log_level)

    is_dry_run = dry_run or config.dry_run
    is_mainnet = config.mainnet and not testnet

    # Banner
    mode = "DRY RUN" if is_dry_run else "LIVE TRADING"
    network = "mainnet" if is_mainnet else "testnet"
    console.print(Panel(
        f"[bold]{mode}[/bold] on [cyan]{network}[/]\n"
        f"Config: [green]{config_path}[/]\n"
        f"Dashboard: [yellow]{'OFF' if no_dashboard else 'ON'}[/]"
        + (f"\nPaire: [yellow]{pair.upper()}[/]" if pair else ""),
        title="Hyperliquid Multi-Pair Market Maker",
    ))

    # Vérifications auth
    if not is_dry_run:
        if not config.secret_key:
            console.print("[red]HL_SECRET_KEY manquant dans .env[/]")
            return
        if not config.account_address:
            console.print("[red]HL_ACCOUNT_ADDRESS manquant dans .env[/]")
            return

    try:
        engine = MMEngine(
            config_path=config_path,
            dry_run=is_dry_run,
            mainnet=is_mainnet,
            secret_key=config.secret_key if not is_dry_run else "",
            account_address=config.account_address,
            show_dashboard=not no_dashboard,
            pair_filter=pair,
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Erreur config: {e}[/]")
        return

    console.print("[green]Bot démarré — Ctrl+C pour arrêter[/]\n")

    # Sur Windows, utiliser SelectorEventLoop pour compatibilité
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        pass

    console.print("\n[bold]Bot arrêté.[/]")


if __name__ == "__main__":
    main()
