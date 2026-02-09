#!/usr/bin/env python3
"""
Test rapide de connexion à Hyperliquid.
Vérifie l'API, les prix, et l'auth.

Usage:
    python scripts/test_connection.py
    python scripts/test_connection.py --testnet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command()
@click.option("--testnet", is_flag=True, help="Utiliser le testnet")
def main(testnet):
    from config import config
    if testnet:
        config.mainnet = False

    network = "testnet" if not config.mainnet else "mainnet"
    console.print(f"\n[bold]🔍 Test connexion Hyperliquid ({network})[/]\n")

    # 1. API publique — prix
    console.print("[cyan]1. Test API publique (prix)...[/]")
    try:
        from src.client.hyperliquid import HyperliquidClient
        client = HyperliquidClient(
            secret_key="",
            account_address=config.account_address or "0x0000000000000000000000000000000000000000",
            mainnet=config.mainnet,
        )

        mids = client.get_all_mids()
        top_coins = ["BTC", "ETH", "SOL", "DOGE", "HYPE"]
        table = Table(title="Prix actuels")
        table.add_column("Coin", style="cyan")
        table.add_column("Mid Price", justify="right")

        for coin in top_coins:
            if coin in mids:
                table.add_row(coin, f"${mids[coin]:,.2f}")

        console.print(table)
        console.print(f"[green]   ✅ {len(mids)} coins disponibles[/]")
    except Exception as e:
        console.print(f"[red]   ❌ Erreur: {e}[/]")
        return

    # 2. Métadonnées
    console.print("\n[cyan]2. Métadonnées...[/]")
    try:
        meta = client.get_meta()
        console.print(f"[green]   ✅ {len(meta['universe'])} perp markets[/]")
    except Exception as e:
        console.print(f"[red]   ❌ Erreur: {e}[/]")

    # 3. Orderbook
    console.print("\n[cyan]3. Orderbook ETH...[/]")
    try:
        bid, ask = client.get_best_bid_ask("ETH")
        spread = (ask - bid) / ((ask + bid) / 2) * 100
        console.print(f"[green]   ✅ Bid: ${bid:,.2f} | Ask: ${ask:,.2f} | Spread: {spread:.4f}%[/]")
    except Exception as e:
        console.print(f"[red]   ❌ Erreur: {e}[/]")

    # 4. Auth (si configuré)
    if config.secret_key and config.account_address:
        console.print("\n[cyan]4. Authentification...[/]")
        try:
            auth_client = HyperliquidClient(
                secret_key=config.secret_key,
                account_address=config.account_address,
                mainnet=config.mainnet,
            )
            account = auth_client.get_account_state()
            console.print(f"[green]   ✅ Equity: ${account.equity:,.2f}[/]")
            console.print(f"[green]   ✅ Available: ${account.available_balance:,.2f}[/]")

            if account.positions:
                console.print(f"[green]   ✅ Positions ouvertes: {len(account.positions)}[/]")
                for pos in account.positions:
                    console.print(f"       {pos.coin}: {pos.size:+.4f} @ ${pos.entry_px:,.2f} (PnL: ${pos.unrealized_pnl:,.2f})")
            else:
                console.print(f"[green]   ✅ Aucune position ouverte[/]")

            orders = auth_client.get_open_orders()
            console.print(f"[green]   ✅ Ordres ouverts: {len(orders)}[/]")

        except Exception as e:
            console.print(f"[red]   ❌ Auth erreur: {e}[/]")
    else:
        console.print("\n[yellow]4. Auth — HL_SECRET_KEY/HL_ACCOUNT_ADDRESS non configurés, skip[/]")

    console.print("\n[bold green]✅ Test terminé![/]\n")


if __name__ == "__main__":
    main()
