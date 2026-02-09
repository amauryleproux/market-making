"""Moteur principal du bot multi-paires.

Charge la configuration, lance une tâche asyncio par paire,
surveille l'exposition totale et le kill switch.
"""

import asyncio
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.client.hyperliquid import HyperliquidClient
from src.client.async_wrapper import AsyncHyperliquidClient
from src.strategies.pair_market_maker import PairMarketMaker, PairConfig, PairState
from src.monitoring.db_logger import DBLogger
from src.monitoring.pnl_tracker import PairPnLTracker
from src.monitoring.dashboard import Dashboard
from src.utils.logger import get_logger

log = get_logger("engine")


@dataclass
class GlobalConfig:
    """Configuration globale chargée depuis pairs_config.json."""
    max_total_inventory_usd: float = 2000
    max_drawdown_pct: float = 5.0
    kill_switch_loss_usd: float = 200
    dashboard_refresh_sec: float = 10
    snapshot_interval_sec: float = 60
    db_path: str = "data/trading_log.db"


class MMEngine:
    """Moteur orchestrant toutes les paires."""

    def __init__(
        self,
        config_path: str = "pairs_config.json",
        dry_run: bool = True,
        mainnet: bool = True,
        secret_key: str = "",
        account_address: str = "",
        show_dashboard: bool = True,
        pair_filter: Optional[str] = None,
    ):
        self._config_path = config_path
        self._dry_run = dry_run
        self._show_dashboard = show_dashboard
        self._pair_filter = pair_filter.upper() if pair_filter else None

        # Charger la config JSON
        self._global_config = GlobalConfig()
        self._pair_configs: dict[str, PairConfig] = {}
        self._load_config(config_path)

        # Client Hyperliquid (un seul, partagé)
        self._sync_client = HyperliquidClient(secret_key, account_address, mainnet)
        self._client = AsyncHyperliquidClient(self._sync_client)

        # Monitoring
        self._db = DBLogger(self._global_config.db_path)
        self._pnl_trackers: dict[str, PairPnLTracker] = {}
        self._pair_makers: dict[str, PairMarketMaker] = {}

        # Kill switch
        self._kill_event = asyncio.Event()
        self._session_start = datetime.now(timezone.utc).isoformat()
        self._shutting_down = False

        # Dashboard
        self._dashboard: Optional[Dashboard] = None

    def _load_config(self, config_path: str) -> None:
        """Charge pairs_config.json."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config introuvable : {config_path}")

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        # Global config
        g = raw.get("global", {})
        self._global_config = GlobalConfig(
            max_total_inventory_usd=g.get("max_total_inventory_usd", 2000),
            max_drawdown_pct=g.get("max_drawdown_pct", 5.0),
            kill_switch_loss_usd=g.get("kill_switch_loss_usd", 200),
            dashboard_refresh_sec=g.get("dashboard_refresh_sec", 10),
            snapshot_interval_sec=g.get("snapshot_interval_sec", 60),
            db_path=g.get("db_path", "data/trading_log.db"),
        )

        # Pair configs
        for coin, pc in raw.get("pairs", {}).items():
            coin = coin.upper()

            # Filtrer si --pair spécifié
            if self._pair_filter and coin != self._pair_filter:
                continue

            if not pc.get("enabled", True):
                continue

            self._pair_configs[coin] = PairConfig(
                coin=coin,
                enabled=True,
                spread_bps=pc.get("spread_bps", 10),
                order_size_usd=pc.get("order_size_usd", 50),
                num_levels=pc.get("num_levels", 3),
                level_spacing_bps=pc.get("level_spacing_bps", 5),
                max_inventory_usd=pc.get("max_inventory_usd", 500),
                inventory_skew_factor=pc.get("inventory_skew_factor", 0.5),
                refresh_interval_sec=pc.get("refresh_interval_sec", 5),
                take_profit_enabled=pc.get("take_profit_enabled", True),
                take_profit_pct=pc.get("take_profit_pct", 0.5),
                take_profit_min_usd=pc.get("take_profit_min_usd", 0.10),
            )

        if not self._pair_configs:
            raise ValueError("Aucune paire activée dans la config")

        log.info("config_loaded",
                 pairs=list(self._pair_configs.keys()),
                 kill_switch=self._global_config.kill_switch_loss_usd)

    async def run(self) -> None:
        """Point d'entrée principal."""
        # Valider les coins sur l'exchange
        log.info("validating_coins")
        valid_coins = []
        for coin in self._pair_configs:
            try:
                await self._client.get_asset_index(coin)
                valid_coins.append(coin)
            except ValueError:
                log.warning("coin_not_found", coin=coin)

        if not valid_coins:
            log.error("no_valid_coins")
            return

        # Retirer les paires invalides
        self._pair_configs = {c: self._pair_configs[c] for c in valid_coins}

        # Créer les composants par paire
        for coin, pc in self._pair_configs.items():
            tracker = PairPnLTracker(coin)
            self._pnl_trackers[coin] = tracker

            maker = PairMarketMaker(
                config=pc,
                client=self._client,
                db=self._db,
                pnl=tracker,
                dry_run=self._dry_run,
                kill_event=self._kill_event,
            )
            self._pair_makers[coin] = maker

        log.info("engine_ready",
                 pairs=valid_coins,
                 dry_run=self._dry_run)

        # Setup signal handlers
        self._setup_signal_handlers()

        # Démarrer le dashboard
        if self._show_dashboard:
            self._dashboard = Dashboard()
            self._dashboard.start()

        # Lancer les tâches
        tasks = []
        for coin, maker in self._pair_makers.items():
            tasks.append(asyncio.create_task(maker.run(), name=f"mm_{coin}"))

        tasks.append(asyncio.create_task(self._monitor_loop(), name="monitor"))

        if self._show_dashboard:
            tasks.append(asyncio.create_task(self._dashboard_loop(), name="dashboard"))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def _monitor_loop(self) -> None:
        """Boucle de surveillance globale (kill switch)."""
        while not self._kill_event.is_set():
            total_pnl = 0.0
            total_inventory = 0.0

            for coin, maker in self._pair_makers.items():
                tracker = self._pnl_trackers[coin]
                mid = maker.state.mid_price
                if mid > 0:
                    total_pnl += tracker.get_total_pnl(mid)
                    total_inventory += abs(tracker.current_inventory * mid)

            # Kill switch : perte totale > seuil
            if total_pnl < -self._global_config.kill_switch_loss_usd:
                log.critical(
                    "kill_switch_triggered",
                    reason="max_loss",
                    total_pnl=round(total_pnl, 2),
                    threshold=-self._global_config.kill_switch_loss_usd,
                )
                self._kill_event.set()
                break

            # Warning exposition totale
            if total_inventory > self._global_config.max_total_inventory_usd:
                log.warning(
                    "exposure_warning",
                    total_inventory_usd=round(total_inventory, 2),
                    max=self._global_config.max_total_inventory_usd,
                )

            try:
                await asyncio.wait_for(self._kill_event.wait(), timeout=5)
                break
            except asyncio.TimeoutError:
                pass

    async def _dashboard_loop(self) -> None:
        """Boucle de rafraîchissement du dashboard."""
        while not self._kill_event.is_set():
            if self._dashboard:
                pair_states = {coin: maker.state for coin, maker in self._pair_makers.items()}
                uptime = time.time() - self._start_time if hasattr(self, '_start_time') else 0
                total_fills = sum(m.state.fills_count for m in self._pair_makers.values())
                self._dashboard.update(
                    pair_states=pair_states,
                    pnl_trackers=self._pnl_trackers,
                    global_info={
                        "uptime_sec": uptime,
                        "total_fills": total_fills,
                        "kill_switch": "OK" if not self._kill_event.is_set() else "TRIGGERED",
                        "dry_run": self._dry_run,
                    },
                )

            try:
                await asyncio.wait_for(
                    self._kill_event.wait(),
                    timeout=self._global_config.dashboard_refresh_sec,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def shutdown(self) -> None:
        """Arrêt propre : cancel tout, flush les logs, écrit les summaries."""
        if self._shutting_down:
            return
        self._shutting_down = True

        log.info("shutdown_started")
        self._kill_event.set()

        # Attendre un peu que les paires nettoient
        await asyncio.sleep(2)

        # Écrire les résumés PnL dans la DB
        session_end = datetime.now(timezone.utc).isoformat()
        for coin, tracker in self._pnl_trackers.items():
            # Calculer le spread moyen capturé depuis la DB
            fills = self._db.query_fills(pair=coin, since=self._session_start)
            avg_spread = 0.0
            if fills:
                spreads = [f["spread_captured_bps"] for f in fills]
                avg_spread = sum(spreads) / len(spreads)

            await self._db.log_session_summary(
                session_start=self._session_start,
                session_end=session_end,
                pair=coin,
                fills_count=tracker.fills_count,
                volume_usd=tracker.state.total_volume_usd,
                gross_pnl=tracker.state.realized_pnl,
                total_fees=tracker.state.total_fees,
                net_pnl=tracker.state.realized_pnl - tracker.state.total_fees,
                avg_spread_captured_bps=avg_spread,
                max_inventory_usd=tracker.state.max_inventory_usd,
                max_drawdown_usd=tracker.state.max_drawdown_usd,
            )

        # Fermer le dashboard
        if self._dashboard:
            self._dashboard.stop()

        self._db.close()
        log.info("shutdown_complete")

    def _setup_signal_handlers(self) -> None:
        """Installe les handlers Ctrl+C."""
        self._start_time = time.time()

        if sys.platform == "win32":
            # Sur Windows, signal.signal est le seul moyen
            def _handler(sig, frame):
                if not self._shutting_down:
                    log.info("signal_received", signal=sig)
                    self._kill_event.set()
            signal.signal(signal.SIGINT, _handler)
        else:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: self._kill_event.set())
