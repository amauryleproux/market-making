"""Moteur de backtesting event-driven.

Itère sur les snapshots L2 chronologiquement, appelle la stratégie
à chaque tick, simule les fills via le matching engine, et accumule
les résultats pour l'analyse.
"""

import time as wall_time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtesting.data.storage import StorageManager
from backtesting.engine.matching import MatchingEngine, SimulatedFill
from backtesting.engine.orderbook import OrderBookSnapshot
from backtesting.strategy.mm_strategy import BacktestStrategy
from src.monitoring.pnl_tracker import PairPnLTracker
from src.strategies.pair_market_maker import PairConfig
from src.utils.logger import get_logger

log = get_logger("bt.engine")


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration d'un run de backtest."""

    coin: str
    start_date: str
    end_date: str
    data_dir: str = "backtesting/data/market_data"

    # Paramètres stratégie (mappés vers PairConfig)
    spread_bps: float = 10.0
    order_size_usd: float = 50.0
    num_levels: int = 3
    level_spacing_bps: float = 5.0
    max_inventory_usd: float = 500.0
    inventory_skew_factor: float = 0.5

    # Take-profit
    take_profit_enabled: bool = False
    take_profit_pct: float = 0.5
    take_profit_min_usd: float = 0.10

    # Matching engine
    latency_ms: int = 300
    maker_fee_bps: float = -2.0
    aggressive_fill: bool = False

    @classmethod
    def from_json(cls, path: str) -> "BacktestConfig":
        import json
        from pathlib import Path

        with open(Path(path), encoding="utf-8") as f:
            raw = json.load(f)
        return cls(**raw)

    def to_pair_config(self) -> PairConfig:
        return PairConfig(
            coin=self.coin,
            enabled=True,
            spread_bps=self.spread_bps,
            order_size_usd=self.order_size_usd,
            num_levels=self.num_levels,
            level_spacing_bps=self.level_spacing_bps,
            max_inventory_usd=self.max_inventory_usd,
            inventory_skew_factor=self.inventory_skew_factor,
            refresh_interval_sec=2.0,
            take_profit_enabled=self.take_profit_enabled,
            take_profit_pct=self.take_profit_pct,
            take_profit_min_usd=self.take_profit_min_usd,
        )


# ------------------------------------------------------------------
# Tick events
# ------------------------------------------------------------------

@dataclass
class TickEvent:
    """Données enregistrées à chaque tick."""
    timestamp_ms: int
    mid_price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    inventory: float
    inventory_usd: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    num_fills: int
    num_orders: int


@dataclass
class BacktestResult:
    """Résultats complets d'un backtest."""
    config: BacktestConfig
    tick_events: list[TickEvent]
    all_fills: list[SimulatedFill]
    pnl_tracker: PairPnLTracker
    elapsed_seconds: float
    total_ticks: int
    total_orders_placed: int
    total_orders_rejected: int


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class BacktestEngine:
    """Moteur event-driven de backtesting."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._storage = StorageManager(config.data_dir)
        self._matching = MatchingEngine(
            latency_ms=config.latency_ms,
            maker_fee_bps=config.maker_fee_bps,
            aggressive_fill=config.aggressive_fill,
        )
        self._strategy = BacktestStrategy(config.to_pair_config())

        self._tick_events: list[TickEvent] = []
        self._all_fills: list[SimulatedFill] = []
        self._total_orders = 0

    def run(self, show_progress: bool = True) -> BacktestResult:
        """Exécute le backtest complet."""
        t0 = wall_time.time()

        # 1. Charger les données
        snap_df = self._storage.read_snapshots(
            self.config.coin, self.config.start_date, self.config.end_date
        )
        trades_df = self._storage.read_trades(
            self.config.coin, self.config.start_date, self.config.end_date
        )

        if snap_df.empty:
            log.error("no_snapshot_data", coin=self.config.coin)
            return self._build_result(wall_time.time() - t0)

        log.info(
            "data_loaded",
            snapshots=len(snap_df),
            trades=len(trades_df),
            coin=self.config.coin,
        )

        # 2. Pré-indexer les trades par intervalle de snapshot
        snap_timestamps = snap_df["timestamp_ms"].tolist()
        trades_index = self._build_trades_index(trades_df, snap_timestamps)

        # 3. Itérer sur chaque snapshot
        iterator = snap_df.itertuples(index=False)
        if show_progress:
            iterator = tqdm(iterator, total=len(snap_df), desc=f"BT {self.config.coin}")

        for row in iterator:
            ts = int(row.timestamp_ms)
            book = OrderBookSnapshot.from_row(row)

            if book.mid <= 0:
                continue

            trades = trades_index.get(ts, [])
            self._process_tick(book, trades, ts)

        elapsed = wall_time.time() - t0
        return self._build_result(elapsed)

    def _build_trades_index(
        self, trades_df: pd.DataFrame, snap_timestamps: list[int]
    ) -> dict[int, list[dict]]:
        """Assigne chaque trade au snapshot suivant (no look-ahead).

        Pour chaque snapshot T[i], ses trades sont ceux avec
        timestamp dans [T[i-1], T[i]).
        """
        if trades_df.empty:
            return {}

        index: dict[int, list[dict]] = {}
        snap_arr = np.array(snap_timestamps, dtype=np.int64)

        # Convertir en list of dicts une seule fois
        trade_records = trades_df.to_dict("records")

        for t in trade_records:
            t_ms = int(t.get("timestamp_ms", 0))
            # Trouver le premier snapshot APRÈS ce trade
            pos = np.searchsorted(snap_arr, t_ms, side="right")
            if pos < len(snap_arr):
                snap_ts = int(snap_arr[pos])
                if snap_ts not in index:
                    index[snap_ts] = []
                index[snap_ts].append(t)

        return index

    def _process_tick(
        self,
        book: OrderBookSnapshot,
        trades: list[dict],
        timestamp_ms: int,
    ) -> None:
        """Traite un tick complet."""
        # 1. Vérifier les fills sur les ordres précédents
        fills = self._matching.process_tick(book, trades, timestamp_ms)

        for fill in fills:
            self._strategy.record_fill(
                side=fill.side,
                price=fill.price,
                size=fill.size,
                fee=fill.fee,
                timestamp_ms=fill.timestamp_ms,
                mid_price=book.mid,
            )
            self._all_fills.append(fill)

        # 2. Obtenir les nouveaux ordres de la stratégie
        new_orders = self._strategy.on_tick(
            mid_price=book.mid,
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            timestamp_ms=timestamp_ms,
        )

        # 3. Remplacer les ordres
        self._matching.cancel_all()
        accepted = self._matching.submit_orders(new_orders, book, timestamp_ms)
        self._total_orders += len(new_orders)

        # 4. Enregistrer le tick
        inv = self._strategy.inventory
        mid = book.mid

        event = TickEvent(
            timestamp_ms=timestamp_ms,
            mid_price=mid,
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            spread_bps=book.spread_bps,
            inventory=inv,
            inventory_usd=inv * mid,
            unrealized_pnl=self._strategy.pnl.get_unrealized_pnl(mid),
            realized_pnl=self._strategy.pnl.state.realized_pnl,
            total_pnl=self._strategy.pnl.get_total_pnl(mid),
            num_fills=len(fills),
            num_orders=len(accepted),
        )
        self._tick_events.append(event)

    def _build_result(self, elapsed: float) -> BacktestResult:
        return BacktestResult(
            config=self.config,
            tick_events=self._tick_events,
            all_fills=self._all_fills,
            pnl_tracker=self._strategy.pnl,
            elapsed_seconds=elapsed,
            total_ticks=len(self._tick_events),
            total_orders_placed=self._total_orders,
            total_orders_rejected=self._matching.rejected_count,
        )
