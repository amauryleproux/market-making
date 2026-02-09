"""Adaptateur de la stratégie MM pour le backtesting.

Réutilise PairConfig et PairPnLTracker du code live.
Sous-classe PriceTracker pour utiliser le temps simulé.
Réplique _compute_quotes() en fonction standalone.

NOTE: compute_quotes() et _round_price() sont répliqués depuis
src/strategies/pair_market_maker.py (PairMarketMaker._compute_quotes).
Si le code live change, il faut synchroniser ici.
"""

from collections import deque
from typing import Optional

from src.strategies.pair_market_maker import PriceTracker, PairConfig
from src.monitoring.pnl_tracker import PairPnLTracker
from src.utils.logger import get_logger

log = get_logger("bt.strategy")

# --- Constantes (répliquées de PairMarketMaker) ---
TREND_WINDOW_SEC = 60.0
TREND_CAUTION_BPS = 15.0
TREND_CUTOFF_BPS = 30.0
VOL_BASELINE_BPS = 3.0
VOL_MAX_MULT = 2.5


# ------------------------------------------------------------------
# PriceTracker avec temps simulé
# ------------------------------------------------------------------

class BacktestPriceTracker(PriceTracker):
    """PriceTracker qui utilise le temps simulé au lieu de time.time()."""

    def __init__(self, window_seconds: float = 60.0, max_samples: int = 200):
        super().__init__(window_seconds, max_samples)
        self._sim_time: float = 0.0

    def set_time(self, timestamp_sec: float) -> None:
        self._sim_time = timestamp_sec

    def record(self, price: float) -> None:
        """Override : utilise le temps simulé."""
        self._prices.append((self._sim_time, price))

    def _trim(self) -> list[tuple[float, float]]:
        """Override : utilise le temps simulé."""
        cutoff = self._sim_time - self._window_sec
        return [(t, p) for t, p in self._prices if t >= cutoff]


# ------------------------------------------------------------------
# Fonctions de quoting standalone
# ------------------------------------------------------------------

def _round_price(price: float) -> float:
    """Répliqué de PairMarketMaker._round_price()."""
    if price > 10000:
        return round(price, 0)
    elif price > 100:
        return round(price, 1)
    elif price > 1:
        return round(price, 2)
    elif price > 0.01:
        return round(price, 4)
    else:
        return round(price, 6)


def compute_quotes(
    config: PairConfig,
    mid_price: float,
    inventory: float,
    trend_bps: float = 0.0,
    vol_mult: float = 1.0,
    quoting_mode: str = "normal",
) -> list[dict]:
    """Calcul des quotes — réplique exacte de PairMarketMaker._compute_quotes().

    Logique :
    1. Skew par mid-shift proportionnel au spread
    2. Spread ajusté par volatilité
    3. Trend protection (normal / cautious / one_sided)
    4. Hard inventory cutoff
    5. Check cumulatif par level
    """
    cfg = config
    inventory_usd = inventory * mid_price
    max_inv = cfg.max_inventory_usd

    # --- Skew ---
    if max_inv > 0:
        skew_ratio = cfg.inventory_skew_factor * (inventory_usd / max_inv)
    else:
        skew_ratio = 0.0
    skew_ratio = max(-1.0, min(1.0, skew_ratio))

    adjusted_mid = mid_price * (1 - skew_ratio * cfg.spread_bps / 10_000)

    # --- Spread ajusté par volatilité ---
    effective_spread_bps = cfg.spread_bps * vol_mult
    half_spread = effective_spread_bps / 10_000

    # --- Trend protection ---
    bid_mult = 1.0
    ask_mult = 1.0
    allow_bids = True
    allow_asks = True

    if quoting_mode == "cautious":
        if trend_bps < 0:
            bid_mult = 1.5
        else:
            ask_mult = 1.5
    elif quoting_mode == "one_sided":
        if trend_bps < 0:
            allow_bids = False
        else:
            allow_asks = False

    # --- Hard inventory cutoff ---
    if inventory_usd >= max_inv:
        allow_bids = False
    elif inventory_usd <= -max_inv:
        allow_asks = False

    orders = []
    sz_per_order = cfg.order_size_usd / mid_price

    # --- Ordres avec check cumulatif ---
    cumulative_buy_usd = 0.0
    cumulative_sell_usd = 0.0

    for i in range(cfg.num_levels):
        level_offset = i * cfg.level_spacing_bps / 10_000

        if allow_bids:
            cumulative_buy_usd += cfg.order_size_usd
            if abs(inventory_usd + cumulative_buy_usd) < max_inv:
                bid_price = adjusted_mid * (1 - half_spread * bid_mult - level_offset)
                bid_price = _round_price(bid_price)
                orders.append({
                    "coin": cfg.coin,
                    "is_buy": True,
                    "size": sz_per_order,
                    "price": bid_price,
                    "post_only": True,
                    "level": i,
                })

        if allow_asks:
            cumulative_sell_usd += cfg.order_size_usd
            if abs(inventory_usd - cumulative_sell_usd) < max_inv:
                ask_price = adjusted_mid * (1 + half_spread * ask_mult + level_offset)
                ask_price = _round_price(ask_price)
                orders.append({
                    "coin": cfg.coin,
                    "is_buy": False,
                    "size": sz_per_order,
                    "price": ask_price,
                    "post_only": True,
                    "level": i,
                })

    return orders


# ------------------------------------------------------------------
# Strategy adapter
# ------------------------------------------------------------------

class BacktestStrategy:
    """Stratégie MM adaptée pour le backtesting.

    Miroir de PairMarketMaker._cycle() sans async, API, ni DBLogger.
    """

    def __init__(self, config: PairConfig):
        self.config = config
        self.pnl = PairPnLTracker(config.coin)
        self._price_tracker = BacktestPriceTracker(
            window_seconds=TREND_WINDOW_SEC,
            max_samples=200,
        )
        self._last_mid: float = 0.0

    def on_tick(
        self,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        timestamp_ms: int,
    ) -> list[dict]:
        """Traite un tick. Retourne les ordres à placer."""
        self._last_mid = mid_price
        timestamp_sec = timestamp_ms / 1000.0

        # 1. Mettre à jour le price tracker avec le temps simulé
        self._price_tracker.set_time(timestamp_sec)
        self._price_tracker.record(mid_price)

        # 2. Calculer trend et volatilité
        trend_bps = self._price_tracker.get_trend_bps()
        vol_bps = self._price_tracker.get_volatility_bps()

        if VOL_BASELINE_BPS > 0 and vol_bps > VOL_BASELINE_BPS:
            vol_mult = min(vol_bps / VOL_BASELINE_BPS, VOL_MAX_MULT)
        else:
            vol_mult = 1.0

        # 3. Mode de quoting
        abs_trend = abs(trend_bps)
        if abs_trend >= TREND_CUTOFF_BPS:
            quoting_mode = "one_sided"
        elif abs_trend >= TREND_CAUTION_BPS:
            quoting_mode = "cautious"
        else:
            quoting_mode = "normal"

        # 4. Calculer les quotes
        inventory = self.pnl.current_inventory
        return compute_quotes(
            config=self.config,
            mid_price=mid_price,
            inventory=inventory,
            trend_bps=trend_bps,
            vol_mult=vol_mult,
            quoting_mode=quoting_mode,
        )

    def record_fill(
        self,
        side: str,
        price: float,
        size: float,
        fee: float,
        timestamp_ms: int,
        mid_price: float,
    ) -> dict:
        """Enregistre un fill dans le PnL tracker."""
        return self.pnl.record_fill(
            side=side,
            price=price,
            size=size,
            fee=fee,
            timestamp=str(timestamp_ms),
            mid_price=mid_price,
        )

    @property
    def inventory(self) -> float:
        return self.pnl.current_inventory

    @property
    def inventory_usd(self) -> float:
        return self.pnl.current_inventory * self._last_mid
