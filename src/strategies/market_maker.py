"""
Stratégie de Market Making pour Hyperliquid.

Place des ordres bid/ask autour du mid-price avec:
- ALO (post-only) pour garantir les maker rebates
- Inventory skew pour gérer le risque directionnel
- Multiple niveaux pour capturer plus de volume

Le profit vient des maker rebates (-0.001% à -0.003%) + spread.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.client.hyperliquid import HyperliquidClient, Position
from src.utils.logger import get_logger

log = get_logger("mm")


@dataclass
class CoinState:
    """État du market making pour un coin."""
    coin: str
    mid_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    position_size: float = 0.0  # en unités du coin
    position_usd: float = 0.0
    active_bids: int = 0
    active_asks: int = 0
    total_fills: int = 0
    total_rebates_usd: float = 0.0
    last_update: float = 0.0


class MarketMaker:
    """
    Market Maker multi-coin pour Hyperliquid.

    Stratégie:
    1. Récupère le mid-price de chaque coin
    2. Calcule le spread optimal (min_spread + inventory skew)
    3. Place des ordres ALO (post-only) bid/ask
    4. Rebalance toutes les N secondes
    5. Gère le risque via position limits et inventory skew
    """

    def __init__(
        self,
        client: HyperliquidClient,
        coins: list[str],
        order_size_usd: float = 50,
        min_spread_pct: float = 0.05,
        num_levels: int = 3,
        level_spacing_pct: float = 0.02,
        max_position_usd: float = 500,
        inventory_skew: float = 0.5,
        dry_run: bool = True,
    ):
        self.client = client
        self.coins = coins
        self.order_size_usd = order_size_usd
        self.min_spread_pct = min_spread_pct / 100  # convertir en ratio
        self.num_levels = num_levels
        self.level_spacing_pct = level_spacing_pct / 100
        self.max_position_usd = max_position_usd
        self.inventory_skew = inventory_skew
        self.dry_run = dry_run

        # État par coin
        self.states: dict[str, CoinState] = {
            coin: CoinState(coin=coin) for coin in coins
        }

        # Stats globales
        self.total_orders_placed = 0
        self.total_fills = 0
        self.start_time = time.time()

    def update_quotes(self):
        """Met à jour tous les quotes pour tous les coins."""
        for coin in self.coins:
            try:
                self._update_coin(coin)
            except Exception as e:
                log.error("update_failed", coin=coin, error=str(e))

    def _update_coin(self, coin: str):
        """Met à jour les quotes pour un coin."""
        state = self.states[coin]

        # 1. Récupérer le prix
        mid = self.client.get_mid(coin)
        best_bid, best_ask = self.client.get_best_bid_ask(coin)
        state.mid_price = mid
        state.best_bid = best_bid
        state.best_ask = best_ask

        # 2. Récupérer la position actuelle
        position_size = 0.0
        account = self.client.get_account_state()
        for pos in account.positions:
            if pos.coin == coin:
                position_size = pos.size
                break

        state.position_size = position_size
        state.position_usd = position_size * mid

        # 3. Calculer le spread avec inventory skew
        half_spread = max(self.min_spread_pct / 2, (best_ask - best_bid) / mid / 2)

        # Skew: si on est long, on baisse le bid et monte l'ask (pour se décharger)
        inventory_ratio = state.position_usd / self.max_position_usd if self.max_position_usd > 0 else 0
        inventory_ratio = max(-1, min(1, inventory_ratio))  # clamp [-1, 1]
        skew = inventory_ratio * self.inventory_skew * half_spread

        bid_offset = half_spread + skew  # + skew = bid plus bas quand long
        ask_offset = half_spread - skew  # - skew = ask plus bas quand long

        # 4. Vérifier les limites de position
        can_buy = abs(state.position_usd + self.order_size_usd) < self.max_position_usd
        can_sell = abs(state.position_usd - self.order_size_usd) < self.max_position_usd

        # 5. Annuler les anciens ordres
        if not self.dry_run:
            self.client.cancel_coin_orders(coin)

        # 6. Construire les nouveaux ordres
        orders = []
        sz_per_order = self.order_size_usd / mid
        sz_decimals = self.client.get_sz_decimals(coin)

        for level in range(self.num_levels):
            extra_offset = level * self.level_spacing_pct

            # Bid
            if can_buy:
                bid_price = mid * (1 - bid_offset - extra_offset)
                bid_price = self._round_price(bid_price, coin)
                orders.append({
                    "coin": coin,
                    "is_buy": True,
                    "size": sz_per_order,
                    "price": bid_price,
                    "post_only": True,
                })

            # Ask
            if can_sell:
                ask_price = mid * (1 + ask_offset + extra_offset)
                ask_price = self._round_price(ask_price, coin)
                orders.append({
                    "coin": coin,
                    "is_buy": False,
                    "size": sz_per_order,
                    "price": ask_price,
                    "post_only": True,
                })

        # 7. Placer les ordres
        if self.dry_run:
            state.active_bids = sum(1 for o in orders if o["is_buy"])
            state.active_asks = sum(1 for o in orders if not o["is_buy"])
            for o in orders:
                side = "BID" if o["is_buy"] else "ASK"
                log.info(
                    "dry_order",
                    coin=coin,
                    side=side,
                    price=o["price"],
                    size=round(o["size"], sz_decimals),
                    mid=mid,
                )
            self.total_orders_placed += len(orders)
        elif orders:
            results = self.client.place_bulk_orders(orders)
            placed = sum(1 for r in results if r.success)
            errors = sum(1 for r in results if not r.success)

            state.active_bids = sum(1 for o, r in zip(orders, results) if o["is_buy"] and r.success)
            state.active_asks = sum(1 for o, r in zip(orders, results) if not o["is_buy"] and r.success)
            self.total_orders_placed += placed

            log.info(
                "quotes_placed",
                coin=coin,
                mid=round(mid, 2),
                bids=state.active_bids,
                asks=state.active_asks,
                errors=errors,
                position=round(state.position_usd, 2),
                skew=round(inventory_ratio, 3),
            )

            for r in results:
                if not r.success:
                    log.warning("order_error", coin=coin, error=r.error)

        state.last_update = time.time()

    def _round_price(self, price: float, coin: str) -> float:
        """Arrondit le prix selon les règles Hyperliquid."""
        # Prix > 1000: arrondir à l'entier
        # Prix > 1: 2 décimales
        # Prix < 1: 4-6 décimales
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

    def get_status(self) -> dict:
        """Retourne un résumé de l'état du bot."""
        total_position = sum(s.position_usd for s in self.states.values())
        uptime = time.time() - self.start_time

        return {
            "uptime_min": round(uptime / 60, 1),
            "coins": len(self.coins),
            "total_orders": self.total_orders_placed,
            "total_position_usd": round(total_position, 2),
            "coin_states": {
                coin: {
                    "mid": round(s.mid_price, 2),
                    "position": round(s.position_usd, 2),
                    "bids": s.active_bids,
                    "asks": s.active_asks,
                }
                for coin, s in self.states.items()
            },
        }
