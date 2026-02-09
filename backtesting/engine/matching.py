"""Moteur de matching conservatif pour le backtesting.

Règles :
1. CROSSING, pas touching : un bid fill seulement si best_ask < bid_price
2. Volume contraint par les trades observés dans l'intervalle
3. Latence simulée : ordre invisible pendant latency_ms après placement
4. Post-only : bid >= best_ask ou ask <= best_bid → rejeté
5. Pas de look-ahead bias
"""

from dataclasses import dataclass, field
from typing import Optional

from backtesting.engine.orderbook import OrderBookSnapshot
from src.utils.logger import get_logger

log = get_logger("bt.matching")


@dataclass
class SimulatedOrder:
    """Un ordre limit en attente dans le backtest."""
    order_id: int
    coin: str
    is_buy: bool
    price: float
    size: float
    remaining: float
    post_only: bool
    placed_at_ms: int
    visible_at_ms: int   # placed_at_ms + latency


@dataclass
class SimulatedFill:
    """Résultat d'un fill simulé."""
    order_id: int
    coin: str
    side: str             # "buy" ou "sell"
    price: float          # prix du fill = prix de l'ordre (limit)
    size: float           # taille remplie
    fee: float            # fee simulée (négative = rebate maker)
    timestamp_ms: int
    mid_at_fill: float


class MatchingEngine:
    """Moteur de matching conservatif pour backtesting MM.

    Args:
        latency_ms: Latence simulée réseau + exchange (default 300ms)
        maker_fee_bps: Fee maker en bps, négatif = rebate (default -2.0)
        aggressive_fill: Si True, fill au "touch" au lieu de "cross" (default False)
    """

    def __init__(
        self,
        latency_ms: int = 300,
        maker_fee_bps: float = -2.0,
        aggressive_fill: bool = False,
    ):
        self._latency_ms = latency_ms
        self._maker_fee_bps = maker_fee_bps
        self._aggressive = aggressive_fill
        self._pending: list[SimulatedOrder] = []
        self._next_id = 1
        self._rejected_count = 0

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def submit_orders(
        self,
        orders: list[dict],
        book: OrderBookSnapshot,
        timestamp_ms: int,
    ) -> list[SimulatedOrder]:
        """Soumet de nouveaux ordres. Applique la validation post-only.

        Retourne la liste des ordres acceptés.
        """
        accepted = []
        for o in orders:
            is_buy = o["is_buy"]
            price = o["price"]

            # Post-only : rejet si l'ordre crosserait le spread
            if o.get("post_only", True):
                if is_buy and price >= book.best_ask and book.best_ask > 0:
                    self._rejected_count += 1
                    continue
                if not is_buy and price <= book.best_bid and book.best_bid > 0:
                    self._rejected_count += 1
                    continue

            order = SimulatedOrder(
                order_id=self._next_id,
                coin=o["coin"],
                is_buy=is_buy,
                price=price,
                size=o["size"],
                remaining=o["size"],
                post_only=o.get("post_only", True),
                placed_at_ms=timestamp_ms,
                visible_at_ms=timestamp_ms + self._latency_ms,
            )
            self._next_id += 1
            self._pending.append(order)
            accepted.append(order)

        return accepted

    def cancel_all(self) -> None:
        """Annule tous les ordres en attente."""
        self._pending.clear()

    def process_tick(
        self,
        book: OrderBookSnapshot,
        trades_in_interval: list[dict],
        timestamp_ms: int,
    ) -> list[SimulatedFill]:
        """Traite un tick. Vérifie les fills sur les ordres visibles.

        Args:
            book: Snapshot L2 courant
            trades_in_interval: Trades observés entre le tick précédent et celui-ci
            timestamp_ms: Timestamp du tick courant

        Returns:
            Liste des fills simulés
        """
        fills: list[SimulatedFill] = []
        still_pending: list[SimulatedOrder] = []

        for order in self._pending:
            # L'ordre n'est pas encore visible (latence)
            if order.visible_at_ms > timestamp_ms:
                still_pending.append(order)
                continue

            fill = self._try_fill(order, book, trades_in_interval, timestamp_ms)
            if fill is not None:
                fills.append(fill)
                # Si partiellement rempli, garder le reste
                if order.remaining > 0:
                    still_pending.append(order)
            else:
                still_pending.append(order)

        self._pending = still_pending
        return fills

    def _try_fill(
        self,
        order: SimulatedOrder,
        book: OrderBookSnapshot,
        trades: list[dict],
        timestamp_ms: int,
    ) -> Optional[SimulatedFill]:
        """Tente de remplir un ordre contre le book et les trades."""
        if order.is_buy:
            return self._try_fill_buy(order, book, trades, timestamp_ms)
        else:
            return self._try_fill_sell(order, book, trades, timestamp_ms)

    def _try_fill_buy(
        self,
        order: SimulatedOrder,
        book: OrderBookSnapshot,
        trades: list[dict],
        timestamp_ms: int,
    ) -> Optional[SimulatedFill]:
        """Fill d'un bid : quelqu'un vend dans notre bid.

        Condition : best_ask < order.price (crossing)
        ou best_ask <= order.price si aggressive_fill=True
        """
        ba = book.best_ask
        if ba <= 0:
            return None

        crosses = ba < order.price if not self._aggressive else ba <= order.price
        if not crosses:
            return None

        # Volume disponible depuis les trades observés
        available = self._contra_volume_from_trades(
            trades, is_buy_order=True, our_price=order.price
        )

        # Fallback : estimation depuis le book si pas de trades
        if available <= 0:
            available = book.volume_through_price("buy", order.price) * 0.1

        if available <= 0:
            return None

        fill_size = min(order.remaining, available)
        fee = self._compute_fee(order.price, fill_size)
        order.remaining -= fill_size

        return SimulatedFill(
            order_id=order.order_id,
            coin=order.coin,
            side="buy",
            price=order.price,
            size=fill_size,
            fee=fee,
            timestamp_ms=timestamp_ms,
            mid_at_fill=book.mid,
        )

    def _try_fill_sell(
        self,
        order: SimulatedOrder,
        book: OrderBookSnapshot,
        trades: list[dict],
        timestamp_ms: int,
    ) -> Optional[SimulatedFill]:
        """Fill d'un ask : quelqu'un achète dans notre ask.

        Condition : best_bid > order.price (crossing)
        """
        bb = book.best_bid
        if bb <= 0:
            return None

        crosses = bb > order.price if not self._aggressive else bb >= order.price
        if not crosses:
            return None

        available = self._contra_volume_from_trades(
            trades, is_buy_order=False, our_price=order.price
        )

        if available <= 0:
            available = book.volume_through_price("sell", order.price) * 0.1

        if available <= 0:
            return None

        fill_size = min(order.remaining, available)
        fee = self._compute_fee(order.price, fill_size)
        order.remaining -= fill_size

        return SimulatedFill(
            order_id=order.order_id,
            coin=order.coin,
            side="sell",
            price=order.price,
            size=fill_size,
            fee=fee,
            timestamp_ms=timestamp_ms,
            mid_at_fill=book.mid,
        )

    def _contra_volume_from_trades(
        self,
        trades: list[dict],
        is_buy_order: bool,
        our_price: float,
    ) -> float:
        """Volume des trades qui traversent notre prix.

        Pour un BUY order : trades sell (side "A") à prix <= notre bid
        Pour un SELL order : trades buy (side "B") à prix >= notre ask
        """
        total = 0.0
        for t in trades:
            price = float(t.get("price", 0))
            size = float(t.get("size", 0))
            side = t.get("side", "")

            if is_buy_order:
                # Cherche des ventes qui traversent notre bid
                if side in ("A", "sell") and price <= our_price:
                    total += size
            else:
                # Cherche des achats qui traversent notre ask
                if side in ("B", "buy") and price >= our_price:
                    total += size

        return total

    def _compute_fee(self, price: float, size: float) -> float:
        """Calcule la fee maker. Négative = rebate."""
        return price * size * self._maker_fee_bps / 10_000
