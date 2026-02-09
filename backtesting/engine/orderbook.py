"""Représentation d'un snapshot L2 d'orderbook pour le backtesting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BookLevel:
    """Un niveau de prix dans l'orderbook."""
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Snapshot L2 reconstruit depuis une row Parquet aplatie."""

    timestamp_ms: int
    coin: str
    bids: list[BookLevel]  # trié descending (best bid en premier)
    asks: list[BookLevel]  # trié ascending (best ask en premier)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb > 0 and ba > 0:
            return (bb + ba) / 2.0
        return 0.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        if mid <= 0:
            return 0.0
        return ((self.best_ask - self.best_bid) / mid) * 10_000

    def volume_through_price(self, side: str, price: float) -> float:
        """Volume total dans le book au-delà d'un prix donné.

        Pour estimer le volume contra disponible quand on n'a pas de trades.
        - side="buy" : somme des asks à prix <= price
        - side="sell" : somme des bids à prix >= price
        """
        total = 0.0
        if side == "buy":
            for lvl in self.asks:
                if lvl.price <= price:
                    total += lvl.size
                else:
                    break
        else:
            for lvl in self.bids:
                if lvl.price >= price:
                    total += lvl.size
                else:
                    break
        return total

    @classmethod
    def from_row(cls, row) -> "OrderBookSnapshot":
        """Construit depuis une row Parquet (namedtuple ou dict).

        Attend les colonnes : timestamp_ms, coin,
        bid_px_0..bid_px_9, bid_sz_0..bid_sz_9,
        ask_px_0..ask_px_9, ask_sz_0..ask_sz_9
        """
        # Support à la fois namedtuple (itertuples) et dict
        def _get(key):
            if isinstance(row, dict):
                return row.get(key, 0)
            return getattr(row, key, 0)

        bids = []
        asks = []
        for i in range(10):
            bp = float(_get(f"bid_px_{i}"))
            bs = float(_get(f"bid_sz_{i}"))
            if bp > 0 and bs > 0:
                bids.append(BookLevel(bp, bs))

            ap = float(_get(f"ask_px_{i}"))
            az = float(_get(f"ask_sz_{i}"))
            if ap > 0 and az > 0:
                asks.append(BookLevel(ap, az))

        # S'assurer du tri
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return cls(
            timestamp_ms=int(_get("timestamp_ms")),
            coin=str(_get("coin")),
            bids=bids,
            asks=asks,
        )
