"""Suivi PnL par paire avec méthode FIFO.

Maintient une file d'attente d'entrées (achat/vente) et calcule
le PnL réalisé quand une position est réduite.
"""

from dataclasses import dataclass, field
from collections import deque


@dataclass
class FillEntry:
    """Une entrée dans la file FIFO."""
    side: str       # "buy" | "sell"
    price: float
    size: float     # toujours positif
    timestamp: str = ""


@dataclass
class PnLState:
    """État PnL courant pour une paire."""
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    total_volume_usd: float = 0.0
    buy_fills: int = 0
    sell_fills: int = 0
    max_inventory_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    peak_pnl: float = 0.0


class PairPnLTracker:
    """Tracker PnL FIFO pour une seule paire."""

    def __init__(self, pair: str):
        self.pair = pair
        self._long_queue: deque[FillEntry] = deque()
        self._short_queue: deque[FillEntry] = deque()
        self.state = PnLState()
        self.current_inventory: float = 0.0  # positif = long, négatif = short

    @property
    def fills_count(self) -> int:
        return self.state.buy_fills + self.state.sell_fills

    def record_fill(
        self,
        side: str,
        price: float,
        size: float,
        fee: float,
        timestamp: str = "",
        mid_price: float = 0.0,
    ) -> dict:
        """Enregistre un fill et calcule le PnL réalisé.

        Returns:
            dict avec realized_pnl_delta, inventory_after, spread_captured_bps
        """
        # Compteurs
        if side == "buy":
            self.state.buy_fills += 1
        else:
            self.state.sell_fills += 1
        self.state.total_fees += fee
        self.state.total_volume_usd += price * size

        # Matching FIFO
        realized = self._match_fifo(side, price, size, timestamp)
        self.state.realized_pnl += realized

        # Recalculer l'inventaire
        self.current_inventory = self._compute_inventory()

        # Tracker max inventory
        inv_usd = abs(self.current_inventory * price)
        if inv_usd > self.state.max_inventory_usd:
            self.state.max_inventory_usd = inv_usd

        # Tracker drawdown (basé sur PnL réalisé uniquement pour la robustesse)
        total = self.state.realized_pnl - self.state.total_fees
        if total > self.state.peak_pnl:
            self.state.peak_pnl = total
        drawdown = self.state.peak_pnl - total
        if drawdown > self.state.max_drawdown_usd:
            self.state.max_drawdown_usd = drawdown

        # Spread capturé
        spread_captured_bps = 0.0
        if mid_price > 0:
            spread_captured_bps = abs(price - mid_price) / mid_price * 10_000

        return {
            "realized_pnl_delta": realized,
            "inventory_after": self.current_inventory,
            "spread_captured_bps": round(spread_captured_bps, 2),
        }

    def _match_fifo(self, side: str, price: float, size: float, timestamp: str) -> float:
        """Matching FIFO : consomme la queue opposée et retourne le PnL réalisé."""
        realized = 0.0
        remaining = size

        if side == "buy":
            # Un achat ferme des positions short (queue short)
            while remaining > 0 and self._short_queue:
                entry = self._short_queue[0]
                matched = min(remaining, entry.size)

                # Short fermé : on avait vendu à entry.price, on rachète à price
                realized += matched * (entry.price - price)

                entry.size -= matched
                remaining -= matched
                if entry.size <= 1e-12:
                    self._short_queue.popleft()

            # Le reste va dans la queue long
            if remaining > 1e-12:
                self._long_queue.append(FillEntry("buy", price, remaining, timestamp))

        else:  # sell
            # Une vente ferme des positions long (queue long)
            while remaining > 0 and self._long_queue:
                entry = self._long_queue[0]
                matched = min(remaining, entry.size)

                # Long fermé : on avait acheté à entry.price, on vend à price
                realized += matched * (price - entry.price)

                entry.size -= matched
                remaining -= matched
                if entry.size <= 1e-12:
                    self._long_queue.popleft()

            # Le reste va dans la queue short
            if remaining > 1e-12:
                self._short_queue.append(FillEntry("sell", price, remaining, timestamp))

        return realized

    def _compute_inventory(self) -> float:
        """Calcule l'inventaire net depuis les queues."""
        long_total = sum(e.size for e in self._long_queue)
        short_total = sum(e.size for e in self._short_queue)
        return long_total - short_total

    def get_avg_entry_price(self) -> float:
        """Prix d'entrée moyen pondéré de l'inventaire restant."""
        if self._long_queue:
            total_value = sum(e.price * e.size for e in self._long_queue)
            total_size = sum(e.size for e in self._long_queue)
            return total_value / total_size if total_size > 0 else 0.0
        elif self._short_queue:
            total_value = sum(e.price * e.size for e in self._short_queue)
            total_size = sum(e.size for e in self._short_queue)
            return total_value / total_size if total_size > 0 else 0.0
        return 0.0

    def get_unrealized_pnl(self, current_mid: float) -> float:
        """PnL non réalisé = inventory × (mid - avg_entry)."""
        if abs(self.current_inventory) < 1e-12:
            return 0.0
        avg_entry = self.get_avg_entry_price()
        if avg_entry <= 0:
            return 0.0
        return self.current_inventory * (current_mid - avg_entry)

    def get_total_pnl(self, current_mid: float) -> float:
        """PnL total = réalisé + non réalisé - fees."""
        return (
            self.state.realized_pnl
            + self.get_unrealized_pnl(current_mid)
            - self.state.total_fees
        )
