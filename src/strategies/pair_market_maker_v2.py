"""Stratégie de Market Making V3 — Smart Quoting avec State Machine.

Le bot ne quote PAS en permanence. Il analyse le flow en temps réel
et ne place des ordres que quand les conditions sont safe.

State machine:
- MONITORING: Observe le flow, aucune quote. Attend flow safe 5s.
- QUOTING: Quotes actives. Pull si flow toxique ou si fill détecté.
- CLOSING: Inventaire à fermer. Ordres close tight uniquement.
- EMERGENCY_CLOSE: Stop-loss market, puis cooldown 60s.

Features conservées du V2:
- Spread dynamique Avellaneda-Stoikov en espace bps
- Reservation price (mid ajusté par inventaire + funding)
- Smart cancel/replace
- Anti-flip protection
- Asymmetric sizing
"""

import asyncio
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.client.async_wrapper import AsyncHyperliquidClient
from src.monitoring.db_logger import DBLogger
from src.monitoring.pnl_tracker import PairPnLTracker
from src.utils.logger import get_logger

# Re-use PairConfig and PairState from V1 for engine compatibility
from src.strategies.pair_market_maker import PairConfig, PairState


# ---------------------------------------------------------------------------
# Micro-structure helpers
# ---------------------------------------------------------------------------

class VolatilityEstimator:
    """Estime la volatilité réalisée (sigma) en temps réel.

    Utilise les returns tick-by-tick sur une fenêtre glissante.
    Retourne sigma en prix/seconde (pas en bps).
    """

    def __init__(self, window_sec: float = 120.0, max_samples: int = 500):
        self._window_sec = window_sec
        self._ticks: deque[tuple[float, float]] = deque(maxlen=max_samples)  # (time, price)

    def record(self, price: float) -> None:
        self._ticks.append((time.time(), price))

    def get_sigma(self) -> float:
        """Retourne sigma annualisé-like mais en unités de prix par seconde.

        Pour AS, on veut sigma² en (prix²/sec) pour que le spread
        soit en unités de prix.
        """
        cutoff = time.time() - self._window_sec
        ticks = [(t, p) for t, p in self._ticks if t >= cutoff]

        if len(ticks) < 10:
            return 0.0

        # Calcul de la variance des returns
        returns = []
        for i in range(1, len(ticks)):
            dt = ticks[i][0] - ticks[i-1][0]
            if dt < 0.01:  # skip duplicate timestamps
                continue
            ret = (ticks[i][1] - ticks[i-1][1]) / ticks[i-1][1]
            # Normaliser par sqrt(dt) pour obtenir vol par seconde
            returns.append(ret / math.sqrt(dt))

        if len(returns) < 5:
            return 0.0

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        sigma = math.sqrt(variance)

        # Retourner en unités de prix (sigma * prix moyen)
        avg_price = sum(p for _, p in ticks) / len(ticks)
        return sigma * avg_price

    def get_sigma_bps(self) -> float:
        """Sigma en bps pour le logging."""
        cutoff = time.time() - self._window_sec
        ticks = [(t, p) for t, p in self._ticks if t >= cutoff]
        if len(ticks) < 10:
            return 0.0

        returns_bps = []
        for i in range(1, len(ticks)):
            if ticks[i-1][1] > 0:
                ret = ((ticks[i][1] - ticks[i-1][1]) / ticks[i-1][1]) * 10_000
                returns_bps.append(ret)

        if len(returns_bps) < 5:
            return 0.0

        mean = sum(returns_bps) / len(returns_bps)
        variance = sum((r - mean) ** 2 for r in returns_bps) / len(returns_bps)
        return math.sqrt(variance)


class BookAnalyzer:
    """Analyse le L2 book pour extraire des signaux."""

    @staticmethod
    def parse_l2(l2_data: dict) -> tuple[float, float, float, float, float]:
        """Parse le L2 snapshot.

        Returns:
            (mid, best_bid, best_ask, bid_depth_usd, ask_depth_usd)
        """
        bids = l2_data.get("levels", [[],[]])[0]
        asks = l2_data.get("levels", [[],[]])[1]

        if not bids or not asks:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        best_bid = float(bids[0]["px"])
        best_ask = float(asks[0]["px"])
        mid = (best_bid + best_ask) / 2.0

        # Profondeur en USD sur les N premiers niveaux
        bid_depth = sum(float(b["px"]) * float(b["sz"]) for b in bids[:5])
        ask_depth = sum(float(a["px"]) * float(a["sz"]) for a in asks[:5])

        return mid, best_bid, best_ask, bid_depth, ask_depth

    @staticmethod
    def get_imbalance(bid_depth: float, ask_depth: float) -> float:
        """Book imbalance: +1 = all bids, -1 = all asks, 0 = balanced."""
        total = bid_depth + ask_depth
        if total < 1e-6:
            return 0.0
        return (bid_depth - ask_depth) / total

    @staticmethod
    def get_microprice(best_bid: float, best_ask: float,
                       bid_size: float, ask_size: float) -> float:
        """Micro-price pondéré par les tailles au best."""
        total = bid_size + ask_size
        if total < 1e-12:
            return (best_bid + best_ask) / 2.0
        return (best_bid * ask_size + best_ask * bid_size) / total


class FlowAnalyzer:
    """Analyse les trades récents pour détecter le flow toxique.

    Flow toxique = les trades récents sont majoritairement dans une direction,
    indiquant la présence de traders informés / momentum.
    Flow safe = les trades sont équilibrés (bruit, retail).
    """

    def __init__(self, window_sec: float = 30.0):
        self._window_sec = window_sec
        self._trades: deque[tuple[float, str, float]] = deque(maxlen=200)
        # (timestamp, side "buy"/"sell", size_usd)

    def record_trade(self, side: str, size_usd: float) -> None:
        """Enregistre un trade observé sur le marché."""
        self._trades.append((time.time(), side, size_usd))

    def get_flow_imbalance(self) -> float:
        """Retourne l'imbalance du flow sur la fenêtre.

        +1.0 = tout le volume est acheteur
        -1.0 = tout le volume est vendeur
        0.0 = équilibré
        """
        cutoff = time.time() - self._window_sec
        recent = [(s, v) for t, s, v in self._trades if t >= cutoff]

        if not recent:
            return 0.0

        buy_vol = sum(v for s, v in recent if s == "buy")
        sell_vol = sum(v for s, v in recent if s == "sell")
        total = buy_vol + sell_vol

        if total < 1e-6:
            return 0.0

        return (buy_vol - sell_vol) / total

    def get_trade_intensity(self) -> float:
        """Nombre de trades par seconde sur la fenêtre."""
        cutoff = time.time() - self._window_sec
        recent = [t for t, _, _ in self._trades if t >= cutoff]

        if len(recent) < 2:
            return 0.0

        return len(recent) / self._window_sec

    def is_safe_to_quote(self) -> tuple[bool, str]:
        """Détermine si les conditions sont safe pour quoter.

        Returns:
            (is_safe, reason)
        """
        imbalance = self.get_flow_imbalance()
        intensity = self.get_trade_intensity()

        # Flow très déséquilibré (> 70% dans une direction)
        if abs(imbalance) > 0.7:
            return False, f"toxic_flow_imbalance={imbalance:.2f}"

        # Très haute intensité de trades + déséquilibre modéré
        if intensity > 3.0 and abs(imbalance) > 0.5:
            return False, f"high_intensity={intensity:.1f}_imbalance={imbalance:.2f}"

        return True, "safe"


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

class PairMarketMakerV2:
    """Market maker V3 — Smart Quoting avec state machine.

    Compatible avec MMEngine (même interface que PairMarketMaker).
    """

    # --- Paramètres AS ---
    GAMMA = 0.5                     # Facteur de sensibilité à la volatilité (vol adjustment)
    MIN_SPREAD_BPS = 10.0           # Spread plancher: couvre fees A/R (3bps) + marge
    MAX_SPREAD_BPS = 30.0           # Spread plafond

    # --- Paramètres event-driven ---
    POLL_INTERVAL_SEC = 5.0         # Fréquence de lecture du prix
    REQUOTE_THRESHOLD_BPS = 4.0     # Seuil de mouvement pour requoter
    FORCE_REQUOTE_SEC = 60.0        # Requote forcé après X secondes même sans mouvement

    # --- Paramètres trend/protection ---
    TREND_WINDOW_SEC = 60.0
    TREND_CUTOFF_BPS = 25.0         # Au-delà: coupe un côté
    TREND_CAUTION_BPS = 12.0        # Au-delà: spread x1.5 côté danger

    # --- Imbalance ---
    IMBALANCE_SKEW_FACTOR = 0.3     # Pondération de l'imbalance sur le skew

    # --- Flow safety ---
    SAFE_FLOW_WAIT_SEC = 5.0        # Secondes de flow safe avant de quoter

    def __init__(
        self,
        config: PairConfig,
        client: AsyncHyperliquidClient,
        db: DBLogger,
        pnl: PairPnLTracker,
        dry_run: bool = True,
        kill_event: Optional[asyncio.Event] = None,
    ):
        self.config = config
        self.client = client
        self.db = db
        self.pnl = pnl
        self.dry_run = dry_run
        self._kill_event = kill_event or asyncio.Event()
        self.state = PairState(coin=config.coin)
        self._log = get_logger(f"mm2.{config.coin}")
        self._known_fill_ids: set = set()

        # Micro-structure components
        self._vol = VolatilityEstimator(window_sec=120.0, max_samples=500)
        self._book = BookAnalyzer()
        self._flow = FlowAnalyzer(window_sec=30.0)
        self._seen_market_trades: set = set()

        # Trend detector (prix sur fenêtre glissante)
        self._price_history: deque[tuple[float, float]] = deque(maxlen=300)

        # État des quotes actuels (pour smart cancel/replace)
        self._current_orders: list[dict] = []  # {"oid", "is_buy", "price", "size"}
        self._last_quoted_mid: float = 0.0
        self._last_quote_time: float = 0.0
        self._cycle_count: int = 0

        # Cache inventaire (refresh seulement lors du requote complet)
        self._cached_inventory: float = 0.0
        self._inventory_last_refresh: float = 0.0

        # Cached funding rate — refreshed every 60s
        self._cached_funding: float = 0.0
        self._funding_fetch_time: float = 0.0

        # Entry price from API
        self._cached_entry_price: float = 0.0

        # State machine
        self._state: str = "MONITORING"
        self._safe_since: float = 0.0       # Timestamp flow safe continu
        self._closing_start: float = 0.0     # Quand on est entré en CLOSING
        self._emergency_cooldown_until: float = 0.0

    # =====================================================================
    # Main loop — event-driven
    # =====================================================================

    async def run(self) -> None:
        """Boucle principale event-driven."""
        self._log.info("v2_started", coin=self.config.coin, dry_run=self.dry_run,
                       mode="smart_quoting")

        await asyncio.sleep(random.uniform(0, 1.0))

        while not self._kill_event.is_set():
            try:
                await self._fast_cycle()
                self.state.error = None
            except Exception as e:
                self.state.error = str(e)
                self._log.error("cycle_error", coin=self.config.coin, error=str(e))
                await asyncio.sleep(3)
                continue

            # Poll rapide
            try:
                await asyncio.wait_for(
                    self._kill_event.wait(),
                    timeout=self.POLL_INTERVAL_SEC,
                )
                break
            except asyncio.TimeoutError:
                pass

        await self._cleanup()
        self.state.is_running = False

    async def _fast_cycle(self) -> None:
        """Cycle rapide: lire le marché, state machine."""
        coin = self.config.coin
        self._cycle_count += 1

        # 1. UN SEUL appel API: L2 book (donne mid + bid/ask + profondeur)
        l2 = await self.client.get_l2_book(coin, n_levels=5)
        mid, best_bid, best_ask, bid_depth, ask_depth = self._book.parse_l2(l2)

        if mid <= 0 or best_bid <= 0 or best_ask <= 0:
            return

        # 2. Enregistrer le prix
        self._vol.record(mid)
        self._price_history.append((time.time(), mid))

        # 3. Récupérer et analyser les trades du marché (tous les 2 cycles)
        if self._cycle_count % 2 == 0:
            try:
                market_trades = await self.client.get_recent_market_trades(coin, 50)
                for t in market_trades:
                    trade_id = (t.get("tid", ""), t.get("time", ""))
                    if trade_id not in self._seen_market_trades:
                        self._seen_market_trades.add(trade_id)
                        side = "buy" if t.get("side") == "B" else "sell"
                        size_usd = float(t.get("px", 0)) * float(t.get("sz", 0))
                        self._flow.record_trade(side, size_usd)
            except Exception:
                pass
            # Trim seen trades set to prevent unbounded growth
            if len(self._seen_market_trades) > 500:
                self._seen_market_trades = set(list(self._seen_market_trades)[-200:])

        # 4. Check fills (tous les 3 cycles)
        if not self.dry_run and self._cycle_count % 3 == 0:
            await self._check_fills(mid)

        # 5. Refresh inventory
        if not self.dry_run and self._cycle_count % 3 == 0:
            inventory = await self._get_inventory()
            self._cached_inventory = inventory
            self._inventory_last_refresh = time.time()

        inventory = self._cached_inventory
        inventory_usd = inventory * mid

        # 6. Compute unrealized PnL from API entry price
        unrealized = self.pnl.get_unrealized_pnl(mid)

        # 7. State machine dispatch
        if self._state == "MONITORING":
            await self._handle_monitoring(mid, best_bid, best_ask, bid_depth, ask_depth)
        elif self._state == "QUOTING":
            await self._handle_quoting(mid, best_bid, best_ask, bid_depth, ask_depth,
                                       inventory, inventory_usd, unrealized)
        elif self._state == "CLOSING":
            await self._handle_closing(mid, inventory, inventory_usd, unrealized)
        elif self._state == "EMERGENCY_CLOSE":
            await self._handle_emergency(mid, inventory)

        # 8. Update state pour le dashboard
        self._update_state(mid, best_bid, best_ask)

        # 9. Snapshot DB (pas chaque cycle)
        if self._cycle_count % 5 == 0:
            await self._log_snapshot(mid, best_bid, best_ask, inventory, inventory_usd)

    # =====================================================================
    # State machine handlers
    # =====================================================================

    async def _handle_monitoring(
        self, mid: float, best_bid: float, best_ask: float,
        bid_depth: float, ask_depth: float,
    ) -> None:
        """MONITORING: Observer le flow, quoter quand c'est safe."""
        # Check emergency cooldown
        if time.time() < self._emergency_cooldown_until:
            return

        is_safe, reason = self._flow.is_safe_to_quote()

        if is_safe:
            if self._safe_since == 0:
                self._safe_since = time.time()
            # Attendre N secondes de flow safe avant de quoter
            if time.time() - self._safe_since >= self.SAFE_FLOW_WAIT_SEC:
                self._state = "QUOTING"
                self._log.info("state_change", new_state="QUOTING", reason="flow_safe")
                # Place les quotes immédiatement
                await self._full_requote(mid, best_bid, best_ask, bid_depth, ask_depth)
                self._last_quoted_mid = mid
                self._last_quote_time = time.time()
        else:
            self._safe_since = 0  # Reset le timer
            # S'assurer qu'il n'y a pas d'ordres dans le book
            if self._current_orders and not self.dry_run:
                try:
                    await self.client.cancel_coin_orders(self.config.coin)
                except Exception:
                    pass
                self._current_orders = []
                self._log.info("quotes_pulled", reason=reason)

    async def _handle_quoting(
        self, mid: float, best_bid: float, best_ask: float,
        bid_depth: float, ask_depth: float,
        inventory: float, inventory_usd: float, unrealized: float,
    ) -> None:
        """QUOTING: Quotes actives. Surveiller le flow et les fills."""
        # Check si le flow est devenu toxique
        is_safe, reason = self._flow.is_safe_to_quote()
        if not is_safe:
            # Pull toutes les quotes
            if not self.dry_run:
                try:
                    await self.client.cancel_coin_orders(self.config.coin)
                except Exception:
                    pass
            self._current_orders = []
            self._state = "MONITORING"
            self._safe_since = 0
            self._log.info("state_change", new_state="MONITORING", reason=reason)
            return

        # Check si on a été fill (inventaire non-nul)
        if abs(inventory_usd) > 10.0:
            self._state = "CLOSING"
            self._closing_start = time.time()
            self._log.info("state_change", new_state="CLOSING",
                           inventory_usd=round(inventory_usd, 2))
            # Cancel quotes normales immédiatement, on va placer des close orders
            if not self.dry_run:
                try:
                    await self.client.cancel_coin_orders(self.config.coin)
                except Exception:
                    pass
                self._current_orders = []
            return

        # Requote normal si nécessaire
        now = time.time()
        mid_moved_bps = 0.0
        if self._last_quoted_mid > 0:
            mid_moved_bps = abs(mid - self._last_quoted_mid) / self._last_quoted_mid * 10_000

        needs_requote = (
            self._last_quoted_mid == 0
            or mid_moved_bps >= self.REQUOTE_THRESHOLD_BPS
            or (now - self._last_quote_time) >= self.FORCE_REQUOTE_SEC
        )

        if needs_requote:
            await self._full_requote(mid, best_bid, best_ask, bid_depth, ask_depth)
            self._last_quoted_mid = mid
            self._last_quote_time = now

    async def _handle_closing(
        self, mid: float, inventory: float, inventory_usd: float,
        unrealized: float,
    ) -> None:
        """CLOSING: Fermer la position en profit si possible."""
        # STOP LOSS: si perte > $0.30, emergency close
        if unrealized < -0.30:
            self._state = "EMERGENCY_CLOSE"
            self._log.info("state_change", new_state="EMERGENCY_CLOSE",
                           unrealized=round(unrealized, 4))
            return

        # Si position fermée, retour en monitoring
        if abs(inventory_usd) < 5.0:
            self._state = "MONITORING"
            self._safe_since = 0
            self._log.info("state_change", new_state="MONITORING",
                           reason="position_closed")
            # Cancel any remaining close orders
            if self._current_orders and not self.dry_run:
                try:
                    await self.client.cancel_coin_orders(self.config.coin)
                except Exception:
                    pass
                self._current_orders = []
            return

        # Placer des ordres close tight
        time_in_closing = time.time() - self._closing_start

        if unrealized > 0:
            # En profit → close à 2 bps
            close_spread_bps = 2.0
        elif time_in_closing > 120:
            # Timeout → élargir le close pour forcer la sortie
            close_spread_bps = 5.0
        else:
            # Légère perte → close à 3 bps
            close_spread_bps = 3.0

        close_offset = close_spread_bps / 10_000

        # Anti-flip: cap size to actual inventory
        close_size = abs(inventory)

        if inventory > 0:
            # Long → sell to close
            close_price = mid * (1 + close_offset)
            close_price = self._round_price(close_price)
            close_order = {
                "coin": self.config.coin,
                "is_buy": False,
                "size": close_size,
                "price": close_price,
                "post_only": True,
                "level": 0,
            }
        else:
            # Short → buy to close
            close_price = mid * (1 - close_offset)
            close_price = self._round_price(close_price)
            close_order = {
                "coin": self.config.coin,
                "is_buy": True,
                "size": close_size,
                "price": close_price,
                "post_only": True,
                "level": 0,
            }

        # Cancel et replace seulement si ordre a changé
        if not self.dry_run and self._orders_changed([close_order]):
            try:
                await self.client.cancel_coin_orders(self.config.coin)
            except Exception:
                pass

            results = await self.client.place_bulk_orders([close_order])
            self._current_orders = []
            if results and results[0].success:
                self._current_orders = [{
                    "oid": results[0].oid,
                    "is_buy": close_order["is_buy"],
                    "price": close_order["price"],
                    "size": close_order["size"],
                }]
            self._log.info("close_order_placed",
                           spread_bps=close_spread_bps,
                           unrealized=round(unrealized, 4),
                           time_in_closing=round(time_in_closing, 0))

    async def _handle_emergency(self, mid: float, inventory: float) -> None:
        """EMERGENCY_CLOSE: Fermer au market immédiatement."""
        if abs(inventory) < 1e-12:
            self._state = "MONITORING"
            self._emergency_cooldown_until = time.time() + 60.0
            self._safe_since = 0
            self._log.info("state_change", new_state="MONITORING",
                           reason="emergency_done", cooldown_sec=60)
            return

        coin = self.config.coin
        self._log.warning(
            "emergency_close",
            coin=coin,
            inventory=round(inventory, 6),
            mid=round(mid, 4),
        )

        try:
            await self.client.cancel_coin_orders(coin)
            self._current_orders = []

            is_buy = inventory < 0
            size = abs(inventory)
            result = await self.client.market_order(
                coin=coin,
                is_buy=is_buy,
                size=size,
                reduce_only=True,
            )

            if result.success:
                self._log.info("emergency_close_done", size=round(size, 6))
            else:
                self._log.error("emergency_close_failed", error=result.error)
        except Exception as e:
            self._log.error("emergency_close_error", error=str(e))

        self._state = "MONITORING"
        self._emergency_cooldown_until = time.time() + 60.0
        self._safe_since = 0
        self._cached_inventory = 0.0

    # =====================================================================
    # Full requote (used in QUOTING state)
    # =====================================================================

    async def _full_requote(
        self, mid: float, best_bid: float, best_ask: float,
        bid_depth: float, ask_depth: float,
    ) -> None:
        """Recalcule et place de nouvelles quotes."""
        coin = self.config.coin

        # --- Inventaire (refresh et cache) ---
        inventory = await self._get_inventory()
        self._cached_inventory = inventory
        self._inventory_last_refresh = time.time()
        inventory_usd = inventory * mid

        # --- Signaux micro-structure ---
        sigma_bps = self._vol.get_sigma_bps()
        imbalance = self._book.get_imbalance(bid_depth, ask_depth)
        trend_bps = self._get_trend_bps()

        # --- Calcul du spread optimal (Avellaneda-Stoikov) ---
        spread_bps = self._compute_as_spread()

        # --- Reservation price (mid ajusté par inventaire) ---
        reservation_mid = self._compute_reservation_price(mid, inventory)

        # --- Application de l'imbalance ---
        imbalance_shift = imbalance * self.IMBALANCE_SKEW_FACTOR * spread_bps / 10_000 * mid
        reservation_mid += imbalance_shift

        # --- Funding rate bias ---
        now = time.time()
        if now - self._funding_fetch_time > 60.0:
            try:
                self._cached_funding = await self.client.get_funding_rate(coin)
                self._funding_fetch_time = now
            except Exception:
                pass

        if abs(self._cached_funding) > 0.00005:
            funding_shift_bps = self._cached_funding * 10_000 * 0.5
            reservation_mid += funding_shift_bps / 10_000 * mid

        # --- Trend protection ---
        quoting_mode = "normal"
        abs_trend = abs(trend_bps)
        if abs_trend >= self.TREND_CUTOFF_BPS:
            quoting_mode = "one_sided"
        elif abs_trend >= self.TREND_CAUTION_BPS:
            quoting_mode = "cautious"

        # --- Construire les ordres ---
        orders = self._build_orders(
            reservation_mid=reservation_mid,
            spread_bps=spread_bps,
            inventory=inventory,
            inventory_usd=inventory_usd,
            trend_bps=trend_bps,
            quoting_mode=quoting_mode,
            mid=mid,
        )

        # --- Smart cancel/replace ---
        if not self.dry_run:
            if self._orders_changed(orders):
                await self.client.cancel_coin_orders(coin)

                if orders:
                    results = await self.client.place_bulk_orders(orders)
                    self._current_orders = []

                    active_bids = 0
                    active_asks = 0
                    errors = 0

                    for order, result in zip(orders, results):
                        if result.success:
                            self._current_orders.append({
                                "oid": result.oid,
                                "is_buy": order["is_buy"],
                                "price": order["price"],
                                "size": order["size"],
                            })
                            if order["is_buy"]:
                                active_bids += 1
                            else:
                                active_asks += 1
                        else:
                            errors += 1
                            self._log.warning("order_error", coin=coin, error=result.error)

                        await self.db.log_order(
                            pair=coin,
                            side="buy" if order["is_buy"] else "sell",
                            price=order["price"],
                            size=order["size"],
                            size_usd=order["price"] * order["size"],
                            level=order.get("level", 0),
                            order_id=result.oid if result.success else 0,
                            status="resting" if result.success else "error",
                        )

                    self._log.info(
                        "quotes_placed",
                        coin=coin,
                        mid=round(mid, 4),
                        reservation=round(reservation_mid, 4),
                        bids=active_bids,
                        asks=active_asks,
                        errors=errors,
                        spread_bps=round(spread_bps, 1),
                        sigma_bps=round(sigma_bps, 1),
                        imbalance=round(imbalance, 2),
                        trend_bps=round(trend_bps, 1),
                        inventory_usd=round(inventory_usd, 2),
                        mode=quoting_mode,
                        state=self._state,
                    )
                else:
                    self._current_orders = []
        else:
            for o in orders:
                side = "BID" if o["is_buy"] else "ASK"
                self._log.info("dry_order", coin=coin, side=side,
                             price=o["price"], size=round(o["size"], 6))

        # Update state
        self.state.spread_bps = round(spread_bps, 1)
        self.state.trend_signal = round(trend_bps, 1)
        self.state.volatility_mult = round(sigma_bps, 1)
        self.state.quoting_mode = self._state
        self.state.active_bids = sum(1 for o in orders if o["is_buy"])
        self.state.active_asks = sum(1 for o in orders if not o["is_buy"])

    # =====================================================================
    # Avellaneda-Stoikov spread computation
    # =====================================================================

    def _compute_as_spread(self) -> float:
        """Calcule le spread optimal AS en bps."""
        sigma_bps = self._vol.get_sigma_bps()

        if sigma_bps < 0.5:
            return self.config.spread_bps

        base_spread = self.config.spread_bps
        vol_adjustment = self.GAMMA * sigma_bps * sigma_bps

        spread = base_spread + vol_adjustment
        return max(self.MIN_SPREAD_BPS, min(self.MAX_SPREAD_BPS, spread))

    def _compute_reservation_price(
        self, mid: float, inventory: float,
    ) -> float:
        """Calcule le reservation price (mid ajusté par l'inventaire)."""
        max_inv = self.config.max_inventory_usd

        if max_inv <= 0 or mid <= 0:
            return mid

        q_normalized = (inventory * mid) / max_inv
        q_normalized = max(-1.0, min(1.0, q_normalized))

        sigma_bps = self._vol.get_sigma_bps()

        if sigma_bps < 0.5:
            skew_bps = q_normalized * self.config.spread_bps * self.config.inventory_skew_factor
        else:
            skew_bps = q_normalized * self.GAMMA * sigma_bps * sigma_bps
            skew_bps = max(-20.0, min(20.0, skew_bps))

        return mid * (1 - skew_bps / 10_000)

    # =====================================================================
    # Order building
    # =====================================================================

    def _build_orders(
        self,
        reservation_mid: float,
        spread_bps: float,
        inventory: float,
        inventory_usd: float,
        trend_bps: float,
        quoting_mode: str,
        mid: float,
    ) -> list[dict]:
        """Construit les ordres de quoting."""
        cfg = self.config
        max_inv = cfg.max_inventory_usd
        half_spread = spread_bps / 2.0 / 10_000

        # Trend protection multipliers
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

        # Hard inventory cutoff
        if inventory_usd >= max_inv:
            allow_bids = False
        elif inventory_usd <= -max_inv:
            allow_asks = False

        orders = []
        sz_per_order = cfg.order_size_usd / mid

        # Asymmetric sizing: accelerate return to flat
        open_sz = sz_per_order
        close_sz = sz_per_order
        if abs(inventory_usd) > 20.0:
            close_sz = sz_per_order * 1.5
            open_sz = sz_per_order * 0.7

        # Anti-flip: cap reduce-side size to never flip position
        if abs(inventory_usd) >= 5.0 and abs(inventory) > 1e-12:
            max_reduce = abs(inventory) / max(1, cfg.num_levels)
            close_sz = min(close_sz, max_reduce)

        cumulative_buy_usd = 0.0
        cumulative_sell_usd = 0.0

        for i in range(cfg.num_levels):
            level_offset = i * cfg.level_spacing_bps / 10_000

            if allow_bids:
                cumulative_buy_usd += cfg.order_size_usd
                if abs(inventory_usd + cumulative_buy_usd) < max_inv:
                    bid_price = reservation_mid * (1 - half_spread * bid_mult - level_offset)
                    bid_price = self._round_price(bid_price)
                    bid_sz = close_sz if inventory < 0 else open_sz
                    orders.append({
                        "coin": cfg.coin,
                        "is_buy": True,
                        "size": bid_sz,
                        "price": bid_price,
                        "post_only": True,
                        "level": i,
                    })

            if allow_asks:
                cumulative_sell_usd += cfg.order_size_usd
                if abs(inventory_usd - cumulative_sell_usd) < max_inv:
                    ask_price = reservation_mid * (1 + half_spread * ask_mult + level_offset)
                    ask_price = self._round_price(ask_price)
                    ask_sz = close_sz if inventory > 0 else open_sz
                    orders.append({
                        "coin": cfg.coin,
                        "is_buy": False,
                        "size": ask_sz,
                        "price": ask_price,
                        "post_only": True,
                        "level": i,
                    })

        return orders

    # =====================================================================
    # Smart cancel/replace
    # =====================================================================

    def _orders_changed(self, new_orders: list[dict]) -> bool:
        """Vérifie si les nouveaux ordres diffèrent significativement des actuels."""
        if len(new_orders) != len(self._current_orders):
            return True

        if not self._current_orders:
            return True

        old_sorted = sorted(self._current_orders, key=lambda o: (o["is_buy"], o["price"]))
        new_sorted = sorted(new_orders, key=lambda o: (o["is_buy"], o["price"]))

        for old, new in zip(old_sorted, new_sorted):
            if old["is_buy"] != new["is_buy"]:
                return True

            if old["price"] > 0:
                price_diff_bps = abs(old["price"] - new["price"]) / old["price"] * 10_000
                if price_diff_bps > 1.0:
                    return True

        return False

    # =====================================================================
    # Trend detection
    # =====================================================================

    def _get_trend_bps(self) -> float:
        """Mouvement directionnel sur la fenêtre glissante, en bps."""
        cutoff = time.time() - self.TREND_WINDOW_SEC
        window = [(t, p) for t, p in self._price_history if t >= cutoff]

        if len(window) < 5:
            return 0.0

        n = len(window)
        t0 = window[0][0]
        sum_t = sum(t - t0 for t, _ in window)
        sum_p = sum(p for _, p in window)
        sum_tp = sum((t - t0) * p for t, p in window)
        sum_t2 = sum((t - t0) ** 2 for t, _ in window)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-12:
            return 0.0

        slope = (n * sum_tp - sum_t * sum_p) / denom
        avg_price = sum_p / n

        if avg_price <= 0:
            return 0.0

        total_move_bps = (slope * self.TREND_WINDOW_SEC / avg_price) * 10_000
        return total_move_bps

    # =====================================================================
    # Inventory / Fills
    # =====================================================================

    async def _get_inventory(self) -> float:
        """Récupère l'inventaire depuis l'API + cache entry_price."""
        if self.dry_run:
            return self.pnl.current_inventory

        try:
            account = await self.client.get_account_state()
            for pos in account.positions:
                if pos.coin == self.config.coin:
                    self._cached_entry_price = pos.entry_px
                    return pos.size
            self._cached_entry_price = 0.0
            return 0.0
        except Exception as e:
            self._log.warning("inventory_fetch_error", error=str(e))
            return self.pnl.current_inventory

    async def _check_fills(self, mid_price: float) -> None:
        """Poll les fills récents."""
        try:
            fills = await self.client.get_recent_fills()
        except Exception as e:
            self._log.warning("fills_poll_error", error=str(e))
            return

        for fill in fills:
            if fill.get("coin") != self.config.coin:
                continue

            fill_id = (fill.get("oid", 0), fill.get("time", ""), fill.get("side", ""))
            if fill_id in self._known_fill_ids:
                continue
            self._known_fill_ids.add(fill_id)

            side = "buy" if fill.get("side") == "B" else "sell"
            price = float(fill.get("px", 0))
            size = abs(float(fill.get("sz", 0)))
            fee = float(fill.get("fee", 0))

            if price <= 0 or size <= 0:
                continue

            result = self.pnl.record_fill(
                side=side, price=price, size=size, fee=fee,
                timestamp=fill.get("time", ""), mid_price=mid_price,
            )

            await self.db.log_fill(
                pair=self.config.coin, side=side, price=price,
                size=size, size_usd=price * size, fee=fee,
                mid_price_at_fill=mid_price,
                spread_captured_bps=result["spread_captured_bps"],
                inventory_after=result["inventory_after"],
                order_id=fill.get("oid", 0),
            )

            self._log.info(
                "fill_detected",
                coin=self.config.coin,
                side=side,
                price=price,
                size=size,
                realized_delta=round(result["realized_pnl_delta"], 4),
                inventory=round(result["inventory_after"], 6),
                state=self._state,
            )

    # =====================================================================
    # State update & DB snapshot
    # =====================================================================

    def _update_state(self, mid: float, best_bid: float, best_ask: float) -> None:
        """Met à jour l'état pour le dashboard."""
        spread_bps = ((best_ask - best_bid) / mid) * 10_000 if mid > 0 else 0
        inventory = self.pnl.current_inventory
        unrealized = self.pnl.get_unrealized_pnl(mid)
        realized = self.pnl.state.realized_pnl
        total = self.pnl.get_total_pnl(mid)

        self.state.mid_price = mid
        self.state.best_bid = best_bid
        self.state.best_ask = best_ask
        self.state.spread_bps = round(spread_bps, 1)
        self.state.inventory = inventory
        self.state.inventory_usd = round(inventory * mid, 2)
        self.state.realized_pnl = round(realized, 4)
        self.state.unrealized_pnl = round(unrealized, 4)
        self.state.total_pnl = round(total, 4)
        self.state.fills_count = self.pnl.fills_count
        self.state.last_update = time.time()
        self.state.quoting_mode = self._state

    async def _log_snapshot(
        self, mid: float, best_bid: float, best_ask: float,
        inventory: float, inventory_usd: float,
    ) -> None:
        """Log snapshot to DB."""
        unrealized = self.pnl.get_unrealized_pnl(mid)
        realized = self.pnl.state.realized_pnl
        total = self.pnl.get_total_pnl(mid)
        await self.db.log_snapshot(
            pair=self.config.coin, mid_price=mid,
            best_bid=best_bid, best_ask=best_ask,
            spread_bps=((best_ask - best_bid) / mid) * 10_000,
            inventory=inventory, inventory_usd=inventory_usd,
            unrealized_pnl=unrealized, realized_pnl=realized, total_pnl=total,
        )

    # =====================================================================
    # Helpers
    # =====================================================================

    @staticmethod
    def _round_price(price: float) -> float:
        """Arrondit le prix selon les règles Hyperliquid."""
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

    async def _cleanup(self) -> None:
        """Annule tous les ordres."""
        self._log.info("v2_stopping", coin=self.config.coin)
        if not self.dry_run:
            try:
                await self.client.cancel_coin_orders(self.config.coin)
                self._log.info("orders_cancelled", coin=self.config.coin)
            except Exception as e:
                self._log.error("cleanup_error", error=str(e))
