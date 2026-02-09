"""Stratégie de Market Making pour une seule paire.

Fonctionne en boucle asyncio, utilise les paramètres par-paire
depuis pairs_config.json, et logue chaque cycle dans SQLite.

AMÉLIORATIONS v3:
- Fix 1: Skew par mid-shift (évite market orders accidentels)
- Fix 2: Check can_buy/can_sell cumulatif par level
- Fix 3: Trend detector — élargit/coupe les quotes quand le prix trend
- Fix 4: Volatility-adjusted spread — élargit le spread quand la vol monte
- Fix 5: Hard inventory cutoff — coupe strictement un côté au-delà du max
"""

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.client.async_wrapper import AsyncHyperliquidClient
from src.monitoring.db_logger import DBLogger
from src.monitoring.pnl_tracker import PairPnLTracker
from src.utils.logger import get_logger


@dataclass
class PairConfig:
    """Configuration pour une paire."""
    coin: str
    enabled: bool
    spread_bps: float
    order_size_usd: float
    num_levels: int
    level_spacing_bps: float
    max_inventory_usd: float
    inventory_skew_factor: float
    refresh_interval_sec: float
    # Take-profit sur inventory
    take_profit_enabled: bool = True
    take_profit_pct: float = 0.5        # % du notionnel pour déclencher
    take_profit_min_usd: float = 0.10   # PnL min en USD pour déclencher


@dataclass
class PairState:
    """État courant d'une paire (exposé au dashboard)."""
    coin: str
    mid_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_bps: float = 0.0
    inventory: float = 0.0      # en unités du coin
    inventory_usd: float = 0.0
    active_bids: int = 0
    active_asks: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    fills_count: int = 0
    last_update: float = 0.0
    is_running: bool = True
    error: Optional[str] = None
    # Nouvelles métriques v3
    trend_signal: float = 0.0       # en bps, négatif = bearish
    volatility_mult: float = 1.0    # multiplicateur de spread
    quoting_mode: str = "normal"    # normal / cautious / one_sided


class PriceTracker:
    """Suit l'historique des prix pour détecter trends et volatilité.

    Utilise une régression linéaire sur une fenêtre glissante pour
    mesurer la direction du prix, et l'écart-type des returns pour
    la volatilité.
    """

    def __init__(self, window_seconds: float = 60.0, max_samples: int = 200):
        self._window_sec = window_seconds
        self._prices: deque[tuple[float, float]] = deque(maxlen=max_samples)

    def record(self, price: float) -> None:
        """Enregistre un nouveau prix."""
        self._prices.append((time.time(), price))

    def _trim(self) -> list[tuple[float, float]]:
        """Retourne les prix dans la fenêtre temporelle."""
        cutoff = time.time() - self._window_sec
        return [(t, p) for t, p in self._prices if t >= cutoff]

    def get_trend_bps(self) -> float:
        """Mouvement directionnel sur la fenêtre, en bps.

        Positif = prix monte, Négatif = prix baisse.
        Régression linéaire pour lisser le bruit.
        """
        window = self._trim()
        if len(window) < 3:
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

        # Slope en prix/sec → bps sur toute la fenêtre
        total_move_bps = (slope * self._window_sec / avg_price) * 10_000
        return total_move_bps

    def get_volatility_bps(self) -> float:
        """Volatilité (écart-type des returns) sur la fenêtre, en bps."""
        window = self._trim()
        if len(window) < 5:
            return 0.0

        prices = [p for _, p in window]
        returns_bps = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                ret = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 10_000
                returns_bps.append(ret)

        if len(returns_bps) < 3:
            return 0.0

        mean = sum(returns_bps) / len(returns_bps)
        variance = sum((r - mean) ** 2 for r in returns_bps) / len(returns_bps)
        return variance ** 0.5


class PairMarketMaker:
    """Market maker pour une seule paire, exécuté en tâche asyncio."""

    # --- Paramètres du trend detector ---
    TREND_WINDOW_SEC = 60.0         # Fenêtre d'observation
    TREND_CAUTION_BPS = 15.0        # Mode cautious : spread x1.5 côté dangereux
    TREND_CUTOFF_BPS = 30.0         # Mode one_sided : coupe côté dangereux
    VOL_BASELINE_BPS = 3.0          # Volatilité "normale" de référence
    VOL_MAX_MULT = 2.5              # Multiplicateur max du spread

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
        self._log = get_logger(f"mm.{config.coin}")
        self._known_fill_ids: set = set()

        # Trend & volatility tracker
        self._price_tracker = PriceTracker(
            window_seconds=self.TREND_WINDOW_SEC,
            max_samples=200,
        )

    async def run(self) -> None:
        """Boucle principale : tourne jusqu'au kill_event."""
        self._log.info("pair_started", coin=self.config.coin, dry_run=self.dry_run)

        # Petit jitter initial pour décaler les paires
        await asyncio.sleep(random.uniform(0, 1.5))

        while not self._kill_event.is_set():
            try:
                await self._cycle()
                self.state.error = None
            except Exception as e:
                self.state.error = str(e)
                self._log.error("cycle_error", coin=self.config.coin, error=str(e))
                await asyncio.sleep(5)
                continue

            # Attente interruptible par le kill switch
            try:
                await asyncio.wait_for(
                    self._kill_event.wait(),
                    timeout=self.config.refresh_interval_sec,
                )
                break
            except asyncio.TimeoutError:
                pass

        await self._cleanup()
        self.state.is_running = False

    async def _cycle(self) -> None:
        """Un cycle complet de quoting."""
        coin = self.config.coin

        # 1. Récupérer le prix
        mid = await self.client.get_mid(coin)
        best_bid, best_ask = await self.client.get_best_bid_ask(coin)

        if mid <= 0 or best_bid <= 0 or best_ask <= 0:
            self._log.warning("invalid_prices", coin=coin, mid=mid, bid=best_bid, ask=best_ask)
            return

        spread_bps = ((best_ask - best_bid) / mid) * 10_000

        # Enregistrer le prix pour le trend detector
        self._price_tracker.record(mid)

        # 2. Récupérer la position via API (source of truth)
        api_inventory = 0.0
        if not self.dry_run:
            account = await self.client.get_account_state()
            for pos in account.positions:
                if pos.coin == coin:
                    api_inventory = pos.size
                    break

        # 3. Vérifier les nouveaux fills
        if not self.dry_run:
            await self._check_fills(mid)

        # Utiliser l'inventaire API pour le skew, le PnL tracker pour le PnL
        inventory = api_inventory if not self.dry_run else self.pnl.current_inventory
        inventory_usd = inventory * mid

        # 3.5 Check take-profit sur l'inventory
        if not self.dry_run:
            tp_executed = await self._check_take_profit(mid, inventory)
            if tp_executed:
                account = await self.client.get_account_state()
                inventory = 0.0
                for pos in account.positions:
                    if pos.coin == coin:
                        inventory = pos.size
                        break
                inventory_usd = inventory * mid

        # 4. Analyser trend et volatilité
        trend_bps = self._price_tracker.get_trend_bps()
        vol_bps = self._price_tracker.get_volatility_bps()

        # Multiplicateur de volatilité
        if self.VOL_BASELINE_BPS > 0 and vol_bps > self.VOL_BASELINE_BPS:
            vol_mult = min(vol_bps / self.VOL_BASELINE_BPS, self.VOL_MAX_MULT)
        else:
            vol_mult = 1.0

        # Mode de quoting
        abs_trend = abs(trend_bps)
        if abs_trend >= self.TREND_CUTOFF_BPS:
            quoting_mode = "one_sided"
        elif abs_trend >= self.TREND_CAUTION_BPS:
            quoting_mode = "cautious"
        else:
            quoting_mode = "normal"

        # 5. Construire les quotes
        orders = self._compute_quotes(
            mid_price=mid,
            inventory=inventory,
            trend_bps=trend_bps,
            vol_mult=vol_mult,
            quoting_mode=quoting_mode,
        )

        # 6. Annuler les anciens ordres
        if not self.dry_run:
            await self.client.cancel_coin_orders(coin)

        # 7. Placer les nouveaux ordres
        active_bids = 0
        active_asks = 0

        if self.dry_run:
            for o in orders:
                side = "BID" if o["is_buy"] else "ASK"
                self._log.info(
                    "dry_order",
                    coin=coin,
                    side=side,
                    price=o["price"],
                    size=round(o["size"], 6),
                    level=o["level"],
                    mid=mid,
                )
            active_bids = sum(1 for o in orders if o["is_buy"])
            active_asks = sum(1 for o in orders if not o["is_buy"])
        elif orders:
            results = await self.client.place_bulk_orders(orders)

            for order, result in zip(orders, results):
                side = "buy" if order["is_buy"] else "sell"
                status = "resting" if result.success else "error"
                if result.success:
                    if order["is_buy"]:
                        active_bids += 1
                    else:
                        active_asks += 1

                await self.db.log_order(
                    pair=coin,
                    side=side,
                    price=order["price"],
                    size=order["size"],
                    size_usd=order["price"] * order["size"],
                    level=order["level"],
                    order_id=result.oid if result.success else 0,
                    status=status,
                )

            errors = sum(1 for r in results if not r.success)
            skew_ratio = inventory_usd / self.config.max_inventory_usd if self.config.max_inventory_usd > 0 else 0
            self._log.info(
                "quotes_placed",
                coin=coin,
                mid=round(mid, 6),
                bids=active_bids,
                asks=active_asks,
                errors=errors,
                inventory_usd=round(inventory_usd, 2),
                skew=round(skew_ratio, 3),
                trend_bps=round(trend_bps, 1),
                vol_mult=round(vol_mult, 2),
                mode=quoting_mode,
            )

            for r in results:
                if not r.success:
                    self._log.warning("order_error", coin=coin, error=r.error)

        # 8. Mettre à jour l'état (lu par le dashboard)
        unrealized = self.pnl.get_unrealized_pnl(mid)
        realized = self.pnl.state.realized_pnl
        total = self.pnl.get_total_pnl(mid)

        self.state.mid_price = mid
        self.state.best_bid = best_bid
        self.state.best_ask = best_ask
        self.state.spread_bps = round(spread_bps, 1)
        self.state.inventory = inventory
        self.state.inventory_usd = round(inventory_usd, 2)
        self.state.active_bids = active_bids
        self.state.active_asks = active_asks
        self.state.realized_pnl = round(realized, 4)
        self.state.unrealized_pnl = round(unrealized, 4)
        self.state.total_pnl = round(total, 4)
        self.state.fills_count = self.pnl.fills_count
        self.state.trend_signal = round(trend_bps, 1)
        self.state.volatility_mult = round(vol_mult, 2)
        self.state.quoting_mode = quoting_mode

        # 9. Log snapshot dans SQLite
        await self.db.log_snapshot(
            pair=coin,
            mid_price=mid,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            inventory=inventory,
            inventory_usd=inventory_usd,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_pnl=total,
        )

    async def _check_take_profit(self, mid_price: float, inventory: float) -> bool:
        """Vérifie si on doit prendre le profit sur l'inventory.

        Retourne True si une fermeture a été exécutée.
        """
        if not self.config.take_profit_enabled or self.dry_run:
            return False

        inventory_usd = abs(inventory * mid_price)
        if inventory_usd < 5.0:
            return False

        unrealized = self.pnl.get_unrealized_pnl(mid_price)

        threshold_usd = max(
            self.config.take_profit_min_usd,
            inventory_usd * self.config.take_profit_pct / 100.0,
        )

        if unrealized >= threshold_usd:
            self._log.info(
                "take_profit_triggered",
                coin=self.config.coin,
                inventory=round(inventory, 4),
                inventory_usd=round(inventory_usd, 2),
                unrealized_pnl=round(unrealized, 4),
                threshold_usd=round(threshold_usd, 4),
            )

            try:
                await self.client.cancel_coin_orders(self.config.coin)

                is_buy = inventory < 0
                size = abs(inventory)

                result = await self.client.market_order(
                    coin=self.config.coin,
                    is_buy=is_buy,
                    size=size,
                    reduce_only=True,
                )

                if result.success:
                    self._log.info(
                        "take_profit_executed",
                        coin=self.config.coin,
                        side="buy" if is_buy else "sell",
                        size=round(size, 4),
                        unrealized_pnl=round(unrealized, 4),
                    )
                    return True
                else:
                    self._log.warning(
                        "take_profit_failed",
                        coin=self.config.coin,
                        error=result.error,
                    )
            except Exception as e:
                self._log.error(
                    "take_profit_error",
                    coin=self.config.coin,
                    error=str(e),
                )

        return False

    async def _check_fills(self, mid_price: float) -> None:
        """Poll les fills récents et enregistre les nouveaux."""
        try:
            fills = await self.client.get_recent_fills()
        except Exception as e:
            self._log.warning("fills_poll_error", coin=self.config.coin, error=str(e))
            return

        for fill in fills:
            if fill.get("coin") != self.config.coin:
                continue

            # Identifiant unique du fill
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
                side=side,
                price=price,
                size=size,
                fee=fee,
                timestamp=fill.get("time", ""),
                mid_price=mid_price,
            )

            await self.db.log_fill(
                pair=self.config.coin,
                side=side,
                price=price,
                size=size,
                size_usd=price * size,
                fee=fee,
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
            )

    def _compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        trend_bps: float = 0.0,
        vol_mult: float = 1.0,
        quoting_mode: str = "normal",
    ) -> list[dict]:
        """Calcule les niveaux de quoting.

        Logique :
        1. Skew par mid-shift : décale le mid selon l'inventaire
        2. Spread ajusté par la volatilité : spread × vol_mult
        3. Trend protection :
           - normal : quotes symétriques
           - cautious : spread ×1.5 du côté dangereux
           - one_sided : coupe les quotes du côté dangereux
        4. Hard inventory cutoff : si inventory >= max, seules les
           réductions sont permises (pas de check cumulatif nécessaire)
        5. Check cumulatif par level pour le reste
        """
        cfg = self.config
        inventory_usd = inventory * mid_price
        max_inv = cfg.max_inventory_usd

        # --- Skew : décalage du mid ---
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
                bid_mult = 1.5   # prix baisse → bids plus loin
            else:
                ask_mult = 1.5   # prix monte → asks plus loin

        elif quoting_mode == "one_sided":
            if trend_bps < 0:
                allow_bids = False
                self._log.info("trend_protection", coin=cfg.coin,
                               action="bids_cut", trend_bps=round(trend_bps, 1))
            else:
                allow_asks = False
                self._log.info("trend_protection", coin=cfg.coin,
                               action="asks_cut", trend_bps=round(trend_bps, 1))

        # --- Hard inventory cutoff ---
        if inventory_usd >= max_inv:
            allow_bids = False
        elif inventory_usd <= -max_inv:
            allow_asks = False

        self._log.debug(
            "quote_params",
            coin=cfg.coin,
            inventory_usd=round(inventory_usd, 2),
            skew=round(skew_ratio, 4),
            adj_mid=round(adjusted_mid, 6),
            spread_bps=round(effective_spread_bps, 1),
            trend=round(trend_bps, 1),
            vol_mult=round(vol_mult, 2),
            mode=quoting_mode,
            bids=allow_bids,
            asks=allow_asks,
        )

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
                    bid_price = self._round_price(bid_price)
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
                    ask_price = self._round_price(ask_price)
                    orders.append({
                        "coin": cfg.coin,
                        "is_buy": False,
                        "size": sz_per_order,
                        "price": ask_price,
                        "post_only": True,
                        "level": i,
                    })

        return orders

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
        """Annule tous les ordres de cette paire."""
        self._log.info("pair_stopping", coin=self.config.coin)
        if not self.dry_run:
            try:
                await self.client.cancel_coin_orders(self.config.coin)
                self._log.info("orders_cancelled", coin=self.config.coin)
            except Exception as e:
                self._log.error("cleanup_error", coin=self.config.coin, error=str(e))
