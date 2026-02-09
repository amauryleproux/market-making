"""Auto-Screener de paires pour le Market Maker.

Tourne en tache asyncio parallele au bot. Scanne periodiquement
toutes les paires Hyperliquid, les score, et declenche une rotation
si une meilleure paire est trouvee.

Deux modes :
  - Scan rapide (toutes les 5 min) : scoring statique uniquement
  - Analyse profonde (toutes les 30 min) : statique + toxicite + backtest
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Awaitable

from src.client.async_wrapper import AsyncHyperliquidClient
from src.strategies.toxicity_analyzer import ToxicityAnalyzer, ToxicityMetrics
from src.strategies.backtest_integrator import BacktestIntegrator, BacktestScore
from src.utils.logger import get_logger


@dataclass
class PairScore:
    """Score d'une paire candidate."""

    coin: str
    score: float  # 0-100, composite final
    static_score: float = 0.0  # score statique (orderbook)
    toxicity_score: float = -1.0  # 0-100, -1 = pas calcule
    backtest_pnl_h: float = 0.0  # PnL/h backtest, 0 = pas de donnees

    # Donnees brutes
    spread_bps: float = 0.0
    volume_24h_usd: float = 0.0
    depth_usd: float = 0.0
    num_levels: int = 0
    funding_rate: float = 0.0
    mid_price: float = 0.0
    tick_spread_bps: float = 0.0

    # Nouvelles metriques
    adverse_selection_5s_bps: float = 0.0
    spread_stability: float = 0.0
    book_imbalance: float = 0.0
    competition: float = 0.0

    timestamp: float = field(default_factory=time.time)


@dataclass
class ScreenerConfig:
    """Configuration du screener."""

    # Intervals
    scan_interval_sec: float = 300.0
    deep_scan_interval_sec: float = 1800.0

    # Filtres
    min_volume_usd: float = 100_000
    max_volume_usd: float = 5_000_000_000  # releve pour inclure ETH/BTC
    min_spread_bps: float = 3.0
    min_score_to_trade: float = 40.0

    # Poids (nouveaux, informes par l'experience)
    w_spread: float = 0.20
    w_volume: float = 0.15
    w_depth: float = 0.10
    w_levels: float = 0.05
    w_funding: float = 0.05
    w_toxicity: float = 0.35
    w_backtest: float = 0.10

    # Rotation
    rotation_threshold: float = 15.0
    cooldown_after_rotation_sec: float = 600.0
    max_active_pairs: int = 5

    # Toxicite
    data_dir: str = "backtesting/data/market_data"
    use_live_fallback: bool = True
    min_trades_for_analysis: int = 100

    # Backtest
    results_dir: str = "backtesting/results"
    min_bt_confidence: float = 0.3

    # Listes
    blacklist: list[str] = field(default_factory=list)
    whitelist_priority: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "ScreenerConfig":
        """Charge la config depuis un fichier JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        s = data.get("screener", data)
        filters = s.get("filters", {})
        weights = s.get("weights", {})
        tox = s.get("toxicity", {})
        bt = s.get("backtest", {})
        rot = s.get("rotation", {})

        return cls(
            scan_interval_sec=s.get("scan_interval_sec", 300),
            deep_scan_interval_sec=s.get("deep_scan_interval_sec", 1800),
            min_volume_usd=filters.get("min_volume_usd", 100_000),
            max_volume_usd=filters.get("max_volume_usd", 5_000_000_000),
            min_spread_bps=filters.get("min_spread_bps", 3.0),
            min_score_to_trade=filters.get("min_score", 40),
            w_spread=weights.get("spread", 0.20),
            w_volume=weights.get("volume", 0.15),
            w_depth=weights.get("depth", 0.10),
            w_levels=weights.get("levels", 0.05),
            w_funding=weights.get("funding", 0.05),
            w_toxicity=weights.get("toxicity", 0.35),
            w_backtest=weights.get("backtest", 0.10),
            data_dir=tox.get("data_dir", "backtesting/data/market_data"),
            use_live_fallback=tox.get("use_live_fallback", True),
            min_trades_for_analysis=tox.get("min_trades_for_analysis", 100),
            results_dir=bt.get("results_dir", "backtesting/results"),
            min_bt_confidence=bt.get("min_confidence", 0.3),
            rotation_threshold=rot.get("threshold_score_diff", 15),
            cooldown_after_rotation_sec=rot.get("cooldown_sec", 600),
            max_active_pairs=rot.get("max_active_pairs", 5),
            blacklist=s.get("blacklist", []),
            whitelist_priority=s.get("whitelist_priority", []),
        )


class PairScreener:
    """Screener automatique de paires, tourne en parallele du bot."""

    def __init__(
        self,
        client: AsyncHyperliquidClient,
        config: Optional[ScreenerConfig] = None,
        on_rotation: Optional[Callable[[str, str, PairScore], Awaitable[None]]] = None,
        kill_event: Optional[asyncio.Event] = None,
    ):
        self.client = client
        self.config = config or ScreenerConfig()
        self._on_rotation = on_rotation
        self._kill_event = kill_event or asyncio.Event()
        self._log = get_logger("screener")

        # Analyseurs
        self._toxicity = ToxicityAnalyzer(self.config.data_dir)
        self._backtest = BacktestIntegrator(self.config.results_dir)

        # Etat
        self._scores: list[PairScore] = []
        self._active_coins: set[str] = set()
        self._last_rotation_time: float = 0.0
        self._last_deep_scan: float = 0.0
        self._scan_count: int = 0

        # Cache toxicite et backtest (rafraichi au deep scan)
        self._toxicity_cache: dict[str, ToxicityMetrics] = {}
        self._backtest_cache: dict[str, BacktestScore] = {}

    @property
    def scores(self) -> list[PairScore]:
        return sorted(self._scores, key=lambda s: s.score, reverse=True)

    @property
    def top_pairs(self) -> list[PairScore]:
        return [
            s
            for s in self.scores
            if s.score >= self.config.min_score_to_trade
            and s.coin not in self.config.blacklist
        ][: self.config.max_active_pairs * 2]

    def set_active_coins(self, coins: set[str]) -> None:
        self._active_coins = coins

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Boucle principale du screener."""
        self._log.info("screener_started", interval=self.config.scan_interval_sec)

        while not self._kill_event.is_set():
            try:
                # Deep scan si le temps est venu
                now = time.time()
                deep = (now - self._last_deep_scan) >= self.config.deep_scan_interval_sec
                await self._scan_cycle(deep=deep)
                if deep:
                    self._last_deep_scan = now
            except Exception as e:
                self._log.error("scan_error", error=str(e))

            try:
                await asyncio.wait_for(
                    self._kill_event.wait(),
                    timeout=self.config.scan_interval_sec,
                )
                break
            except asyncio.TimeoutError:
                pass

        self._log.info("screener_stopped")

    async def force_scan(self, deep: bool = False) -> list[PairScore]:
        """Force un scan immediat."""
        await self._scan_cycle(deep=deep)
        return self.scores

    # ------------------------------------------------------------------
    # Scan cycle
    # ------------------------------------------------------------------

    async def _scan_cycle(self, deep: bool = False) -> None:
        self._scan_count += 1
        start = time.time()
        mode = "deep" if deep else "quick"

        # 1. Recuperer tous les coins + stats en un seul appel
        all_stats = await self.client.get_all_market_stats()

        self._log.info(
            "scan_started",
            total_coins=len(all_stats),
            scan_num=self._scan_count,
            mode=mode,
        )

        # 2. Si deep scan, rafraichir les caches toxicite/backtest
        if deep:
            self._backtest_cache = self._backtest.get_all_scores()
            # Toxicite sera calculee par coin individuellement

        # 3. Scorer chaque paire
        scores = []
        for ctx in all_stats:
            coin = ctx.get("coin", "")
            if not coin or coin in self.config.blacklist:
                continue

            try:
                score = await self._score_pair(coin, ctx, deep=deep)
                if score is not None:
                    scores.append(score)
            except Exception as e:
                self._log.debug("score_error", coin=coin, error=str(e))

        self._scores = scores
        elapsed = time.time() - start

        # 4. Log top resultats
        top = self.scores[:10]
        self._log.info(
            "scan_complete",
            scored=len(scores),
            elapsed_sec=round(elapsed, 1),
            mode=mode,
            top_3=[f"{s.coin}({s.score:.0f})" for s in top[:3]],
        )

        for i, s in enumerate(top[:5]):
            active = "*" if s.coin in self._active_coins else " "
            self._log.info(
                "pair_ranked",
                rank=i + 1,
                coin=s.coin,
                score=round(s.score, 1),
                static=round(s.static_score, 1),
                toxic=round(s.toxicity_score, 1),
                spread_bps=round(s.spread_bps, 1),
                volume_k=round(s.volume_24h_usd / 1000, 0),
                active=active,
            )

        # 5. Rotation
        await self._check_rotation()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def _score_pair(
        self, coin: str, ctx: dict, deep: bool = False
    ) -> Optional[PairScore]:
        """Score une paire. ctx vient de get_all_market_stats."""
        cfg = self.config

        # --- Donnees de base depuis le contexte ---
        volume_24h = float(ctx.get("dayNtlVlm", 0))
        funding_rate = float(ctx.get("funding", 0))
        mark_px = float(ctx.get("markPx", 0))

        # Filtres eliminatoires rapides (avant l'appel L2)
        if volume_24h < cfg.min_volume_usd or volume_24h > cfg.max_volume_usd:
            return None

        # --- L2 book ---
        book = await self.client.get_l2_book(coin)
        bids = book.get("levels", [[]])[0] if book.get("levels") else []
        asks = (
            book.get("levels", [[], []])[1] if len(book.get("levels", [])) > 1 else []
        )

        if not bids or not asks:
            return None

        best_bid = float(bids[0].get("px", 0))
        best_ask = float(asks[0].get("px", 0))
        mid = (best_bid + best_ask) / 2

        if mid <= 0:
            return None

        spread_bps = ((best_ask - best_bid) / mid) * 10_000

        if spread_bps < cfg.min_spread_bps:
            return None

        # Tick spread
        if mid > 10000:
            tick = 1.0
        elif mid > 100:
            tick = 0.1
        elif mid > 1:
            tick = 0.01
        elif mid > 0.01:
            tick = 0.0001
        else:
            tick = 0.000001
        tick_spread_bps = (tick / mid) * 10_000

        # Depth (notionnel a +-0.5% du mid)
        depth = 0.0
        for level in bids[:10]:
            px, sz = float(level.get("px", 0)), float(level.get("sz", 0))
            if px >= mid * 0.995:
                depth += px * sz
        for level in asks[:10]:
            px, sz = float(level.get("px", 0)), float(level.get("sz", 0))
            if px <= mid * 1.005:
                depth += px * sz

        num_levels = len(bids) + len(asks)

        # --- Sous-scores statiques (courbes revisees) ---

        # Spread : seuil minimum releve a 8 bps
        if spread_bps < 8:
            spread_score = 10.0
        elif spread_bps < 12:
            spread_score = 30 + (spread_bps - 8) * 10  # 30-70
        elif spread_bps < 25:
            spread_score = 70 + (spread_bps - 12) * 2.3  # 70-100
        elif spread_bps < 60:
            spread_score = 100.0
        elif spread_bps < 150:
            spread_score = 100 - (spread_bps - 60) * 0.5  # 100-55
        else:
            spread_score = max(20, 55 - (spread_bps - 150) * 0.2)

        # Volume : bimodal — tres gros (ETH/BTC tier) ou petit niche
        if volume_24h < 100_000:
            vol_score = 10.0
        elif volume_24h < 500_000:
            vol_score = 30 + (volume_24h - 100_000) / 400_000 * 30  # 30-60
        elif volume_24h <= 5_000_000:
            vol_score = 60.0  # zone dangereuse
        elif volume_24h <= 50_000_000:
            vol_score = 60 + (volume_24h - 5_000_000) / 45_000_000 * 20  # 60-80
        else:
            vol_score = 80 + min(
                20, (volume_24h - 50_000_000) / 100_000_000 * 20
            )  # 80-100

        # Depth : moins = moins de concurrence
        if depth < 1_000:
            depth_score = 100.0
        elif depth < 5_000:
            depth_score = 100 - (depth - 1_000) / 4_000 * 30
        elif depth < 20_000:
            depth_score = 70 - (depth - 5_000) / 15_000 * 30
        elif depth < 100_000:
            depth_score = 40 - (depth - 20_000) / 80_000 * 20
        else:
            depth_score = 20.0

        # Levels : peu = book fin
        if num_levels <= 10:
            levels_score = 100.0
        elif num_levels <= 30:
            levels_score = 100 - (num_levels - 10) * 2.5
        elif num_levels <= 60:
            levels_score = 50 - (num_levels - 30)
        else:
            levels_score = 20.0

        # Funding : proche de 0
        abs_funding = abs(funding_rate)
        if abs_funding < 0.0001:
            funding_score = 100.0
        elif abs_funding < 0.0005:
            funding_score = 80.0
        elif abs_funding < 0.001:
            funding_score = 50.0
        else:
            funding_score = 20.0

        # Tick penalty
        tick_penalty = 1.0
        if tick_spread_bps > 8:
            tick_penalty = max(0.3, 1.0 - (tick_spread_bps - 8) / 20)

        # Score statique (normalise sur 0-100)
        static_weight_sum = (
            cfg.w_spread + cfg.w_volume + cfg.w_depth + cfg.w_levels + cfg.w_funding
        )
        static_score = (
            spread_score * cfg.w_spread
            + vol_score * cfg.w_volume
            + depth_score * cfg.w_depth
            + levels_score * cfg.w_levels
            + funding_score * cfg.w_funding
        )
        if static_weight_sum > 0:
            static_score = static_score / static_weight_sum
        static_score *= tick_penalty

        # --- Toxicite (deep scan only) ---
        tox_score = -1.0
        as_5s = 0.0
        spread_stab = 0.0
        book_imb = 0.0
        competition = 0.0

        if deep:
            tox = self._toxicity.analyze_from_parquet(coin)
            if tox is not None and tox.sample_size >= cfg.min_trades_for_analysis:
                self._toxicity_cache[coin] = tox
                tox_score = tox.toxicity_score
                as_5s = tox.adverse_selection_5s_bps
                spread_stab = tox.spread_stability
                book_imb = tox.book_imbalance
                competition = tox.competition_score

        # Utiliser le cache si on a une valeur
        if tox_score < 0 and coin in self._toxicity_cache:
            cached = self._toxicity_cache[coin]
            tox_score = cached.toxicity_score
            as_5s = cached.adverse_selection_5s_bps
            spread_stab = cached.spread_stability
            book_imb = cached.book_imbalance
            competition = cached.competition_score

        # --- Backtest ---
        bt_pnl_h = 0.0
        bt_score_val = 50.0  # neutre par defaut
        bt = self._backtest_cache.get(coin)
        if bt is not None and bt.confidence >= cfg.min_bt_confidence:
            bt_pnl_h = bt.best_pnl_per_hour
            if bt.best_pnl_per_hour > 0:
                bt_score_val = min(100, 60 + bt.best_pnl_per_hour * 100)
            elif bt.pct_profitable_configs < 0.1:
                bt_score_val = 10.0
            else:
                bt_score_val = max(10, 50 + bt.best_pnl_per_hour * 50)

        # --- Score final composite ---
        if tox_score >= 0:
            # Avec toxicite : score complet
            final = (
                static_score * static_weight_sum
                + tox_score * cfg.w_toxicity
                + bt_score_val * cfg.w_backtest
            )
        else:
            # Sans toxicite : statique seulement (rescale)
            final = static_score

        # Override fort du backtest si haute confiance
        if bt is not None and bt.confidence >= 0.5:
            if bt.best_pnl_per_hour <= 0 and bt.pct_profitable_configs < 0.1:
                final = min(final, 20)
            elif bt.best_pnl_per_hour > 0:
                final = min(100, final + min(20, bt.best_pnl_per_hour * 80))

        return PairScore(
            coin=coin,
            score=round(max(0, min(100, final)), 1),
            static_score=round(static_score, 1),
            toxicity_score=round(tox_score, 1),
            backtest_pnl_h=round(bt_pnl_h, 4),
            spread_bps=round(spread_bps, 1),
            volume_24h_usd=volume_24h,
            depth_usd=depth,
            num_levels=num_levels,
            funding_rate=funding_rate,
            mid_price=mid,
            tick_spread_bps=round(tick_spread_bps, 2),
            adverse_selection_5s_bps=round(as_5s, 2),
            spread_stability=round(spread_stab, 4),
            book_imbalance=round(book_imb, 4),
            competition=round(competition, 1),
        )

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    async def _check_rotation(self) -> None:
        cfg = self.config
        now = time.time()

        if now - self._last_rotation_time < cfg.cooldown_after_rotation_sec:
            return
        if not self._active_coins or not self._scores:
            return

        active_scores = {
            s.coin: s.score for s in self._scores if s.coin in self._active_coins
        }
        if not active_scores:
            return

        worst_active = min(active_scores, key=active_scores.get)
        worst_score = active_scores[worst_active]

        best_candidates = [
            s
            for s in self.scores
            if s.coin not in self._active_coins
            and s.coin not in cfg.blacklist
            and s.score >= cfg.min_score_to_trade
        ]

        if not best_candidates:
            return

        best_candidate = best_candidates[0]
        gap = best_candidate.score - worst_score

        if gap >= cfg.rotation_threshold:
            self._log.info(
                "rotation_proposed",
                old=worst_active,
                old_score=round(worst_score, 1),
                new=best_candidate.coin,
                new_score=round(best_candidate.score, 1),
                gap=round(gap, 1),
            )

            if self._on_rotation:
                await self._on_rotation(
                    worst_active, best_candidate.coin, best_candidate
                )
                self._last_rotation_time = now
