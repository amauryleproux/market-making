"""Analyse de toxicite des flux pour le scoring de paires.

Mesure l'adverse selection, la directionnalite et le clustering
des trades pour determiner si un token est viable pour le market making.

Supporte deux modes :
  - Batch : analyse des fichiers Parquet du collecteur
  - Live  : appels API directs (moins precis, pas d'historique profond)
"""

import time
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger("toxicity")


@dataclass
class ToxicityMetrics:
    """Metriques de toxicite des flux pour un token."""

    coin: str

    # Adverse selection : mouvement moyen post-trade dans la direction du taker (bps)
    adverse_selection_1s_bps: float = 0.0
    adverse_selection_5s_bps: float = 0.0
    adverse_selection_30s_bps: float = 0.0
    adverse_selection_60s_bps: float = 0.0

    # Distribution des tailles de trades
    avg_trade_size_usd: float = 0.0
    median_trade_size_usd: float = 0.0
    large_trade_pct: float = 0.0  # % de trades > 5x la mediane

    # Directionnalite
    buy_sell_imbalance: float = 0.0  # |buys - sells| / total
    autocorrelation_1min: float = 0.0  # correl returns successifs

    # Clustering temporel
    burst_ratio: float = 0.0  # max_trades_per_sec / avg_trades_per_sec

    # Spread stability
    spread_stability: float = 0.0  # 1 - (std(spreads) / mean(spreads))

    # Book imbalance
    book_imbalance: float = 0.0  # (bid_depth - ask_depth) / (bid_depth + ask_depth)

    # Competition
    competition_score: float = 0.0  # 0 = hyper competitif, 100 = peu de concurrence

    # Score synthetique
    toxicity_score: float = 0.0  # 0 = tres toxique, 100 = flux propres

    # Meta
    sample_size: int = 0
    data_hours: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ToxicityAnalyzer:
    """Analyse la toxicite des flux d'un token."""

    HORIZONS_SEC = [1, 5, 30, 60]

    def __init__(self, data_dir: str = "backtesting/data/market_data"):
        self._data_dir = Path(data_dir)

    def analyze_from_parquet(
        self, coin: str, date: Optional[str] = None
    ) -> Optional[ToxicityMetrics]:
        """Analyse a partir des fichiers Parquet du collecteur.

        Args:
            coin: Le token a analyser.
            date: Date au format YYYY-MM-DD. Si None, utilise aujourd'hui.

        Returns:
            ToxicityMetrics ou None si pas assez de donnees.
        """
        if date is None:
            from datetime import datetime, timezone

            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        snap_path = self._data_dir / "l2_snapshots" / coin.upper() / f"{date}.parquet"
        trade_path = self._data_dir / "trades" / coin.upper() / f"{date}.parquet"

        if not snap_path.exists() or not trade_path.exists():
            return None

        snapshots = pd.read_parquet(snap_path)
        trades = pd.read_parquet(trade_path)

        if len(trades) < 20 or len(snapshots) < 5:
            return None

        return self._compute(coin, snapshots, trades)

    def analyze_live(
        self, coin: str, snapshots: list[dict], trades: list[dict]
    ) -> Optional[ToxicityMetrics]:
        """Analyse a partir de donnees live (listes de dicts).

        Args:
            coin: Le token.
            snapshots: Liste de dicts avec au moins timestamp_ms, mid, best_bid, best_ask.
            trades: Liste de dicts avec timestamp_ms, price, size, side.

        Returns:
            ToxicityMetrics ou None si pas assez de donnees.
        """
        if len(trades) < 10 or len(snapshots) < 3:
            return None

        snap_df = pd.DataFrame(snapshots)
        trade_df = pd.DataFrame(trades)
        return self._compute(coin, snap_df, trade_df)

    def _compute(
        self, coin: str, snapshots: pd.DataFrame, trades: pd.DataFrame
    ) -> ToxicityMetrics:
        """Calcul principal des metriques de toxicite."""
        snapshots = snapshots.sort_values("timestamp_ms").reset_index(drop=True)
        trades = trades.sort_values("timestamp_ms").reset_index(drop=True)

        snap_ts = snapshots["timestamp_ms"].values
        snap_mids = snapshots["mid"].values

        # Duration en heures
        ts_range = (snap_ts[-1] - snap_ts[0]) / 1000.0
        data_hours = max(ts_range / 3600.0, 0.001)

        # --- Adverse selection ---
        as_results = {}
        for horizon in self.HORIZONS_SEC:
            as_results[horizon] = self._adverse_selection(
                trades, snap_ts, snap_mids, horizon
            )

        # --- Trade size distribution ---
        trade_sizes_usd = (trades["price"] * trades["size"]).values
        if len(trade_sizes_usd) > 0:
            avg_size = float(np.mean(trade_sizes_usd))
            median_size = float(np.median(trade_sizes_usd))
            large_threshold = median_size * 5.0 if median_size > 0 else float("inf")
            large_pct = float(np.mean(trade_sizes_usd > large_threshold))
        else:
            avg_size = median_size = large_pct = 0.0

        # --- Buy/sell imbalance ---
        if len(trades) > 0 and "side" in trades.columns:
            buys = (trades["side"].str.upper() == "B").sum() + (
                trades["side"].str.upper() == "BUY"
            ).sum()
            total = len(trades)
            sells = total - buys
            imbalance = abs(buys - sells) / total if total > 0 else 0.0
        else:
            imbalance = 0.0

        # --- Autocorrelation (returns 1min) ---
        autocorr = self._autocorrelation(snap_ts, snap_mids)

        # --- Burst ratio ---
        burst = self._burst_ratio(trades["timestamp_ms"].values)

        # --- Spread stability ---
        spread_stab = self._spread_stability(snapshots)

        # --- Book imbalance ---
        book_imb = self._book_imbalance(snapshots)

        # --- Competition ---
        competition = self._competition_score(snapshots)

        metrics = ToxicityMetrics(
            coin=coin,
            adverse_selection_1s_bps=as_results.get(1, 0.0),
            adverse_selection_5s_bps=as_results.get(5, 0.0),
            adverse_selection_30s_bps=as_results.get(30, 0.0),
            adverse_selection_60s_bps=as_results.get(60, 0.0),
            avg_trade_size_usd=avg_size,
            median_trade_size_usd=median_size,
            large_trade_pct=large_pct,
            buy_sell_imbalance=imbalance,
            autocorrelation_1min=autocorr,
            burst_ratio=burst,
            spread_stability=spread_stab,
            book_imbalance=book_imb,
            competition_score=competition,
            sample_size=len(trades),
            data_hours=round(data_hours, 2),
        )

        # Score synthetique
        metrics.toxicity_score = self._compute_toxicity_score(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Adverse selection
    # ------------------------------------------------------------------

    def _adverse_selection(
        self,
        trades: pd.DataFrame,
        snap_ts: np.ndarray,
        snap_mids: np.ndarray,
        horizon_sec: float,
    ) -> float:
        """Calcule l'adverse selection moyenne a un horizon donne.

        Pour chaque trade :
        - Trade BUY : si mid monte apres -> adverse selection (MM a vendu avant hausse)
        - Trade SELL : si mid baisse apres -> adverse selection (MM a achete avant baisse)

        Retourne la moyenne en bps. Positif = adverse (mauvais pour MM).
        """
        horizon_ms = horizon_sec * 1000
        results = []

        for row in trades.itertuples():
            trade_ts = row.timestamp_ms
            trade_side = getattr(row, "side", "")

            # Mid au moment du trade
            idx_at = bisect_left(snap_ts, trade_ts)
            if idx_at >= len(snap_ts):
                idx_at = len(snap_ts) - 1
            if idx_at > 0 and abs(snap_ts[idx_at - 1] - trade_ts) < abs(
                snap_ts[idx_at] - trade_ts
            ):
                idx_at -= 1
            mid_at = snap_mids[idx_at]

            if mid_at <= 0:
                continue

            # Mid apres horizon
            target_ts = trade_ts + horizon_ms
            idx_after = bisect_left(snap_ts, target_ts)
            if idx_after >= len(snap_ts):
                continue  # pas assez de donnees apres
            mid_after = snap_mids[idx_after]

            if mid_after <= 0:
                continue

            change_bps = (mid_after - mid_at) / mid_at * 10_000

            side_upper = str(trade_side).upper()
            if side_upper in ("B", "BUY"):
                results.append(change_bps)  # taker achete -> MM a vendu
            elif side_upper in ("A", "SELL", "S"):
                results.append(-change_bps)  # taker vend -> MM a achete

        return round(float(np.mean(results)), 2) if results else 0.0

    # ------------------------------------------------------------------
    # Autocorrelation
    # ------------------------------------------------------------------

    def _autocorrelation(
        self, snap_ts: np.ndarray, snap_mids: np.ndarray
    ) -> float:
        """Autocorrelation des returns successifs (approximation 1 min)."""
        if len(snap_mids) < 10:
            return 0.0

        # Returns entre snapshots consecutifs
        valid = snap_mids[:-1] > 0
        if valid.sum() < 5:
            return 0.0

        returns = np.diff(snap_mids) / np.where(snap_mids[:-1] > 0, snap_mids[:-1], 1.0)

        if len(returns) < 5:
            return 0.0

        # Autocorrelation lag-1
        r1 = returns[:-1]
        r2 = returns[1:]
        if np.std(r1) < 1e-10 or np.std(r2) < 1e-10:
            return 0.0

        corr = float(np.corrcoef(r1, r2)[0, 1])
        return round(max(0, corr), 4)  # clamp to positive (trending)

    # ------------------------------------------------------------------
    # Burst ratio
    # ------------------------------------------------------------------

    def _burst_ratio(self, trade_timestamps_ms: np.ndarray) -> float:
        """Ratio de clustering temporel des trades."""
        if len(trade_timestamps_ms) < 10:
            return 1.0

        # Bin trades par seconde
        ts_sec = trade_timestamps_ms // 1000
        unique_secs, counts = np.unique(ts_sec, return_counts=True)

        if len(counts) == 0:
            return 1.0

        max_per_sec = float(counts.max())
        total_secs = max(1, (ts_sec[-1] - ts_sec[0]))
        avg_per_sec = len(trade_timestamps_ms) / total_secs

        if avg_per_sec < 0.01:
            return 1.0

        return round(max_per_sec / avg_per_sec, 2)

    # ------------------------------------------------------------------
    # Spread stability
    # ------------------------------------------------------------------

    def _spread_stability(self, snapshots: pd.DataFrame) -> float:
        """1 - (std(spreads) / mean(spreads)). 1.0 = stable, 0 = variable."""
        if "best_bid" not in snapshots.columns or "best_ask" not in snapshots.columns:
            return 0.5

        bids = snapshots["best_bid"].values
        asks = snapshots["best_ask"].values
        mids = (bids + asks) / 2.0
        valid = mids > 0
        if valid.sum() < 3:
            return 0.5

        spreads_bps = np.where(valid, (asks - bids) / mids * 10_000, 0)
        spreads_bps = spreads_bps[valid]

        mean_s = np.mean(spreads_bps)
        if mean_s < 0.01:
            return 1.0

        std_s = np.std(spreads_bps)
        stability = 1.0 - min(1.0, std_s / mean_s)
        return round(max(0, stability), 4)

    # ------------------------------------------------------------------
    # Book imbalance
    # ------------------------------------------------------------------

    def _book_imbalance(self, snapshots: pd.DataFrame) -> float:
        """Desequilibre moyen du book (bid vs ask depth)."""
        imbalances = []
        for _, row in snapshots.iterrows():
            bid_depth = sum(
                row.get(f"bid_sz_{i}", 0) * row.get(f"bid_px_{i}", 0)
                for i in range(5)
            )
            ask_depth = sum(
                row.get(f"ask_sz_{i}", 0) * row.get(f"ask_px_{i}", 0)
                for i in range(5)
            )
            total = bid_depth + ask_depth
            if total > 0:
                imbalances.append((bid_depth - ask_depth) / total)

        if not imbalances:
            return 0.0
        return round(float(np.mean(np.abs(imbalances))), 4)

    # ------------------------------------------------------------------
    # Competition score
    # ------------------------------------------------------------------

    def _competition_score(self, snapshots: pd.DataFrame) -> float:
        """Estime la competition MM. 0 = hyper competitif, 100 = peu de concurrence."""
        if snapshots.empty:
            return 50.0

        # Utiliser le dernier snapshot
        row = snapshots.iloc[-1]
        mid = row.get("mid", 0)
        if mid <= 0:
            return 50.0

        # Compter les niveaux a ≤10 bps du mid
        tight_levels = 0
        tight_depth = 0.0
        threshold = mid * 0.001  # 10 bps

        for side in ("bid", "ask"):
            for i in range(10):
                px = row.get(f"{side}_px_{i}", 0)
                sz = row.get(f"{side}_sz_{i}", 0)
                if px <= 0:
                    continue
                if abs(px - mid) <= threshold:
                    tight_levels += 1
                    tight_depth += px * sz

        # Plus de niveaux serres + plus de profondeur = plus de competition
        if tight_levels <= 2:
            score = 100.0
        elif tight_levels <= 5:
            score = 80.0 - (tight_levels - 2) * 5
        elif tight_levels <= 10:
            score = 65.0 - (tight_levels - 5) * 5
        else:
            score = 40.0

        # Penaliser si grosse profondeur pres du mid
        if tight_depth > 50000:
            score -= 20
        elif tight_depth > 10000:
            score -= 10

        return max(0, min(100, score))

    # ------------------------------------------------------------------
    # Score synthetique
    # ------------------------------------------------------------------

    def _compute_toxicity_score(self, m: ToxicityMetrics) -> float:
        """Score composite. 0 = tres toxique, 100 = flux propres."""
        score = 100.0

        # Adverse selection 5s : le plus important
        as_5s = m.adverse_selection_5s_bps
        if as_5s > 5:
            score -= 40
        elif as_5s > 2:
            score -= 20 + (as_5s - 2) * 6.67
        elif as_5s > 1:
            score -= 10 + (as_5s - 1) * 10
        else:
            score -= max(0, as_5s * 10)

        # Adverse selection 30s : confirme si structurel
        as_30s = m.adverse_selection_30s_bps
        if as_30s > 10:
            score -= 20
        elif as_30s > 5:
            score -= 10 + (as_30s - 5) * 2

        # Large trades
        if m.large_trade_pct > 0.2:
            score -= 15
        elif m.large_trade_pct > 0.1:
            score -= 7

        # Autocorrelation
        if m.autocorrelation_1min > 0.3:
            score -= 10
        elif m.autocorrelation_1min > 0.15:
            score -= 5

        # Buy/sell imbalance
        if m.buy_sell_imbalance > 0.3:
            score -= 10
        elif m.buy_sell_imbalance > 0.15:
            score -= 5

        # Burst ratio
        if m.burst_ratio > 10:
            score -= 5

        return round(max(0, min(100, score)), 1)
