"""Integre les resultats de backtest dans le scoring du screener.

Lit les fichiers de resultats de sweep et de backtest dans
backtesting/results/ pour alimenter le score final d'une paire.
"""

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

log = get_logger("bt.integrator")


@dataclass
class BacktestScore:
    """Score derive des resultats de backtest."""

    coin: str
    best_pnl_per_hour: float = 0.0
    best_config: dict = field(default_factory=dict)
    pct_profitable_configs: float = 0.0
    avg_pnl_per_hour: float = 0.0
    max_drawdown: float = 0.0
    total_fills: int = 0
    fill_rate: float = 0.0
    total_hours_tested: float = 0.0
    backtest_date: str = ""
    confidence: float = 0.0  # 0-1


class BacktestIntegrator:
    """Lit les resultats de backtest pour les integrer au screener."""

    def __init__(self, results_dir: str = "backtesting/results"):
        self._results_dir = Path(results_dir)

    def get_score(self, coin: str) -> Optional[BacktestScore]:
        """Recupere le BacktestScore pour un coin.

        Cherche dans l'ordre :
        1. sweep_results.csv (multi-coin sweep)
        2. {COIN}_*_summary.txt (single backtest)
        """
        coin = coin.upper()

        # Try sweep results first
        score = self._from_sweep_csv(coin)
        if score is not None:
            return score

        # Try individual backtest summary
        score = self._from_summary(coin)
        return score

    def get_all_scores(self) -> dict[str, BacktestScore]:
        """Recupere les scores pour tous les coins avec des resultats."""
        scores = {}

        # Sweep CSV
        sweep_path = self._results_dir / "sweep_results.csv"
        if sweep_path.exists():
            coins = self._coins_from_sweep(sweep_path)
            for coin in coins:
                s = self._from_sweep_csv(coin)
                if s is not None:
                    scores[coin] = s

        # Individual summaries
        for f in self._results_dir.glob("*_summary.txt"):
            coin = f.name.split("_")[0].upper()
            if coin not in scores:
                s = self._from_summary(coin)
                if s is not None:
                    scores[coin] = s

        return scores

    def _from_sweep_csv(self, coin: str) -> Optional[BacktestScore]:
        """Parse le sweep_results.csv pour un coin."""
        sweep_path = self._results_dir / "sweep_results.csv"
        if not sweep_path.exists():
            return None

        rows = []
        try:
            with open(sweep_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            log.warning("sweep_csv_error", error=str(e))
            return None

        if not rows:
            return None

        # Le sweep CSV peut contenir tous les coins ou un seul
        # Filtrer par coin si une colonne coin existe
        # Sinon prendre tous les rows (single coin sweep)

        # Calculer les metriques
        pnls = []
        for r in rows:
            try:
                pnl = float(r.get("total_pnl", 0))
                pnls.append((pnl, r))
            except (ValueError, TypeError):
                continue

        if not pnls:
            return None

        # Trier par PnL desc
        pnls.sort(key=lambda x: x[0], reverse=True)

        best_pnl, best_row = pnls[0]
        profitable = sum(1 for p, _ in pnls if p > 0)
        pct_profitable = profitable / len(pnls) if pnls else 0.0
        avg_pnl = sum(p for p, _ in pnls) / len(pnls) if pnls else 0.0

        # Estimer PnL/h (on utilise pnl_per_hour si disponible)
        best_pnl_h = float(best_row.get("pnl_per_hour", best_pnl))
        avg_pnl_h = avg_pnl  # approximate

        best_config = {
            "spread_bps": best_row.get("spread_bps", ""),
            "order_size_usd": best_row.get("order_size_usd", ""),
            "num_levels": best_row.get("num_levels", ""),
            "max_inventory_usd": best_row.get("max_inventory_usd", ""),
        }

        dd = float(best_row.get("max_drawdown", 0))
        fills = int(float(best_row.get("total_fills", 0)))
        fr = float(best_row.get("fill_rate", 0))

        # Confidence basee sur le nombre de configs testees
        confidence = min(1.0, len(pnls) / 50.0)

        return BacktestScore(
            coin=coin,
            best_pnl_per_hour=round(best_pnl_h, 4),
            best_config=best_config,
            pct_profitable_configs=round(pct_profitable, 4),
            avg_pnl_per_hour=round(avg_pnl_h, 4),
            max_drawdown=round(dd, 4),
            total_fills=fills,
            fill_rate=round(fr, 4),
            confidence=round(confidence, 2),
        )

    def _from_summary(self, coin: str) -> Optional[BacktestScore]:
        """Parse un fichier *_summary.txt de backtest individuel."""
        # Chercher le fichier le plus recent pour ce coin
        pattern = f"{coin}_*_summary.txt"
        files = sorted(self._results_dir.glob(pattern), reverse=True)
        if not files:
            return None

        summary_file = files[0]
        try:
            text = summary_file.read_text(encoding="utf-8")
        except Exception:
            return None

        # Parser les lignes cles du summary
        metrics = {}
        for line in text.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            key = parts[0].strip().lower()
            val = parts[1].strip()
            # Nettoyer les valeurs ($, %, etc.)
            val_clean = val.replace("$", "").replace("%", "").replace(",", "").strip()
            try:
                metrics[key] = float(val_clean)
            except (ValueError, TypeError):
                metrics[key] = val

        total_pnl = metrics.get("total pnl", 0.0)
        pnl_h = metrics.get("pnl/hour", total_pnl)
        dd = abs(metrics.get("max drawdown", 0.0))
        fills_raw = metrics.get("fills", 0)
        try:
            fills = int(float(str(fills_raw).split("(")[0].strip()))
        except (ValueError, TypeError):
            fills = 0
        fr = metrics.get("fill rate", 0.0)

        # Date depuis le nom du fichier
        parts = summary_file.stem.split("_")
        date_str = parts[1] if len(parts) > 1 else ""

        # Confidence basse pour un seul backtest
        confidence = 0.3 if fills > 10 else 0.1

        return BacktestScore(
            coin=coin,
            best_pnl_per_hour=float(pnl_h) if isinstance(pnl_h, (int, float)) else 0.0,
            pct_profitable_configs=1.0 if float(total_pnl) > 0 else 0.0,
            avg_pnl_per_hour=float(pnl_h) if isinstance(pnl_h, (int, float)) else 0.0,
            max_drawdown=float(dd) if isinstance(dd, (int, float)) else 0.0,
            total_fills=fills,
            fill_rate=float(fr) if isinstance(fr, (int, float)) else 0.0,
            backtest_date=date_str,
            confidence=confidence,
        )

    def _coins_from_sweep(self, path: Path) -> set[str]:
        """Extrait la liste des coins uniques d'un sweep CSV."""
        coins = set()
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    c = row.get("coin", "").upper()
                    if c:
                        coins.add(c)
        except Exception:
            pass
        return coins
