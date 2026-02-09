"""
Grid search d'optimisation de paramètres.

Usage:
    python -m backtesting.run_parameter_sweep --coin MEGA --start 2026-02-09 --end 2026-02-09
    python -m backtesting.run_parameter_sweep --config backtesting/config/sweep_config.json
"""

import sys
import csv
import json
import argparse
import itertools
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.engine.backtest_engine import BacktestEngine, BacktestConfig
from backtesting.analysis.metrics import compute_metrics
from src.utils.logger import get_logger

log = get_logger("bt.sweep")

# Plages par défaut
DEFAULT_SWEEP = {
    "spread_bps": [8, 10, 12, 15, 20],
    "order_size_usd": [25, 50, 75, 100],
    "num_levels": [1, 2, 3, 4],
    "max_inventory_usd": [100, 250, 500, 750],
}


def run_single_backtest(params: dict) -> dict:
    """Exécute un backtest unique. Fonction top-level pour multiprocessing."""
    try:
        config = BacktestConfig(**params)
        engine = BacktestEngine(config)
        result = engine.run(show_progress=False)
        metrics = compute_metrics(result)

        row = {}
        # Paramètres sweepés
        for key in ["spread_bps", "order_size_usd", "num_levels", "max_inventory_usd",
                     "inventory_skew_factor"]:
            row[key] = params.get(key, "")
        # Métriques clés
        row.update({
            "total_pnl": round(metrics.total_pnl, 4),
            "realized_pnl": round(metrics.realized_pnl, 4),
            "pnl_per_hour": round(metrics.pnl_per_hour, 4),
            "max_drawdown": round(metrics.max_drawdown_usd, 4),
            "total_fills": metrics.total_fills,
            "total_volume": round(metrics.total_volume_usd, 2),
            "fill_rate": round(metrics.fill_rate, 4),
            "avg_adverse_1m": round(metrics.avg_pnl_1min_bps, 2),
            "avg_adverse_5m": round(metrics.avg_pnl_5min_bps, 2),
            "avg_inventory": round(metrics.avg_inventory_usd, 2),
            "max_inventory": round(metrics.max_inventory_usd, 2),
            "spread_captured": round(metrics.avg_spread_captured_bps, 2),
            "profit_factor": round(metrics.profit_factor, 2),
        })
        return row
    except Exception as e:
        return {"error": str(e), **params}


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep backtest")
    parser.add_argument("--config", type=str, help="Sweep config JSON")
    parser.add_argument("--coin", type=str)
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--data-dir", type=str, default="backtesting/data/market_data")
    parser.add_argument("--output", type=str, default="backtesting/results/sweep_results.csv")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            sweep_config = json.load(f)
        base_params = sweep_config.get("base", {})
        sweep_ranges = sweep_config.get("sweep", DEFAULT_SWEEP)
    else:
        if not args.coin or not args.start or not args.end:
            parser.error("Either --config or --coin/--start/--end are required")
        base_params = {
            "coin": args.coin.upper(),
            "start_date": args.start,
            "end_date": args.end,
            "data_dir": args.data_dir,
        }
        sweep_ranges = DEFAULT_SWEEP

    # Générer toutes les combinaisons
    param_names = list(sweep_ranges.keys())
    param_values = list(sweep_ranges.values())
    combinations = list(itertools.product(*param_values))

    all_params = []
    for combo in combinations:
        params = {**base_params}
        for name, value in zip(param_names, combo):
            params[name] = value
        all_params.append(params)

    total = len(all_params)
    workers = args.workers or max(1, cpu_count() - 1)
    print(f"Running {total} combinations on {workers} workers...")

    results = []
    with Pool(processes=workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_single_backtest, all_params)):
            results.append(row)
            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                print(f"  [{pct:5.1f}%] {i + 1}/{total} done")

    # Filtrer les erreurs
    errors = [r for r in results if "error" in r]
    results = [r for r in results if "error" not in r]

    if errors:
        print(f"\n{len(errors)} combinations failed.")

    # Trier par PnL
    results.sort(key=lambda r: r.get("total_pnl", -999), reverse=True)

    # Écrire CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if results:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    # Afficher top 10
    print(f"\n{'='*90}")
    print(f"  TOP 10 PARAMETER COMBINATIONS (out of {total})")
    print(f"{'='*90}")

    all_negative = all(r.get("total_pnl", 0) < 0 for r in results) if results else True

    for i, r in enumerate(results[:10]):
        pnl = r.get("total_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        print(
            f"  #{i+1:2d}  PnL={sign}${pnl:.4f}  |  "
            f"spread={r.get('spread_bps', '?')}bps  "
            f"size=${r.get('order_size_usd', '?')}  "
            f"levels={r.get('num_levels', '?')}  "
            f"max_inv=${r.get('max_inventory_usd', '?')}  |  "
            f"fills={r.get('total_fills', 0)}  "
            f"dd=${r.get('max_drawdown', 0):.4f}"
        )

    if all_negative and results:
        print(f"\n  *** TOUTES les combinaisons perdent de l'argent. "
              f"Ce token n'est probablement pas viable pour le MM. ***")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
