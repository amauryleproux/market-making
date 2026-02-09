"""
Point d'entrée CLI pour un backtest unique.

Usage:
    python -m backtesting.run_backtest --config backtesting/config/backtest_config.json
    python -m backtesting.run_backtest --coin MEGA --start 2026-02-09 --end 2026-02-09
    python -m backtesting.run_backtest --coin MEGA --start 2026-02-09 --end 2026-02-09 --spread-bps 15 --size 50
"""

import sys
import argparse
from pathlib import Path

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.engine.backtest_engine import BacktestEngine, BacktestConfig
from backtesting.analysis.metrics import compute_metrics
from backtesting.analysis.report import generate_report
from src.utils.logger import get_logger

log = get_logger("bt.run")


def main():
    parser = argparse.ArgumentParser(description="Run market making backtest")
    parser.add_argument("--config", type=str, help="Path to JSON config")
    parser.add_argument("--coin", type=str, help="Coin to backtest")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--spread-bps", type=float, default=12.0)
    parser.add_argument("--size", type=float, default=25.0, help="Order size USD")
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--max-inv", type=float, default=100.0)
    parser.add_argument("--skew", type=float, default=0.8, help="Inventory skew factor")
    parser.add_argument("--latency", type=int, default=300, help="Latency ms")
    parser.add_argument("--data-dir", type=str, default="backtesting/data/market_data")
    parser.add_argument("--output-dir", type=str, default="backtesting/results")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.config:
        config = BacktestConfig.from_json(args.config)
    else:
        if not args.coin or not args.start or not args.end:
            parser.error("Either --config or --coin/--start/--end are required")
        config = BacktestConfig(
            coin=args.coin.upper(),
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir,
            spread_bps=args.spread_bps,
            order_size_usd=args.size,
            num_levels=args.levels,
            max_inventory_usd=args.max_inv,
            inventory_skew_factor=args.skew,
            latency_ms=args.latency,
        )

    log.info(
        "backtest_starting",
        coin=config.coin,
        start=config.start_date,
        end=config.end_date,
        spread=config.spread_bps,
        size=config.order_size_usd,
    )

    engine = BacktestEngine(config)
    result = engine.run(show_progress=not args.no_progress)

    metrics = compute_metrics(result)
    report_path = generate_report(
        result, metrics,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )

    print(metrics.summary())

    if report_path:
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
