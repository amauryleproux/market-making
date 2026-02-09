"""
One-shot screener: scanne toutes les paires et affiche le classement.

Usage:
    python run_screener.py                    # scan rapide (statique)
    python run_screener.py --deep             # analyse profonde (+ toxicite + backtest)
    python run_screener.py --top 50           # top 50
    python run_screener.py --config screener_config.json
"""

import asyncio
import argparse
from pathlib import Path

from src.client.hyperliquid import HyperliquidClient
from src.client.async_wrapper import AsyncHyperliquidClient
from src.strategies.pair_screener import PairScreener, ScreenerConfig


def _fmt_volume(v: float) -> str:
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${v:.0f}"


def _fmt_bt(pnl: float) -> str:
    if pnl == 0:
        return "  ---  "
    sign = "+" if pnl >= 0 else ""
    return f"{sign}${pnl:.2f}"


async def main(args):
    sync_client = HyperliquidClient(secret_key="", account_address="", mainnet=True)
    client = AsyncHyperliquidClient(sync_client)

    if args.config and Path(args.config).exists():
        config = ScreenerConfig.from_json(args.config)
    else:
        config = ScreenerConfig(
            min_volume_usd=args.min_volume,
            max_volume_usd=args.max_volume,
            min_spread_bps=args.min_spread,
            blacklist=args.blacklist.split(",") if args.blacklist else [],
        )

    screener = PairScreener(client=client, config=config)

    mode = "deep" if args.deep else "quick"
    print(f"Scanning all pairs ({mode} mode)...")
    scores = await screener.force_scan(deep=args.deep)

    if not scores:
        print("No pairs found matching criteria.")
        return

    # Header
    is_deep = args.deep
    print(f"\n{'=' * 100}")
    title = "DEEP ANALYSIS" if is_deep else "QUICK SCAN"
    print(f"  TOP {args.top} PAIRS FOR MARKET MAKING ({title})")
    print(f"{'=' * 100}")

    if is_deep:
        print(
            f"  {'#':<4} {'Coin':<10} {'Final':<7} {'Static':<8} {'Toxic':<7} "
            f"{'BT $/h':<9} {'Spread':<9} {'Volume 24h':<13} {'AS_5s':<8} {'Depth':<10}"
        )
        print(f"  {'-' * 93}")
    else:
        print(
            f"  {'#':<4} {'Coin':<10} {'Score':<7} {'Spread':<10} "
            f"{'Volume 24h':<14} {'Depth':<12} {'Levels':<8} {'Funding':<10} {'Tick Sprd':<10}"
        )
        print(f"  {'-' * 86}")

    for i, s in enumerate(scores[: args.top], 1):
        if is_deep:
            tox_str = f"{s.toxicity_score:.0f}" if s.toxicity_score >= 0 else " --- "
            as_str = f"{s.adverse_selection_5s_bps:.1f}bp" if s.toxicity_score >= 0 else " --- "
            bt_str = _fmt_bt(s.backtest_pnl_h)
            print(
                f"  {i:<4} {s.coin:<10} {s.score:<7.1f} {s.static_score:<8.1f} "
                f"{tox_str:<7} {bt_str:<9} {s.spread_bps:<9.1f} "
                f"{_fmt_volume(s.volume_24h_usd):<13} {as_str:<8} "
                f"${s.depth_usd:,.0f}"
            )
        else:
            vol_str = _fmt_volume(s.volume_24h_usd)
            depth_str = f"${s.depth_usd:,.0f}"
            funding_str = f"{s.funding_rate * 100:.4f}%"
            print(
                f"  {i:<4} {s.coin:<10} {s.score:<7.1f} {s.spread_bps:<10.1f} "
                f"{vol_str:<14} {depth_str:<12} {s.num_levels:<8} "
                f"{funding_str:<10} {s.tick_spread_bps:<10.2f}"
            )

    print(f"\n  Total pairs scored: {len(scores)}")

    # Recommendations (deep mode only)
    if is_deep:
        recommended = []
        needs_data = []
        avoid = []

        for s in scores[:20]:
            has_bt = s.backtest_pnl_h != 0
            has_tox = s.toxicity_score >= 0

            if has_bt and s.backtest_pnl_h > 0:
                recommended.append(s)
            elif has_tox and s.toxicity_score < 30:
                avoid.append(s)
            elif has_bt and s.backtest_pnl_h < -0.5:
                avoid.append(s)
            elif has_tox and s.toxicity_score >= 60 and s.score >= 50:
                recommended.append(s)
            else:
                needs_data.append(s)

        if recommended:
            print(f"\n  RECOMMENDED FOR LIVE:")
            for s in recommended[:5]:
                reason = ""
                if s.backtest_pnl_h > 0:
                    reason = f"backtest positive (+${s.backtest_pnl_h:.2f}/h)"
                elif s.toxicity_score >= 60:
                    reason = f"low toxicity ({s.toxicity_score:.0f}/100)"
                print(f"     -> {s.coin}: score {s.score:.0f}, {reason}")

        if needs_data:
            print(f"\n  NEEDS MORE DATA:")
            for s in needs_data[:5]:
                print(f"     ?  {s.coin}: score {s.score:.0f}, collect more data for toxicity/backtest")

        if avoid:
            print(f"\n  AVOID:")
            for s in avoid[:5]:
                reason = ""
                if s.backtest_pnl_h < -0.5:
                    reason = f"backtest negative (${s.backtest_pnl_h:.2f}/h)"
                elif s.toxicity_score < 30:
                    reason = f"high toxicity ({s.toxicity_score:.0f}/100)"
                print(f"     X  {s.coin}: score {s.score:.0f}, {reason}")
    else:
        sweet = [s for s in scores if s.spread_bps >= 10 and s.score >= 50]
        if sweet:
            print(f"\n  SWEET SPOT (spread >= 10 bps, score >= 50):")
            for s in sweet[:5]:
                print(
                    f"     -> {s.coin}: score {s.score}, spread {s.spread_bps} bps, "
                    f"vol {_fmt_volume(s.volume_24h_usd)}"
                )

        print(f"\n  Tip: run with --deep for toxicity analysis + backtest integration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan pairs for market making")
    parser.add_argument("--top", type=int, default=20, help="Number of top pairs to show")
    parser.add_argument("--deep", action="store_true", help="Deep analysis (toxicity + backtest)")
    parser.add_argument("--config", type=str, help="Screener config JSON")
    parser.add_argument("--min-volume", type=float, default=100_000, help="Min 24h volume USD")
    parser.add_argument("--max-volume", type=float, default=5_000_000_000, help="Max 24h volume USD")
    parser.add_argument("--min-spread", type=float, default=3.0, help="Min spread in bps")
    parser.add_argument("--blacklist", type=str, default="", help="Comma-separated blacklist")
    args = parser.parse_args()

    asyncio.run(main(args))
