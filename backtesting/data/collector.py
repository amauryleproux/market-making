"""Collecteur de données marché en continu pour le backtesting.

Collecte des snapshots L2, trades récents et candles depuis l'API
Hyperliquid, et les persiste en Parquet via StorageManager.

Usage:
    python -m backtesting.data.collector --coins MEGA,AR --interval 2
    python -m backtesting.data.collector --coins MEGA --mode candles --days 7
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

from backtesting.data.storage import StorageManager
from src.utils.logger import get_logger

BASE_URL = "https://api.hyperliquid.xyz/info"
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0

log = get_logger("data.collector")


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------

def _post(payload: dict) -> dict | list:
    """POST vers l'API Hyperliquid avec retry + backoff."""
    delay = RETRY_BACKOFF
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(BASE_URL, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
    raise ConnectionError(f"API failed after {MAX_RETRIES} retries: {last_error}")


def fetch_l2_snapshot(coin: str) -> dict:
    """Fetch L2 orderbook snapshot."""
    return _post({"type": "l2Book", "coin": coin})


def fetch_recent_trades(coin: str) -> list[dict]:
    """Fetch recent trades."""
    return _post({"type": "recentTrades", "coin": coin})


def fetch_candles(
    coin: str, interval: str, start_ms: int, end_ms: int
) -> list[dict]:
    """Fetch candle data."""
    return _post({
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    })


# ------------------------------------------------------------------
# Snapshot normalization
# ------------------------------------------------------------------

def _normalize_snapshot(coin: str, raw: dict, timestamp_ms: int) -> dict:
    """Transforme un snapshot L2 brut en dict à colonnes aplaties."""
    levels = raw.get("levels", [[], []])
    bids_raw = levels[0] if len(levels) > 0 else []
    asks_raw = levels[1] if len(levels) > 1 else []

    row: dict = {"timestamp_ms": timestamp_ms, "coin": coin}

    # Aplatir les 10 premiers niveaux
    for i in range(10):
        if i < len(bids_raw):
            row[f"bid_px_{i}"] = float(bids_raw[i].get("px", 0))
            row[f"bid_sz_{i}"] = float(bids_raw[i].get("sz", 0))
        else:
            row[f"bid_px_{i}"] = 0.0
            row[f"bid_sz_{i}"] = 0.0

        if i < len(asks_raw):
            row[f"ask_px_{i}"] = float(asks_raw[i].get("px", 0))
            row[f"ask_sz_{i}"] = float(asks_raw[i].get("sz", 0))
        else:
            row[f"ask_px_{i}"] = 0.0
            row[f"ask_sz_{i}"] = 0.0

    bb = row["bid_px_0"]
    ba = row["ask_px_0"]
    row["best_bid"] = bb
    row["best_ask"] = ba
    row["mid"] = (bb + ba) / 2.0 if bb > 0 and ba > 0 else 0.0
    return row


def _normalize_trade(coin: str, raw: dict) -> dict:
    """Normalise un trade brut de l'API."""
    return {
        "timestamp_ms": int(raw.get("time", 0)),
        "coin": coin,
        "price": float(raw.get("px", 0)),
        "size": float(raw.get("sz", 0)),
        "side": raw.get("side", ""),
        "tid": int(raw.get("tid", 0)),
    }


def _normalize_candle(coin: str, interval: str, raw: dict) -> dict:
    """Normalise une candle brute."""
    return {
        "timestamp_ms": int(raw.get("t", 0)),
        "close_ms": int(raw.get("T", 0)),
        "coin": coin,
        "interval": interval,
        "open": float(raw.get("o", 0)),
        "high": float(raw.get("h", 0)),
        "low": float(raw.get("l", 0)),
        "close": float(raw.get("c", 0)),
        "volume": float(raw.get("v", 0)),
        "num_trades": int(raw.get("n", 0)),
    }


# ------------------------------------------------------------------
# Collector class
# ------------------------------------------------------------------

class DataCollector:
    """Daemon de collecte continue de données marché."""

    def __init__(
        self,
        coins: list[str],
        data_dir: str = "backtesting/data/market_data",
        snapshot_interval_sec: float = 2.0,
        candle_interval: str = "1m",
        flush_interval_sec: float = 30.0,
    ):
        self._coins = [c.upper() for c in coins]
        self._storage = StorageManager(data_dir)
        self._snapshot_interval = snapshot_interval_sec
        self._candle_interval = candle_interval
        self._flush_interval = flush_interval_sec

        # Buffers in-memory
        self._snapshot_buffers: dict[str, list[dict]] = {c: [] for c in self._coins}
        self._trade_buffers: dict[str, list[dict]] = {c: [] for c in self._coins}
        self._candle_buffers: dict[str, list[dict]] = {c: [] for c in self._coins}

        # Deduplication
        self._seen_tids: dict[str, set[int]] = {c: set() for c in self._coins}
        self._last_candle_fetch: float = 0.0
        self._last_flush: float = 0.0

    def run(self) -> None:
        """Boucle principale. Tourne jusqu'à Ctrl+C."""
        log.info("collector_started", coins=self._coins, interval=self._snapshot_interval)
        self._last_flush = time.time()
        self._last_candle_fetch = time.time()

        total_snapshots = 0
        total_trades = 0

        try:
            while True:
                cycle_start = time.time()
                now_ms = int(cycle_start * 1000)

                # Fetch L2 + trades pour tous les coins en parallèle
                with ThreadPoolExecutor(max_workers=len(self._coins) * 2) as pool:
                    l2_futures = {
                        pool.submit(fetch_l2_snapshot, coin): coin
                        for coin in self._coins
                    }
                    trade_futures = {
                        pool.submit(fetch_recent_trades, coin): coin
                        for coin in self._coins
                    }

                    for fut in as_completed(l2_futures):
                        coin = l2_futures[fut]
                        try:
                            raw = fut.result()
                            snap = _normalize_snapshot(coin, raw, now_ms)
                            self._snapshot_buffers[coin].append(snap)
                            total_snapshots += 1
                        except Exception as e:
                            log.warning("snapshot_error", coin=coin, error=str(e))

                    for fut in as_completed(trade_futures):
                        coin = trade_futures[fut]
                        try:
                            raw_trades = fut.result()
                            seen = self._seen_tids[coin]
                            for t in raw_trades:
                                tid = int(t.get("tid", 0))
                                if tid and tid not in seen:
                                    seen.add(tid)
                                    self._trade_buffers[coin].append(
                                        _normalize_trade(coin, t)
                                    )
                                    total_trades += 1
                            if len(seen) > 15000:
                                sorted_tids = sorted(seen)
                                self._seen_tids[coin] = set(sorted_tids[-10000:])
                        except Exception as e:
                            log.warning("trades_error", coin=coin, error=str(e))

                # Candles toutes les 60s (aussi en parallèle)
                now = time.time()
                if now - self._last_candle_fetch >= 60:
                    with ThreadPoolExecutor(max_workers=len(self._coins)) as pool:
                        end_ms = int(now * 1000)
                        start_ms = end_ms - 120_000
                        candle_futures = {
                            pool.submit(
                                fetch_candles, coin, self._candle_interval,
                                start_ms, end_ms
                            ): coin
                            for coin in self._coins
                        }
                        for fut in as_completed(candle_futures):
                            coin = candle_futures[fut]
                            try:
                                raw_candles = fut.result()
                                for c in raw_candles:
                                    self._candle_buffers[coin].append(
                                        _normalize_candle(
                                            coin, self._candle_interval, c
                                        )
                                    )
                            except Exception as e:
                                log.warning("candles_error", coin=coin, error=str(e))
                    self._last_candle_fetch = now

                # Flush vers Parquet
                if now - self._last_flush >= self._flush_interval:
                    self._flush()
                    self._last_flush = now
                    log.info(
                        "collector_status",
                        total_snapshots=total_snapshots,
                        total_trades=total_trades,
                    )

                # Attente
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self._snapshot_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("collector_stopping")
            self._flush()
            log.info("collector_stopped", snapshots=total_snapshots, trades=total_trades)

    def _flush(self) -> None:
        """Flush tous les buffers vers Parquet."""
        for coin in self._coins:
            if self._snapshot_buffers[coin]:
                self._storage.write_snapshots(coin, self._snapshot_buffers[coin])
                self._snapshot_buffers[coin] = []

            if self._trade_buffers[coin]:
                self._storage.write_trades(coin, self._trade_buffers[coin])
                self._trade_buffers[coin] = []

            if self._candle_buffers[coin]:
                self._storage.write_candles(coin, self._candle_buffers[coin])
                self._candle_buffers[coin] = []


# ------------------------------------------------------------------
# Candle backfill mode
# ------------------------------------------------------------------

def backfill_candles(
    coins: list[str],
    days: int,
    interval: str = "1m",
    data_dir: str = "backtesting/data/market_data",
) -> None:
    """Télécharge les candles historiques pour N jours."""
    storage = StorageManager(data_dir)
    now_ms = int(time.time() * 1000)

    for coin in coins:
        coin = coin.upper()
        log.info("backfill_start", coin=coin, days=days, interval=interval)

        # Découper en chunks de 24h pour éviter les limites API
        day_ms = 86_400_000
        for d in range(days):
            end_ms = now_ms - d * day_ms
            start_ms = end_ms - day_ms

            try:
                raw = fetch_candles(coin, interval, start_ms, end_ms)
                candles = [_normalize_candle(coin, interval, c) for c in raw]
                storage.write_candles(coin, candles)
                log.info("backfill_day", coin=coin, day=d + 1, candles=len(candles))
                time.sleep(0.2)  # rate limit
            except Exception as e:
                log.error("backfill_error", coin=coin, day=d + 1, error=str(e))

        log.info("backfill_done", coin=coin)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect market data for backtesting")
    parser.add_argument(
        "--coins", type=str, required=True, help="Comma-separated coin list"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Snapshot interval in seconds"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtesting/data/market_data",
        help="Output directory",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=30.0,
        help="Flush to disk interval in seconds",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "candles"],
        default="live",
        help="Collection mode",
    )
    parser.add_argument("--days", type=int, default=7, help="Days to backfill (candle mode)")
    args = parser.parse_args()

    coins = [c.strip() for c in args.coins.split(",")]

    if args.mode == "candles":
        backfill_candles(coins, args.days, data_dir=args.output)
    else:
        collector = DataCollector(
            coins=coins,
            data_dir=args.output,
            snapshot_interval_sec=args.interval,
            flush_interval_sec=args.flush_interval,
        )
        collector.run()


if __name__ == "__main__":
    main()
