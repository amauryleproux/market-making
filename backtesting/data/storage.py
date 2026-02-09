"""Gestion du stockage Parquet pour les données de backtesting.

Structure :
    {data_dir}/
        l2_snapshots/{coin}/{YYYY-MM-DD}.parquet
        trades/{coin}/{YYYY-MM-DD}.parquet
        candles/{coin}/{YYYY-MM-DD}.parquet
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

log = get_logger("data.storage")


class StorageManager:
    """Lecture/écriture de données marché en Parquet partitionné par jour/coin."""

    def __init__(self, data_dir: str = "backtesting/data/market_data"):
        self._root = Path(data_dir)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_snapshots(self, coin: str, snapshots: list[dict]) -> None:
        """Écrit des snapshots L2 en Parquet, partitionnés par date."""
        if not snapshots:
            return
        df = pd.DataFrame(snapshots)
        self._append_by_date(df, "l2_snapshots", coin)

    def write_trades(self, coin: str, trades: list[dict]) -> None:
        """Écrit des trades en Parquet, partitionnés par date."""
        if not trades:
            return
        df = pd.DataFrame(trades)
        self._append_by_date(df, "trades", coin)

    def write_candles(self, coin: str, candles: list[dict]) -> None:
        """Écrit des candles en Parquet, partitionnés par date."""
        if not candles:
            return
        df = pd.DataFrame(candles)
        self._append_by_date(df, "candles", coin)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_snapshots(
        self, coin: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Lit les snapshots L2 pour une plage de dates. Retourne un DataFrame trié."""
        return self._read_range("l2_snapshots", coin, start_date, end_date)

    def read_trades(
        self, coin: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Lit les trades pour une plage de dates."""
        return self._read_range("trades", coin, start_date, end_date)

    def read_candles(
        self, coin: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Lit les candles pour une plage de dates."""
        return self._read_range("candles", coin, start_date, end_date)

    def list_available_dates(
        self, coin: str, data_type: str = "l2_snapshots"
    ) -> list[str]:
        """Liste les dates disponibles pour un coin et type de données."""
        folder = self._root / data_type / coin.upper()
        if not folder.exists():
            return []
        dates = sorted(p.stem for p in folder.glob("*.parquet"))
        return dates

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_path(self, data_type: str, coin: str, date_str: str) -> Path:
        return self._root / data_type / coin.upper() / f"{date_str}.parquet"

    def _append_by_date(
        self, df: pd.DataFrame, data_type: str, coin: str
    ) -> None:
        """Append un DataFrame à des fichiers Parquet partitionnés par date.

        Groupe les rows par date (extraite de timestamp_ms), puis pour
        chaque date : lit le fichier existant s'il y a, concat, et réécrit.
        """
        if "timestamp_ms" not in df.columns:
            log.warning("no_timestamp_column", data_type=data_type, coin=coin)
            return

        df["_date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.strftime(
            "%Y-%m-%d"
        )

        for date_str, group in df.groupby("_date"):
            group = group.drop(columns=["_date"])
            path = self._get_path(data_type, coin, date_str)
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, group], ignore_index=True)
                # Déduplique sur timestamp_ms pour éviter les doublons en cas de re-flush
                if "tid" in combined.columns:
                    combined = combined.drop_duplicates(subset=["tid"], keep="last")
                else:
                    combined = combined.drop_duplicates(
                        subset=["timestamp_ms"], keep="last"
                    )
                combined = combined.sort_values("timestamp_ms").reset_index(drop=True)
            else:
                combined = group.sort_values("timestamp_ms").reset_index(drop=True)

            combined.to_parquet(path, index=False)
            log.debug(
                "parquet_written",
                data_type=data_type,
                coin=coin,
                date=date_str,
                rows=len(combined),
            )

    def _read_range(
        self, data_type: str, coin: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Lit et concatène les fichiers Parquet d'une plage de dates."""
        folder = self._root / data_type / coin.upper()
        if not folder.exists():
            log.warning("no_data_folder", data_type=data_type, coin=coin)
            return pd.DataFrame()

        frames = []
        for path in sorted(folder.glob("*.parquet")):
            date_str = path.stem
            if start_date <= date_str <= end_date:
                frames.append(pd.read_parquet(path))

        if not frames:
            log.warning(
                "no_data_in_range",
                data_type=data_type,
                coin=coin,
                start=start_date,
                end=end_date,
            )
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        log.info(
            "data_loaded",
            data_type=data_type,
            coin=coin,
            rows=len(df),
            start=start_date,
            end=end_date,
        )
        return df
