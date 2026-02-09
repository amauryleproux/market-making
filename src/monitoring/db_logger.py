"""Logger SQLite pour le suivi des ordres, fills, snapshots et PnL.

Crée la base et les 4 tables au démarrage. Les écritures sont
sérialisées via asyncio.Lock pour thread-safety.
"""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class DBLogger:
    """Gestionnaire de base de données SQLite pour le trading."""

    def __init__(self, db_path: str = "data/trading_log.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Crée les 4 tables si elles n'existent pas."""
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                size_usd REAL NOT NULL,
                level INTEGER NOT NULL,
                order_id INTEGER NOT NULL,
                status TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                size_usd REAL NOT NULL,
                fee REAL NOT NULL DEFAULT 0,
                mid_price_at_fill REAL NOT NULL,
                spread_captured_bps REAL NOT NULL,
                inventory_after REAL NOT NULL,
                order_id INTEGER NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                mid_price REAL NOT NULL,
                best_bid REAL NOT NULL,
                best_ask REAL NOT NULL,
                spread_bps REAL NOT NULL,
                inventory REAL NOT NULL,
                inventory_usd REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS pnl_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start TEXT NOT NULL,
                session_end TEXT NOT NULL,
                pair TEXT NOT NULL,
                fills_count INTEGER NOT NULL,
                volume_usd REAL NOT NULL,
                gross_pnl REAL NOT NULL,
                total_fees REAL NOT NULL,
                net_pnl REAL NOT NULL,
                avg_spread_captured_bps REAL NOT NULL,
                max_inventory_usd REAL NOT NULL,
                max_drawdown_usd REAL NOT NULL
            )
        """)

        # Index pour les requêtes fréquentes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fills_pair ON fills(pair)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_pair ON snapshots(pair)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_pair ON orders(pair)")

        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # === Insertions async ===

    async def log_order(
        self,
        pair: str,
        side: str,
        price: float,
        size: float,
        size_usd: float,
        level: int,
        order_id: int,
        status: str,
    ) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT INTO orders (timestamp, pair, side, price, size, size_usd, level, order_id, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self._now(), pair, side, price, size, size_usd, level, order_id, status),
            )
            self._conn.commit()

    async def log_fill(
        self,
        pair: str,
        side: str,
        price: float,
        size: float,
        size_usd: float,
        fee: float,
        mid_price_at_fill: float,
        spread_captured_bps: float,
        inventory_after: float,
        order_id: int,
    ) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT INTO fills (timestamp, pair, side, price, size, size_usd, fee, "
                "mid_price_at_fill, spread_captured_bps, inventory_after, order_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self._now(), pair, side, price, size, size_usd, fee,
                 mid_price_at_fill, spread_captured_bps, inventory_after, order_id),
            )
            self._conn.commit()

    async def log_snapshot(
        self,
        pair: str,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        spread_bps: float,
        inventory: float,
        inventory_usd: float,
        unrealized_pnl: float,
        realized_pnl: float,
        total_pnl: float,
    ) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT INTO snapshots (timestamp, pair, mid_price, best_bid, best_ask, "
                "spread_bps, inventory, inventory_usd, unrealized_pnl, realized_pnl, total_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self._now(), pair, mid_price, best_bid, best_ask, spread_bps,
                 inventory, inventory_usd, unrealized_pnl, realized_pnl, total_pnl),
            )
            self._conn.commit()

    async def log_session_summary(
        self,
        session_start: str,
        session_end: str,
        pair: str,
        fills_count: int,
        volume_usd: float,
        gross_pnl: float,
        total_fees: float,
        net_pnl: float,
        avg_spread_captured_bps: float,
        max_inventory_usd: float,
        max_drawdown_usd: float,
    ) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT INTO pnl_summary (session_start, session_end, pair, fills_count, "
                "volume_usd, gross_pnl, total_fees, net_pnl, avg_spread_captured_bps, "
                "max_inventory_usd, max_drawdown_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_start, session_end, pair, fills_count, volume_usd,
                 gross_pnl, total_fees, net_pnl, avg_spread_captured_bps,
                 max_inventory_usd, max_drawdown_usd),
            )
            self._conn.commit()

    # === Queries sync (pour analyse et dashboard standalone) ===

    def query_fills(self, pair: Optional[str] = None, since: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM fills WHERE 1=1"
        params = []
        if pair:
            sql += " AND pair = ?"
            params.append(pair)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_snapshots(self, pair: Optional[str] = None, since: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM snapshots WHERE 1=1"
        params = []
        if pair:
            sql += " AND pair = ?"
            params.append(pair)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_orders(self, pair: Optional[str] = None, since: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM orders WHERE 1=1"
        params = []
        if pair:
            sql += " AND pair = ?"
            params.append(pair)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_pnl_summary(self, pair: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM pnl_summary WHERE 1=1"
        params = []
        if pair:
            sql += " AND pair = ?"
            params.append(pair)
        sql += " ORDER BY session_start DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_latest_snapshots(self) -> list[dict]:
        """Retourne le dernier snapshot par paire."""
        sql = """
            SELECT s.* FROM snapshots s
            INNER JOIN (
                SELECT pair, MAX(timestamp) as max_ts FROM snapshots GROUP BY pair
            ) latest ON s.pair = latest.pair AND s.timestamp = latest.max_ts
        """
        rows = self._conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
