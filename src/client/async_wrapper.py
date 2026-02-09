"""Wrapper asynchrone autour du HyperliquidClient synchrone.

Utilise asyncio.loop.run_in_executor pour appeler les méthodes sync
du SDK Hyperliquid sans bloquer la boucle événementielle.
"""

import asyncio
from functools import partial
from typing import Optional

from src.client.hyperliquid import HyperliquidClient, OrderResult, AccountState


class AsyncHyperliquidClient:
    """Adaptateur async pour HyperliquidClient."""

    def __init__(self, sync_client: HyperliquidClient):
        self._sync = sync_client

    async def _run(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    # --- Market Data ---

    async def get_mid(self, coin: str) -> float:
        return await self._run(self._sync.get_mid, coin)

    async def get_best_bid_ask(self, coin: str) -> tuple[float, float]:
        return await self._run(self._sync.get_best_bid_ask, coin)

    async def get_l2(self, coin: str, n_levels: int = 10) -> dict:
        return await self._run(self._sync.get_l2, coin, n_levels)

    async def get_l2_book(self, coin: str, n_levels: int = 10) -> dict:
        return await self.get_l2(coin, n_levels)

    async def get_all_mids(self) -> dict[str, float]:
        return await self._run(self._sync.get_all_mids)

    # --- Account ---

    async def get_account_state(self) -> AccountState:
        return await self._run(self._sync.get_account_state)

    async def get_open_orders(self, coin: Optional[str] = None) -> list[dict]:
        return await self._run(self._sync.get_open_orders, coin)

    async def get_recent_fills(self) -> list[dict]:
        return await self._run(self._sync.get_recent_fills)

    # --- Orders ---

    async def market_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        reduce_only: bool = False,
        slippage_pct: float = 1.0,
    ) -> OrderResult:
        return await self._run(
            self._sync.market_order, coin, is_buy, size,
            reduce_only=reduce_only, slippage_pct=slippage_pct,
        )

    async def place_bulk_orders(self, orders: list[dict]) -> list[OrderResult]:
        return await self._run(self._sync.place_bulk_orders, orders)

    async def cancel_coin_orders(self, coin: str) -> bool:
        return await self._run(self._sync.cancel_coin_orders, coin)

    async def cancel_all(self, coin: Optional[str] = None) -> bool:
        return await self._run(self._sync.cancel_all, coin)

    # --- Metadata ---

    async def get_meta(self) -> dict:
        return await self._run(self._sync.get_meta)

    async def get_sz_decimals(self, coin: str) -> int:
        return await self._run(self._sync.get_sz_decimals, coin)

    async def get_asset_index(self, coin: str) -> int:
        return await self._run(self._sync.get_asset_index, coin)

    async def get_market_stats(self, coin: str) -> dict:
        return await self._run(self._sync.get_market_stats, coin)

    async def get_all_market_stats(self) -> list[dict]:
        return await self._run(self._sync.get_all_market_stats)

    async def get_funding_rate(self, coin: str) -> float:
        return await self._run(self._sync.get_funding_rate, coin)
