"""
Client Hyperliquid — wrapper autour du SDK officiel.

Gère la connexion, les données de marché, et le placement d'ordres.
Pas de Cloudflare, pas de proxy, juste une API REST propre.
"""

import time
from dataclasses import dataclass
from typing import Optional

import eth_account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from src.utils.logger import get_logger

log = get_logger("client")


@dataclass
class OrderResult:
    success: bool
    oid: int = 0
    status: str = ""
    filled_sz: float = 0.0
    avg_px: float = 0.0
    error: str = ""


@dataclass
class Position:
    coin: str
    size: float  # positif = long, négatif = short
    entry_px: float
    unrealized_pnl: float
    liquidation_px: float = 0.0
    margin_used: float = 0.0


@dataclass
class AccountState:
    equity: float
    available_balance: float
    positions: list[Position]
    total_unrealized_pnl: float


class HyperliquidClient:
    """Client pour l'API Hyperliquid."""

    def __init__(self, secret_key: str, account_address: str, mainnet: bool = True):
        self.account_address = account_address
        self.mainnet = mainnet

        base_url = constants.MAINNET_API_URL if mainnet else constants.TESTNET_API_URL

        # Info API (lecture seule, pas besoin de clé)
        self.info = Info(base_url, skip_ws=True)

        # Exchange API (trading)
        self._exchange: Optional[Exchange] = None
        if secret_key:
            wallet = eth_account.Account.from_key(secret_key)
            self._exchange = Exchange(
                wallet,
                base_url,
                account_address=account_address,
            )
            log.info("client_ready", mode="authenticated", network="mainnet" if mainnet else "testnet")
        else:
            log.info("client_ready", mode="read-only")

        # Cache des métadonnées
        self._meta = None
        self._asset_map: dict[str, int] = {}

    # =========================================================================
    # METADATA
    # =========================================================================

    def get_meta(self) -> dict:
        """Récupère les métadonnées (liste d'assets, etc.)."""
        if self._meta is None:
            self._meta = self.info.meta()
            # Construire le mapping coin -> asset index
            for i, asset in enumerate(self._meta["universe"]):
                self._asset_map[asset["name"]] = i
            log.info("meta_loaded", assets=len(self._meta["universe"]))
        return self._meta

    def get_asset_index(self, coin: str) -> int:
        """Retourne l'index d'un asset (nécessaire pour les ordres)."""
        self.get_meta()
        if coin not in self._asset_map:
            raise ValueError(f"Coin {coin} not found. Available: {list(self._asset_map.keys())[:20]}...")
        return self._asset_map[coin]

    def get_sz_decimals(self, coin: str) -> int:
        """Retourne le nombre de décimales pour la taille d'un coin."""
        self.get_meta()
        idx = self.get_asset_index(coin)
        return self._meta["universe"][idx]["szDecimals"]

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_all_mids(self) -> dict[str, float]:
        """Récupère tous les mid-prices."""
        raw = self.info.all_mids()
        return {k: float(v) for k, v in raw.items()}

    def get_mid(self, coin: str) -> float:
        """Récupère le mid-price d'un coin."""
        mids = self.get_all_mids()
        if coin not in mids:
            raise ValueError(f"No mid price for {coin}")
        return mids[coin]

    def get_l2(self, coin: str, n_levels: int = 10) -> dict:
        """Récupère l'orderbook L2."""
        return self.info.l2_snapshot(coin)

    def get_best_bid_ask(self, coin: str) -> tuple[float, float]:
        """Retourne (best_bid, best_ask)."""
        l2 = self.get_l2(coin, 1)
        bids = l2["levels"][0]
        asks = l2["levels"][1]
        best_bid = float(bids[0]["px"]) if bids else 0
        best_ask = float(asks[0]["px"]) if asks else 0
        return best_bid, best_ask

    def get_meta_and_ctxs(self) -> tuple[dict, list[dict]]:
        """Récupère les métadonnées + contextes de marché pour tous les assets."""
        result = self.info.meta_and_asset_ctxs()
        meta = result[0]
        ctxs = result[1]
        # Enrichir chaque ctx avec le nom du coin
        universe = meta.get("universe", [])
        for i, ctx in enumerate(ctxs):
            if i < len(universe):
                ctx["coin"] = universe[i]["name"]
        return meta, ctxs

    def get_market_stats(self, coin: str) -> dict:
        """Récupère les stats de marché pour un coin (volume, funding, etc.)."""
        _, ctxs = self.get_meta_and_ctxs()
        for ctx in ctxs:
            if ctx.get("coin") == coin:
                return ctx
        return {}

    def get_all_market_stats(self) -> list[dict]:
        """Récupère les stats de marché pour tous les coins."""
        _, ctxs = self.get_meta_and_ctxs()
        return ctxs

    def get_funding_rate(self, coin: str) -> float:
        """Récupère le funding rate actuel d'un coin."""
        stats = self.get_market_stats(coin)
        return float(stats.get("funding", 0.0))

    # =========================================================================
    # ACCOUNT
    # =========================================================================

    def get_account_state(self) -> AccountState:
        """Récupère l'état du compte."""
        state = self.info.user_state(self.account_address)

        margin_summary = state.get("marginSummary", {})
        equity = float(margin_summary.get("accountValue", 0))
        available = float(margin_summary.get("totalMarginUsed", 0))
        available_balance = equity - available

        positions = []
        total_pnl = 0.0
        for pos_data in state.get("assetPositions", []):
            p = pos_data["position"]
            size = float(p.get("szi", 0))
            if abs(size) < 1e-10:
                continue
            entry = float(p.get("entryPx", 0))
            pnl = float(p.get("unrealizedPnl", 0))
            liq = float(p.get("liquidationPx", 0) or 0)
            margin = float(p.get("marginUsed", 0))
            positions.append(Position(
                coin=p["coin"],
                size=size,
                entry_px=entry,
                unrealized_pnl=pnl,
                liquidation_px=liq,
                margin_used=margin,
            ))
            total_pnl += pnl

        return AccountState(
            equity=equity,
            available_balance=available_balance,
            positions=positions,
            total_unrealized_pnl=total_pnl,
        )

    def get_open_orders(self, coin: Optional[str] = None) -> list[dict]:
        """Récupère les ordres ouverts."""
        orders = self.info.open_orders(self.account_address)
        if coin:
            orders = [o for o in orders if o.get("coin") == coin]
        return orders

    def get_recent_fills(self) -> list[dict]:
        """Récupère les derniers fills."""
        return self.info.user_fills(self.account_address)

    # =========================================================================
    # ORDERS
    # =========================================================================

    def place_limit_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        price: float,
        post_only: bool = True,
        reduce_only: bool = False,
    ) -> OrderResult:
        """
        Place un ordre limit.

        Args:
            coin: "ETH", "BTC", etc.
            is_buy: True = buy, False = sell
            size: Taille en unités du coin
            price: Prix limite
            post_only: True = ALO (maker only, pour les rebates)
            reduce_only: True = ne peut que réduire une position
        """
        if self._exchange is None:
            return OrderResult(success=False, error="Not authenticated")

        tif = "Alo" if post_only else "Gtc"
        order_type = {"limit": {"tif": tif}}

        # Arrondir la taille aux décimales correctes
        sz_decimals = self.get_sz_decimals(coin)
        size = round(size, sz_decimals)

        if size <= 0:
            return OrderResult(success=False, error=f"Size too small after rounding: {size}")

        try:
            result = self._exchange.order(
                coin, is_buy, size, price, order_type, reduce_only=reduce_only
            )

            if result["status"] == "ok":
                statuses = result["response"]["data"]["statuses"]
                if statuses:
                    s = statuses[0]
                    if "resting" in s:
                        return OrderResult(
                            success=True,
                            oid=s["resting"]["oid"],
                            status="resting",
                        )
                    elif "filled" in s:
                        return OrderResult(
                            success=True,
                            oid=s["filled"]["oid"],
                            status="filled",
                            filled_sz=float(s["filled"]["totalSz"]),
                            avg_px=float(s["filled"]["avgPx"]),
                        )
                    elif "error" in s:
                        return OrderResult(success=False, error=s["error"])
                return OrderResult(success=True, status="ok")
            else:
                return OrderResult(success=False, error=str(result))

        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def market_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        reduce_only: bool = False,
        slippage_pct: float = 1.0,
    ) -> OrderResult:
        """Place un ordre au marché via un limit agressif (GTC).

        Utilise le mid price +/- slippage pour garantir l'exécution.
        """
        mid = self.get_mid(coin)
        if mid <= 0:
            return OrderResult(success=False, error=f"Cannot get mid price for {coin}")

        # Prix agressif pour garantir le fill
        if is_buy:
            price = round(mid * (1 + slippage_pct / 100), 6)
        else:
            price = round(mid * (1 - slippage_pct / 100), 6)

        return self.place_limit_order(
            coin=coin,
            is_buy=is_buy,
            size=size,
            price=price,
            post_only=False,
            reduce_only=reduce_only,
        )

    def place_bulk_orders(self, orders: list[dict]) -> list[OrderResult]:
        """
        Place plusieurs ordres en une seule transaction atomique.

        orders: list de {"coin", "is_buy", "size", "price", "post_only", "reduce_only"}
        """
        if self._exchange is None:
            return [OrderResult(success=False, error="Not authenticated")]

        order_requests = []
        for o in orders:
            tif = "Alo" if o.get("post_only", True) else "Gtc"
            sz_decimals = self.get_sz_decimals(o["coin"])
            size = round(o["size"], sz_decimals)
            if size <= 0:
                continue

            order_requests.append({
                "coin": o["coin"],
                "is_buy": o["is_buy"],
                "sz": size,
                "limit_px": o["price"],
                "order_type": {"limit": {"tif": tif}},
                "reduce_only": o.get("reduce_only", False),
            })

        if not order_requests:
            return []

        try:
            result = self._exchange.bulk_orders(order_requests)
            results = []

            if result["status"] == "ok":
                for s in result["response"]["data"]["statuses"]:
                    if "resting" in s:
                        results.append(OrderResult(success=True, oid=s["resting"]["oid"], status="resting"))
                    elif "filled" in s:
                        results.append(OrderResult(
                            success=True,
                            oid=s["filled"]["oid"],
                            status="filled",
                            filled_sz=float(s["filled"]["totalSz"]),
                            avg_px=float(s["filled"]["avgPx"]),
                        ))
                    elif "error" in s:
                        results.append(OrderResult(success=False, error=s["error"]))
                    else:
                        results.append(OrderResult(success=True, status="unknown"))
            else:
                results.append(OrderResult(success=False, error=str(result)))

            return results

        except Exception as e:
            return [OrderResult(success=False, error=str(e))]

    def cancel_all(self, coin: Optional[str] = None) -> bool:
        """Annule tous les ordres (ou ceux d'un coin spécifique).
        
        Uses bulk cancel for speed — critical for anti-flip.
        """
        if self._exchange is None:
            return False

        try:
            orders = self.get_open_orders(coin)
            if not orders:
                return True

            # Use bulk cancel via cancel_by_cloid or batch cancel
            # Group by coin for the API
            by_coin: dict[str, list[int]] = {}
            for o in orders:
                c = o["coin"]
                if c not in by_coin:
                    by_coin[c] = []
                by_coin[c].append(o["oid"])

            for cancel_coin, oids in by_coin.items():
                if len(oids) == 1:
                    self._exchange.cancel(cancel_coin, oids[0])
                else:
                    # Bulk cancel: send all cancels for this coin at once
                    # The SDK's bulk_cancel expects list of {"coin": str, "oid": int}
                    try:
                        cancel_requests = [{"coin": cancel_coin, "oid": oid} for oid in oids]
                        self._exchange.bulk_cancel(cancel_requests)
                    except AttributeError:
                        # Fallback if SDK doesn't have bulk_cancel
                        # Use cancel with list of oids (some SDK versions support this)
                        try:
                            self._exchange.cancel(cancel_coin, oids[0])
                            for oid in oids[1:]:
                                self._exchange.cancel(cancel_coin, oid)
                        except Exception:
                            pass
                    except Exception as e:
                        # Last resort: cancel one by one
                        log.warning("bulk_cancel_fallback", coin=cancel_coin, error=str(e))
                        for oid in oids:
                            try:
                                self._exchange.cancel(cancel_coin, oid)
                            except Exception:
                                pass

            return True
        except Exception as e:
            log.warning("cancel_failed", error=str(e))
            return False

    def cancel_coin_orders(self, coin: str) -> bool:
        """Annule tous les ordres d'un coin."""
        return self.cancel_all(coin)

    def get_recent_market_trades(self, coin: str, count: int = 50) -> list[dict]:
        """Récupère les trades récents du marché via l'API info."""
        import requests
        base_url = constants.MAINNET_API_URL if self.mainnet else constants.TESTNET_API_URL
        try:
            resp = requests.post(
                base_url + "/info",
                json={"type": "recentTrades", "coin": coin},
                timeout=5,
            )
            if resp.status_code == 200:
                trades = resp.json()
                return trades[-count:] if len(trades) > count else trades
        except Exception:
            pass
        return []