"""Wrapper API Hyperliquid — lecture seule, pas d'auth nécessaire."""

import time
import requests
from scanner.config import BASE_URL, MAX_RETRIES, RETRY_BACKOFF


def _post(payload: dict) -> dict | list:
    """POST vers l'API Hyperliquid avec retry + backoff exponentiel."""
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


def get_meta() -> dict:
    """Récupère les métadonnées (liste de tous les assets)."""
    return _post({"type": "meta"})


def get_meta_and_asset_ctxs() -> list:
    """Récupère meta + contextes des assets (volume, funding, OI, etc.)."""
    return _post({"type": "metaAndAssetCtxs"})


def get_l2_book(coin: str) -> dict:
    """Récupère l'orderbook L2 pour un coin."""
    return _post({"type": "l2Book", "coin": coin})
