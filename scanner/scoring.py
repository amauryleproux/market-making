"""Logique de calcul du score composite pour le market making."""

from scanner.config import (
    WEIGHTS,
    SPREAD_CAP_BPS,
    VOLUME_SWEET_LOW,
    VOLUME_SWEET_HIGH,
    DEPTH_LOW,
    DEPTH_HIGH,
    LEVELS_LOW,
    LEVELS_HIGH,
    FUNDING_THRESHOLD,
)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def score_spread(spread_bps: float) -> float:
    """Plus le spread est large, mieux c'est. Plafonné à SPREAD_CAP_BPS."""
    if spread_bps <= 0:
        return 0.0
    return _clamp((spread_bps / SPREAD_CAP_BPS) * 100)


def score_volume(volume_usd: float) -> float:
    """Sweet spot entre VOLUME_SWEET_LOW et VOLUME_SWEET_HIGH.

    Score max (100) dans le sweet spot, décroît linéairement en dehors.
    """
    if VOLUME_SWEET_LOW <= volume_usd <= VOLUME_SWEET_HIGH:
        return 100.0

    if volume_usd < VOLUME_SWEET_LOW:
        # Décroît de 100 à 0 entre MIN_VOLUME et SWEET_LOW
        ratio = volume_usd / VOLUME_SWEET_LOW
        return _clamp(ratio * 100)

    # volume > SWEET_HIGH
    # Décroît de 100 à 0 entre SWEET_HIGH et MAX_VOLUME (50M)
    overshoot = (volume_usd - VOLUME_SWEET_HIGH) / (50_000_000 - VOLUME_SWEET_HIGH)
    return _clamp((1.0 - overshoot) * 100)


def score_depth(depth_usd: float) -> float:
    """Moins de profondeur = moins de concurrence = mieux.

    Score linéaire inversé entre DEPTH_LOW (score 100) et DEPTH_HIGH (score 0).
    """
    if depth_usd <= DEPTH_LOW:
        return 100.0
    if depth_usd >= DEPTH_HIGH:
        return 0.0
    return _clamp((1.0 - (depth_usd - DEPTH_LOW) / (DEPTH_HIGH - DEPTH_LOW)) * 100)


def score_levels(num_levels: int) -> float:
    """Moins de niveaux = moins de market makers = mieux.

    Score linéaire inversé entre LEVELS_LOW (score 100) et LEVELS_HIGH (score 0).
    """
    if num_levels <= LEVELS_LOW:
        return 100.0
    if num_levels >= LEVELS_HIGH:
        return 0.0
    return _clamp((1.0 - (num_levels - LEVELS_LOW) / (LEVELS_HIGH - LEVELS_LOW)) * 100)


def score_funding(funding_rate: float) -> float:
    """Plus le funding est proche de 0, mieux c'est.

    Score décroît de 100 (funding=0) à 0 (funding >= FUNDING_THRESHOLD).
    """
    abs_funding = abs(funding_rate)
    if abs_funding >= FUNDING_THRESHOLD:
        return 0.0
    return _clamp((1.0 - abs_funding / FUNDING_THRESHOLD) * 100)


def compute_composite_score(metrics: dict) -> dict:
    """Calcule le score composite à partir des métriques brutes.

    Args:
        metrics: dict avec les clés spread_bps, volume_24h, depth_usd,
                 num_levels, funding_rate

    Returns:
        dict avec les sous-scores et le score composite
    """
    sub_scores = {
        "spread": score_spread(metrics["spread_bps"]),
        "volume": score_volume(metrics["volume_24h"]),
        "depth": score_depth(metrics["depth_usd"]),
        "levels": score_levels(metrics["num_levels"]),
        "funding": score_funding(metrics["funding_rate"]),
    }

    composite = sum(sub_scores[k] * WEIGHTS[k] for k in WEIGHTS)

    return {
        "sub_scores": sub_scores,
        "composite": round(composite, 1),
    }
