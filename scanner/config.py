"""Configuration du scanner — seuils, poids du scoring, filtres."""

# API Hyperliquid
BASE_URL = "https://api.hyperliquid.xyz/info"
REQUEST_DELAY = 0.1  # secondes entre les requêtes d'orderbook

# Retry
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # secondes (doublé à chaque retry)

# === Filtres (exclure les paires qui ne passent pas) ===
MIN_VOLUME_24H = 100_000      # $100k minimum
MAX_VOLUME_24H = 50_000_000   # $50M maximum
MIN_SPREAD_BPS = 2.0          # 2 bps minimum

# === Poids du score composite ===
WEIGHTS = {
    "spread": 0.30,
    "volume": 0.25,
    "depth": 0.20,
    "levels": 0.15,
    "funding": 0.10,
}

# === Paramètres de scoring ===
# Spread : plafonné à 50 bps, linéaire
SPREAD_CAP_BPS = 50.0

# Volume : sweet spot $500k - $10M
VOLUME_SWEET_LOW = 500_000
VOLUME_SWEET_HIGH = 10_000_000

# Profondeur du book : seuils pour le scoring (en USD)
DEPTH_LOW = 10_000    # très peu de profondeur = score max
DEPTH_HIGH = 500_000  # beaucoup de profondeur = score min

# Niveaux de prix dans le book
LEVELS_LOW = 5        # peu de niveaux = score max
LEVELS_HIGH = 50      # beaucoup de niveaux = score min

# Funding rate : seuil pour pénaliser (en valeur absolue)
FUNDING_THRESHOLD = 0.001  # 0.1% = funding rate élevé → score 0

# Nombre de niveaux du book à analyser pour la profondeur
BOOK_DEPTH_LEVELS = 5
