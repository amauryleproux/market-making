"""Configuration pour le bot Hyperliquid."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # === Auth ===
    # Clé privée du wallet API (généré sur https://app.hyperliquid.xyz/API)
    secret_key: str = os.getenv("HL_SECRET_KEY", "")
    # Adresse publique du wallet PRINCIPAL (pas le wallet API)
    account_address: str = os.getenv("HL_ACCOUNT_ADDRESS", "")

    # === Network ===
    # True = mainnet, False = testnet
    mainnet: bool = os.getenv("HL_MAINNET", "false").lower() == "true"

    # === Trading ===
    # Coins à market-maker (séparés par virgule)
    coins: str = os.getenv("HL_COINS", "ETH,BTC")
    # Taille d'ordre par défaut en USD
    order_size_usd: float = float(os.getenv("HL_ORDER_SIZE_USD", "50"))
    # Spread minimum (en %) — ex: 0.05 = 0.05%
    min_spread_pct: float = float(os.getenv("HL_MIN_SPREAD_PCT", "0.05"))
    # Nombre de niveaux de chaque côté
    num_levels: int = int(os.getenv("HL_NUM_LEVELS", "3"))
    # Écart entre niveaux (en %)
    level_spacing_pct: float = float(os.getenv("HL_LEVEL_SPACING_PCT", "0.02"))

    # === Risk ===
    # Exposition max par coin (en USD)
    max_position_usd: float = float(os.getenv("HL_MAX_POSITION_USD", "500"))
    # Exposition totale max (en USD)
    max_total_exposure_usd: float = float(os.getenv("HL_MAX_TOTAL_EXPOSURE_USD", "2000"))
    # Skew du spread quand on a une position (inventory management)
    inventory_skew: float = float(os.getenv("HL_INVENTORY_SKEW", "0.5"))

    # === Timing ===
    # Intervalle entre les mises à jour des quotes (secondes)
    update_interval: float = float(os.getenv("HL_UPDATE_INTERVAL", "10"))
    # Intervalle pour le monitoring (secondes)
    monitor_interval: float = float(os.getenv("HL_MONITOR_INTERVAL", "60"))

    # === Mode ===
    dry_run: bool = os.getenv("HL_DRY_RUN", "true").lower() == "true"
    log_level: str = os.getenv("HL_LOG_LEVEL", "INFO")

    @property
    def coin_list(self) -> list[str]:
        return [c.strip().upper() for c in self.coins.split(",") if c.strip()]


config = Config()
