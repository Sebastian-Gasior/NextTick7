"""Konfigurationsmodul für NextTick."""

from pathlib import Path
from typing import Dict, Any
import yaml

# Projektpfade
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TA_FEATURES_DIR = DATA_DIR / "ta_features"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Stelle sicher, dass alle Verzeichnisse existieren
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TA_FEATURES_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Trading-Parameter
TRADING_CONFIG: Dict[str, Any] = {
    "lookback_period": 60,  # Tage für Feature-Berechnung
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "learning_rate": 0.001,
}

# Technische Analyse Parameter
TA_CONFIG: Dict[str, Any] = {
    "sma_periods": [20, 50, 200],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2,
}

# Zentrale Liste der zu analysierenden Aktien
SYMBOLS = [
    'AAPL',
    'MSFT',
    'GOOGL',
    'IBM',
    'ORCL',
    'SAP',
    'ADBE',
    'AMZN',
    'NVDA',
    'TSM',
    'QCOM',
    'CSCO',
    'CMCSA',
    'INTC',
    'META',
    'NFLX',
    'TSLA',
    
]

def get_symbols() -> list:
    """Gibt die zentrale Liste der zu analysierenden Aktien zurück."""
    return SYMBOLS

def load_config(config_path: Path) -> Dict[str, Any]:
    """Lade benutzerdefinierte Konfiguration aus YAML-Datei."""
    if not config_path.exists():
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def update_config(custom_config: Dict[str, Any]) -> None:
    """Aktualisiere die Standardkonfiguration mit benutzerdefinierten Werten."""
    TRADING_CONFIG.update(custom_config.get("trading", {}))
    TA_CONFIG.update(custom_config.get("technical_analysis", {})) 