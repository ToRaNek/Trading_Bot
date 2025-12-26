"""
User Settings Manager - Permet de modifier les settings depuis le dashboard
Les settings utilisateur surchargent les valeurs par defaut de settings.py
"""
import json
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

# Chemin du fichier de settings utilisateur
USER_SETTINGS_FILE = Path(__file__).parent / "user_settings.json"

# Settings par defaut (copie de settings.py)
DEFAULT_SETTINGS = {
    # Capital & Risk
    "INITIAL_CAPITAL": 10000,
    "RISK_PER_TRADE": 0.02,
    "MIN_RISK_REWARD": 3.0,
    "DAILY_MAX_LOSS": 0.05,
    "MAX_OPEN_POSITIONS": 5,
    "MAX_POSITION_PERCENT": 0.20,
    "MAX_SECTOR_EXPOSURE": 0.30,

    # Psychologie
    "MAX_TRADES_PER_DAY": 5,
    "MAX_CONSECUTIVE_LOSSES": 3,
    "COOLDOWN_AFTER_LOSSES_MINUTES": 60,
    "DRAWDOWN_STOP_TRADING": 0.10,

    # Timeframes
    "PRIMARY_TIMEFRAME": "1d",
    "SECONDARY_TIMEFRAME": "1h",
    "LOOKBACK_DAYS": 90,

    # Strategy
    "ACTIVE_STRATEGY": "combined",
    "ENABLED_STRATEGIES": {
        "swing": True,
        "wyckoff": True,
        "elliott": True,
        "ichimoku": True,
        "volume_profile": True
    },
    "STRATEGY_WEIGHTS": {
        "swing": 1.0,
        "wyckoff": 1.0,
        "elliott": 1.0,
        "ichimoku": 1.0,
        "volume_profile": 1.0
    },

    # Indicateurs
    "INDICATORS": {
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "sma_short": 20,
        "sma_medium": 50,
        "sma_long": 200,
        "ema_short": 9,
        "ema_medium": 21,
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "volume_sma": 20
    },

    # Zones
    "ZONES_CONFIG": {
        "min_touches": 2,
        "zone_threshold": 0.02,
        "max_age_days": 90,
        "min_zone_strength": 0.6
    },

    # Signaux
    "SIGNAL_CONFIG": {
        "min_signal_strength": 0.7,
        "require_volume_confirmation": True,
        "require_trend_alignment": True,
        "min_rejection_ratio": 0.5
    },

    # Take Profit
    "TAKE_PROFIT_CONFIG": {
        "tp1_percent": 0.25,
        "tp2_percent": 0.50,
        "move_to_breakeven": True
    },

    # API
    "ALPACA_API_KEY": "",
    "ALPACA_SECRET_KEY": "",
    "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",

    # Notifications
    "DISCORD_WEBHOOK_URL": "",
    "ENABLE_TRADE_NOTIFICATIONS": True,
    "ENABLE_SIGNAL_NOTIFICATIONS": True,
    "ENABLE_DAILY_SUMMARY": True
}


def load_user_settings() -> Dict[str, Any]:
    """Charge les settings utilisateur depuis le fichier JSON"""
    settings = DEFAULT_SETTINGS.copy()

    if USER_SETTINGS_FILE.exists():
        try:
            with open(USER_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                user_settings = json.load(f)
                # Merge avec les defaults (user settings prennent le dessus)
                settings = _deep_merge(settings, user_settings)
                logger.info("User settings loaded")
        except Exception as e:
            logger.error(f"Error loading user settings: {e}")

    return settings


def save_user_settings(settings: Dict[str, Any]) -> bool:
    """Sauvegarde les settings utilisateur dans le fichier JSON"""
    try:
        with open(USER_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        logger.info("User settings saved")
        return True
    except Exception as e:
        logger.error(f"Error saving user settings: {e}")
        return False


def get_setting(key: str, default: Any = None) -> Any:
    """Recupere une valeur de setting"""
    settings = load_user_settings()
    return settings.get(key, default)


def set_setting(key: str, value: Any) -> bool:
    """Modifie une valeur de setting"""
    settings = load_user_settings()
    settings[key] = value
    return save_user_settings(settings)


def reset_to_defaults() -> bool:
    """Reset tous les settings aux valeurs par defaut"""
    return save_user_settings(DEFAULT_SETTINGS.copy())


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Merge recursif de deux dictionnaires"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# Charger les settings au demarrage
_current_settings = load_user_settings()


def get_all_settings() -> Dict[str, Any]:
    """Retourne tous les settings actuels"""
    global _current_settings
    _current_settings = load_user_settings()
    return _current_settings
