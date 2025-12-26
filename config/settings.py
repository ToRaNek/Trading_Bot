"""
Configuration generale du Trading Bot
Basee sur MASTER_TRADING_SKILL
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "storage"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "trading_bot.db"

# Creer les dossiers si necessaire
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TRADING MODE
# =============================================================================
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" ou "live"

# =============================================================================
# CAPITAL & RISK MANAGEMENT (PARTIE XI du MASTER_TRADING_SKILL)
# =============================================================================
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 10000))

# Risk par trade (JAMAIS plus de 2%)
RISK_PER_TRADE = 0.02  # 2%

# Risk/Reward minimum (PARTIE XII - minimum 1:3)
MIN_RISK_REWARD = 3.0

# Daily max loss (stop trading si atteint)
DAILY_MAX_LOSS = 0.05  # 5% du capital

# Max positions ouvertes simultanement
MAX_OPEN_POSITIONS = 5

# Max % du capital par position
MAX_POSITION_PERCENT = 0.20  # 20%

# Max % par secteur (diversification)
MAX_SECTOR_EXPOSURE = 0.30  # 30%

# =============================================================================
# PSYCHOLOGIE TRADING (PARTIE X du MASTER_TRADING_SKILL)
# =============================================================================
# Anti-CATS (Crack Addict Trading Syndrome)
MAX_TRADES_PER_DAY = 5

# Cooldown apres pertes consecutives
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_AFTER_LOSSES_MINUTES = 60

# Pas de trading si drawdown important
DRAWDOWN_STOP_TRADING = 0.10  # 10% drawdown = stop

# =============================================================================
# TIMEFRAMES (PARTIE XII - Swing Trading Hybride)
# =============================================================================
PRIMARY_TIMEFRAME = "1d"    # Daily pour direction
SECONDARY_TIMEFRAME = "1h"  # H1 pour precision
LOOKBACK_DAYS = 90          # 2-3 mois de donnees

# =============================================================================
# INDICATEURS TECHNIQUES (PARTIE V)
# =============================================================================
INDICATORS = {
    # RSI
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,

    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # Bollinger Bands
    "bb_period": 20,
    "bb_std": 2,

    # Moving Averages
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 200,
    "ema_short": 9,
    "ema_medium": 21,

    # ATR pour Stop Loss
    "atr_period": 14,
    "atr_multiplier": 2.0,

    # Volume
    "volume_sma": 20,
}

# =============================================================================
# ZONES HAUTE PROBABILITE (PARTIE XII)
# =============================================================================
ZONES_CONFIG = {
    "min_touches": 2,           # Minimum 2 points de contact
    "zone_threshold": 0.02,     # 2% de tolerance pour regrouper
    "max_age_days": 90,         # Zones des 3 derniers mois
    "min_zone_strength": 0.6,   # Score minimum
}

# =============================================================================
# SIGNAUX TRADING
# =============================================================================
SIGNAL_CONFIG = {
    # Force du signal (0-1)
    "min_signal_strength": 0.7,

    # Confirmations requises
    "require_volume_confirmation": True,
    "require_trend_alignment": True,

    # Rejection candle
    "min_rejection_ratio": 0.5,  # Meche >= 50% du corps
}

# =============================================================================
# TAKE PROFIT (PARTIE XII)
# =============================================================================
TAKE_PROFIT_CONFIG = {
    "tp1_percent": 0.25,    # Prendre 25% au TP1
    "tp2_percent": 0.50,    # Prendre 50% au TP2 (si existe)
    "move_to_breakeven": True,  # Passer en BE apres TP1
}

# =============================================================================
# ALPACA API (Live Trading - Gratuit pour US)
# =============================================================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # Paper par defaut

# =============================================================================
# NOTIFICATIONS (Discord webhook)
# =============================================================================
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
ENABLE_NOTIFICATIONS = bool(DISCORD_WEBHOOK_URL)

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# TRADING HOURS
# =============================================================================
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Ne pas trader pendant les premieres et dernieres minutes
AVOID_FIRST_MINUTES = 15
AVOID_LAST_MINUTES = 15

# =============================================================================
# BACKTESTING
# =============================================================================
BACKTEST_CONFIG = {
    "commission": 0.001,    # 0.1% commission
    "slippage": 0.001,      # 0.1% slippage
    "initial_capital": INITIAL_CAPITAL,
}
