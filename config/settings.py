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

# =============================================================================
# STRATEGY SELECTION (Multi-Strategy Support)
# =============================================================================
# Strategies disponibles: swing, wyckoff, elliott, ichimoku, volume_profile, combined
ACTIVE_STRATEGY = os.getenv("ACTIVE_STRATEGY", "combined")

# Strategies activees pour le mode combine
ENABLED_STRATEGIES = {
    "swing": True,
    "wyckoff": True,
    "elliott": True,
    "ichimoku": True,
    "volume_profile": True,
}

# Poids des strategies dans le vote combine (plus eleve = plus d'influence)
STRATEGY_WEIGHTS = {
    "swing": 1.0,
    "wyckoff": 0.9,
    "elliott": 0.7,
    "ichimoku": 0.8,
    "volume_profile": 0.85,
}

# Score de consensus minimum pour signal combine
MIN_CONSENSUS_SCORE = 0.6

# =============================================================================
# WYCKOFF METHOD (PARTIE XIII du MASTER_TRADING_SKILL)
# =============================================================================
WYCKOFF_CONFIG = {
    # Detection des phases
    "min_range_periods": 10,      # Minimum 10 periodes pour un trading range
    "range_threshold": 0.05,      # 5% de variation max pour considerer un range

    # Volume analysis
    "volume_spike_multiplier": 2.0,   # Volume > 2x moyenne = spike
    "volume_dry_threshold": 0.5,      # Volume < 50% moyenne = dry up

    # Spring/Upthrust
    "spring_penetration_max": 0.02,   # Max 2% de penetration sous support
    "upthrust_penetration_max": 0.02, # Max 2% de penetration au-dessus resistance

    # Effort vs Result
    "effort_result_threshold": 1.5,   # Ratio effort/result pour divergence

    # Signal strength
    "min_confidence": 0.6,
}

# =============================================================================
# ELLIOTT WAVE (PARTIE XIV du MASTER_TRADING_SKILL)
# =============================================================================
ELLIOTT_CONFIG = {
    # Swing detection
    "swing_lookback": 5,          # Periodes pour identifier un pivot

    # Wave validation
    "wave2_max_retrace": 1.0,     # Vague 2 ne retrace pas plus de 100%
    "wave4_min_retrace": 0.236,   # Vague 4 retrace au minimum 23.6%
    "wave4_max_retrace": 0.50,    # Vague 4 ne depasse pas 50%

    # Fibonacci ratios
    "fib_tolerance": 0.02,        # 2% de tolerance sur les niveaux Fib

    # Targets
    "wave3_extension_target": 1.618,  # Target typique pour vague 3
    "wave5_extension_target": 1.0,    # Target typique pour vague 5

    # Signal strength
    "min_confidence": 0.6,
}

# =============================================================================
# ICHIMOKU (PARTIE XV du MASTER_TRADING_SKILL)
# =============================================================================
ICHIMOKU_CONFIG = {
    # Periodes standard (Hosoda)
    "tenkan_period": 9,
    "kijun_period": 26,
    "senkou_span_b_period": 52,
    "chikou_displacement": 26,
    "senkou_displacement": 26,

    # Signal strength requirements
    "require_price_above_kumo": True,     # Prix doit etre au-dessus du nuage (achat)
    "require_tk_cross": True,             # Tenkan doit croiser Kijun
    "require_chikou_confirmation": False, # Chikou Span confirmation (optionnel)

    # Cloud thickness threshold
    "thin_kumo_threshold": 0.01,  # Kumo < 1% = nuage fin (faible)

    # Signal strength
    "min_confidence": 0.6,
}

# =============================================================================
# VOLUME PROFILE (PARTIE XVI du MASTER_TRADING_SKILL)
# =============================================================================
VOLUME_PROFILE_CONFIG = {
    # Profile calculation
    "num_bins": 50,               # Nombre de niveaux de prix
    "lookback_periods": 30,       # Periodes pour calculer le profil
    "value_area_percent": 0.70,   # 70% du volume = Value Area

    # HVN/LVN detection
    "hvn_threshold": 1.5,         # > 1.5x volume moyen = HVN
    "lvn_threshold": 0.5,         # < 0.5x volume moyen = LVN

    # Trading signals
    "poc_bounce_tolerance": 0.005,    # 0.5% de tolerance pour rebond sur POC
    "va_breakout_confirmation": 2,    # 2 bougies pour confirmer breakout VA

    # Signal strength
    "min_confidence": 0.6,
}
