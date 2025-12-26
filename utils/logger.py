"""
Logging Configuration
Systeme de logs complet pour le trading bot
"""
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Niveaux de log
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Format des logs
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# CUSTOM FORMATTER
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour la console"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Coloriser le niveau
        record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


class TradeFormatter(logging.Formatter):
    """Formatter special pour les logs de trades"""

    def format(self, record):
        # Ajouter emoji selon le type
        if hasattr(record, 'trade_type'):
            if record.trade_type == 'BUY':
                record.msg = f"🟢 {record.msg}"
            elif record.trade_type == 'SELL':
                record.msg = f"🔴 {record.msg}"
            elif record.trade_type == 'CLOSE':
                record.msg = f"⚪ {record.msg}"
            elif record.trade_type == 'PROFIT':
                record.msg = f"💰 {record.msg}"
            elif record.trade_type == 'LOSS':
                record.msg = f"📉 {record.msg}"

        return super().format(record)


# =============================================================================
# LOGGER SETUP
# =============================================================================

def setup_logger(
    name: str = "trading_bot",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure et retourne un logger

    Args:
        name: Nom du logger
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Ecrire dans un fichier
        log_to_console: Afficher dans la console
        max_bytes: Taille max du fichier log
        backup_count: Nombre de fichiers de backup

    Returns:
        Logger configure
    """
    logger = logging.getLogger(name)

    # Eviter duplication des handlers
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))

    # Handler Console
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter(CONSOLE_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)

    # Handler Fichier (rotation par taille)
    if log_to_file:
        log_file = LOG_DIR / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def setup_trade_logger() -> logging.Logger:
    """
    Configure un logger special pour les trades

    Log separe pour historique des trades
    """
    logger = logging.getLogger("trades")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Fichier de trades (rotation journaliere)
    trade_file = LOG_DIR / "trades.log"
    handler = TimedRotatingFileHandler(
        trade_file,
        when='midnight',
        interval=1,
        backupCount=30,  # Garder 30 jours
        encoding='utf-8'
    )
    handler.setFormatter(TradeFormatter(
        "%(asctime)s | %(message)s",
        DATE_FORMAT
    ))
    logger.addHandler(handler)

    # Console aussi
    console = logging.StreamHandler()
    console.setFormatter(TradeFormatter(CONSOLE_FORMAT, DATE_FORMAT))
    logger.addHandler(console)

    return logger


# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

def log_trade(
    action: str,
    symbol: str,
    quantity: int,
    price: float,
    **kwargs
):
    """
    Log une action de trading

    Args:
        action: Type d'action (BUY, SELL, CLOSE)
        symbol: Symbole
        quantity: Quantite
        price: Prix
        **kwargs: Details supplementaires (pnl, reason, etc.)
    """
    trade_logger = logging.getLogger("trades")

    msg = f"{action} {quantity} {symbol} @ {price:.2f}"

    if 'stop_loss' in kwargs and 'take_profit' in kwargs:
        msg += f" | SL: {kwargs['stop_loss']:.2f} | TP: {kwargs['take_profit']:.2f}"

    if 'pnl' in kwargs:
        pnl = kwargs['pnl']
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        msg += f" | P&L: {pnl_str}"

    if 'reason' in kwargs:
        msg += f" | Reason: {kwargs['reason']}"

    # Ajouter attribut pour le formatter
    record = trade_logger.makeRecord(
        trade_logger.name, logging.INFO, "", 0, msg, (), None
    )
    record.trade_type = action
    if 'pnl' in kwargs:
        record.trade_type = 'PROFIT' if kwargs['pnl'] >= 0 else 'LOSS'

    trade_logger.handle(record)


def log_signal(
    symbol: str,
    signal_type: str,
    strength: float,
    reasons: list = None
):
    """
    Log un signal de trading
    """
    logger = logging.getLogger("signals")

    if not logger.handlers:
        setup_logger("signals", level="INFO")

    reasons_str = ", ".join(reasons) if reasons else "N/A"
    msg = f"SIGNAL {signal_type.upper()} | {symbol} | Strength: {strength:.2f} | Reasons: {reasons_str}"

    logger.info(msg)


def log_error(
    error: Exception,
    context: str = "",
    critical: bool = False
):
    """
    Log une erreur avec contexte
    """
    logger = logging.getLogger("errors")

    if not logger.handlers:
        error_file = LOG_DIR / "errors.log"
        handler = RotatingFileHandler(
            error_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

    msg = f"{context} | {type(error).__name__}: {str(error)}"

    if critical:
        logger.critical(msg, exc_info=True)
    else:
        logger.error(msg, exc_info=True)


def log_performance(metrics: dict):
    """
    Log les metriques de performance
    """
    logger = logging.getLogger("performance")

    if not logger.handlers:
        perf_file = LOG_DIR / "performance.log"
        handler = TimedRotatingFileHandler(
            perf_file,
            when='midnight',
            interval=1,
            backupCount=90,  # 3 mois
            encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s",
            DATE_FORMAT
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    msg_parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            msg_parts.append(f"{key}: {value:.2f}")
        else:
            msg_parts.append(f"{key}: {value}")

    logger.info(" | ".join(msg_parts))


# =============================================================================
# INITIALIZE DEFAULT LOGGERS
# =============================================================================

def init_all_loggers(level: str = "INFO"):
    """
    Initialise tous les loggers du systeme
    """
    # Logger principal
    setup_logger("trading_bot", level)

    # Logger trades
    setup_trade_logger()

    # Loggers des modules
    modules = [
        "data.fetcher",
        "data.database",
        "analysis.indicators",
        "analysis.zones",
        "analysis.patterns",
        "analysis.signals",
        "strategy.risk_management",
        "strategy.position_sizing",
        "strategy.swing_trading",
        "execution.orders",
        "execution.paper_trader",
        "execution.live_trader",
        "backtest.backtester",
        "dashboard"
    ]

    for module in modules:
        setup_logger(module, level, log_to_file=True, log_to_console=False)

    # Log demarrage
    main_logger = logging.getLogger("trading_bot")
    main_logger.info("=" * 60)
    main_logger.info("TRADING BOT STARTED")
    main_logger.info(f"Log level: {level}")
    main_logger.info(f"Log directory: {LOG_DIR}")
    main_logger.info("=" * 60)


# =============================================================================
# SINGLETON
# =============================================================================

_initialized = False

def get_logger(name: str = "trading_bot") -> logging.Logger:
    """
    Retourne un logger (initialise si necessaire)
    """
    global _initialized

    if not _initialized:
        init_all_loggers()
        _initialized = True

    return logging.getLogger(name)
