"""Point d'entrée principal du bot de trading modulaire"""

import logging
import os
from dotenv import load_dotenv

# Configuration
load_dotenv()

# Importer la configuration
from config import WATCHLIST, VALIDATION_THRESHOLD, LOG_FILE, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingBot')

# Imports modulaires
# Note: Certains modules sont encore en transition et importent depuis trading_bot_main.py
from analyzers import TechnicalAnalyzer, HistoricalNewsAnalyzer, RedditSentimentAnalyzer
from backtest import RealisticBacktestEngine  # En transition
from bot import bot  # En transition

if __name__ == "__main__":
    DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

    if not DISCORD_TOKEN:
        logger.error("❌ Token Discord manquant dans .env")
        logger.error("Ajoutez: DISCORD_BOT_TOKEN=votre_token")
        exit(1)

    logger.info("=" * 80)
    logger.info("🚀 TRADING BOT - ARCHITECTURE MODULAIRE")
    logger.info("=" * 80)
    logger.info(f"📁 Modules chargés:")
    logger.info(f"   ✅ analyzers.TechnicalAnalyzer")
    logger.info(f"   ✅ analyzers.HistoricalNewsAnalyzer")
    logger.info(f"   ✅ analyzers.RedditSentimentAnalyzer")
    logger.info(f"   ⚠️  backtest.RealisticBacktestEngine (transition)")
    logger.info(f"   ⚠️  bot.TradingBot (transition)")
    logger.info(f"")
    logger.info(f"📊 Configuration:")
    logger.info(f"   • Watchlist: {len(WATCHLIST)} actions")
    logger.info(f"   • Seuil validation: {VALIDATION_THRESHOLD}/100")
    logger.info(f"   • Log: {LOG_FILE}")
    logger.info("=" * 80)
    logger.info("")

    try:
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Arrêt du bot...")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        raise
