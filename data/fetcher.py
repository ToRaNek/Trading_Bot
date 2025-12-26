"""
Data Fetcher - Recuperation des donnees de marche
Utilise yfinance (gratuit)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

from config.settings import LOOKBACK_DAYS, PRIMARY_TIMEFRAME, SECONDARY_TIMEFRAME

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Recupere les donnees de marche via Yahoo Finance
    Supporte Daily et Hourly pour le Swing Trading Hybride
    """

    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=5)

    def get_stock_data(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Recupere les donnees OHLCV pour un symbole

        Args:
            symbol: Symbole de l'action (ex: "AAPL")
            period: Periode (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Intervalle (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame avec Open, High, Low, Close, Volume
        """
        cache_key = f"{symbol}_{period}_{interval}"

        # Check cache
        if cache_key in self.cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
                logger.debug(f"Cache hit for {symbol}")
                return self.cache[cache_key].copy()

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Nettoyer les colonnes
            df.columns = [col.lower() for col in df.columns]

            # Garder seulement OHLCV
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in required_cols if col in df.columns]]

            # Ajouter le symbole
            df['symbol'] = symbol

            # Cache
            self.cache[cache_key] = df.copy()
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration

            logger.info(f"Fetched {len(df)} rows for {symbol} ({interval})")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def get_daily_data(self, symbol: str, days: int = LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
        """
        Recupere les donnees Daily (timeframe principal du Swing Trading)
        """
        if days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"

        return self.get_stock_data(symbol, period=period, interval="1d")

    def get_hourly_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Recupere les donnees Hourly (timeframe secondaire pour precision)
        Note: yfinance limite les donnees hourly a ~730 heures (~30 jours)
        """
        # yfinance limite: hourly data max 730 data points
        if days > 30:
            days = 30
            logger.warning(f"Hourly data limited to 30 days for {symbol}")

        period = f"{days}d"
        return self.get_stock_data(symbol, period=period, interval="1h")

    def get_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "3mo",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Recupere les donnees pour plusieurs symboles
        """
        data = {}
        for symbol in symbols:
            df = self.get_stock_data(symbol, period=period, interval=interval)
            if df is not None:
                data[symbol] = df
        return data

    def get_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """
        Recupere le prix en temps reel (avec delai Yahoo ~15min)
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            quote = {
                'symbol': symbol,
                'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'open': info.get('regularMarketOpen'),
                'high': info.get('regularMarketDayHigh'),
                'low': info.get('regularMarketDayLow'),
                'volume': info.get('regularMarketVolume'),
                'previous_close': info.get('regularMarketPreviousClose'),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'timestamp': datetime.now()
            }

            return quote

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Recupere les informations de base sur une action
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName') or info.get('shortName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
            }

        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {e}")
            return None

    def is_market_open(self) -> bool:
        """
        Verifie si le marche US est ouvert
        """
        now = datetime.now()
        # Simplification: verifier heure US (pas de gestion timezone complete)
        # Marche ouvert 9:30-16:00 ET, lundi-vendredi
        if now.weekday() >= 5:  # Weekend
            return False

        # TODO: Ajouter gestion timezone correcte et jours feries
        return True

    def clear_cache(self):
        """
        Vide le cache
        """
        self.cache.clear()
        self.cache_expiry.clear()
        logger.info("Cache cleared")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def download_historical_data(
    symbols: List[str],
    start_date: str,
    end_date: str = None,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Telecharge les donnees historiques pour backtesting

    Args:
        symbols: Liste de symboles
        start_date: Date de debut (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD), defaut = aujourd'hui
        interval: Intervalle des donnees

    Returns:
        Dict avec les DataFrames par symbole
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = {}
    for symbol in symbols:
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                df['symbol'] = symbol
                data[symbol] = df
                logger.info(f"Downloaded {len(df)} rows for {symbol}")
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")

    return data


def validate_symbol(symbol: str) -> bool:
    """
    Verifie si un symbole est valide
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('regularMarketPrice') is not None
    except:
        return False


# =============================================================================
# SINGLETON
# =============================================================================
_fetcher_instance = None

def get_fetcher() -> DataFetcher:
    """Retourne l'instance singleton du DataFetcher"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = DataFetcher()
    return _fetcher_instance
