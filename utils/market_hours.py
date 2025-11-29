"""
Module pour vérifier les horaires d'ouverture des marchés

Horaires des marchés (heure française - UTC+1 en hiver, UTC+2 en été):
- US (NYSE/NASDAQ): 15h30 - 22h00 (heure française)
- France (Euronext Paris): 09h00 - 17h30 (heure française)
"""

from datetime import datetime, time
from typing import Tuple
import pytz
import logging

logger = logging.getLogger('TradingBot')


class MarketHours:
    """Gestion des horaires d'ouverture des marchés"""

    # Timezone Paris
    PARIS_TZ = pytz.timezone('Europe/Paris')

    # Horaires des marchés en heure française
    US_OPEN = time(15, 30)      # 15:30 heure française
    US_CLOSE = time(22, 0)      # 22:00 heure française
    US_LAST_TRADE = time(21, 45) # Dernière analyse à 21:45 pour avoir le temps de trader

    FRANCE_OPEN = time(9, 0)     # 09:00 heure française
    FRANCE_CLOSE = time(17, 30)  # 17:30 heure française
    FRANCE_LAST_TRADE = time(17, 15) # Dernière analyse à 17:15

    @classmethod
    def is_us_stock(cls, symbol: str) -> bool:
        """Vérifie si c'est une action US"""
        # Actions françaises finissent par .PA
        return not symbol.endswith('.PA')

    @classmethod
    def get_current_paris_time(cls) -> datetime:
        """Retourne l'heure actuelle à Paris"""
        return datetime.now(cls.PARIS_TZ)

    @classmethod
    def is_market_open(cls, symbol: str, check_time: datetime = None) -> Tuple[bool, str]:
        """
        Vérifie si le marché est ouvert pour cette action

        Args:
            symbol: Ticker de l'action
            check_time: Heure à vérifier (None = maintenant)

        Returns:
            (is_open, reason) - True si ouvert, False sinon + raison
        """
        if check_time is None:
            check_time = cls.get_current_paris_time()

        # Convertir en timezone Paris si nécessaire
        if check_time.tzinfo is None:
            check_time = cls.PARIS_TZ.localize(check_time)
        else:
            check_time = check_time.astimezone(cls.PARIS_TZ)

        # Vérifier si c'est un jour de semaine
        if check_time.weekday() >= 5:  # Samedi=5, Dimanche=6
            return False, f"Marché fermé (week-end)"

        current_time = check_time.time()

        if cls.is_us_stock(symbol):
            # Marché US
            if current_time < cls.US_OPEN:
                return False, f"Marché US fermé (ouverture à {cls.US_OPEN.strftime('%H:%M')})"
            elif current_time >= cls.US_CLOSE:
                return False, f"Marché US fermé (fermeture à {cls.US_CLOSE.strftime('%H:%M')})"
            else:
                return True, "Marché US ouvert"
        else:
            # Marché français
            if current_time < cls.FRANCE_OPEN:
                return False, f"Marché français fermé (ouverture à {cls.FRANCE_OPEN.strftime('%H:%M')})"
            elif current_time >= cls.FRANCE_CLOSE:
                return False, f"Marché français fermé (fermeture à {cls.FRANCE_CLOSE.strftime('%H:%M')})"
            else:
                return True, "Marché français ouvert"

    @classmethod
    def can_trade_now(cls, symbol: str, check_time: datetime = None) -> Tuple[bool, str]:
        """
        Vérifie si on peut trader maintenant (avec marge de sécurité avant la fermeture)

        Args:
            symbol: Ticker de l'action
            check_time: Heure à vérifier (None = maintenant)

        Returns:
            (can_trade, reason) - True si on peut trader, False sinon + raison
        """
        if check_time is None:
            check_time = cls.get_current_paris_time()

        # Convertir en timezone Paris si nécessaire
        if check_time.tzinfo is None:
            check_time = cls.PARIS_TZ.localize(check_time)
        else:
            check_time = check_time.astimezone(cls.PARIS_TZ)

        # Vérifier si c'est un jour de semaine
        if check_time.weekday() >= 5:
            return False, "Marché fermé (week-end)"

        current_time = check_time.time()

        if cls.is_us_stock(symbol):
            # Marché US - doit être après l'ouverture et avant la dernière analyse
            if current_time < cls.US_OPEN:
                return False, f"Marché US pas encore ouvert (ouverture à {cls.US_OPEN.strftime('%H:%M')})"
            elif current_time >= cls.US_LAST_TRADE:
                return False, f"Trop tard pour trader (dernière analyse à {cls.US_LAST_TRADE.strftime('%H:%M')})"
            else:
                return True, "Trading autorisé (marché US)"
        else:
            # Marché français
            if current_time < cls.FRANCE_OPEN:
                return False, f"Marché français pas encore ouvert (ouverture à {cls.FRANCE_OPEN.strftime('%H:%M')})"
            elif current_time >= cls.FRANCE_LAST_TRADE:
                return False, f"Trop tard pour trader (dernière analyse à {cls.FRANCE_LAST_TRADE.strftime('%H:%M')})"
            else:
                return True, "Trading autorisé (marché français)"

    @classmethod
    def get_next_market_open(cls, symbol: str, from_time: datetime = None) -> datetime:
        """
        Retourne la prochaine ouverture du marché

        Args:
            symbol: Ticker de l'action
            from_time: À partir de quelle heure (None = maintenant)

        Returns:
            Datetime de la prochaine ouverture
        """
        if from_time is None:
            from_time = cls.get_current_paris_time()

        # Convertir en timezone Paris si nécessaire
        if from_time.tzinfo is None:
            from_time = cls.PARIS_TZ.localize(from_time)
        else:
            from_time = from_time.astimezone(cls.PARIS_TZ)

        is_us = cls.is_us_stock(symbol)
        market_open = cls.US_OPEN if is_us else cls.FRANCE_OPEN

        # Si on est avant l'ouverture du jour, retourner aujourd'hui
        if from_time.time() < market_open and from_time.weekday() < 5:
            return from_time.replace(hour=market_open.hour, minute=market_open.minute, second=0, microsecond=0)

        # Sinon, chercher le prochain jour ouvrable
        from datetime import timedelta
        next_day = from_time + timedelta(days=1)

        # Sauter le week-end
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        return next_day.replace(hour=market_open.hour, minute=market_open.minute, second=0, microsecond=0)

    @classmethod
    def log_market_status(cls, symbol: str, check_time: datetime = None):
        """Log le statut du marché pour debug"""
        if check_time is None:
            check_time = cls.get_current_paris_time()

        is_open, reason_open = cls.is_market_open(symbol, check_time)
        can_trade, reason_trade = cls.can_trade_now(symbol, check_time)

        market_type = "US" if cls.is_us_stock(symbol) else "FR"

        logger.info(f"[MarketHours] {symbol} ({market_type}) à {check_time.strftime('%H:%M')}:")
        logger.info(f"   • Marché ouvert: {is_open} - {reason_open}")
        logger.info(f"   • Trading autorisé: {can_trade} - {reason_trade}")

        if not can_trade:
            next_open = cls.get_next_market_open(symbol, check_time)
            logger.info(f"   • Prochaine ouverture: {next_open.strftime('%Y-%m-%d %H:%M')}")
