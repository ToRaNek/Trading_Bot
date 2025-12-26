"""
Patterns de Prix et Bougies
Base sur MASTER_TRADING_SKILL PARTIE V - Sections 36-44
Detection automatique des patterns
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CandlePattern:
    """Represente un pattern de bougie detecte"""
    name: str
    signal: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1
    index: int
    description: str


class PatternDetector:
    """
    Detecte les patterns de prix et de bougies
    """

    def __init__(self):
        self.min_body_ratio = 0.6  # Ratio corps/range pour bougie de force

    # =========================================================================
    # CANDLESTICK PATTERNS (PARTIE V - Section 36-44)
    # =========================================================================

    def detect_all_patterns(self, df: pd.DataFrame) -> List[CandlePattern]:
        """
        Detecte tous les patterns sur les dernieres bougies
        """
        patterns = []

        if len(df) < 5:
            return patterns

        # Single candle patterns
        patterns.extend(self._detect_doji(df))
        patterns.extend(self._detect_hammer(df))
        patterns.extend(self._detect_shooting_star(df))
        patterns.extend(self._detect_marubozu(df))

        # Double candle patterns
        patterns.extend(self._detect_engulfing(df))
        patterns.extend(self._detect_harami(df))
        patterns.extend(self._detect_tweezer(df))

        # Triple candle patterns
        patterns.extend(self._detect_morning_evening_star(df))
        patterns.extend(self._detect_three_soldiers_crows(df))

        return patterns

    def detect_rejection_candle(
        self,
        df: pd.DataFrame,
        direction: str = 'buy'
    ) -> bool:
        """
        Detecte une bougie de rejection (PARTIE XII - Section 97)

        Une bougie de rejection montre que les acheteurs/vendeurs
        ont repousse le prix dans notre direction.
        """
        if len(df) < 1:
            return False

        candle = df.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return False

        if direction == 'buy':
            # Bougie haussiere avec longue meche basse
            is_bullish = candle['close'] > candle['open']
            has_rejection = lower_wick >= body * 0.5  # Meche >= 50% du corps
            return is_bullish and has_rejection

        elif direction == 'sell':
            # Bougie baissiere avec longue meche haute
            is_bearish = candle['close'] < candle['open']
            has_rejection = upper_wick >= body * 0.5
            return is_bearish and has_rejection

        return False

    def detect_momentum_candle(self, df: pd.DataFrame, direction: str = 'buy') -> bool:
        """
        Detecte une bougie de momentum/force

        Caracteristiques:
        - Grand corps
        - Petites meches
        - Dans la direction attendue
        """
        if len(df) < 1:
            return False

        candle = df.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return False

        body_ratio = body / total_range

        if direction == 'buy':
            is_bullish = candle['close'] > candle['open']
            return is_bullish and body_ratio >= self.min_body_ratio

        elif direction == 'sell':
            is_bearish = candle['close'] < candle['open']
            return is_bearish and body_ratio >= self.min_body_ratio

        return False

    # =========================================================================
    # SINGLE CANDLE PATTERNS
    # =========================================================================

    def _detect_doji(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte les Doji (Section 36)"""
        patterns = []
        candle = df.iloc[-1]

        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']

        if total_range > 0 and body / total_range < 0.1:
            patterns.append(CandlePattern(
                name='doji',
                signal='neutral',
                strength=0.5,
                index=len(df) - 1,
                description='Doji - Indecision du marche'
            ))

        return patterns

    def _detect_hammer(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte les Hammer/Hanging Man (Section 37)"""
        patterns = []
        candle = df.iloc[-1]

        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_range = candle['high'] - candle['low']

        if total_range == 0 or body == 0:
            return patterns

        # Hammer: petite meche haute, longue meche basse, petit corps en haut
        if lower_wick >= body * 2 and upper_wick <= body * 0.3:
            # Verifier contexte (apres baisse = bullish, apres hausse = bearish)
            if len(df) >= 5:
                recent_trend = df['close'].iloc[-5] - df['close'].iloc[-2]
                if recent_trend < 0:  # Apres baisse = Hammer bullish
                    patterns.append(CandlePattern(
                        name='hammer',
                        signal='bullish',
                        strength=0.7,
                        index=len(df) - 1,
                        description='Hammer - Signal de retournement haussier'
                    ))
                else:  # Apres hausse = Hanging Man bearish
                    patterns.append(CandlePattern(
                        name='hanging_man',
                        signal='bearish',
                        strength=0.6,
                        index=len(df) - 1,
                        description='Hanging Man - Signal de retournement baissier'
                    ))

        return patterns

    def _detect_shooting_star(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte les Shooting Star/Inverted Hammer (Section 38)"""
        patterns = []
        candle = df.iloc[-1]

        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']

        if body == 0:
            return patterns

        # Shooting Star: longue meche haute, petite meche basse
        if upper_wick >= body * 2 and lower_wick <= body * 0.3:
            if len(df) >= 5:
                recent_trend = df['close'].iloc[-5] - df['close'].iloc[-2]
                if recent_trend > 0:  # Apres hausse = Shooting Star bearish
                    patterns.append(CandlePattern(
                        name='shooting_star',
                        signal='bearish',
                        strength=0.7,
                        index=len(df) - 1,
                        description='Shooting Star - Signal de retournement baissier'
                    ))
                else:  # Apres baisse = Inverted Hammer bullish
                    patterns.append(CandlePattern(
                        name='inverted_hammer',
                        signal='bullish',
                        strength=0.6,
                        index=len(df) - 1,
                        description='Inverted Hammer - Signal de retournement haussier'
                    ))

        return patterns

    def _detect_marubozu(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte les Marubozu (bougie pleine sans meches)"""
        patterns = []
        candle = df.iloc[-1]

        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return patterns

        if body / total_range >= 0.95:
            if candle['close'] > candle['open']:
                patterns.append(CandlePattern(
                    name='marubozu_bullish',
                    signal='bullish',
                    strength=0.8,
                    index=len(df) - 1,
                    description='Marubozu Haussier - Forte pression acheteuse'
                ))
            else:
                patterns.append(CandlePattern(
                    name='marubozu_bearish',
                    signal='bearish',
                    strength=0.8,
                    index=len(df) - 1,
                    description='Marubozu Baissier - Forte pression vendeuse'
                ))

        return patterns

    # =========================================================================
    # DOUBLE CANDLE PATTERNS
    # =========================================================================

    def _detect_engulfing(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte les Engulfing patterns (Section 39)"""
        patterns = []

        if len(df) < 2:
            return patterns

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])

        # Bullish Engulfing
        if (prev['close'] < prev['open'] and  # Prev bearish
            curr['close'] > curr['open'] and  # Curr bullish
            curr['open'] <= prev['close'] and
            curr['close'] >= prev['open'] and
            curr_body > prev_body):
            patterns.append(CandlePattern(
                name='bullish_engulfing',
                signal='bullish',
                strength=0.8,
                index=len(df) - 1,
                description='Bullish Engulfing - Fort signal haussier'
            ))

        # Bearish Engulfing
        if (prev['close'] > prev['open'] and  # Prev bullish
            curr['close'] < curr['open'] and  # Curr bearish
            curr['open'] >= prev['close'] and
            curr['close'] <= prev['open'] and
            curr_body > prev_body):
            patterns.append(CandlePattern(
                name='bearish_engulfing',
                signal='bearish',
                strength=0.8,
                index=len(df) - 1,
                description='Bearish Engulfing - Fort signal baissier'
            ))

        return patterns

    def _detect_harami(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte les Harami patterns (Section 40)"""
        patterns = []

        if len(df) < 2:
            return patterns

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])

        # Bullish Harami (petit corps haussier dans grand corps baissier)
        if (prev['close'] < prev['open'] and
            curr_body < prev_body * 0.5 and
            curr['high'] < prev['open'] and
            curr['low'] > prev['close']):
            patterns.append(CandlePattern(
                name='bullish_harami',
                signal='bullish',
                strength=0.6,
                index=len(df) - 1,
                description='Bullish Harami - Potentiel retournement haussier'
            ))

        # Bearish Harami
        if (prev['close'] > prev['open'] and
            curr_body < prev_body * 0.5 and
            curr['high'] < prev['close'] and
            curr['low'] > prev['open']):
            patterns.append(CandlePattern(
                name='bearish_harami',
                signal='bearish',
                strength=0.6,
                index=len(df) - 1,
                description='Bearish Harami - Potentiel retournement baissier'
            ))

        return patterns

    def _detect_tweezer(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte les Tweezer tops/bottoms"""
        patterns = []

        if len(df) < 2:
            return patterns

        prev = df.iloc[-2]
        curr = df.iloc[-1]
        tolerance = (prev['high'] - prev['low']) * 0.02

        # Tweezer Bottom
        if abs(prev['low'] - curr['low']) <= tolerance:
            if prev['close'] < prev['open'] and curr['close'] > curr['open']:
                patterns.append(CandlePattern(
                    name='tweezer_bottom',
                    signal='bullish',
                    strength=0.65,
                    index=len(df) - 1,
                    description='Tweezer Bottom - Support confirme'
                ))

        # Tweezer Top
        if abs(prev['high'] - curr['high']) <= tolerance:
            if prev['close'] > prev['open'] and curr['close'] < curr['open']:
                patterns.append(CandlePattern(
                    name='tweezer_top',
                    signal='bearish',
                    strength=0.65,
                    index=len(df) - 1,
                    description='Tweezer Top - Resistance confirmee'
                ))

        return patterns

    # =========================================================================
    # TRIPLE CANDLE PATTERNS
    # =========================================================================

    def _detect_morning_evening_star(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte Morning Star et Evening Star (Section 41)"""
        patterns = []

        if len(df) < 3:
            return patterns

        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]

        first_body = abs(first['close'] - first['open'])
        second_body = abs(second['close'] - second['open'])
        third_body = abs(third['close'] - third['open'])

        # Morning Star (bullish reversal)
        if (first['close'] < first['open'] and  # Big bearish
            second_body < first_body * 0.3 and  # Small body
            third['close'] > third['open'] and  # Big bullish
            third['close'] > (first['open'] + first['close']) / 2):
            patterns.append(CandlePattern(
                name='morning_star',
                signal='bullish',
                strength=0.85,
                index=len(df) - 1,
                description='Morning Star - Fort signal de retournement haussier'
            ))

        # Evening Star (bearish reversal)
        if (first['close'] > first['open'] and  # Big bullish
            second_body < first_body * 0.3 and  # Small body
            third['close'] < third['open'] and  # Big bearish
            third['close'] < (first['open'] + first['close']) / 2):
            patterns.append(CandlePattern(
                name='evening_star',
                signal='bearish',
                strength=0.85,
                index=len(df) - 1,
                description='Evening Star - Fort signal de retournement baissier'
            ))

        return patterns

    def _detect_three_soldiers_crows(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detecte Three White Soldiers et Three Black Crows (Section 42)"""
        patterns = []

        if len(df) < 3:
            return patterns

        candles = df.iloc[-3:]

        # Three White Soldiers
        all_bullish = all(c['close'] > c['open'] for _, c in candles.iterrows())
        higher_closes = all(
            candles.iloc[i]['close'] > candles.iloc[i-1]['close']
            for i in range(1, 3)
        )

        if all_bullish and higher_closes:
            patterns.append(CandlePattern(
                name='three_white_soldiers',
                signal='bullish',
                strength=0.9,
                index=len(df) - 1,
                description='Three White Soldiers - Tres fort signal haussier'
            ))

        # Three Black Crows
        all_bearish = all(c['close'] < c['open'] for _, c in candles.iterrows())
        lower_closes = all(
            candles.iloc[i]['close'] < candles.iloc[i-1]['close']
            for i in range(1, 3)
        )

        if all_bearish and lower_closes:
            patterns.append(CandlePattern(
                name='three_black_crows',
                signal='bearish',
                strength=0.9,
                index=len(df) - 1,
                description='Three Black Crows - Tres fort signal baissier'
            ))

        return patterns

    # =========================================================================
    # CHART PATTERNS
    # =========================================================================

    def detect_trend(self, df: pd.DataFrame, period: int = 20) -> str:
        """
        Detecte la tendance generale

        Returns: 'uptrend', 'downtrend', ou 'sideways'
        """
        if len(df) < period:
            return 'sideways'

        closes = df['close'].iloc[-period:]

        # Regression lineaire simple
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        # Normaliser par le prix moyen
        normalized_slope = slope / closes.mean()

        if normalized_slope > 0.001:
            return 'uptrend'
        elif normalized_slope < -0.001:
            return 'downtrend'
        else:
            return 'sideways'

    def detect_accumulation(self, df: pd.DataFrame) -> bool:
        """
        Detecte une accumulation (PARTIE XII - Section 98)

        Caracteristiques:
        - Mouvement impulsif puissant
        - Peu ou pas de retracement
        """
        if len(df) < 10:
            return False

        recent = df.iloc[-10:]
        total_move = abs(recent['close'].iloc[-1] - recent['close'].iloc[0])
        max_range = recent['high'].max() - recent['low'].min()

        # Si le mouvement total est proche du range max, c'est une accumulation
        efficiency = total_move / max_range if max_range > 0 else 0

        return efficiency > 0.7  # Mouvement efficace (peu de retracement)


# =============================================================================
# SINGLETON
# =============================================================================
_pattern_detector = None

def get_pattern_detector() -> PatternDetector:
    """Retourne l'instance singleton"""
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = PatternDetector()
    return _pattern_detector
