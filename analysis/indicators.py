"""
Indicateurs Techniques
Base sur MASTER_TRADING_SKILL PARTIE V
Utilise pandas-ta pour les calculs
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional, Tuple
import logging

from config.settings import INDICATORS

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calcul des indicateurs techniques pour l'analyse
    Implementer selon le MASTER_TRADING_SKILL
    """

    def __init__(self, config: Dict = None):
        self.config = config or INDICATORS

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute tous les indicateurs techniques au DataFrame
        """
        df = df.copy()

        # Trend Indicators
        df = self.add_moving_averages(df)
        df = self.add_macd(df)

        # Momentum Indicators
        df = self.add_rsi(df)
        df = self.add_stochastic(df)

        # Volatility Indicators
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)

        # Volume Indicators
        df = self.add_volume_indicators(df)

        # Support/Resistance
        df = self.add_pivot_points(df)

        return df

    # =========================================================================
    # TREND INDICATORS
    # =========================================================================

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute SMA et EMA
        MASTER_TRADING_SKILL Section 28-29
        """
        # SMA
        df['sma_20'] = ta.sma(df['close'], length=self.config['sma_short'])
        df['sma_50'] = ta.sma(df['close'], length=self.config['sma_medium'])
        df['sma_200'] = ta.sma(df['close'], length=self.config['sma_long'])

        # EMA
        df['ema_9'] = ta.ema(df['close'], length=self.config['ema_short'])
        df['ema_21'] = ta.ema(df['close'], length=self.config['ema_medium'])

        # Trend direction based on MAs (avec gestion des NaN)
        sma50_filled = df['sma_50'].fillna(df['close'])
        df['trend_sma'] = np.where(
            df['close'] > sma50_filled, 1,
            np.where(df['close'] < sma50_filled, -1, 0)
        )

        ema9_filled = df['ema_9'].fillna(df['close'])
        ema21_filled = df['ema_21'].fillna(df['close'])
        df['trend_ema'] = np.where(
            ema9_filled > ema21_filled, 1,
            np.where(ema9_filled < ema21_filled, -1, 0)
        )

        # Golden Cross / Death Cross (avec gestion des NaN)
        df['ma_cross'] = 0  # Default
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            sma50 = df['sma_50'].fillna(0)
            sma200 = df['sma_200'].fillna(0)
            sma50_prev = sma50.shift(1).fillna(0)
            sma200_prev = sma200.shift(1).fillna(0)

            golden = (sma50 > sma200) & (sma50_prev <= sma200_prev) & (sma200 > 0)
            death = (sma50 < sma200) & (sma50_prev >= sma200_prev) & (sma200 > 0)

            df.loc[golden, 'ma_cross'] = 1
            df.loc[death, 'ma_cross'] = -1

        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute MACD
        MASTER_TRADING_SKILL Section 30
        """
        macd = ta.macd(
            df['close'],
            fast=self.config['macd_fast'],
            slow=self.config['macd_slow'],
            signal=self.config['macd_signal']
        )

        if macd is not None:
            df['macd'] = macd.iloc[:, 0]  # MACD line
            df['macd_signal'] = macd.iloc[:, 2]  # Signal line
            df['macd_hist'] = macd.iloc[:, 1]  # Histogram

            # MACD Cross signals
            df['macd_cross'] = np.where(
                (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
                np.where(
                    (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1,
                    0
                )
            )

            # MACD Divergence
            df['macd_divergence'] = self._detect_divergence(df, 'close', 'macd')

        return df

    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute RSI
        MASTER_TRADING_SKILL Section 31
        """
        df['rsi'] = ta.rsi(df['close'], length=self.config['rsi_period'])

        # RSI Zones
        df['rsi_overbought'] = df['rsi'] > self.config['rsi_overbought']
        df['rsi_oversold'] = df['rsi'] < self.config['rsi_oversold']

        # RSI Signal
        df['rsi_signal'] = np.where(
            df['rsi'] < self.config['rsi_oversold'], 1,  # Buy signal
            np.where(df['rsi'] > self.config['rsi_overbought'], -1, 0)  # Sell signal
        )

        # RSI Divergence
        df['rsi_divergence'] = self._detect_divergence(df, 'close', 'rsi')

        return df

    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute Stochastic Oscillator
        MASTER_TRADING_SKILL Section 32
        """
        stoch = ta.stoch(df['high'], df['low'], df['close'])

        if stoch is not None:
            df['stoch_k'] = stoch.iloc[:, 0]  # %K
            df['stoch_d'] = stoch.iloc[:, 1]  # %D

            # Stochastic signal
            df['stoch_signal'] = np.where(
                (df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d']), 1,
                np.where(
                    (df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d']), -1,
                    0
                )
            )

        return df

    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute Bollinger Bands
        MASTER_TRADING_SKILL Section 33
        """
        bb = ta.bbands(
            df['close'],
            length=self.config['bb_period'],
            std=self.config['bb_std']
        )

        if bb is not None:
            df['bb_upper'] = bb.iloc[:, 2]  # Upper band
            df['bb_middle'] = bb.iloc[:, 1]  # Middle band (SMA)
            df['bb_lower'] = bb.iloc[:, 0]  # Lower band

            # BB Width (volatility measure)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # BB Position (-1 to 1)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # BB Signal
            df['bb_signal'] = np.where(
                df['close'] <= df['bb_lower'], 1,  # Buy at lower band
                np.where(df['close'] >= df['bb_upper'], -1, 0)  # Sell at upper band
            )

        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute ATR (Average True Range)
        MASTER_TRADING_SKILL Section 35 - Utilise pour Stop Loss
        """
        df['atr'] = ta.atr(
            df['high'],
            df['low'],
            df['close'],
            length=self.config['atr_period']
        )

        # ATR en pourcentage du prix
        df['atr_percent'] = (df['atr'] / df['close']) * 100

        # Stop loss suggere (prix - ATR * multiplier)
        df['atr_stop_long'] = df['close'] - (df['atr'] * self.config['atr_multiplier'])
        df['atr_stop_short'] = df['close'] + (df['atr'] * self.config['atr_multiplier'])

        return df

    # =========================================================================
    # VOLUME INDICATORS
    # =========================================================================

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les indicateurs de volume
        MASTER_TRADING_SKILL Section 26
        """
        # Volume SMA
        df['volume_sma'] = ta.sma(df['volume'], length=self.config['volume_sma'])

        # Volume ratio (current vs average)
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # High volume (>1.5x average)
        df['high_volume'] = df['volume_ratio'] > 1.5

        # OBV (On-Balance Volume)
        df['obv'] = ta.obv(df['close'], df['volume'])

        # Volume Price Trend
        df['vpt'] = ta.pvt(df['close'], df['volume'])

        return df

    # =========================================================================
    # SUPPORT / RESISTANCE
    # =========================================================================

    def add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les Pivot Points
        MASTER_TRADING_SKILL Section 25
        """
        # Standard Pivot Points (pour chaque jour)
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))

        return df

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _detect_divergence(
        self,
        df: pd.DataFrame,
        price_col: str,
        indicator_col: str,
        lookback: int = 14
    ) -> pd.Series:
        """
        Detecte les divergences prix/indicateur
        Retourne: 1 = bullish divergence, -1 = bearish divergence, 0 = none
        """
        divergence = pd.Series(0, index=df.index)

        for i in range(lookback, len(df)):
            try:
                # Prix et indicateur sur la periode
                price_window = df[price_col].iloc[i-lookback:i+1].values
                ind_window = df[indicator_col].iloc[i-lookback:i+1].values

                # Bullish: Prix fait lower low, indicateur fait higher low
                price_ll = price_window[-1] < price_window[:-1].min()
                ind_min_idx = ind_window[:-1].argmin()
                ind_hl = ind_window[-1] > ind_window[ind_min_idx]

                # Bearish: Prix fait higher high, indicateur fait lower high
                price_hh = price_window[-1] > price_window[:-1].max()
                ind_max_idx = ind_window[:-1].argmax()
                ind_lh = ind_window[-1] < ind_window[ind_max_idx]

                if price_ll and ind_hl:
                    divergence.iloc[i] = 1  # Bullish
                elif price_hh and ind_lh:
                    divergence.iloc[i] = -1  # Bearish
            except Exception:
                continue

        return divergence

    def get_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calcule la force de la tendance (-1 a 1)
        """
        if df.empty or len(df) < 2:
            return 0

        latest = df.iloc[-1]
        score = 0
        count = 0

        # MA alignment
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                score += 1
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                score -= 1
            count += 1

        # RSI
        if 'rsi' in df.columns:
            if latest['rsi'] > 50:
                score += (latest['rsi'] - 50) / 50
            else:
                score -= (50 - latest['rsi']) / 50
            count += 1

        # MACD
        if 'macd_hist' in df.columns:
            if latest['macd_hist'] > 0:
                score += 0.5
            else:
                score -= 0.5
            count += 1

        return score / count if count > 0 else 0

    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        Resume tous les signaux des indicateurs
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        return {
            'rsi': latest.get('rsi'),
            'rsi_signal': latest.get('rsi_signal'),
            'macd': latest.get('macd'),
            'macd_signal_line': latest.get('macd_signal'),
            'macd_cross': latest.get('macd_cross'),
            'bb_position': latest.get('bb_position'),
            'bb_signal': latest.get('bb_signal'),
            'trend_sma': latest.get('trend_sma'),
            'trend_ema': latest.get('trend_ema'),
            'volume_ratio': latest.get('volume_ratio'),
            'atr': latest.get('atr'),
            'trend_strength': self.get_trend_strength(df)
        }


# =============================================================================
# SINGLETON
# =============================================================================
_indicators_instance = None

def get_indicators() -> TechnicalIndicators:
    """Retourne l'instance singleton"""
    global _indicators_instance
    if _indicators_instance is None:
        _indicators_instance = TechnicalIndicators()
    return _indicators_instance
