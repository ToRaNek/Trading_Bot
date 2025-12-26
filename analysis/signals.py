"""
Generation de Signaux de Trading
Base sur MASTER_TRADING_SKILL - Strategie Swing Trading Hybride (PARTIE XII)

Combine:
- Indicateurs techniques
- Zones haute probabilite
- Patterns de prix
- Confirmation multi-timeframe
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from config.settings import SIGNAL_CONFIG, MIN_RISK_REWARD, INDICATORS
from analysis.indicators import TechnicalIndicators, get_indicators
from analysis.zones import ZoneDetector, Zone, find_swing_points, detect_breakout, get_zone_detector
from analysis.patterns import PatternDetector, get_pattern_detector

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Represente un signal de trading"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'close_long', 'close_short'
    strength: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Details
    timeframe: str = "daily"
    zone: Optional[Zone] = None
    patterns: List[str] = field(default_factory=list)
    indicators: Dict = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Verifie si le signal est valide"""
        return (
            self.strength >= SIGNAL_CONFIG.get('min_signal_strength', 0.7) and
            self.risk_reward >= MIN_RISK_REWARD
        )

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward': self.risk_reward,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'patterns': self.patterns,
            'reasons': self.reasons
        }


class SignalGenerator:
    """
    Genere les signaux de trading selon le MASTER_TRADING_SKILL

    Strategie Swing Trading Hybride (PARTIE XII):
    1. Daily: Determiner direction + zones
    2. H1: Precision entree
    3. Confirmation indicateurs
    """

    def __init__(self):
        self.indicators = get_indicators()
        self.zones = get_zone_detector()
        self.patterns = get_pattern_detector()
        self.config = SIGNAL_CONFIG

    def generate_signals(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_hourly: pd.DataFrame = None
    ) -> List[Signal]:
        """
        Genere les signaux pour un symbole

        Args:
            symbol: Symbole de l'action
            df_daily: DataFrame Daily avec OHLCV
            df_hourly: DataFrame H1 (optionnel pour precision)

        Returns:
            Liste de signaux
        """
        signals = []

        if df_daily.empty or len(df_daily) < 20:
            return signals

        # 1. Ajouter indicateurs
        df_daily = self.indicators.add_all_indicators(df_daily)

        # 2. Trouver swing points et detecter breakout
        swing_highs, swing_lows = find_swing_points(df_daily)
        breakout = detect_breakout(df_daily, swing_highs, swing_lows)

        # 3. Trouver zones haute probabilite
        zones = self.zones.find_zones(df_daily)

        # 4. Detecter patterns
        candle_patterns = self.patterns.detect_all_patterns(df_daily)

        # 5. Analyser pour signal
        current_price = df_daily['close'].iloc[-1]

        # Signal d'achat
        buy_signal = self._check_buy_signal(
            symbol, df_daily, breakout, zones, candle_patterns, current_price
        )
        if buy_signal:
            signals.append(buy_signal)

        # Signal de vente
        sell_signal = self._check_sell_signal(
            symbol, df_daily, breakout, zones, candle_patterns, current_price
        )
        if sell_signal:
            signals.append(sell_signal)

        # 6. Affiner avec H1 si disponible
        if df_hourly is not None and not df_hourly.empty:
            signals = self._refine_with_hourly(signals, df_hourly)

        return signals

    def _check_buy_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        breakout: Optional[Dict],
        zones: List[Zone],
        patterns: List,
        current_price: float
    ) -> Optional[Signal]:
        """
        Verifie les conditions pour un signal d'achat

        Conditions (PARTIE XII):
        1. Cassure point haut recent (breakout bullish)
        2. Retracement vers zone haute probabilite
        3. Bougie de rejection haussiere
        4. Indicateurs favorables (RSI pas surchat, etc.)
        """
        reasons = []
        score = 0

        # 1. Breakout haussier
        if breakout and breakout['type'] == 'bullish':
            score += 0.25
            reasons.append(f"Cassure haussiere @ {breakout['breakout_price']:.2f}")
        else:
            # Pas de breakout = pas de signal
            return None

        # 2. Prix dans une zone (retracement)
        zone_in = self.zones.is_price_in_zone(current_price, zones)
        if zone_in and zone_in.zone_type in ['support', 'both']:
            score += 0.25
            reasons.append(f"Prix dans zone support ({zone_in.midpoint:.2f})")
        else:
            # Chercher zone proche en dessous
            nearest_zone = self.zones.get_nearest_zone(zones, current_price, 'below')
            if nearest_zone:
                distance = (current_price - nearest_zone.midpoint) / current_price
                if distance < 0.03:  # A moins de 3% de la zone
                    score += 0.15
                    reasons.append(f"Proche zone support ({nearest_zone.midpoint:.2f})")

        # 3. Pattern de bougie haussier
        bullish_patterns = [p for p in patterns if p.signal == 'bullish']
        if bullish_patterns:
            best_pattern = max(bullish_patterns, key=lambda x: x.strength)
            score += best_pattern.strength * 0.2
            reasons.append(f"Pattern: {best_pattern.name}")

        # 4. Rejection candle
        if self.patterns.detect_rejection_candle(df, 'buy'):
            score += 0.1
            reasons.append("Bougie de rejection haussiere")

        # 5. Indicateurs
        latest = df.iloc[-1]
        indicator_score = 0
        indicator_count = 0

        # RSI pas en surachat
        if 'rsi' in latest and not pd.isna(latest['rsi']):
            if latest['rsi'] < INDICATORS['rsi_overbought']:
                indicator_score += 1
                if latest['rsi'] < 50:
                    reasons.append(f"RSI favorable ({latest['rsi']:.1f})")
            indicator_count += 1

        # Trend EMA
        if 'trend_ema' in latest and latest['trend_ema'] == 1:
            indicator_score += 1
            reasons.append("EMA en tendance haussiere")
            indicator_count += 1

        # MACD
        if 'macd_hist' in latest and latest['macd_hist'] > 0:
            indicator_score += 0.5
            indicator_count += 1

        # Volume
        if 'volume_ratio' in latest and latest['volume_ratio'] > 1:
            indicator_score += 0.5
            reasons.append("Volume eleve")
            indicator_count += 1

        if indicator_count > 0:
            score += (indicator_score / indicator_count) * 0.2

        # Calculer SL et TP
        atr = latest.get('atr', current_price * 0.02)
        stop_loss = current_price - (atr * INDICATORS['atr_multiplier'])

        # TP base sur zone de resistance au-dessus ou ratio R:R
        nearest_resistance = self.zones.get_nearest_zone(zones, current_price, 'above')
        if nearest_resistance:
            take_profit = nearest_resistance.midpoint
        else:
            # TP = 3x le risque minimum
            risk = current_price - stop_loss
            take_profit = current_price + (risk * MIN_RISK_REWARD)

        # Calculer R:R
        risk = current_price - stop_loss
        reward = take_profit - current_price
        risk_reward = reward / risk if risk > 0 else 0

        # Verifier R:R minimum
        if risk_reward < MIN_RISK_REWARD:
            return None

        if score < self.config.get('min_signal_strength', 0.7):
            return None

        return Signal(
            symbol=symbol,
            signal_type='buy',
            strength=min(score, 1.0),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            zone=zone_in,
            patterns=[p.name for p in bullish_patterns],
            indicators={
                'rsi': latest.get('rsi'),
                'macd_hist': latest.get('macd_hist'),
                'trend': latest.get('trend_ema')
            },
            reasons=reasons
        )

    def _check_sell_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        breakout: Optional[Dict],
        zones: List[Zone],
        patterns: List,
        current_price: float
    ) -> Optional[Signal]:
        """
        Verifie les conditions pour un signal de vente (short)

        Conditions:
        1. Cassure point bas recent (breakout bearish)
        2. Retracement vers zone resistance
        3. Bougie de rejection baissiere
        4. Indicateurs favorables
        """
        reasons = []
        score = 0

        # 1. Breakout baissier
        if breakout and breakout['type'] == 'bearish':
            score += 0.25
            reasons.append(f"Cassure baissiere @ {breakout['breakout_price']:.2f}")
        else:
            return None

        # 2. Prix dans une zone resistance
        zone_in = self.zones.is_price_in_zone(current_price, zones)
        if zone_in and zone_in.zone_type in ['resistance', 'both']:
            score += 0.25
            reasons.append(f"Prix dans zone resistance ({zone_in.midpoint:.2f})")
        else:
            nearest_zone = self.zones.get_nearest_zone(zones, current_price, 'above')
            if nearest_zone:
                distance = (nearest_zone.midpoint - current_price) / current_price
                if distance < 0.03:
                    score += 0.15
                    reasons.append(f"Proche zone resistance ({nearest_zone.midpoint:.2f})")

        # 3. Pattern baissier
        bearish_patterns = [p for p in patterns if p.signal == 'bearish']
        if bearish_patterns:
            best_pattern = max(bearish_patterns, key=lambda x: x.strength)
            score += best_pattern.strength * 0.2
            reasons.append(f"Pattern: {best_pattern.name}")

        # 4. Rejection candle
        if self.patterns.detect_rejection_candle(df, 'sell'):
            score += 0.1
            reasons.append("Bougie de rejection baissiere")

        # 5. Indicateurs
        latest = df.iloc[-1]
        indicator_score = 0
        indicator_count = 0

        if 'rsi' in latest and not pd.isna(latest['rsi']):
            if latest['rsi'] > INDICATORS['rsi_oversold']:
                indicator_score += 1
                if latest['rsi'] > 50:
                    reasons.append(f"RSI favorable ({latest['rsi']:.1f})")
            indicator_count += 1

        if 'trend_ema' in latest and latest['trend_ema'] == -1:
            indicator_score += 1
            reasons.append("EMA en tendance baissiere")
            indicator_count += 1

        if 'macd_hist' in latest and latest['macd_hist'] < 0:
            indicator_score += 0.5
            indicator_count += 1

        if 'volume_ratio' in latest and latest['volume_ratio'] > 1:
            indicator_score += 0.5
            reasons.append("Volume eleve")
            indicator_count += 1

        if indicator_count > 0:
            score += (indicator_score / indicator_count) * 0.2

        # SL et TP
        atr = latest.get('atr', current_price * 0.02)
        stop_loss = current_price + (atr * INDICATORS['atr_multiplier'])

        nearest_support = self.zones.get_nearest_zone(zones, current_price, 'below')
        if nearest_support:
            take_profit = nearest_support.midpoint
        else:
            risk = stop_loss - current_price
            take_profit = current_price - (risk * MIN_RISK_REWARD)

        risk = stop_loss - current_price
        reward = current_price - take_profit
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < MIN_RISK_REWARD:
            return None

        if score < self.config.get('min_signal_strength', 0.7):
            return None

        return Signal(
            symbol=symbol,
            signal_type='sell',
            strength=min(score, 1.0),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            zone=zone_in,
            patterns=[p.name for p in bearish_patterns],
            indicators={
                'rsi': latest.get('rsi'),
                'macd_hist': latest.get('macd_hist'),
                'trend': latest.get('trend_ema')
            },
            reasons=reasons
        )

    def _refine_with_hourly(
        self,
        signals: List[Signal],
        df_hourly: pd.DataFrame
    ) -> List[Signal]:
        """
        Affine les signaux avec les donnees H1 (PARTIE XII - Section 97)

        Processus:
        1. Confirmer cassure sur H1
        2. Trouver zones H1
        3. Attendre retracement + rejection H1
        """
        if not signals or df_hourly.empty:
            return signals

        refined = []
        df_hourly = self.indicators.add_all_indicators(df_hourly)

        for signal in signals:
            # Verifier confirmation H1
            swing_highs, swing_lows = find_swing_points(df_hourly, lookback=3)
            h1_breakout = detect_breakout(df_hourly, swing_highs, swing_lows)

            confirmation = False

            if signal.signal_type == 'buy' and h1_breakout:
                if h1_breakout['type'] == 'bullish':
                    signal.strength = min(signal.strength + 0.1, 1.0)
                    signal.reasons.append("Confirme sur H1")
                    confirmation = True

            elif signal.signal_type == 'sell' and h1_breakout:
                if h1_breakout['type'] == 'bearish':
                    signal.strength = min(signal.strength + 0.1, 1.0)
                    signal.reasons.append("Confirme sur H1")
                    confirmation = True

            # Affiner entry avec H1
            if confirmation:
                h1_current = df_hourly['close'].iloc[-1]
                # Ajuster entry si H1 donne meilleur prix
                if signal.signal_type == 'buy' and h1_current < signal.entry_price:
                    signal.entry_price = h1_current
                elif signal.signal_type == 'sell' and h1_current > signal.entry_price:
                    signal.entry_price = h1_current

            refined.append(signal)

        return refined

    def check_exit_signals(
        self,
        symbol: str,
        position_side: str,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        take_profit: float,
        df: pd.DataFrame
    ) -> Optional[Signal]:
        """
        Verifie les signaux de sortie pour une position existante

        Args:
            position_side: 'long' ou 'short'
            entry_price: Prix d'entree
            current_price: Prix actuel
            stop_loss: SL defini
            take_profit: TP defini
            df: DataFrame avec prix actuels
        """
        # 1. Stop Loss touche
        if position_side == 'long' and current_price <= stop_loss:
            return Signal(
                symbol=symbol,
                signal_type='close_long',
                strength=1.0,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=0,
                reasons=['Stop Loss touche']
            )

        if position_side == 'short' and current_price >= stop_loss:
            return Signal(
                symbol=symbol,
                signal_type='close_short',
                strength=1.0,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=0,
                reasons=['Stop Loss touche']
            )

        # 2. Take Profit touche
        if position_side == 'long' and current_price >= take_profit:
            return Signal(
                symbol=symbol,
                signal_type='close_long',
                strength=1.0,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=0,
                reasons=['Take Profit atteint']
            )

        if position_side == 'short' and current_price <= take_profit:
            return Signal(
                symbol=symbol,
                signal_type='close_short',
                strength=1.0,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=0,
                reasons=['Take Profit atteint']
            )

        # 3. Signal inverse fort
        df = self.indicators.add_all_indicators(df)
        patterns = self.patterns.detect_all_patterns(df)

        if position_side == 'long':
            bearish_patterns = [p for p in patterns if p.signal == 'bearish' and p.strength > 0.8]
            if bearish_patterns:
                return Signal(
                    symbol=symbol,
                    signal_type='close_long',
                    strength=0.8,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=0,
                    patterns=[p.name for p in bearish_patterns],
                    reasons=['Signal baissier fort detecte']
                )

        if position_side == 'short':
            bullish_patterns = [p for p in patterns if p.signal == 'bullish' and p.strength > 0.8]
            if bullish_patterns:
                return Signal(
                    symbol=symbol,
                    signal_type='close_short',
                    strength=0.8,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=0,
                    patterns=[p.name for p in bullish_patterns],
                    reasons=['Signal haussier fort detecte']
                )

        return None


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def scan_for_signals(
    symbols: List[str],
    fetcher,
    min_strength: float = 0.7
) -> List[Signal]:
    """
    Scanne plusieurs symboles pour trouver des signaux

    Args:
        symbols: Liste de symboles a scanner
        fetcher: DataFetcher instance
        min_strength: Force minimum du signal

    Returns:
        Liste de signaux tries par force
    """
    generator = SignalGenerator()
    all_signals = []

    for symbol in symbols:
        try:
            df_daily = fetcher.get_daily_data(symbol)
            df_hourly = fetcher.get_hourly_data(symbol)

            if df_daily is not None:
                signals = generator.generate_signals(symbol, df_daily, df_hourly)
                valid_signals = [s for s in signals if s.strength >= min_strength]
                all_signals.extend(valid_signals)

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")

    # Trier par force
    all_signals.sort(key=lambda x: x.strength, reverse=True)
    return all_signals


# =============================================================================
# SINGLETON
# =============================================================================
_signal_generator = None

def get_signal_generator() -> SignalGenerator:
    """Retourne l'instance singleton"""
    global _signal_generator
    if _signal_generator is None:
        _signal_generator = SignalGenerator()
    return _signal_generator
