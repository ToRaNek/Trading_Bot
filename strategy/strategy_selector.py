"""
Systeme de Selection de Strategies
Permet de choisir et combiner differentes strategies de trading

Strategies disponibles:
1. Swing Trading (base) - MASTER_TRADING_SKILL PARTIE XII
2. Wyckoff Method - MASTER_TRADING_SKILL PARTIE XIII
3. Elliott Wave - MASTER_TRADING_SKILL PARTIE XIV
4. Ichimoku - MASTER_TRADING_SKILL PARTIE XV
5. Volume Profile - MASTER_TRADING_SKILL PARTIE XVI
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types de strategies disponibles"""
    SWING = "swing"
    WYCKOFF = "wyckoff"
    ELLIOTT = "elliott"
    ICHIMOKU = "ichimoku"
    VOLUME_PROFILE = "volume_profile"
    COMBINED = "combined"  # Combinaison de plusieurs strategies


class SignalStrength(Enum):
    """Force du signal"""
    STRONG = 3
    MODERATE = 2
    WEAK = 1
    NEUTRAL = 0


@dataclass
class UnifiedSignal:
    """Signal unifie provenant de n'importe quelle strategie"""
    symbol: str
    direction: str  # 'buy', 'sell', 'neutral'
    strength: SignalStrength
    strategy: StrategyType
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float  # 0.0 - 1.0
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'strength': self.strength.name,
            'strategy': self.strategy.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward': self.risk_reward,
            'confidence': self.confidence,
            'reasons': self.reasons,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class CombinedSignal:
    """Signal combine provenant de plusieurs strategies"""
    symbol: str
    direction: str
    consensus_score: float  # Score de consensus (0-1)
    total_weight: float
    contributing_strategies: List[StrategyType]
    signals: List[UnifiedSignal]
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    final_confidence: float
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'consensus_score': self.consensus_score,
            'total_weight': self.total_weight,
            'contributing_strategies': [s.value for s in self.contributing_strategies],
            'num_signals': len(self.signals),
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward': self.risk_reward,
            'final_confidence': self.final_confidence,
            'reasons': self.reasons,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class StrategyConfig:
    """Configuration pour une strategie"""
    enabled: bool = True
    weight: float = 1.0  # Poids dans le vote combine
    min_confidence: float = 0.6  # Confiance minimum pour signal valide
    timeframes: List[str] = field(default_factory=lambda: ['daily', 'h1'])
    parameters: Dict[str, Any] = field(default_factory=dict)


class StrategySelector:
    """
    Selecteur et combineur de strategies

    Permet de:
    1. Selectionner une strategie unique
    2. Combiner plusieurs strategies avec ponderation
    3. Obtenir un consensus entre strategies
    """

    # Poids par defaut pour chaque strategie
    DEFAULT_WEIGHTS = {
        StrategyType.SWING: 1.0,
        StrategyType.WYCKOFF: 0.9,
        StrategyType.ELLIOTT: 0.7,
        StrategyType.ICHIMOKU: 0.8,
        StrategyType.VOLUME_PROFILE: 0.85
    }

    def __init__(self):
        self.configs: Dict[StrategyType, StrategyConfig] = {}
        self.active_strategy: Optional[StrategyType] = None
        self.combination_mode: bool = True  # Mode combine par defaut

        # Initialiser les configs par defaut
        self._init_default_configs()

        # Lazy loading des analyzers
        self._analyzers = {}

    def _init_default_configs(self):
        """Initialise les configurations par defaut"""
        for strategy_type in StrategyType:
            if strategy_type != StrategyType.COMBINED:
                self.configs[strategy_type] = StrategyConfig(
                    enabled=True,
                    weight=self.DEFAULT_WEIGHTS.get(strategy_type, 1.0)
                )

    def _get_analyzer(self, strategy_type: StrategyType):
        """Charge un analyzer de maniere lazy"""
        if strategy_type not in self._analyzers:
            if strategy_type == StrategyType.WYCKOFF:
                from analysis.wyckoff import get_wyckoff_analyzer
                self._analyzers[strategy_type] = get_wyckoff_analyzer()
            elif strategy_type == StrategyType.ELLIOTT:
                from analysis.elliott_wave import get_elliott_analyzer
                self._analyzers[strategy_type] = get_elliott_analyzer()
            elif strategy_type == StrategyType.ICHIMOKU:
                from analysis.ichimoku import get_ichimoku_analyzer
                self._analyzers[strategy_type] = get_ichimoku_analyzer()
            elif strategy_type == StrategyType.VOLUME_PROFILE:
                from analysis.volume_profile import get_volume_profile_analyzer
                self._analyzers[strategy_type] = get_volume_profile_analyzer()
            elif strategy_type == StrategyType.SWING:
                from strategy.swing_trading import get_swing_strategy
                self._analyzers[strategy_type] = get_swing_strategy()

        return self._analyzers.get(strategy_type)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_active_strategy(self, strategy_type: StrategyType):
        """Definit la strategie active unique"""
        if strategy_type == StrategyType.COMBINED:
            self.combination_mode = True
            self.active_strategy = None
        else:
            self.combination_mode = False
            self.active_strategy = strategy_type
        logger.info(f"Active strategy set to: {strategy_type.value}")

    def enable_strategy(self, strategy_type: StrategyType, enabled: bool = True):
        """Active/desactive une strategie"""
        if strategy_type in self.configs:
            self.configs[strategy_type].enabled = enabled
            logger.info(f"Strategy {strategy_type.value} {'enabled' if enabled else 'disabled'}")

    def set_strategy_weight(self, strategy_type: StrategyType, weight: float):
        """Definit le poids d'une strategie"""
        if strategy_type in self.configs:
            self.configs[strategy_type].weight = max(0.0, min(2.0, weight))
            logger.info(f"Strategy {strategy_type.value} weight set to {weight}")

    def configure_strategy(self, strategy_type: StrategyType, config: StrategyConfig):
        """Configure une strategie complete"""
        self.configs[strategy_type] = config

    def get_enabled_strategies(self) -> List[StrategyType]:
        """Retourne les strategies actives"""
        return [
            st for st, cfg in self.configs.items()
            if cfg.enabled and st != StrategyType.COMBINED
        ]

    # =========================================================================
    # ANALYSE
    # =========================================================================

    def analyze(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None
    ) -> Optional[UnifiedSignal]:
        """
        Analyse un symbole avec la strategie active

        Returns:
            UnifiedSignal si signal valide, None sinon
        """
        if self.combination_mode:
            combined = self.analyze_combined(symbol, df_daily, df_h1)
            if combined and combined.final_confidence >= 0.6:
                return self._combined_to_unified(combined)
            return None

        if self.active_strategy:
            return self._analyze_single(self.active_strategy, symbol, df_daily, df_h1)

        return None

    def analyze_combined(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None,
        min_consensus: float = 0.6
    ) -> Optional[CombinedSignal]:
        """
        Analyse avec toutes les strategies actives et combine les signaux

        Args:
            symbol: Symbole a analyser
            df_daily: Donnees Daily
            df_h1: Donnees H1 (optionnel)
            min_consensus: Score de consensus minimum requis

        Returns:
            CombinedSignal si consensus atteint, None sinon
        """
        enabled_strategies = self.get_enabled_strategies()
        if not enabled_strategies:
            logger.warning("No strategies enabled for combined analysis")
            return None

        signals: List[UnifiedSignal] = []

        # Collecter les signaux de chaque strategie
        for strategy_type in enabled_strategies:
            try:
                signal = self._analyze_single(strategy_type, symbol, df_daily, df_h1)
                if signal and signal.confidence >= self.configs[strategy_type].min_confidence:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing with {strategy_type.value}: {e}")

        if not signals:
            return None

        # Combiner les signaux
        return self._combine_signals(symbol, signals, min_consensus)

    def analyze_all(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None
    ) -> Dict[StrategyType, Optional[UnifiedSignal]]:
        """
        Analyse avec TOUTES les strategies actives et retourne les resultats individuels

        Args:
            symbol: Symbole a analyser
            df_daily: Donnees Daily
            df_h1: Donnees H1 (optionnel)

        Returns:
            Dict avec les signaux de chaque strategie
        """
        enabled_strategies = self.get_enabled_strategies()
        results: Dict[StrategyType, Optional[UnifiedSignal]] = {}

        for strategy_type in enabled_strategies:
            try:
                signal = self._analyze_single(strategy_type, symbol, df_daily, df_h1)
                results[strategy_type] = signal
            except Exception as e:
                logger.error(f"Error analyzing with {strategy_type.value}: {e}")
                results[strategy_type] = None

        return results

    def get_combined_signal(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None,
        min_consensus: float = 0.6
    ) -> Optional[CombinedSignal]:
        """
        Alias pour analyze_combined - pour compatibilite avec le dashboard

        Returns:
            CombinedSignal si consensus atteint, None sinon
        """
        return self.analyze_combined(symbol, df_daily, df_h1, min_consensus)

    def _analyze_single(
        self,
        strategy_type: StrategyType,
        symbol: str,
        df_daily: pd.DataFrame,
        df_h1: Optional[pd.DataFrame]
    ) -> Optional[UnifiedSignal]:
        """Analyse avec une strategie unique"""

        analyzer = self._get_analyzer(strategy_type)
        if not analyzer:
            return None

        try:
            if strategy_type == StrategyType.WYCKOFF:
                return self._analyze_wyckoff(analyzer, symbol, df_daily)

            elif strategy_type == StrategyType.ELLIOTT:
                return self._analyze_elliott(analyzer, symbol, df_daily)

            elif strategy_type == StrategyType.ICHIMOKU:
                return self._analyze_ichimoku(analyzer, symbol, df_daily)

            elif strategy_type == StrategyType.VOLUME_PROFILE:
                return self._analyze_volume_profile(analyzer, symbol, df_daily)

            elif strategy_type == StrategyType.SWING:
                return self._analyze_swing(analyzer, symbol)

        except Exception as e:
            logger.error(f"Error in {strategy_type.value} analysis: {e}")
            return None

        return None

    def _analyze_wyckoff(
        self,
        analyzer,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[UnifiedSignal]:
        """Analyse Wyckoff"""
        result = analyzer.analyze(df)
        signal = result.get('signal') if result else None

        if not signal or signal.signal_type == 'avoid' or signal.signal_type == 'neutral':
            return None

        # Calculer risk_reward si non present
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        return UnifiedSignal(
            symbol=symbol,
            direction=signal.signal_type,  # 'buy' ou 'sell'
            strength=self._map_wyckoff_strength(signal.strength),
            strategy=StrategyType.WYCKOFF,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward=risk_reward,
            confidence=signal.strength,  # strength est 0-1, comme confidence
            reasons=signal.reasons,
            metadata={
                'phase': signal.phase.value if signal.phase else None,
                'event': signal.event.value if signal.event else None,
                'volume_confirmation': signal.volume_confirmation
            }
        )

    def _analyze_elliott(
        self,
        analyzer,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[UnifiedSignal]:
        """Analyse Elliott Wave"""
        result = analyzer.analyze(df)
        signal = result.get('signal') if result else None

        if not signal or signal.signal_type == 'neutral':
            return None

        # Calculer risk_reward
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        return UnifiedSignal(
            symbol=symbol,
            direction=signal.signal_type,  # 'buy' ou 'sell'
            strength=self._map_confidence_to_strength(signal.strength),
            strategy=StrategyType.ELLIOTT,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward=risk_reward,
            confidence=signal.strength,
            reasons=signal.reasons,
            metadata={
                'current_wave': signal.current_wave.value if signal.current_wave else None,
                'trend': signal.trend.value if signal.trend else None,
                'fib_target': signal.fibonacci_target
            }
        )

    def _analyze_ichimoku(
        self,
        analyzer,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[UnifiedSignal]:
        """Analyse Ichimoku"""
        result = analyzer.analyze(df)
        signal = result.get('signal') if result else None

        if not signal or signal.signal_type == 'neutral':
            return None

        # Calculer risk_reward
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        # Mapper la force Ichimoku vers un float pour confidence
        strength_map = {'strong': 0.9, 'medium': 0.7, 'weak': 0.5}
        confidence = strength_map.get(signal.strength.value, 0.6) if hasattr(signal.strength, 'value') else 0.6

        return UnifiedSignal(
            symbol=symbol,
            direction=signal.signal_type,  # 'buy' ou 'sell'
            strength=self._map_ichimoku_strength(signal.strength.value if hasattr(signal.strength, 'value') else 'medium'),
            strategy=StrategyType.ICHIMOKU,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward=risk_reward,
            confidence=confidence,
            reasons=signal.reasons,
            metadata={
                'trend': signal.trend.value if hasattr(signal.trend, 'value') else str(signal.trend),
                'checklist': signal.checklist
            }
        )

    def _analyze_volume_profile(
        self,
        analyzer,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[UnifiedSignal]:
        """Analyse Volume Profile"""
        result = analyzer.analyze(df)
        signal = result.get('signal') if result else None

        if not signal or signal.signal_type == 'neutral':
            return None

        # Calculer risk_reward
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        return UnifiedSignal(
            symbol=symbol,
            direction=signal.signal_type,  # 'buy' ou 'sell'
            strength=self._map_confidence_to_strength(signal.strength),
            strategy=StrategyType.VOLUME_PROFILE,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward=risk_reward,
            confidence=signal.strength,
            reasons=signal.reasons,
            metadata={
                'profile_shape': signal.profile_shape.value if hasattr(signal.profile_shape, 'value') else str(signal.profile_shape),
                'key_level': signal.key_level
            }
        )

    def _analyze_swing(
        self,
        analyzer,
        symbol: str
    ) -> Optional[UnifiedSignal]:
        """Analyse Swing Trading"""
        setup = analyzer.analyze_symbol(symbol)

        if not setup:
            return None

        return UnifiedSignal(
            symbol=symbol,
            direction=setup.direction,
            strength=self._map_confidence_to_strength(setup.signal_strength),
            strategy=StrategyType.SWING,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit_final,
            risk_reward=setup.risk_reward,
            confidence=setup.signal_strength,
            reasons=setup.reasons,
            metadata={
                'tp1': setup.take_profit_1,
                'tp2': setup.take_profit_2,
                'daily_confirmation': setup.daily_confirmation,
                'h1_confirmation': setup.h1_confirmation
            }
        )

    # =========================================================================
    # COMBINAISON DE SIGNAUX
    # =========================================================================

    def _combine_signals(
        self,
        symbol: str,
        signals: List[UnifiedSignal],
        min_consensus: float
    ) -> Optional[CombinedSignal]:
        """Combine plusieurs signaux en un signal consensus"""

        if not signals:
            return None

        # Compter les votes par direction
        buy_votes = 0.0
        sell_votes = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = self.configs[signal.strategy].weight
            if signal.direction == 'buy':
                buy_votes += weight * signal.confidence
            elif signal.direction == 'sell':
                sell_votes += weight * signal.confidence
            total_weight += weight

        if total_weight == 0:
            return None

        # Determiner la direction majoritaire
        buy_score = buy_votes / total_weight
        sell_score = sell_votes / total_weight

        if buy_score > sell_score and buy_score >= min_consensus:
            direction = 'buy'
            consensus_score = buy_score
            relevant_signals = [s for s in signals if s.direction == 'buy']
        elif sell_score > buy_score and sell_score >= min_consensus:
            direction = 'sell'
            consensus_score = sell_score
            relevant_signals = [s for s in signals if s.direction == 'sell']
        else:
            return None  # Pas de consensus

        if not relevant_signals:
            return None

        # Calculer les niveaux combines (moyenne ponderee)
        entry_price = self._weighted_average(relevant_signals, 'entry_price')
        stop_loss = self._weighted_average(relevant_signals, 'stop_loss')
        take_profit = self._weighted_average(relevant_signals, 'take_profit')

        # Risk/Reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        # Confiance finale
        confidences = [s.confidence for s in relevant_signals]
        weights = [self.configs[s.strategy].weight for s in relevant_signals]
        final_confidence = np.average(confidences, weights=weights)

        # Raisons combinees
        all_reasons = []
        for signal in relevant_signals:
            strategy_name = signal.strategy.value.upper()
            for reason in signal.reasons:
                all_reasons.append(f"[{strategy_name}] {reason}")

        return CombinedSignal(
            symbol=symbol,
            direction=direction,
            consensus_score=consensus_score,
            total_weight=total_weight,
            contributing_strategies=[s.strategy for s in relevant_signals],
            signals=relevant_signals,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            final_confidence=final_confidence,
            reasons=all_reasons
        )

    def _weighted_average(self, signals: List[UnifiedSignal], attribute: str) -> float:
        """Calcule la moyenne ponderee d'un attribut"""
        values = [getattr(s, attribute) for s in signals]
        weights = [self.configs[s.strategy].weight for s in signals]
        return np.average(values, weights=weights)

    def _combined_to_unified(self, combined: CombinedSignal) -> UnifiedSignal:
        """Convertit un CombinedSignal en UnifiedSignal"""
        return UnifiedSignal(
            symbol=combined.symbol,
            direction=combined.direction,
            strength=self._map_confidence_to_strength(combined.final_confidence),
            strategy=StrategyType.COMBINED,
            entry_price=combined.entry_price,
            stop_loss=combined.stop_loss,
            take_profit=combined.take_profit,
            risk_reward=combined.risk_reward,
            confidence=combined.final_confidence,
            reasons=combined.reasons,
            metadata={
                'consensus_score': combined.consensus_score,
                'contributing_strategies': [s.value for s in combined.contributing_strategies],
                'num_signals': len(combined.signals)
            }
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _map_wyckoff_strength(self, strength) -> SignalStrength:
        """Mappe la force Wyckoff vers SignalStrength"""
        # Si c'est un float (0-1), utiliser la meme logique que confidence
        if isinstance(strength, (int, float)):
            return self._map_confidence_to_strength(strength)
        # Si c'est une string
        mapping = {
            'strong': SignalStrength.STRONG,
            'moderate': SignalStrength.MODERATE,
            'weak': SignalStrength.WEAK
        }
        return mapping.get(str(strength).lower(), SignalStrength.NEUTRAL)

    def _map_ichimoku_strength(self, strength) -> SignalStrength:
        """Mappe la force Ichimoku vers SignalStrength"""
        # Si c'est un float (0-1), utiliser la meme logique que confidence
        if isinstance(strength, (int, float)):
            return self._map_confidence_to_strength(strength)
        # Si c'est une string
        mapping = {
            'strong': SignalStrength.STRONG,
            'medium': SignalStrength.MODERATE,
            'weak': SignalStrength.WEAK
        }
        return mapping.get(str(strength).lower(), SignalStrength.NEUTRAL)

    def _map_confidence_to_strength(self, confidence: float) -> SignalStrength:
        """Mappe une confiance numerique vers SignalStrength"""
        if confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.6:
            return SignalStrength.MODERATE
        elif confidence >= 0.4:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NEUTRAL

    # =========================================================================
    # SCAN MULTIPLE
    # =========================================================================

    def scan_watchlist(
        self,
        symbols: List[str],
        data_fetcher=None
    ) -> List[UnifiedSignal]:
        """
        Scanne une liste de symboles

        Args:
            symbols: Liste des symboles
            data_fetcher: DataFetcher pour recuperer les donnees

        Returns:
            Liste de signaux tries par confiance
        """
        if not data_fetcher:
            from data.fetcher import get_fetcher
            data_fetcher = get_fetcher()

        signals = []

        for symbol in symbols:
            try:
                df_daily = data_fetcher.get_daily_data(symbol)
                df_h1 = data_fetcher.get_hourly_data(symbol)

                if df_daily is None or len(df_daily) < 50:
                    continue

                signal = self.analyze(symbol, df_daily, df_h1)
                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        # Trier par confiance decroissante
        signals.sort(key=lambda x: x.confidence, reverse=True)

        return signals

    def get_strategy_summary(self) -> Dict:
        """Resume de la configuration actuelle"""
        return {
            'active_strategy': self.active_strategy.value if self.active_strategy else 'combined',
            'combination_mode': self.combination_mode,
            'enabled_strategies': [s.value for s in self.get_enabled_strategies()],
            'configs': {
                st.value: {
                    'enabled': cfg.enabled,
                    'weight': cfg.weight,
                    'min_confidence': cfg.min_confidence
                }
                for st, cfg in self.configs.items()
            }
        }


# =============================================================================
# SINGLETON
# =============================================================================
_selector = None

def get_strategy_selector() -> StrategySelector:
    """Retourne l'instance singleton"""
    global _selector
    if _selector is None:
        _selector = StrategySelector()
    return _selector


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def quick_analyze(
    symbol: str,
    strategy: str = 'combined',
    df_daily: pd.DataFrame = None,
    df_h1: pd.DataFrame = None
) -> Optional[UnifiedSignal]:
    """
    Fonction helper pour analyse rapide

    Args:
        symbol: Symbole a analyser
        strategy: 'swing', 'wyckoff', 'elliott', 'ichimoku', 'volume_profile', 'combined'
        df_daily: Donnees Daily (optionnel, sera fetch si absent)
        df_h1: Donnees H1 (optionnel)

    Returns:
        UnifiedSignal ou None
    """
    selector = get_strategy_selector()

    # Mapper le string vers StrategyType
    strategy_map = {
        'swing': StrategyType.SWING,
        'wyckoff': StrategyType.WYCKOFF,
        'elliott': StrategyType.ELLIOTT,
        'ichimoku': StrategyType.ICHIMOKU,
        'volume_profile': StrategyType.VOLUME_PROFILE,
        'combined': StrategyType.COMBINED
    }

    strategy_type = strategy_map.get(strategy.lower(), StrategyType.COMBINED)
    selector.set_active_strategy(strategy_type)

    # Fetch data si necessaire
    if df_daily is None:
        from data.fetcher import get_fetcher
        fetcher = get_fetcher()
        df_daily = fetcher.get_daily_data(symbol)
        df_h1 = fetcher.get_hourly_data(symbol)

    if df_daily is None:
        return None

    return selector.analyze(symbol, df_daily, df_h1)
