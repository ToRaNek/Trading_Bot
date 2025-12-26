"""
Analyse Elliott Wave
Base sur MASTER_TRADING_SKILL PARTIE XIV - Theorie des Vagues d'Elliott

Implementation des concepts:
- Structure de base 5-3 (5 vagues impulsives + 3 correctives)
- Les 3 regles inviolables
- Ratios Fibonacci (retracements et extensions)
- Patterns correctifs (Zigzag, Flat, Triangle)
- Signaux de trading bases sur le comptage des vagues
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WaveType(Enum):
    """Types de vagues Elliott"""
    IMPULSE_1 = "wave_1"
    IMPULSE_2 = "wave_2"  # Corrective
    IMPULSE_3 = "wave_3"
    IMPULSE_4 = "wave_4"  # Corrective
    IMPULSE_5 = "wave_5"
    CORRECTIVE_A = "wave_a"
    CORRECTIVE_B = "wave_b"
    CORRECTIVE_C = "wave_c"
    UNKNOWN = "unknown"


class TrendDirection(Enum):
    """Direction de la tendance"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    UNKNOWN = "unknown"


class CorrectivePattern(Enum):
    """Types de patterns correctifs"""
    ZIGZAG = "zigzag"      # 5-3-5
    FLAT = "flat"          # 3-3-5
    TRIANGLE = "triangle"  # 3-3-3-3-3
    COMBINATION = "combination"  # W-X-Y
    UNKNOWN = "unknown"


# Ratios Fibonacci
FIBONACCI_RATIOS = {
    'retracements': [0.236, 0.382, 0.500, 0.618, 0.786],
    'extensions': [1.000, 1.272, 1.618, 2.000, 2.618],
    # Ratios specifiques aux vagues
    'wave2': [0.382, 0.500, 0.618],
    'wave3': [1.618, 2.618],
    'wave4': [0.236, 0.382],
    'wave5': [0.618, 1.000, 1.618],
    'waveA': [0.382, 0.500, 0.618],
    'waveB': [0.382, 0.500, 0.618],
    'waveC': [1.000, 1.618],
}


@dataclass
class Wave:
    """Represente une vague Elliott"""
    wave_type: WaveType
    start_price: float
    end_price: float
    start_index: int
    end_index: int
    direction: TrendDirection
    retracement: Optional[float] = None  # % de retracement
    extension: Optional[float] = None    # % d'extension

    @property
    def length(self) -> float:
        return abs(self.end_price - self.start_price)

    @property
    def is_impulse(self) -> bool:
        return self.wave_type in [
            WaveType.IMPULSE_1, WaveType.IMPULSE_3, WaveType.IMPULSE_5
        ]


@dataclass
class WaveCount:
    """Comptage complet des vagues"""
    waves: List[Wave] = field(default_factory=list)
    trend: TrendDirection = TrendDirection.UNKNOWN
    current_wave: WaveType = WaveType.UNKNOWN
    corrective_pattern: CorrectivePattern = CorrectivePattern.UNKNOWN
    confidence: float = 0.0
    is_valid: bool = False
    violations: List[str] = field(default_factory=list)

    def add_wave(self, wave: Wave):
        self.waves.append(wave)

    def get_wave(self, wave_type: WaveType) -> Optional[Wave]:
        for wave in self.waves:
            if wave.wave_type == wave_type:
                return wave
        return None


@dataclass
class ElliottSignal:
    """Signal de trading Elliott Wave"""
    symbol: str
    signal_type: str  # 'buy', 'sell'
    current_wave: WaveType
    trend: TrendDirection
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    fibonacci_target: float
    timestamp: datetime = field(default_factory=datetime.now)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'current_wave': self.current_wave.value,
            'trend': self.trend.value,
            'strength': self.strength,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'fibonacci_target': self.fibonacci_target,
            'timestamp': self.timestamp.isoformat(),
            'reasons': self.reasons
        }


class ElliottWaveAnalyzer:
    """
    Analyseur Elliott Wave selon MASTER_TRADING_SKILL

    Les 3 Regles Inviolables:
    1. La vague 2 ne retrace JAMAIS plus de 100% de la vague 1
    2. La vague 3 n'est JAMAIS la plus courte des vagues impulsives
    3. La vague 4 ne penetre JAMAIS le territoire de la vague 1
    """

    def __init__(self, min_wave_size: float = 0.01):
        self.min_wave_size = min_wave_size  # 1% minimum

    # =========================================================================
    # ANALYSE PRINCIPALE
    # =========================================================================

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Analyse complete Elliott Wave

        Returns:
            Dict avec comptage, phase actuelle et signal
        """
        if len(df) < 50:
            return {'wave_count': None, 'signal': None}

        # 1. Trouver les pivots (swing highs/lows)
        pivots = self._find_pivots(df)

        if len(pivots) < 5:
            return {'wave_count': None, 'signal': None}

        # 2. Tenter le comptage des vagues
        wave_count = self._count_waves(df, pivots)

        # 3. Valider le comptage (les 3 regles)
        wave_count = self._validate_wave_count(wave_count)

        # 4. Calculer les niveaux Fibonacci
        fib_levels = self._calculate_fibonacci_levels(wave_count, df)

        # 5. Generer signal si setup valide
        signal = self._generate_signal(
            symbol=symbol,
            df=df,
            wave_count=wave_count,
            fib_levels=fib_levels
        )

        return {
            'wave_count': wave_count,
            'fibonacci_levels': fib_levels,
            'signal': signal
        }

    # =========================================================================
    # DETECTION DES PIVOTS
    # =========================================================================

    def _find_pivots(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> List[Dict]:
        """
        Trouve les points pivots (swing highs et lows)

        Un pivot est un point ou le prix change de direction
        """
        pivots = []
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        for i in range(lookback, len(df) - lookback):
            # Swing High
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                pivots.append({
                    'type': 'high',
                    'price': highs[i],
                    'index': i,
                    'close': closes[i]
                })

            # Swing Low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                pivots.append({
                    'type': 'low',
                    'price': lows[i],
                    'index': i,
                    'close': closes[i]
                })

        # Trier par index
        pivots.sort(key=lambda x: x['index'])

        # Filtrer les pivots trop proches
        pivots = self._filter_close_pivots(pivots)

        return pivots

    def _filter_close_pivots(
        self,
        pivots: List[Dict],
        min_bars: int = 3
    ) -> List[Dict]:
        """Filtre les pivots trop proches"""
        if len(pivots) < 2:
            return pivots

        filtered = [pivots[0]]
        for pivot in pivots[1:]:
            if pivot['index'] - filtered[-1]['index'] >= min_bars:
                # Eviter deux pivots du meme type consecutifs
                if pivot['type'] != filtered[-1]['type']:
                    filtered.append(pivot)
                elif pivot['type'] == 'high' and pivot['price'] > filtered[-1]['price']:
                    filtered[-1] = pivot
                elif pivot['type'] == 'low' and pivot['price'] < filtered[-1]['price']:
                    filtered[-1] = pivot

        return filtered

    # =========================================================================
    # COMPTAGE DES VAGUES
    # =========================================================================

    def _count_waves(
        self,
        df: pd.DataFrame,
        pivots: List[Dict]
    ) -> WaveCount:
        """
        Compte les vagues Elliott a partir des pivots

        Structure: 5 vagues impulsives + 3 correctives
        """
        wave_count = WaveCount()

        if len(pivots) < 5:
            return wave_count

        # Determiner la direction de la tendance
        first_pivot = pivots[0]
        last_pivot = pivots[-1]

        if last_pivot['price'] > first_pivot['price']:
            if first_pivot['type'] == 'low':
                wave_count.trend = TrendDirection.BULLISH
            else:
                wave_count.trend = TrendDirection.BEARISH
        else:
            if first_pivot['type'] == 'high':
                wave_count.trend = TrendDirection.BEARISH
            else:
                wave_count.trend = TrendDirection.BULLISH

        # Assigner les vagues selon la structure 5-3
        wave_types = [
            WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3,
            WaveType.IMPULSE_4, WaveType.IMPULSE_5,
            WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C
        ]

        for i in range(min(len(pivots) - 1, len(wave_types))):
            start = pivots[i]
            end = pivots[i + 1]

            direction = TrendDirection.BULLISH if end['price'] > start['price'] else TrendDirection.BEARISH

            wave = Wave(
                wave_type=wave_types[i],
                start_price=start['price'],
                end_price=end['price'],
                start_index=start['index'],
                end_index=end['index'],
                direction=direction
            )

            # Calculer retracement si c'est une vague corrective
            if wave_types[i] in [WaveType.IMPULSE_2, WaveType.IMPULSE_4]:
                prev_wave = wave_count.waves[-1] if wave_count.waves else None
                if prev_wave:
                    retracement = wave.length / prev_wave.length
                    wave.retracement = retracement

            wave_count.add_wave(wave)

        # Determiner la vague actuelle
        if len(wave_count.waves) > 0:
            last_wave = wave_count.waves[-1]
            wave_count.current_wave = last_wave.wave_type

            # La prochaine vague probable
            if last_wave.wave_type == WaveType.IMPULSE_5:
                wave_count.current_wave = WaveType.CORRECTIVE_A
            elif last_wave.wave_type == WaveType.CORRECTIVE_C:
                wave_count.current_wave = WaveType.IMPULSE_1

        return wave_count

    # =========================================================================
    # VALIDATION (LES 3 REGLES)
    # =========================================================================

    def _validate_wave_count(self, wave_count: WaveCount) -> WaveCount:
        """
        Valide le comptage selon les 3 regles inviolables

        1. Vague 2 ne retrace pas plus de 100% de vague 1
        2. Vague 3 n'est pas la plus courte
        3. Vague 4 ne penetre pas le territoire de vague 1
        """
        wave_count.is_valid = True
        wave_count.violations = []
        confidence = 1.0

        wave1 = wave_count.get_wave(WaveType.IMPULSE_1)
        wave2 = wave_count.get_wave(WaveType.IMPULSE_2)
        wave3 = wave_count.get_wave(WaveType.IMPULSE_3)
        wave4 = wave_count.get_wave(WaveType.IMPULSE_4)
        wave5 = wave_count.get_wave(WaveType.IMPULSE_5)

        # Regle 1: Vague 2 < 100% de Vague 1
        if wave1 and wave2:
            if wave2.length > wave1.length:
                wave_count.is_valid = False
                wave_count.violations.append("Vague 2 retrace plus de 100% de vague 1")
                confidence -= 0.4

        # Regle 2: Vague 3 n'est pas la plus courte
        if wave1 and wave3 and wave5:
            lengths = [wave1.length, wave3.length, wave5.length]
            if wave3.length == min(lengths):
                wave_count.is_valid = False
                wave_count.violations.append("Vague 3 est la plus courte")
                confidence -= 0.4

        # Regle 3: Vague 4 ne penetre pas vague 1
        if wave1 and wave4:
            if wave_count.trend == TrendDirection.BULLISH:
                if wave4.end_price < wave1.end_price:
                    wave_count.is_valid = False
                    wave_count.violations.append("Vague 4 penetre le territoire de vague 1")
                    confidence -= 0.4
            else:
                if wave4.end_price > wave1.end_price:
                    wave_count.is_valid = False
                    wave_count.violations.append("Vague 4 penetre le territoire de vague 1")
                    confidence -= 0.4

        # Guidelines (moins strictes)
        # Alternance: Si vague 2 est simple, vague 4 devrait etre complexe
        if wave2 and wave4:
            # Approximation: vague plus longue = plus complexe
            ratio_2_to_4 = wave2.length / wave4.length if wave4.length > 0 else 1
            if 0.8 < ratio_2_to_4 < 1.2:  # Trop similaires
                confidence -= 0.1

        # Vague 3 devrait etre la plus longue (guideline)
        if wave1 and wave3 and wave5:
            if wave3.length < wave1.length or wave3.length < wave5.length:
                confidence -= 0.15

        wave_count.confidence = max(0, confidence)
        return wave_count

    # =========================================================================
    # NIVEAUX FIBONACCI
    # =========================================================================

    def _calculate_fibonacci_levels(
        self,
        wave_count: WaveCount,
        df: pd.DataFrame
    ) -> Dict:
        """
        Calcule les niveaux Fibonacci pour les prochains mouvements

        - Retracements pour vagues correctives
        - Extensions pour vagues impulsives
        """
        levels = {
            'retracements': [],
            'extensions': [],
            'targets': []
        }

        if not wave_count.waves:
            return levels

        current = df['close'].iloc[-1]
        last_wave = wave_count.waves[-1]

        # Calculer retracements de la derniere vague
        wave_high = max(last_wave.start_price, last_wave.end_price)
        wave_low = min(last_wave.start_price, last_wave.end_price)
        wave_range = wave_high - wave_low

        for ratio in FIBONACCI_RATIOS['retracements']:
            if last_wave.direction == TrendDirection.BULLISH:
                level = wave_high - (wave_range * ratio)
            else:
                level = wave_low + (wave_range * ratio)
            levels['retracements'].append({
                'ratio': ratio,
                'price': level
            })

        # Calculer extensions pour la prochaine vague
        wave1 = wave_count.get_wave(WaveType.IMPULSE_1)
        wave3 = wave_count.get_wave(WaveType.IMPULSE_3)

        if wave1:
            for ratio in FIBONACCI_RATIOS['extensions']:
                if wave_count.trend == TrendDirection.BULLISH:
                    level = wave_low + (wave1.length * ratio)
                else:
                    level = wave_high - (wave1.length * ratio)
                levels['extensions'].append({
                    'ratio': ratio,
                    'price': level
                })

        # Targets specifiques selon la vague actuelle
        levels['targets'] = self._calculate_wave_targets(wave_count, current)

        return levels

    def _calculate_wave_targets(
        self,
        wave_count: WaveCount,
        current_price: float
    ) -> List[Dict]:
        """Calcule les targets pour la vague en cours"""
        targets = []

        wave1 = wave_count.get_wave(WaveType.IMPULSE_1)
        wave2 = wave_count.get_wave(WaveType.IMPULSE_2)
        wave3 = wave_count.get_wave(WaveType.IMPULSE_3)

        current_wave = wave_count.current_wave

        if current_wave == WaveType.IMPULSE_3 and wave1:
            # Vague 3: 161.8% ou 261.8% de vague 1
            for ext in [1.618, 2.618]:
                if wave_count.trend == TrendDirection.BULLISH:
                    target = wave2.end_price if wave2 else wave1.end_price
                    target += wave1.length * ext
                else:
                    target = wave2.end_price if wave2 else wave1.end_price
                    target -= wave1.length * ext
                targets.append({
                    'wave': 'wave_3',
                    'extension': ext,
                    'price': target
                })

        elif current_wave == WaveType.IMPULSE_5 and wave1 and wave3:
            # Vague 5: 61.8%, 100% ou 161.8% de vague 1-3
            wave1_to_3_length = abs(wave3.end_price - wave1.start_price)
            for ext in [0.618, 1.0, 1.618]:
                if wave_count.trend == TrendDirection.BULLISH:
                    target = wave3.end_price + (wave1.length * ext)
                else:
                    target = wave3.end_price - (wave1.length * ext)
                targets.append({
                    'wave': 'wave_5',
                    'extension': ext,
                    'price': target
                })

        elif current_wave == WaveType.CORRECTIVE_C:
            # Vague C: 100% ou 161.8% de vague A
            wave_a = wave_count.get_wave(WaveType.CORRECTIVE_A)
            if wave_a:
                for ext in [1.0, 1.618]:
                    if wave_count.trend == TrendDirection.BULLISH:
                        target = current_price - (wave_a.length * ext)
                    else:
                        target = current_price + (wave_a.length * ext)
                    targets.append({
                        'wave': 'wave_c',
                        'extension': ext,
                        'price': target
                    })

        return targets

    # =========================================================================
    # GENERATION DE SIGNAL
    # =========================================================================

    def _generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        wave_count: WaveCount,
        fib_levels: Dict
    ) -> Optional[ElliottSignal]:
        """
        Genere un signal de trading base sur l'analyse Elliott

        Meilleurs setups:
        - Fin de vague 2 (debut de vague 3)
        - Fin de vague 4 (debut de vague 5)
        - Fin de vague C (debut nouveau cycle)
        """
        if not wave_count.is_valid or wave_count.confidence < 0.6:
            return None

        current_price = df['close'].iloc[-1]
        current_wave = wave_count.current_wave

        signal_type = None
        reasons = []
        entry_price = current_price
        stop_loss = 0
        take_profit = 0
        fib_target = 0

        # Setup 1: Fin de vague 2 - ACHAT
        if current_wave == WaveType.IMPULSE_2:
            wave1 = wave_count.get_wave(WaveType.IMPULSE_1)
            wave2 = wave_count.get_wave(WaveType.IMPULSE_2)

            if wave1 and wave2 and wave_count.trend == TrendDirection.BULLISH:
                # Verifier retracement acceptable (38.2% - 61.8%)
                retracement = wave2.length / wave1.length
                if 0.382 <= retracement <= 0.618:
                    signal_type = 'buy'
                    stop_loss = wave1.start_price - (wave1.length * 0.1)
                    # Target: 161.8% de wave1 depuis fin wave2
                    take_profit = wave2.end_price + (wave1.length * 1.618)
                    fib_target = 1.618
                    reasons = [
                        "Fin de vague 2 (correction terminee)",
                        f"Retracement {retracement*100:.1f}% (ideal 38-62%)",
                        "Debut de vague 3 (vague la plus forte)"
                    ]

        # Setup 2: Fin de vague 4 - ACHAT
        elif current_wave == WaveType.IMPULSE_4:
            wave1 = wave_count.get_wave(WaveType.IMPULSE_1)
            wave3 = wave_count.get_wave(WaveType.IMPULSE_3)
            wave4 = wave_count.get_wave(WaveType.IMPULSE_4)

            if wave1 and wave3 and wave4 and wave_count.trend == TrendDirection.BULLISH:
                # Vague 4 retrace souvent 38.2% de wave 3
                retracement = wave4.length / wave3.length
                if 0.236 <= retracement <= 0.5:
                    signal_type = 'buy'
                    stop_loss = wave1.end_price - (wave1.length * 0.1)  # Sous wave 1
                    take_profit = wave3.end_price + (wave1.length * 0.618)
                    fib_target = 0.618
                    reasons = [
                        "Fin de vague 4 (correction terminee)",
                        f"Retracement {retracement*100:.1f}% de wave 3",
                        "Debut de vague 5"
                    ]

        # Setup 3: Fin de vague C - ACHAT (nouveau cycle)
        elif current_wave == WaveType.CORRECTIVE_C:
            wave_c = wave_count.get_wave(WaveType.CORRECTIVE_C)
            wave_a = wave_count.get_wave(WaveType.CORRECTIVE_A)

            if wave_a and wave_c:
                # Verifier que wave C a atteint un niveau Fibonacci
                c_to_a_ratio = wave_c.length / wave_a.length
                if 0.9 <= c_to_a_ratio <= 1.7:
                    signal_type = 'buy'
                    stop_loss = wave_c.end_price - (wave_a.length * 0.2)
                    take_profit = wave_c.end_price + (wave_a.length * 1.618)
                    fib_target = 1.618
                    reasons = [
                        "Fin de vague C (correction ABC terminee)",
                        f"Ratio C/A: {c_to_a_ratio:.2f}",
                        "Debut nouveau cycle impulsif"
                    ]

        if signal_type is None:
            return None

        # Inverser pour tendance baissiere
        if wave_count.trend == TrendDirection.BEARISH:
            signal_type = 'sell' if signal_type == 'buy' else 'buy'
            stop_loss, take_profit = take_profit, stop_loss

        strength = wave_count.confidence * 0.8 + 0.2  # Base de 0.2

        return ElliottSignal(
            symbol=symbol,
            signal_type=signal_type,
            current_wave=current_wave,
            trend=wave_count.trend,
            strength=min(strength, 1.0),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fibonacci_target=fib_target,
            reasons=reasons
        )


# =============================================================================
# SINGLETON
# =============================================================================
_elliott_analyzer = None

def get_elliott_analyzer() -> ElliottWaveAnalyzer:
    """Retourne l'instance singleton"""
    global _elliott_analyzer
    if _elliott_analyzer is None:
        _elliott_analyzer = ElliottWaveAnalyzer()
    return _elliott_analyzer
