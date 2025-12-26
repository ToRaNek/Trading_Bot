"""
Analyse Wyckoff
Base sur MASTER_TRADING_SKILL PARTIE XIII - Methode Wyckoff

Implementation des concepts:
- Les 3 Lois de Wyckoff (Offre/Demande, Cause/Effet, Effort/Resultat)
- Les 4 Phases du marche (Accumulation, Markup, Distribution, Markdown)
- Schema d'accumulation (PS, SC, AR, ST, Spring, SOS, LPS)
- Schema de distribution (PSY, BC, AR, ST, UTAD, SOW, LPSY)
- Detection des pieges (Spring, Upthrust)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WyckoffPhase(Enum):
    """Phases du cycle Wyckoff"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class WyckoffEvent(Enum):
    """Evenements Wyckoff dans les schemas"""
    # Accumulation
    PS = "preliminary_support"      # Arret initial de la baisse
    SC = "selling_climax"          # Capitulation, volume extreme
    AR = "automatic_rally"         # Rebond automatique
    ST = "secondary_test"          # Retest du SC
    SPRING = "spring"              # Faux breakout sous support (piege vendeurs)
    TEST = "test"                  # Test du Spring
    SOS = "sign_of_strength"       # Breakout avec volume
    LPS = "last_point_support"     # Dernier pullback = entree

    # Distribution
    PSY = "preliminary_supply"     # Arret initial de la hausse
    BC = "buying_climax"           # Euphorie, volume extreme
    UTAD = "upthrust_after_dist"   # Faux breakout (piege acheteurs)
    SOW = "sign_of_weakness"       # Breakout baissier
    LPSY = "last_point_supply"     # Dernier pullback = entree short


@dataclass
class WyckoffZone:
    """Zone de trading range Wyckoff"""
    support: float
    resistance: float
    start_date: datetime
    end_date: Optional[datetime] = None
    phase: WyckoffPhase = WyckoffPhase.UNKNOWN
    events: List[Dict] = field(default_factory=list)
    volume_profile: Dict = field(default_factory=dict)

    @property
    def range_size(self) -> float:
        return self.resistance - self.support

    @property
    def midpoint(self) -> float:
        return (self.support + self.resistance) / 2


@dataclass
class WyckoffSignal:
    """Signal de trading Wyckoff"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'avoid'
    phase: WyckoffPhase
    event: WyckoffEvent
    strength: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime = field(default_factory=datetime.now)
    reasons: List[str] = field(default_factory=list)
    volume_confirmation: bool = False

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'phase': self.phase.value,
            'event': self.event.value,
            'strength': self.strength,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat(),
            'reasons': self.reasons,
            'volume_confirmation': self.volume_confirmation
        }


class WyckoffAnalyzer:
    """
    Analyseur Wyckoff selon MASTER_TRADING_SKILL

    Les 3 Lois:
    1. Offre et Demande - Le prix monte quand demande > offre
    2. Cause et Effet - Amplitude = taille de l'accumulation/distribution
    3. Effort vs Resultat - Volume doit confirmer le prix
    """

    def __init__(self, lookback: int = 100, volume_ma: int = 20):
        self.lookback = lookback
        self.volume_ma = volume_ma

    # =========================================================================
    # ANALYSE PRINCIPALE
    # =========================================================================

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Analyse complete Wyckoff d'un DataFrame

        Returns:
            Dict avec phase, zone, evenements et signal
        """
        if len(df) < self.lookback:
            return {'phase': WyckoffPhase.UNKNOWN, 'signal': None}

        # 1. Detecter la zone de trading range
        zone = self._detect_trading_range(df)

        # 2. Identifier la phase
        phase = self._identify_phase(df, zone)

        # 3. Detecter les evenements Wyckoff
        events = self._detect_wyckoff_events(df, zone, phase)

        # 4. Analyser effort vs resultat (volume)
        volume_analysis = self._analyze_effort_result(df)

        # 5. Generer signal si applicable
        signal = self._generate_signal(
            symbol=symbol,
            df=df,
            zone=zone,
            phase=phase,
            events=events,
            volume_analysis=volume_analysis
        )

        return {
            'phase': phase,
            'zone': zone,
            'events': events,
            'volume_analysis': volume_analysis,
            'signal': signal
        }

    # =========================================================================
    # DETECTION ZONE DE TRADING RANGE
    # =========================================================================

    def _detect_trading_range(self, df: pd.DataFrame) -> Optional[WyckoffZone]:
        """
        Detecte une zone de trading range (consolidation)

        Criteres:
        - Prix oscille entre support et resistance
        - Minimum 10-20 bougies dans la zone
        - Range defini (pas de tendance forte)
        """
        recent = df.tail(self.lookback)

        # Calculer highs et lows locaux
        high_max = recent['high'].max()
        low_min = recent['low'].min()

        # Chercher les niveaux les plus testes
        support, resistance = self._find_key_levels(recent)

        if support is None or resistance is None:
            return None

        # Verifier qu'on est en range (pas en tendance)
        price_range = resistance - support
        avg_price = (support + resistance) / 2
        range_percent = price_range / avg_price

        # Range doit etre entre 5% et 30% du prix
        if range_percent < 0.03 or range_percent > 0.30:
            return None

        # Compter combien de fois le prix touche les niveaux
        touches_support = self._count_level_touches(recent, support, 'support')
        touches_resistance = self._count_level_touches(recent, resistance, 'resistance')

        if touches_support < 2 or touches_resistance < 2:
            return None

        return WyckoffZone(
            support=support,
            resistance=resistance,
            start_date=recent.index[0] if hasattr(recent.index[0], 'date') else datetime.now(),
            volume_profile=self._calculate_volume_profile(recent, support, resistance)
        )

    def _find_key_levels(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Trouve les niveaux de support et resistance les plus importants"""
        # Utiliser les pivots
        highs = df['high'].values
        lows = df['low'].values

        # Trouver les swing highs et lows
        swing_highs = []
        swing_lows = []

        for i in range(2, len(df) - 2):
            # Swing high
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])

            # Swing low
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])

        if not swing_highs or not swing_lows:
            return None, None

        # Regrouper les niveaux proches (clustering)
        resistance = self._cluster_levels(swing_highs)
        support = self._cluster_levels(swing_lows)

        return support, resistance

    def _cluster_levels(self, levels: List[float], threshold: float = 0.02) -> Optional[float]:
        """Regroupe les niveaux proches et retourne le plus significant"""
        if not levels:
            return None

        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        clusters.append(current_cluster)

        # Retourner le niveau du cluster le plus grand
        best_cluster = max(clusters, key=len)
        return np.mean(best_cluster)

    def _count_level_touches(
        self,
        df: pd.DataFrame,
        level: float,
        level_type: str,
        tolerance: float = 0.02
    ) -> int:
        """Compte le nombre de fois que le prix touche un niveau"""
        count = 0
        for _, row in df.iterrows():
            if level_type == 'support':
                if abs(row['low'] - level) / level < tolerance:
                    count += 1
            else:
                if abs(row['high'] - level) / level < tolerance:
                    count += 1
        return count

    def _calculate_volume_profile(
        self,
        df: pd.DataFrame,
        support: float,
        resistance: float
    ) -> Dict:
        """Calcule le profil de volume dans la zone"""
        # Diviser la zone en 10 niveaux
        levels = np.linspace(support, resistance, 10)
        volume_at_level = {}

        for i in range(len(levels) - 1):
            low_level = levels[i]
            high_level = levels[i + 1]
            mid = (low_level + high_level) / 2

            # Volume dans cette tranche
            mask = (df['low'] <= high_level) & (df['high'] >= low_level)
            vol = df.loc[mask, 'volume'].sum()
            volume_at_level[mid] = vol

        # Point de controle (POC)
        poc = max(volume_at_level.keys(), key=lambda x: volume_at_level[x])

        return {
            'levels': volume_at_level,
            'poc': poc,
            'total_volume': sum(volume_at_level.values())
        }

    # =========================================================================
    # IDENTIFICATION DE PHASE
    # =========================================================================

    def _identify_phase(
        self,
        df: pd.DataFrame,
        zone: Optional[WyckoffZone]
    ) -> WyckoffPhase:
        """
        Identifie la phase Wyckoff actuelle

        Phases:
        - ACCUMULATION: Smart money achete discretement
        - MARKUP: Tendance haussiere
        - DISTRIBUTION: Smart money vend
        - MARKDOWN: Tendance baissiere
        """
        if zone is None:
            # Pas de zone = probablement en tendance
            return self._identify_trend_phase(df)

        current_price = df['close'].iloc[-1]

        # Analyser la position du prix par rapport a la zone
        zone_position = (current_price - zone.support) / zone.range_size

        # Analyser le volume
        recent_volume = df['volume'].tail(10).mean()
        avg_volume = df['volume'].tail(50).mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        # Analyser le momentum
        momentum = self._calculate_momentum(df)

        # Logique de detection
        if zone_position < 0.3 and momentum < 0:
            # Prix bas dans la zone, momentum negatif
            return WyckoffPhase.ACCUMULATION
        elif zone_position > 0.7 and momentum > 0:
            # Prix haut dans la zone, momentum positif
            return WyckoffPhase.DISTRIBUTION
        elif current_price > zone.resistance:
            return WyckoffPhase.MARKUP
        elif current_price < zone.support:
            return WyckoffPhase.MARKDOWN

        return WyckoffPhase.UNKNOWN

    def _identify_trend_phase(self, df: pd.DataFrame) -> WyckoffPhase:
        """Identifie la phase quand pas de zone claire"""
        # Utiliser les moyennes mobiles
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        current = df['close'].iloc[-1]

        if current > sma20 > sma50:
            return WyckoffPhase.MARKUP
        elif current < sma20 < sma50:
            return WyckoffPhase.MARKDOWN

        return WyckoffPhase.UNKNOWN

    def _calculate_momentum(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcule le momentum"""
        if len(df) < period:
            return 0
        return (df['close'].iloc[-1] / df['close'].iloc[-period] - 1) * 100

    # =========================================================================
    # DETECTION EVENEMENTS WYCKOFF
    # =========================================================================

    def _detect_wyckoff_events(
        self,
        df: pd.DataFrame,
        zone: Optional[WyckoffZone],
        phase: WyckoffPhase
    ) -> List[Dict]:
        """
        Detecte les evenements Wyckoff dans les donnees

        Accumulation: PS, SC, AR, ST, Spring, Test, SOS, LPS
        Distribution: PSY, BC, AR, ST, UTAD, SOW, LPSY
        """
        events = []

        if zone is None:
            return events

        if phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.UNKNOWN]:
            events.extend(self._detect_accumulation_events(df, zone))

        if phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.UNKNOWN]:
            events.extend(self._detect_distribution_events(df, zone))

        return events

    def _detect_accumulation_events(
        self,
        df: pd.DataFrame,
        zone: WyckoffZone
    ) -> List[Dict]:
        """Detecte les evenements d'accumulation"""
        events = []
        recent = df.tail(30)

        # 1. Selling Climax (SC) - Volume extreme + gros range + close bas
        for i in range(5, len(recent)):
            row = recent.iloc[i]
            vol_ratio = row['volume'] / recent['volume'].mean()
            candle_range = row['high'] - row['low']
            avg_range = recent['high'].sub(recent['low']).mean()

            if vol_ratio > 2 and candle_range > avg_range * 1.5:
                if row['close'] < row['open']:  # Bougie rouge
                    if row['low'] <= zone.support * 1.01:
                        events.append({
                            'event': WyckoffEvent.SC,
                            'price': row['close'],
                            'volume_ratio': vol_ratio,
                            'index': i
                        })

        # 2. Spring - Faux breakout sous support avec retour rapide
        spring = self._detect_spring(recent, zone)
        if spring:
            events.append(spring)

        # 3. Sign of Strength (SOS) - Breakout resistance avec volume
        for i in range(5, len(recent)):
            row = recent.iloc[i]
            if row['close'] > zone.resistance:
                vol_ratio = row['volume'] / recent['volume'].mean()
                if vol_ratio > 1.5:
                    events.append({
                        'event': WyckoffEvent.SOS,
                        'price': row['close'],
                        'volume_ratio': vol_ratio,
                        'index': i
                    })

        # 4. Last Point of Support (LPS) - Pullback apres SOS
        sos_events = [e for e in events if e['event'] == WyckoffEvent.SOS]
        if sos_events:
            last_sos = sos_events[-1]
            for i in range(last_sos['index'] + 1, len(recent)):
                row = recent.iloc[i]
                # Pullback vers l'ancienne resistance
                if abs(row['low'] - zone.resistance) / zone.resistance < 0.02:
                    events.append({
                        'event': WyckoffEvent.LPS,
                        'price': row['close'],
                        'index': i
                    })

        return events

    def _detect_distribution_events(
        self,
        df: pd.DataFrame,
        zone: WyckoffZone
    ) -> List[Dict]:
        """Detecte les evenements de distribution"""
        events = []
        recent = df.tail(30)

        # 1. Buying Climax (BC) - Volume extreme + close haut
        for i in range(5, len(recent)):
            row = recent.iloc[i]
            vol_ratio = row['volume'] / recent['volume'].mean()

            if vol_ratio > 2:
                if row['close'] > row['open']:  # Bougie verte
                    if row['high'] >= zone.resistance * 0.99:
                        events.append({
                            'event': WyckoffEvent.BC,
                            'price': row['close'],
                            'volume_ratio': vol_ratio,
                            'index': i
                        })

        # 2. Upthrust After Distribution (UTAD) - Faux breakout au-dessus resistance
        utad = self._detect_upthrust(recent, zone)
        if utad:
            events.append(utad)

        # 3. Sign of Weakness (SOW) - Breakout support avec volume
        for i in range(5, len(recent)):
            row = recent.iloc[i]
            if row['close'] < zone.support:
                vol_ratio = row['volume'] / recent['volume'].mean()
                if vol_ratio > 1.5:
                    events.append({
                        'event': WyckoffEvent.SOW,
                        'price': row['close'],
                        'volume_ratio': vol_ratio,
                        'index': i
                    })

        # 4. Last Point of Supply (LPSY)
        sow_events = [e for e in events if e['event'] == WyckoffEvent.SOW]
        if sow_events:
            last_sow = sow_events[-1]
            for i in range(last_sow['index'] + 1, len(recent)):
                row = recent.iloc[i]
                if abs(row['high'] - zone.support) / zone.support < 0.02:
                    events.append({
                        'event': WyckoffEvent.LPSY,
                        'price': row['close'],
                        'index': i
                    })

        return events

    def _detect_spring(self, df: pd.DataFrame, zone: WyckoffZone) -> Optional[Dict]:
        """
        Detecte un Spring (piege vendeurs)

        Caracteristiques:
        - Prix perce le support brievement
        - Retour rapide au-dessus du support
        - Volume peut etre eleve ou faible (faible = piege)
        """
        for i in range(3, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # Prix perce sous le support
            if row['low'] < zone.support:
                # Mais cloture au-dessus ou proche
                if row['close'] > zone.support * 0.98:
                    # Confirmation: la bougie suivante est haussiere
                    if next_row['close'] > next_row['open']:
                        return {
                            'event': WyckoffEvent.SPRING,
                            'price': row['low'],
                            'recovery_price': next_row['close'],
                            'index': i
                        }
        return None

    def _detect_upthrust(self, df: pd.DataFrame, zone: WyckoffZone) -> Optional[Dict]:
        """
        Detecte un Upthrust (piege acheteurs)

        Caracteristiques:
        - Prix perce la resistance brievement
        - Retour rapide en dessous
        - Volume souvent eleve (faux breakout)
        """
        for i in range(3, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # Prix perce au-dessus de la resistance
            if row['high'] > zone.resistance:
                # Mais cloture en dessous ou proche
                if row['close'] < zone.resistance * 1.02:
                    # Confirmation: la bougie suivante est baissiere
                    if next_row['close'] < next_row['open']:
                        return {
                            'event': WyckoffEvent.UTAD,
                            'price': row['high'],
                            'rejection_price': next_row['close'],
                            'index': i
                        }
        return None

    # =========================================================================
    # ANALYSE EFFORT VS RESULTAT
    # =========================================================================

    def _analyze_effort_result(self, df: pd.DataFrame) -> Dict:
        """
        Analyse la loi Effort vs Resultat

        - Volume (effort) doit confirmer le prix (resultat)
        - Divergence = signal de renversement
        """
        recent = df.tail(20)

        # Calculer les variations
        price_changes = recent['close'].pct_change()
        volume_changes = recent['volume'].pct_change()

        # Correlation prix-volume
        correlation = price_changes.corr(volume_changes)

        # Detecter divergences
        divergence = None

        # Prix monte + volume baisse = divergence baissiere
        if price_changes.tail(5).mean() > 0 and volume_changes.tail(5).mean() < 0:
            divergence = 'bearish'
        # Prix baisse + volume baisse = possible fin de baisse
        elif price_changes.tail(5).mean() < 0 and volume_changes.tail(5).mean() < 0:
            divergence = 'bullish_potential'

        # Volume climax
        current_vol = recent['volume'].iloc[-1]
        avg_vol = recent['volume'].mean()
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

        is_climax = vol_ratio > 2.5

        return {
            'correlation': correlation,
            'divergence': divergence,
            'volume_ratio': vol_ratio,
            'is_climax': is_climax,
            'harmonious': correlation > 0.3  # Volume confirme le prix
        }

    # =========================================================================
    # GENERATION DE SIGNAL
    # =========================================================================

    def _generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        zone: Optional[WyckoffZone],
        phase: WyckoffPhase,
        events: List[Dict],
        volume_analysis: Dict
    ) -> Optional[WyckoffSignal]:
        """Genere un signal de trading base sur l'analyse Wyckoff"""

        if not events or zone is None:
            return None

        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)

        # Chercher les meilleurs setups

        # 1. Spring + confirmation = BUY fort
        springs = [e for e in events if e['event'] == WyckoffEvent.SPRING]
        if springs:
            spring = springs[-1]
            return WyckoffSignal(
                symbol=symbol,
                signal_type='buy',
                phase=WyckoffPhase.ACCUMULATION,
                event=WyckoffEvent.SPRING,
                strength=0.9,
                entry_price=current_price,
                stop_loss=spring['price'] - atr,
                take_profit=zone.resistance + (zone.range_size * 0.5),
                reasons=[
                    "Spring detecte (piege vendeurs)",
                    "Prix recupere au-dessus du support",
                    "Setup accumulation classique"
                ],
                volume_confirmation=volume_analysis.get('harmonious', False)
            )

        # 2. LPS = Buy (entree ideale apres SOS)
        lps_events = [e for e in events if e['event'] == WyckoffEvent.LPS]
        if lps_events:
            return WyckoffSignal(
                symbol=symbol,
                signal_type='buy',
                phase=WyckoffPhase.ACCUMULATION,
                event=WyckoffEvent.LPS,
                strength=0.85,
                entry_price=current_price,
                stop_loss=zone.support - atr,
                take_profit=zone.resistance + (zone.range_size),
                reasons=[
                    "Last Point of Support (LPS)",
                    "Pullback apres Sign of Strength",
                    "Entree ideale sur breakout"
                ],
                volume_confirmation=volume_analysis.get('harmonious', False)
            )

        # 3. Upthrust = SELL fort
        upthrusts = [e for e in events if e['event'] == WyckoffEvent.UTAD]
        if upthrusts:
            utad = upthrusts[-1]
            return WyckoffSignal(
                symbol=symbol,
                signal_type='sell',
                phase=WyckoffPhase.DISTRIBUTION,
                event=WyckoffEvent.UTAD,
                strength=0.9,
                entry_price=current_price,
                stop_loss=utad['price'] + atr,
                take_profit=zone.support - (zone.range_size * 0.5),
                reasons=[
                    "Upthrust After Distribution (UTAD)",
                    "Faux breakout au-dessus resistance",
                    "Setup distribution classique"
                ],
                volume_confirmation=volume_analysis.get('is_climax', False)
            )

        # 4. LPSY = Sell (entree ideale apres SOW)
        lpsy_events = [e for e in events if e['event'] == WyckoffEvent.LPSY]
        if lpsy_events:
            return WyckoffSignal(
                symbol=symbol,
                signal_type='sell',
                phase=WyckoffPhase.DISTRIBUTION,
                event=WyckoffEvent.LPSY,
                strength=0.85,
                entry_price=current_price,
                stop_loss=zone.resistance + atr,
                take_profit=zone.support - (zone.range_size),
                reasons=[
                    "Last Point of Supply (LPSY)",
                    "Pullback apres Sign of Weakness",
                    "Entree ideale sur breakdown"
                ],
                volume_confirmation=volume_analysis.get('harmonious', False)
            )

        return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcule l'ATR"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close),
            'lc': abs(low - close)
        }).max(axis=1)

        return tr.rolling(period).mean().iloc[-1]


# =============================================================================
# SINGLETON
# =============================================================================
_wyckoff_analyzer = None

def get_wyckoff_analyzer() -> WyckoffAnalyzer:
    """Retourne l'instance singleton"""
    global _wyckoff_analyzer
    if _wyckoff_analyzer is None:
        _wyckoff_analyzer = WyckoffAnalyzer()
    return _wyckoff_analyzer
