"""
Analyse Ichimoku Kinko Hyo
Base sur MASTER_TRADING_SKILL PARTIE XV - Ichimoku Cloud Trading

Implementation des concepts:
- Les 5 lignes (Tenkan, Kijun, Senkou Span A/B, Chikou Span)
- Le Kumo (nuage) - support/resistance dynamique
- Signaux TK Cross
- Confirmation Chikou Span
- Strategie complete avec checklist
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IchimokuSignalStrength(Enum):
    """Force du signal Ichimoku"""
    STRONG = "strong"      # Signal au-dessus/dessous du nuage
    MEDIUM = "medium"      # Signal dans le nuage
    WEAK = "weak"          # Signal contraire au nuage


class TrendDirection(Enum):
    """Direction de la tendance"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class IchimokuLines:
    """Les 5 lignes Ichimoku"""
    tenkan_sen: float      # Ligne de conversion (9 periodes)
    kijun_sen: float       # Ligne de base (26 periodes)
    senkou_span_a: float   # Leading Span A (projete 26 periodes)
    senkou_span_b: float   # Leading Span B (projete 26 periodes)
    chikou_span: float     # Lagging Span (decale 26 periodes en arriere)

    @property
    def kumo_top(self) -> float:
        """Haut du nuage"""
        return max(self.senkou_span_a, self.senkou_span_b)

    @property
    def kumo_bottom(self) -> float:
        """Bas du nuage"""
        return min(self.senkou_span_a, self.senkou_span_b)

    @property
    def kumo_thickness(self) -> float:
        """Epaisseur du nuage"""
        return abs(self.senkou_span_a - self.senkou_span_b)

    @property
    def is_bullish_kumo(self) -> bool:
        """Nuage haussier (Span A > Span B)"""
        return self.senkou_span_a > self.senkou_span_b


@dataclass
class IchimokuAnalysis:
    """Resultat de l'analyse Ichimoku"""
    lines: IchimokuLines
    trend: TrendDirection
    price_vs_kumo: str  # 'above', 'below', 'inside'
    tk_cross: Optional[str]  # 'bullish', 'bearish', None
    chikou_confirmation: bool
    kumo_future_trend: TrendDirection
    signal_strength: IchimokuSignalStrength
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)


@dataclass
class IchimokuSignal:
    """Signal de trading Ichimoku"""
    symbol: str
    signal_type: str  # 'buy', 'sell'
    strength: IchimokuSignalStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    trend: TrendDirection
    timestamp: datetime = field(default_factory=datetime.now)
    reasons: List[str] = field(default_factory=list)
    checklist: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': self.strength.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trend': self.trend.value,
            'timestamp': self.timestamp.isoformat(),
            'reasons': self.reasons,
            'checklist': self.checklist
        }


class IchimokuAnalyzer:
    """
    Analyseur Ichimoku selon MASTER_TRADING_SKILL

    Les 5 Composants:
    1. Tenkan-Sen (9 periodes) - Signal rapide
    2. Kijun-Sen (26 periodes) - Signal moyen terme, S/R dynamique
    3. Senkou Span A - Bord du nuage (moyenne Tenkan+Kijun projete)
    4. Senkou Span B - Bord du nuage (52 periodes projete)
    5. Chikou Span - Confirmation (prix actuel decale)
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

    # =========================================================================
    # CALCUL DES LIGNES
    # =========================================================================

    def calculate_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule toutes les lignes Ichimoku

        Formules:
        - Tenkan = (High9 + Low9) / 2
        - Kijun = (High26 + Low26) / 2
        - Senkou A = (Tenkan + Kijun) / 2, projete 26 periodes
        - Senkou B = (High52 + Low52) / 2, projete 26 periodes
        - Chikou = Close, decale 26 periodes en arriere
        """
        df = df.copy()

        # Tenkan-Sen (Conversion Line)
        high_tenkan = df['high'].rolling(self.tenkan_period).max()
        low_tenkan = df['low'].rolling(self.tenkan_period).min()
        df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2

        # Kijun-Sen (Base Line)
        high_kijun = df['high'].rolling(self.kijun_period).max()
        low_kijun = df['low'].rolling(self.kijun_period).min()
        df['kijun_sen'] = (high_kijun + low_kijun) / 2

        # Senkou Span A (Leading Span A) - projete 26 periodes
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.displacement)

        # Senkou Span B (Leading Span B) - projete 26 periodes
        high_senkou = df['high'].rolling(self.senkou_b_period).max()
        low_senkou = df['low'].rolling(self.senkou_b_period).min()
        df['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(self.displacement)

        # Chikou Span (Lagging Span) - decale 26 periodes en arriere
        df['chikou_span'] = df['close'].shift(-self.displacement)

        # Kumo top et bottom
        df['kumo_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        df['kumo_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        return df

    def get_current_lines(self, df: pd.DataFrame) -> Optional[IchimokuLines]:
        """Retourne les lignes actuelles"""
        if len(df) < self.senkou_b_period + self.displacement:
            return None

        df = self.calculate_lines(df)
        latest = df.iloc[-1]

        return IchimokuLines(
            tenkan_sen=latest['tenkan_sen'],
            kijun_sen=latest['kijun_sen'],
            senkou_span_a=latest['senkou_span_a'],
            senkou_span_b=latest['senkou_span_b'],
            chikou_span=df['close'].iloc[-1]  # Chikou actuel
        )

    # =========================================================================
    # ANALYSE PRINCIPALE
    # =========================================================================

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Analyse Ichimoku complete

        Returns:
            Dict avec analyse, signaux et niveaux
        """
        if len(df) < self.senkou_b_period + self.displacement:
            return {'analysis': None, 'signal': None}

        # Calculer les lignes
        df = self.calculate_lines(df)

        # Obtenir les lignes actuelles
        lines = self.get_current_lines(df)
        if lines is None:
            return {'analysis': None, 'signal': None}

        # Analyse de la tendance
        analysis = self._analyze_trend(df, lines)

        # Generer signal
        signal = self._generate_signal(symbol, df, lines, analysis)

        return {
            'lines': lines,
            'analysis': analysis,
            'signal': signal,
            'df_with_ichimoku': df
        }

    def _analyze_trend(
        self,
        df: pd.DataFrame,
        lines: IchimokuLines
    ) -> IchimokuAnalysis:
        """Analyse la tendance actuelle"""
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]

        # Position du prix par rapport au nuage
        if current_price > lines.kumo_top:
            price_vs_kumo = 'above'
        elif current_price < lines.kumo_bottom:
            price_vs_kumo = 'below'
        else:
            price_vs_kumo = 'inside'

        # Tendance principale
        if price_vs_kumo == 'above' and lines.is_bullish_kumo:
            trend = TrendDirection.BULLISH
        elif price_vs_kumo == 'below' and not lines.is_bullish_kumo:
            trend = TrendDirection.BEARISH
        else:
            trend = TrendDirection.NEUTRAL

        # Detection TK Cross
        tk_cross = self._detect_tk_cross(df)

        # Confirmation Chikou
        chikou_confirmation = self._check_chikou_confirmation(df, trend)

        # Tendance future du nuage
        kumo_future_trend = self._analyze_future_kumo(df)

        # Force du signal
        signal_strength = self._calculate_signal_strength(
            price_vs_kumo, tk_cross, trend
        )

        # Niveaux de support/resistance
        support_levels = [lines.kijun_sen, lines.kumo_bottom]
        resistance_levels = [lines.kumo_top]

        if lines.tenkan_sen < current_price:
            support_levels.append(lines.tenkan_sen)
        else:
            resistance_levels.append(lines.tenkan_sen)

        return IchimokuAnalysis(
            lines=lines,
            trend=trend,
            price_vs_kumo=price_vs_kumo,
            tk_cross=tk_cross,
            chikou_confirmation=chikou_confirmation,
            kumo_future_trend=kumo_future_trend,
            signal_strength=signal_strength,
            support_levels=sorted(support_levels),
            resistance_levels=sorted(resistance_levels)
        )

    def _detect_tk_cross(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detecte le croisement Tenkan/Kijun

        TK Cross Bullish: Tenkan croise Kijun vers le HAUT
        TK Cross Bearish: Tenkan croise Kijun vers le BAS
        """
        if len(df) < 3:
            return None

        curr_tenkan = df['tenkan_sen'].iloc[-1]
        prev_tenkan = df['tenkan_sen'].iloc[-2]
        curr_kijun = df['kijun_sen'].iloc[-1]
        prev_kijun = df['kijun_sen'].iloc[-2]

        # Cross bullish
        if prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun:
            return 'bullish'

        # Cross bearish
        if prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun:
            return 'bearish'

        return None

    def _check_chikou_confirmation(
        self,
        df: pd.DataFrame,
        trend: TrendDirection
    ) -> bool:
        """
        Verifie si Chikou Span confirme la tendance

        Bullish: Chikou au-dessus du prix (26 periodes avant)
        Bearish: Chikou en-dessous du prix (26 periodes avant)
        """
        if len(df) < self.displacement + 1:
            return False

        current_close = df['close'].iloc[-1]
        price_26_ago = df['close'].iloc[-self.displacement - 1]

        if trend == TrendDirection.BULLISH:
            return current_close > price_26_ago
        elif trend == TrendDirection.BEARISH:
            return current_close < price_26_ago

        return False

    def _analyze_future_kumo(self, df: pd.DataFrame) -> TrendDirection:
        """
        Analyse la tendance future du nuage (projete)

        Le nuage futur indique la tendance a venir
        """
        # Calculer Senkou Spans pour le futur
        high_recent = df['high'].tail(self.tenkan_period).max()
        low_recent = df['low'].tail(self.tenkan_period).min()
        tenkan_future = (high_recent + low_recent) / 2

        high_kijun = df['high'].tail(self.kijun_period).max()
        low_kijun = df['low'].tail(self.kijun_period).min()
        kijun_future = (high_kijun + low_kijun) / 2

        span_a_future = (tenkan_future + kijun_future) / 2

        high_senkou = df['high'].tail(self.senkou_b_period).max()
        low_senkou = df['low'].tail(self.senkou_b_period).min()
        span_b_future = (high_senkou + low_senkou) / 2

        if span_a_future > span_b_future:
            return TrendDirection.BULLISH
        elif span_a_future < span_b_future:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def _calculate_signal_strength(
        self,
        price_vs_kumo: str,
        tk_cross: Optional[str],
        trend: TrendDirection
    ) -> IchimokuSignalStrength:
        """
        Calcule la force du signal

        STRONG: Signal conforme a la position du prix et au nuage
        MEDIUM: Signal dans le nuage
        WEAK: Signal contraire au contexte
        """
        if price_vs_kumo == 'inside':
            return IchimokuSignalStrength.MEDIUM

        if trend == TrendDirection.BULLISH:
            if price_vs_kumo == 'above' and tk_cross == 'bullish':
                return IchimokuSignalStrength.STRONG
            elif price_vs_kumo == 'below':
                return IchimokuSignalStrength.WEAK

        elif trend == TrendDirection.BEARISH:
            if price_vs_kumo == 'below' and tk_cross == 'bearish':
                return IchimokuSignalStrength.STRONG
            elif price_vs_kumo == 'above':
                return IchimokuSignalStrength.WEAK

        return IchimokuSignalStrength.MEDIUM

    # =========================================================================
    # GENERATION DE SIGNAL
    # =========================================================================

    def _generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        lines: IchimokuLines,
        analysis: IchimokuAnalysis
    ) -> Optional[IchimokuSignal]:
        """
        Genere un signal de trading Ichimoku

        Checklist pour ACHAT:
        - Prix au-dessus du nuage
        - Tenkan au-dessus de Kijun
        - Chikou au-dessus du prix (26 periodes avant)
        - Nuage futur haussier (Span A > Span B)
        - Prix rebondit sur Kijun ou bord du nuage
        """
        current_price = df['close'].iloc[-1]

        # Construire la checklist
        checklist = {
            'price_above_kumo': analysis.price_vs_kumo == 'above',
            'price_below_kumo': analysis.price_vs_kumo == 'below',
            'tenkan_above_kijun': lines.tenkan_sen > lines.kijun_sen,
            'tenkan_below_kijun': lines.tenkan_sen < lines.kijun_sen,
            'chikou_confirmation': analysis.chikou_confirmation,
            'bullish_kumo_future': analysis.kumo_future_trend == TrendDirection.BULLISH,
            'bearish_kumo_future': analysis.kumo_future_trend == TrendDirection.BEARISH,
            'tk_cross_bullish': analysis.tk_cross == 'bullish',
            'tk_cross_bearish': analysis.tk_cross == 'bearish',
        }

        # Conditions pour signal ACHAT
        buy_conditions = [
            checklist['price_above_kumo'],
            checklist['tenkan_above_kijun'],
            checklist['chikou_confirmation'],
            checklist['bullish_kumo_future']
        ]

        # Conditions pour signal VENTE
        sell_conditions = [
            checklist['price_below_kumo'],
            checklist['tenkan_below_kijun'],
            checklist['chikou_confirmation'],
            checklist['bearish_kumo_future']
        ]

        signal_type = None
        reasons = []

        # Signal ACHAT
        if sum(buy_conditions) >= 3 and analysis.tk_cross == 'bullish':
            signal_type = 'buy'
            reasons = [
                "Prix au-dessus du nuage" if checklist['price_above_kumo'] else "",
                "Tenkan au-dessus de Kijun" if checklist['tenkan_above_kijun'] else "",
                "Chikou confirme la tendance haussiere" if checklist['chikou_confirmation'] else "",
                "TK Cross haussier detecte" if checklist['tk_cross_bullish'] else "",
                "Nuage futur haussier" if checklist['bullish_kumo_future'] else ""
            ]
            reasons = [r for r in reasons if r]

        # Signal VENTE
        elif sum(sell_conditions) >= 3 and analysis.tk_cross == 'bearish':
            signal_type = 'sell'
            reasons = [
                "Prix sous le nuage" if checklist['price_below_kumo'] else "",
                "Tenkan sous Kijun" if checklist['tenkan_below_kijun'] else "",
                "Chikou confirme la tendance baissiere" if checklist['chikou_confirmation'] else "",
                "TK Cross baissier detecte" if checklist['tk_cross_bearish'] else "",
                "Nuage futur baissier" if checklist['bearish_kumo_future'] else ""
            ]
            reasons = [r for r in reasons if r]

        # Signal sur rebond Kijun (entry classique)
        elif self._detect_kijun_bounce(df, lines, analysis.trend):
            if analysis.trend == TrendDirection.BULLISH:
                signal_type = 'buy'
                reasons = ["Rebond sur Kijun-Sen (support dynamique)", "Tendance haussiere confirmee"]
            elif analysis.trend == TrendDirection.BEARISH:
                signal_type = 'sell'
                reasons = ["Rejet sur Kijun-Sen (resistance dynamique)", "Tendance baissiere confirmee"]

        if signal_type is None:
            return None

        # Calculer SL et TP
        if signal_type == 'buy':
            stop_loss = lines.kumo_bottom - (lines.kumo_thickness * 0.5)
            take_profit = current_price + (current_price - stop_loss) * 3
        else:
            stop_loss = lines.kumo_top + (lines.kumo_thickness * 0.5)
            take_profit = current_price - (stop_loss - current_price) * 3

        return IchimokuSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=analysis.signal_strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trend=analysis.trend,
            reasons=reasons,
            checklist=checklist
        )

    def _detect_kijun_bounce(
        self,
        df: pd.DataFrame,
        lines: IchimokuLines,
        trend: TrendDirection
    ) -> bool:
        """
        Detecte un rebond sur le Kijun-Sen

        En tendance haussiere: prix touche Kijun et rebondit
        En tendance baissiere: prix touche Kijun et est rejete
        """
        if len(df) < 3:
            return False

        low_recent = df['low'].iloc[-2]
        high_recent = df['high'].iloc[-2]
        close_current = df['close'].iloc[-1]
        kijun = lines.kijun_sen

        # Tolerance de 0.5%
        tolerance = kijun * 0.005

        if trend == TrendDirection.BULLISH:
            # Prix touche Kijun par le bas et rebondit
            if abs(low_recent - kijun) < tolerance and close_current > kijun:
                return True

        elif trend == TrendDirection.BEARISH:
            # Prix touche Kijun par le haut et est rejete
            if abs(high_recent - kijun) < tolerance and close_current < kijun:
                return True

        return False


# =============================================================================
# SINGLETON
# =============================================================================
_ichimoku_analyzer = None

def get_ichimoku_analyzer() -> IchimokuAnalyzer:
    """Retourne l'instance singleton"""
    global _ichimoku_analyzer
    if _ichimoku_analyzer is None:
        _ichimoku_analyzer = IchimokuAnalyzer()
    return _ichimoku_analyzer
