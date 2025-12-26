"""
Analyse Volume Profile
Base sur MASTER_TRADING_SKILL PARTIE XVI - Volume Profile (Trader Dale)

Implementation des concepts:
- Point de Controle (POC) - niveau avec le plus gros volume
- Value Area (VA) - zone contenant 70% du volume
- High Volume Nodes (HVN) - zones de fort volume
- Low Volume Nodes (LVN) - zones de faible volume
- Formes de profil (D-Shape, P-Shape, b-Shape)
- Signaux de trading bases sur le profil
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProfileShape(Enum):
    """Formes du profil de volume"""
    D_SHAPE = "d_shape"    # Distribution normale, POC au milieu = equilibre
    P_SHAPE = "p_shape"    # Volume concentre en haut = accumulation
    B_SHAPE = "b_shape"    # Volume concentre en bas = distribution
    DOUBLE = "double"      # Deux POC = marche indecis
    THIN = "thin"          # Volume faible = zone de passage rapide
    UNKNOWN = "unknown"


class NodeType(Enum):
    """Type de noeud de volume"""
    HVN = "hvn"  # High Volume Node
    LVN = "lvn"  # Low Volume Node


@dataclass
class VolumeNode:
    """Noeud de volume (HVN ou LVN)"""
    price_level: float
    volume: float
    node_type: NodeType
    strength: float  # 0-1, importance relative

    @property
    def is_hvn(self) -> bool:
        return self.node_type == NodeType.HVN

    @property
    def is_lvn(self) -> bool:
        return self.node_type == NodeType.LVN


@dataclass
class ValueArea:
    """Zone de valeur (70% du volume)"""
    vah: float  # Value Area High
    val: float  # Value Area Low
    poc: float  # Point of Control
    total_volume: float
    width_percent: float  # Largeur en % du prix

    @property
    def mid(self) -> float:
        return (self.vah + self.val) / 2


@dataclass
class VolumeProfile:
    """Profil de volume complet"""
    price_levels: List[float]
    volume_at_price: Dict[float, float]
    poc: float
    value_area: ValueArea
    shape: ProfileShape
    hvn_nodes: List[VolumeNode] = field(default_factory=list)
    lvn_nodes: List[VolumeNode] = field(default_factory=list)
    developing: bool = False  # Profil en cours de formation

    def get_volume_at_price(self, price: float, tolerance: float = 0.01) -> float:
        """Retourne le volume a un niveau de prix donne"""
        for level, volume in self.volume_at_price.items():
            if abs(level - price) / price < tolerance:
                return volume
        return 0


@dataclass
class VolumeProfileSignal:
    """Signal de trading Volume Profile"""
    symbol: str
    signal_type: str  # 'buy', 'sell'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    profile_shape: ProfileShape
    key_level: str  # 'poc', 'vah', 'val', 'hvn', 'lvn'
    timestamp: datetime = field(default_factory=datetime.now)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'profile_shape': self.profile_shape.value,
            'key_level': self.key_level,
            'timestamp': self.timestamp.isoformat(),
            'reasons': self.reasons
        }


class VolumeProfileAnalyzer:
    """
    Analyseur Volume Profile selon MASTER_TRADING_SKILL

    Concepts cles:
    - POC: Niveau ou le plus de volume a ete echange (aimant)
    - Value Area: Zone avec 70% du volume (equilibre)
    - HVN: Zone de fort volume (S/R fort, prix s'attarde)
    - LVN: Zone de faible volume (prix traverse vite)
    """

    def __init__(
        self,
        num_levels: int = 50,
        value_area_percent: float = 0.70
    ):
        self.num_levels = num_levels
        self.value_area_percent = value_area_percent

    # =========================================================================
    # CALCUL DU PROFIL
    # =========================================================================

    def calculate_profile(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None
    ) -> VolumeProfile:
        """
        Calcule le profil de volume

        Args:
            df: DataFrame avec OHLCV
            period: Nombre de bougies a analyser (None = tout)

        Returns:
            VolumeProfile complet
        """
        if period:
            df = df.tail(period).copy()

        # Determiner la plage de prix
        price_high = df['high'].max()
        price_low = df['low'].min()
        price_range = price_high - price_low

        if price_range == 0:
            return self._empty_profile()

        # Creer les niveaux de prix
        price_levels = np.linspace(price_low, price_high, self.num_levels)
        level_size = price_levels[1] - price_levels[0]

        # Calculer le volume a chaque niveau
        volume_at_price = {}
        for level in price_levels:
            # Volume des bougies qui traversent ce niveau
            mask = (df['low'] <= level + level_size) & (df['high'] >= level)
            vol = df.loc[mask, 'volume'].sum()
            volume_at_price[level] = vol

        # Trouver le POC
        poc = max(volume_at_price.keys(), key=lambda x: volume_at_price[x])

        # Calculer la Value Area (70% du volume)
        value_area = self._calculate_value_area(volume_at_price, poc, price_levels)

        # Identifier les HVN et LVN
        hvn_nodes, lvn_nodes = self._identify_nodes(volume_at_price, price_levels)

        # Determiner la forme du profil
        shape = self._determine_shape(volume_at_price, poc, value_area, price_high, price_low)

        return VolumeProfile(
            price_levels=list(price_levels),
            volume_at_price=volume_at_price,
            poc=poc,
            value_area=value_area,
            shape=shape,
            hvn_nodes=hvn_nodes,
            lvn_nodes=lvn_nodes
        )

    def _empty_profile(self) -> VolumeProfile:
        """Retourne un profil vide"""
        return VolumeProfile(
            price_levels=[],
            volume_at_price={},
            poc=0,
            value_area=ValueArea(0, 0, 0, 0, 0),
            shape=ProfileShape.UNKNOWN
        )

    def _calculate_value_area(
        self,
        volume_at_price: Dict[float, float],
        poc: float,
        price_levels: np.ndarray
    ) -> ValueArea:
        """
        Calcule la Value Area (zone avec 70% du volume)

        Methode:
        1. Commencer au POC
        2. Ajouter les niveaux adjacents jusqu'a atteindre 70%
        """
        total_volume = sum(volume_at_price.values())
        target_volume = total_volume * self.value_area_percent

        if total_volume == 0:
            return ValueArea(poc, poc, poc, 0, 0)

        # Trouver l'index du POC
        poc_idx = np.abs(price_levels - poc).argmin()

        # Etendre depuis le POC
        included_volume = volume_at_price.get(poc, 0)
        lower_idx = poc_idx
        upper_idx = poc_idx

        while included_volume < target_volume:
            # Comparer le volume au-dessus et en-dessous
            vol_above = volume_at_price.get(price_levels[upper_idx + 1], 0) if upper_idx + 1 < len(price_levels) else 0
            vol_below = volume_at_price.get(price_levels[lower_idx - 1], 0) if lower_idx > 0 else 0

            if vol_above >= vol_below and upper_idx + 1 < len(price_levels):
                upper_idx += 1
                included_volume += vol_above
            elif lower_idx > 0:
                lower_idx -= 1
                included_volume += vol_below
            else:
                break

        vah = price_levels[upper_idx]
        val = price_levels[lower_idx]
        width_percent = (vah - val) / poc * 100 if poc > 0 else 0

        return ValueArea(
            vah=vah,
            val=val,
            poc=poc,
            total_volume=total_volume,
            width_percent=width_percent
        )

    def _identify_nodes(
        self,
        volume_at_price: Dict[float, float],
        price_levels: np.ndarray
    ) -> Tuple[List[VolumeNode], List[VolumeNode]]:
        """
        Identifie les High Volume Nodes (HVN) et Low Volume Nodes (LVN)

        HVN: Volume > moyenne + 1 ecart-type
        LVN: Volume < moyenne - 0.5 ecart-type
        """
        volumes = list(volume_at_price.values())
        if not volumes:
            return [], []

        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)
        max_vol = max(volumes)

        hvn_nodes = []
        lvn_nodes = []

        for level in price_levels:
            vol = volume_at_price.get(level, 0)

            # HVN: Volume significativement au-dessus de la moyenne
            if vol > mean_vol + std_vol:
                strength = (vol - mean_vol) / (max_vol - mean_vol) if max_vol > mean_vol else 0.5
                hvn_nodes.append(VolumeNode(
                    price_level=level,
                    volume=vol,
                    node_type=NodeType.HVN,
                    strength=min(strength, 1.0)
                ))

            # LVN: Volume significativement en-dessous de la moyenne
            elif vol < mean_vol - 0.5 * std_vol:
                strength = 1 - (vol / mean_vol) if mean_vol > 0 else 0.5
                lvn_nodes.append(VolumeNode(
                    price_level=level,
                    volume=vol,
                    node_type=NodeType.LVN,
                    strength=min(strength, 1.0)
                ))

        return hvn_nodes, lvn_nodes

    def _determine_shape(
        self,
        volume_at_price: Dict[float, float],
        poc: float,
        value_area: ValueArea,
        price_high: float,
        price_low: float
    ) -> ProfileShape:
        """
        Determine la forme du profil

        Shapes:
        - D-Shape: Distribution normale, POC au milieu
        - P-Shape: Volume concentre en haut (accumulation)
        - b-Shape: Volume concentre en bas (distribution)
        - Double: Deux pics de volume
        - Thin: Volume faible partout
        """
        price_range = price_high - price_low
        if price_range == 0:
            return ProfileShape.UNKNOWN

        # Position relative du POC
        poc_position = (poc - price_low) / price_range

        # Calculer le skew du volume
        volumes = list(volume_at_price.values())
        if not volumes:
            return ProfileShape.UNKNOWN

        total_vol = sum(volumes)
        if total_vol == 0:
            return ProfileShape.THIN

        # Volume dans la moitie superieure vs inferieure
        mid_price = (price_high + price_low) / 2
        vol_upper = sum(v for p, v in volume_at_price.items() if p >= mid_price)
        vol_lower = sum(v for p, v in volume_at_price.items() if p < mid_price)

        vol_ratio = vol_upper / vol_lower if vol_lower > 0 else float('inf')

        # Detecter les pics multiples
        peak_count = self._count_peaks(volumes)

        # Determiner la forme
        if peak_count >= 2:
            return ProfileShape.DOUBLE

        if vol_ratio > 1.5 and poc_position > 0.6:
            return ProfileShape.P_SHAPE  # Volume en haut = accumulation

        if vol_ratio < 0.67 and poc_position < 0.4:
            return ProfileShape.B_SHAPE  # Volume en bas = distribution

        if 0.35 <= poc_position <= 0.65:
            return ProfileShape.D_SHAPE  # Distribution normale

        return ProfileShape.UNKNOWN

    def _count_peaks(self, volumes: List[float], min_distance: int = 5) -> int:
        """Compte le nombre de pics de volume"""
        if len(volumes) < 3:
            return 1

        mean_vol = np.mean(volumes)
        peaks = 0
        last_peak_idx = -min_distance

        for i in range(1, len(volumes) - 1):
            if volumes[i] > volumes[i-1] and volumes[i] > volumes[i+1]:
                if volumes[i] > mean_vol * 1.2:  # Pic significatif
                    if i - last_peak_idx >= min_distance:
                        peaks += 1
                        last_peak_idx = i

        return max(peaks, 1)

    # =========================================================================
    # ANALYSE PRINCIPALE
    # =========================================================================

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        period: int = 50
    ) -> Dict:
        """
        Analyse Volume Profile complete

        Returns:
            Dict avec profil, niveaux cles et signal
        """
        if len(df) < period:
            period = len(df)

        # Calculer le profil
        profile = self.calculate_profile(df, period)

        if profile.poc == 0:
            return {'profile': None, 'signal': None}

        # Analyser la position du prix par rapport au profil
        current_price = df['close'].iloc[-1]
        price_analysis = self._analyze_price_position(current_price, profile)

        # Generer signal
        signal = self._generate_signal(symbol, df, profile, price_analysis)

        return {
            'profile': profile,
            'price_analysis': price_analysis,
            'signal': signal
        }

    def _analyze_price_position(
        self,
        current_price: float,
        profile: VolumeProfile
    ) -> Dict:
        """Analyse la position du prix par rapport au profil"""
        va = profile.value_area

        # Position par rapport a la Value Area
        if current_price > va.vah:
            va_position = 'above'
        elif current_price < va.val:
            va_position = 'below'
        else:
            va_position = 'inside'

        # Distance au POC
        poc_distance = abs(current_price - profile.poc) / profile.poc * 100

        # Plus proche HVN et LVN
        nearest_hvn = None
        if profile.hvn_nodes:
            nearest_hvn = min(
                profile.hvn_nodes,
                key=lambda x: abs(x.price_level - current_price)
            )

        nearest_lvn = None
        if profile.lvn_nodes:
            nearest_lvn = min(
                profile.lvn_nodes,
                key=lambda x: abs(x.price_level - current_price)
            )

        return {
            'va_position': va_position,
            'poc_distance_percent': poc_distance,
            'near_poc': poc_distance < 1,  # < 1%
            'near_vah': abs(current_price - va.vah) / va.vah * 100 < 1,
            'near_val': abs(current_price - va.val) / va.val * 100 < 1,
            'nearest_hvn': nearest_hvn,
            'nearest_lvn': nearest_lvn
        }

    # =========================================================================
    # GENERATION DE SIGNAL
    # =========================================================================

    def _generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        profile: VolumeProfile,
        price_analysis: Dict
    ) -> Optional[VolumeProfileSignal]:
        """
        Genere un signal de trading base sur le Volume Profile

        Strategies:
        1. Rebond sur POC (support/resistance majeur)
        2. Rebond sur VAH/VAL (limites de la value area)
        3. Traversee de LVN (acceleration attendue)
        4. Rejet sur HVN (prix s'attarde, possible reversal)
        """
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        va = profile.value_area

        signal_type = None
        reasons = []
        key_level = None
        strength = 0.5

        # 1. Signal sur rebond POC
        if price_analysis['near_poc']:
            # Prix proche du POC - chercher une direction
            if prev_close < profile.poc and current_price > profile.poc:
                signal_type = 'buy'
                key_level = 'poc'
                reasons.append("Cassure haussiere du POC")
                reasons.append(f"POC @ {profile.poc:.2f} (niveau majeur)")
                strength = 0.8
            elif prev_close > profile.poc and current_price < profile.poc:
                signal_type = 'sell'
                key_level = 'poc'
                reasons.append("Cassure baissiere du POC")
                reasons.append(f"POC @ {profile.poc:.2f} (niveau majeur)")
                strength = 0.8

        # 2. Signal sur VAH/VAL
        elif price_analysis['near_vah'] and price_analysis['va_position'] == 'above':
            # Prix sort de la VA par le haut
            signal_type = 'buy'
            key_level = 'vah'
            reasons.append("Cassure de la Value Area High")
            reasons.append(f"VAH @ {va.vah:.2f}")
            strength = 0.75

        elif price_analysis['near_val'] and price_analysis['va_position'] == 'below':
            # Prix sort de la VA par le bas
            signal_type = 'sell'
            key_level = 'val'
            reasons.append("Cassure de la Value Area Low")
            reasons.append(f"VAL @ {va.val:.2f}")
            strength = 0.75

        # 3. Signal sur LVN (acceleration)
        elif price_analysis['nearest_lvn']:
            lvn = price_analysis['nearest_lvn']
            lvn_distance = abs(current_price - lvn.price_level) / current_price
            if lvn_distance < 0.01:  # < 1%
                # Direction basee sur la position par rapport au LVN
                if current_price > lvn.price_level and prev_close < lvn.price_level:
                    signal_type = 'buy'
                    key_level = 'lvn'
                    reasons.append("Traversee haussiere d'un Low Volume Node")
                    reasons.append("Acceleration attendue (peu de resistance)")
                    strength = 0.7
                elif current_price < lvn.price_level and prev_close > lvn.price_level:
                    signal_type = 'sell'
                    key_level = 'lvn'
                    reasons.append("Traversee baissiere d'un Low Volume Node")
                    reasons.append("Acceleration attendue (peu de support)")
                    strength = 0.7

        # 4. Signal base sur la forme du profil
        if signal_type is None and profile.shape in [ProfileShape.P_SHAPE, ProfileShape.B_SHAPE]:
            if profile.shape == ProfileShape.P_SHAPE and price_analysis['va_position'] == 'below':
                signal_type = 'buy'
                key_level = 'val'
                reasons.append("Profil P-Shape (accumulation en haut)")
                reasons.append("Prix sous la Value Area - potentiel retour au POC")
                strength = 0.65

            elif profile.shape == ProfileShape.B_SHAPE and price_analysis['va_position'] == 'above':
                signal_type = 'sell'
                key_level = 'vah'
                reasons.append("Profil b-Shape (distribution en bas)")
                reasons.append("Prix au-dessus de la Value Area - potentiel retour au POC")
                strength = 0.65

        if signal_type is None:
            return None

        # Calculer SL et TP
        if signal_type == 'buy':
            # SL sous le prochain support (VAL ou LVN)
            stop_loss = va.val - (va.vah - va.val) * 0.2
            # TP au prochain HVN ou 2x le risque
            if profile.hvn_nodes:
                higher_hvns = [h for h in profile.hvn_nodes if h.price_level > current_price]
                if higher_hvns:
                    take_profit = min(higher_hvns, key=lambda x: x.price_level).price_level
                else:
                    take_profit = current_price + (current_price - stop_loss) * 2
            else:
                take_profit = current_price + (current_price - stop_loss) * 2
        else:
            stop_loss = va.vah + (va.vah - va.val) * 0.2
            if profile.hvn_nodes:
                lower_hvns = [h for h in profile.hvn_nodes if h.price_level < current_price]
                if lower_hvns:
                    take_profit = max(lower_hvns, key=lambda x: x.price_level).price_level
                else:
                    take_profit = current_price - (stop_loss - current_price) * 2
            else:
                take_profit = current_price - (stop_loss - current_price) * 2

        return VolumeProfileSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            profile_shape=profile.shape,
            key_level=key_level,
            reasons=reasons
        )


# =============================================================================
# SINGLETON
# =============================================================================
_volume_profile_analyzer = None

def get_volume_profile_analyzer() -> VolumeProfileAnalyzer:
    """Retourne l'instance singleton"""
    global _volume_profile_analyzer
    if _volume_profile_analyzer is None:
        _volume_profile_analyzer = VolumeProfileAnalyzer()
    return _volume_profile_analyzer
