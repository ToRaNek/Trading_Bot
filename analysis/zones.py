"""
Zones de Haute Probabilite
Base sur MASTER_TRADING_SKILL PARTIE XII - Section 96
Detection automatique des zones de support/resistance
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from config.settings import ZONES_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Represente une zone de haute probabilite"""
    price_low: float
    price_high: float
    touches: int
    strength: float
    zone_type: str  # 'support', 'resistance', 'both'
    first_touch: datetime
    last_touch: datetime
    is_valid: bool = True

    @property
    def midpoint(self) -> float:
        return (self.price_low + self.price_high) / 2

    @property
    def width(self) -> float:
        return self.price_high - self.price_low

    def contains_price(self, price: float, tolerance: float = 0.005) -> bool:
        """Verifie si un prix est dans la zone"""
        extended_low = self.price_low * (1 - tolerance)
        extended_high = self.price_high * (1 + tolerance)
        return extended_low <= price <= extended_high


class ZoneDetector:
    """
    Detecte les zones de haute probabilite selon le MASTER_TRADING_SKILL

    Criteres (Section 96):
    1. Minimum 2-3 points de contact
    2. Recence (2-3 derniers mois)
    3. Clarte (zone visible)
    """

    def __init__(self, config: Dict = None):
        self.config = config or ZONES_CONFIG
        self.min_touches = self.config.get('min_touches', 2)
        self.zone_threshold = self.config.get('zone_threshold', 0.02)
        self.max_age_days = self.config.get('max_age_days', 90)
        self.min_strength = self.config.get('min_zone_strength', 0.6)

    def find_zones(self, df: pd.DataFrame) -> List[Zone]:
        """
        Trouve toutes les zones de haute probabilite

        Args:
            df: DataFrame avec OHLC data

        Returns:
            Liste de zones triees par force
        """
        if df.empty or len(df) < 20:
            return []

        # 1. Trouver les pivots (points hauts et bas locaux)
        pivots = self._find_pivots(df)

        # 2. Regrouper les pivots en zones
        zones = self._cluster_pivots(pivots, df)

        # 3. Filtrer par criteres
        zones = self._filter_zones(zones, df)

        # 4. Calculer la force de chaque zone
        zones = self._calculate_zone_strength(zones, df)

        # 5. Trier par force
        zones.sort(key=lambda z: z.strength, reverse=True)

        logger.info(f"Found {len(zones)} high probability zones")
        return zones

    def _find_pivots(self, df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """
        Trouve les points pivots (hauts et bas locaux)
        """
        pivots = []

        highs = df['high'].values
        lows = df['low'].values
        dates = df.index

        for i in range(window, len(df) - window):
            # Pivot High
            if highs[i] == max(highs[i-window:i+window+1]):
                pivots.append({
                    'price': highs[i],
                    'type': 'high',
                    'date': dates[i],
                    'index': i
                })

            # Pivot Low
            if lows[i] == min(lows[i-window:i+window+1]):
                pivots.append({
                    'price': lows[i],
                    'type': 'low',
                    'date': dates[i],
                    'index': i
                })

        return pivots

    def _cluster_pivots(
        self,
        pivots: List[Dict],
        df: pd.DataFrame
    ) -> List[Zone]:
        """
        Regroupe les pivots proches en zones
        """
        if not pivots:
            return []

        # Trier par prix
        pivots.sort(key=lambda x: x['price'])

        zones = []
        current_cluster = [pivots[0]]

        avg_price = df['close'].mean()
        threshold = avg_price * self.zone_threshold

        for pivot in pivots[1:]:
            # Si proche du cluster actuel, ajouter
            cluster_avg = np.mean([p['price'] for p in current_cluster])
            if abs(pivot['price'] - cluster_avg) <= threshold:
                current_cluster.append(pivot)
            else:
                # Creer zone du cluster actuel si assez de touches
                if len(current_cluster) >= self.min_touches:
                    zone = self._create_zone_from_cluster(current_cluster)
                    if zone:
                        zones.append(zone)
                # Commencer nouveau cluster
                current_cluster = [pivot]

        # Dernier cluster
        if len(current_cluster) >= self.min_touches:
            zone = self._create_zone_from_cluster(current_cluster)
            if zone:
                zones.append(zone)

        return zones

    def _create_zone_from_cluster(self, cluster: List[Dict]) -> Optional[Zone]:
        """
        Cree une zone a partir d'un cluster de pivots
        """
        prices = [p['price'] for p in cluster]
        dates = [p['date'] for p in cluster]
        types = [p['type'] for p in cluster]

        price_low = min(prices)
        price_high = max(prices)

        # Determiner le type de zone
        high_count = types.count('high')
        low_count = types.count('low')

        if high_count > low_count * 2:
            zone_type = 'resistance'
        elif low_count > high_count * 2:
            zone_type = 'support'
        else:
            zone_type = 'both'

        return Zone(
            price_low=price_low,
            price_high=price_high,
            touches=len(cluster),
            strength=0,  # Sera calcule apres
            zone_type=zone_type,
            first_touch=min(dates),
            last_touch=max(dates)
        )

    def _filter_zones(self, zones: List[Zone], df: pd.DataFrame) -> List[Zone]:
        """
        Filtre les zones selon les criteres du MASTER_TRADING_SKILL
        """
        current_date = df.index[-1]
        current_price = df['close'].iloc[-1]
        filtered = []

        for zone in zones:
            # 1. Verifier la recence (max 2-3 mois)
            try:
                age = (current_date - zone.last_touch).days
                if age > self.max_age_days:
                    zone.is_valid = False
                    continue
            except:
                pass

            # 2. Verifier le nombre de touches
            if zone.touches < self.min_touches:
                zone.is_valid = False
                continue

            # 3. Zone pas trop large (max 5% du prix)
            if zone.width / zone.midpoint > 0.05:
                zone.is_valid = False
                continue

            # 4. Zone pas trop loin du prix actuel (max 20%)
            distance = abs(zone.midpoint - current_price) / current_price
            if distance > 0.20:
                zone.is_valid = False
                continue

            filtered.append(zone)

        return filtered

    def _calculate_zone_strength(
        self,
        zones: List[Zone],
        df: pd.DataFrame
    ) -> List[Zone]:
        """
        Calcule la force de chaque zone (0-1)

        Facteurs:
        - Nombre de touches
        - Recence
        - Volume aux touches
        - Reactions du prix (rejections)
        """
        current_date = df.index[-1]

        for zone in zones:
            score = 0

            # 1. Score touches (max 0.3)
            touch_score = min(zone.touches / 5, 1) * 0.3
            score += touch_score

            # 2. Score recence (max 0.2)
            try:
                age = (current_date - zone.last_touch).days
                recency_score = max(0, 1 - age / self.max_age_days) * 0.2
                score += recency_score
            except:
                score += 0.1

            # 3. Score type (max 0.2)
            if zone.zone_type == 'both':
                score += 0.2  # Zones mixtes sont plus fortes
            else:
                score += 0.1

            # 4. Score precision (max 0.15)
            width_ratio = zone.width / zone.midpoint
            precision_score = max(0, 1 - width_ratio / 0.05) * 0.15
            score += precision_score

            # 5. Score position relative (max 0.15)
            current_price = df['close'].iloc[-1]
            distance = abs(zone.midpoint - current_price) / current_price
            proximity_score = max(0, 1 - distance / 0.10) * 0.15
            score += proximity_score

            zone.strength = min(score, 1.0)

        return zones

    def get_nearest_zone(
        self,
        zones: List[Zone],
        current_price: float,
        direction: str = 'any'
    ) -> Optional[Zone]:
        """
        Trouve la zone la plus proche

        Args:
            zones: Liste des zones
            current_price: Prix actuel
            direction: 'above', 'below', ou 'any'
        """
        valid_zones = []

        for zone in zones:
            if not zone.is_valid:
                continue

            if direction == 'above' and zone.midpoint <= current_price:
                continue
            if direction == 'below' and zone.midpoint >= current_price:
                continue

            valid_zones.append(zone)

        if not valid_zones:
            return None

        # Trier par distance
        valid_zones.sort(key=lambda z: abs(z.midpoint - current_price))
        return valid_zones[0]

    def is_price_in_zone(self, price: float, zones: List[Zone]) -> Optional[Zone]:
        """
        Verifie si le prix est dans une zone
        """
        for zone in zones:
            if zone.is_valid and zone.contains_price(price):
                return zone
        return None

    def get_zones_for_setup(
        self,
        df: pd.DataFrame,
        direction: str,
        breakout_point: float = None
    ) -> List[Zone]:
        """
        Retourne les zones valides pour un setup

        REGLE CRITIQUE (Section 96.3):
        - Setup ACHAT: zones au-dessus du dernier point bas casse
        - Setup VENTE: zones en-dessous du dernier point haut

        Args:
            df: DataFrame avec prix
            direction: 'buy' ou 'sell'
            breakout_point: Point de cassure (si connu)

        Returns:
            Zones valides pour le setup
        """
        zones = self.find_zones(df)

        if not zones:
            return []

        current_price = df['close'].iloc[-1]
        valid_zones = []

        for zone in zones:
            if not zone.is_valid or zone.strength < self.min_strength:
                continue

            if direction == 'buy':
                # Zone doit etre en dessous du prix actuel (pour retracement)
                # Mais au-dessus du breakout point (si defini)
                if zone.midpoint > current_price:
                    continue
                if breakout_point and zone.midpoint < breakout_point:
                    continue
                valid_zones.append(zone)

            elif direction == 'sell':
                # Zone doit etre au-dessus du prix actuel
                # Mais en-dessous du breakout point
                if zone.midpoint < current_price:
                    continue
                if breakout_point and zone.midpoint > breakout_point:
                    continue
                valid_zones.append(zone)

        return valid_zones


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Tuple[List, List]:
    """
    Trouve les swing highs et swing lows

    Returns:
        (swing_highs, swing_lows) avec index et prix
    """
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        # Swing High
        if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
            swing_highs.append({
                'index': i,
                'price': df['high'].iloc[i],
                'date': df.index[i]
            })

        # Swing Low
        if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback+1].min():
            swing_lows.append({
                'index': i,
                'price': df['low'].iloc[i],
                'date': df.index[i]
            })

    return swing_highs, swing_lows


def detect_breakout(
    df: pd.DataFrame,
    swing_highs: List,
    swing_lows: List
) -> Optional[Dict]:
    """
    Detecte une cassure de structure (PARTIE XII - Section 95)

    Returns:
        Dict avec type ('bullish' ou 'bearish'), prix casse, etc.
    """
    if len(df) < 5 or not swing_highs or not swing_lows:
        return None

    current_price = df['close'].iloc[-1]
    recent_high = max(swing_highs[-3:], key=lambda x: x['price'])['price'] if len(swing_highs) >= 3 else None
    recent_low = min(swing_lows[-3:], key=lambda x: x['price'])['price'] if len(swing_lows) >= 3 else None

    # Breakout haussier: prix casse le recent swing high
    if recent_high and current_price > recent_high:
        return {
            'type': 'bullish',
            'breakout_price': recent_high,
            'current_price': current_price,
            'direction': 'buy'
        }

    # Breakout baissier: prix casse le recent swing low
    if recent_low and current_price < recent_low:
        return {
            'type': 'bearish',
            'breakout_price': recent_low,
            'current_price': current_price,
            'direction': 'sell'
        }

    return None


# =============================================================================
# SINGLETON
# =============================================================================
_zone_detector = None

def get_zone_detector() -> ZoneDetector:
    """Retourne l'instance singleton"""
    global _zone_detector
    if _zone_detector is None:
        _zone_detector = ZoneDetector()
    return _zone_detector
