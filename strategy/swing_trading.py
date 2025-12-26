"""
Strategie Swing Trading Hybride
Base sur MASTER_TRADING_SKILL PARTIE XII

Implementation complete de la strategie:
1. Daily: Direction du marche + zones haute probabilite
2. H1: Precision de l'entree
3. Risk Management integre
4. Gestion des positions (TP1 + BE)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from config.settings import (
    MIN_RISK_REWARD, TAKE_PROFIT_CONFIG, INDICATORS
)
from config.symbols import get_sector
from data.fetcher import DataFetcher, get_fetcher
from analysis.signals import SignalGenerator, Signal, get_signal_generator
from analysis.indicators import get_indicators
from analysis.zones import get_zone_detector, find_swing_points, detect_breakout
from strategy.risk_management import RiskManager, get_risk_manager
from strategy.position_sizing import PositionSizer, get_position_sizer, calculate_risk_reward

logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    """Setup de trade complet"""
    symbol: str
    direction: str  # 'buy' ou 'sell'
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    take_profit_final: float
    position_size: int
    risk_reward: float
    signal_strength: float
    reasons: List[str] = field(default_factory=list)
    daily_confirmation: bool = False
    h1_confirmation: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.position_size

    @property
    def potential_profit(self) -> float:
        return abs(self.take_profit_final - self.entry_price) * self.position_size


class SwingTradingStrategy:
    """
    Strategie Swing Trading Hybride selon MASTER_TRADING_SKILL

    PARTIE XII - Sections 93-104:
    - Timeframe hybride Daily + H1
    - Zones haute probabilite
    - Cassures structurelles
    - R:R minimum 1:3
    - TP1 + Break-Even
    """

    def __init__(
        self,
        fetcher: DataFetcher = None,
        risk_manager: RiskManager = None,
        position_sizer: PositionSizer = None
    ):
        self.fetcher = fetcher or get_fetcher()
        self.risk_manager = risk_manager or get_risk_manager()
        self.position_sizer = position_sizer or get_position_sizer()
        self.signal_generator = get_signal_generator()
        self.indicators = get_indicators()
        self.zones = get_zone_detector()

        # Tracking
        self.active_setups: Dict[str, TradeSetup] = {}
        self.pending_entries: Dict[str, TradeSetup] = {}

    # =========================================================================
    # ANALYSE PRINCIPALE
    # =========================================================================

    def analyze_symbol(self, symbol: str) -> Optional[TradeSetup]:
        """
        Analyse complete d'un symbole selon la strategie

        Process (PARTIE XII):
        1. Recuperer donnees Daily et H1
        2. Analyser Daily pour direction + zones
        3. Confirmer sur H1
        4. Calculer entry, SL, TP
        5. Valider risk management
        6. Retourner setup si valide
        """
        logger.info(f"Analyzing {symbol}...")

        # 1. Recuperer donnees
        df_daily = self.fetcher.get_daily_data(symbol)
        df_h1 = self.fetcher.get_hourly_data(symbol)

        if df_daily is None or len(df_daily) < 20:
            logger.warning(f"Insufficient daily data for {symbol}")
            return None

        # 2. Analyse Daily
        daily_analysis = self._analyze_daily(symbol, df_daily)
        if not daily_analysis:
            return None

        # 3. Confirmation H1
        h1_confirmed = False
        if df_h1 is not None and len(df_h1) > 10:
            h1_confirmed = self._confirm_on_h1(df_h1, daily_analysis['direction'])

        # 4. Calculer setup complet
        setup = self._create_trade_setup(
            symbol=symbol,
            df_daily=df_daily,
            df_h1=df_h1,
            daily_analysis=daily_analysis,
            h1_confirmed=h1_confirmed
        )

        if not setup:
            return None

        # 5. Valider avec risk management
        sector = get_sector(symbol)
        can_trade, reason = self.risk_manager.can_open_trade(
            symbol=symbol,
            side=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit_final,
            quantity=setup.position_size,
            sector=sector
        )

        if not can_trade:
            logger.info(f"Trade blocked for {symbol}: {reason}")
            return None

        logger.info(f"Valid setup found for {symbol}: {setup.direction} @ {setup.entry_price}")
        return setup

    def _analyze_daily(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyse le timeframe Daily

        Determine:
        - Direction (cassure point haut/bas)
        - Zones haute probabilite
        - Signal de confirmation
        """
        # Ajouter indicateurs
        df = self.indicators.add_all_indicators(df)

        # Trouver swing points
        swing_highs, swing_lows = find_swing_points(df)

        # Detecter breakout
        breakout = detect_breakout(df, swing_highs, swing_lows)

        if not breakout:
            logger.debug(f"No breakout detected for {symbol}")
            return None

        # Trouver zones
        zones = self.zones.find_zones(df)
        valid_zones = self.zones.get_zones_for_setup(
            df=df,
            direction=breakout['direction'],
            breakout_point=breakout['breakout_price']
        )

        if not valid_zones:
            logger.debug(f"No valid zones for {symbol}")
            return None

        # Verifier prix dans zone ou proche
        current_price = df['close'].iloc[-1]
        zone_in = self.zones.is_price_in_zone(current_price, valid_zones)

        # Verifier bougie de rejection
        from analysis.patterns import get_pattern_detector
        patterns = get_pattern_detector()
        has_rejection = patterns.detect_rejection_candle(df, breakout['direction'])

        # Indicateurs
        latest = df.iloc[-1]
        rsi = latest.get('rsi', 50)
        trend = latest.get('trend_ema', 0)

        # Conditions selon direction
        if breakout['direction'] == 'buy':
            indicators_ok = rsi < INDICATORS['rsi_overbought'] and trend >= 0
        else:
            indicators_ok = rsi > INDICATORS['rsi_oversold'] and trend <= 0

        # Score du signal
        score = 0
        if breakout:
            score += 0.3
        if zone_in:
            score += 0.25
        if has_rejection:
            score += 0.2
        if indicators_ok:
            score += 0.15
        if valid_zones:
            score += 0.1

        if score < 0.6:
            return None

        return {
            'direction': breakout['direction'],
            'breakout_price': breakout['breakout_price'],
            'current_price': current_price,
            'zones': valid_zones,
            'zone_in': zone_in,
            'has_rejection': has_rejection,
            'rsi': rsi,
            'trend': trend,
            'atr': latest.get('atr', current_price * 0.02),
            'score': score
        }

    def _confirm_on_h1(self, df_h1: pd.DataFrame, daily_direction: str) -> bool:
        """
        Confirme le signal sur H1 (PARTIE XII - Section 97)

        Cherche:
        - Cassure dans meme direction que Daily
        - Retracement vers zone H1
        - Bougie de rejection H1
        """
        # Ajouter indicateurs
        df_h1 = self.indicators.add_all_indicators(df_h1)

        # Trouver swing points H1
        swing_highs, swing_lows = find_swing_points(df_h1, lookback=3)

        # Detecter breakout H1
        h1_breakout = detect_breakout(df_h1, swing_highs, swing_lows)

        if not h1_breakout:
            return False

        # Verifier meme direction
        if daily_direction == 'buy' and h1_breakout['type'] == 'bullish':
            return True
        if daily_direction == 'sell' and h1_breakout['type'] == 'bearish':
            return True

        return False

    def _create_trade_setup(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_h1: pd.DataFrame,
        daily_analysis: Dict,
        h1_confirmed: bool
    ) -> Optional[TradeSetup]:
        """
        Cree le setup de trade complet avec tous les niveaux
        """
        direction = daily_analysis['direction']
        current_price = daily_analysis['current_price']
        atr = daily_analysis['atr']
        zones = daily_analysis['zones']

        # Entry price (prix actuel ou ajuste avec H1)
        entry_price = current_price
        if h1_confirmed and df_h1 is not None:
            h1_price = df_h1['close'].iloc[-1]
            if direction == 'buy' and h1_price < current_price:
                entry_price = h1_price
            elif direction == 'sell' and h1_price > current_price:
                entry_price = h1_price

        # Stop Loss (ATR-based)
        sl_distance = atr * INDICATORS['atr_multiplier']
        if direction == 'buy':
            stop_loss = entry_price - sl_distance
            # Ajuster sous la zone si possible
            if daily_analysis['zone_in']:
                zone_bottom = daily_analysis['zone_in'].price_low
                stop_loss = min(stop_loss, zone_bottom - (atr * 0.5))
        else:
            stop_loss = entry_price + sl_distance
            if daily_analysis['zone_in']:
                zone_top = daily_analysis['zone_in'].price_high
                stop_loss = max(stop_loss, zone_top + (atr * 0.5))

        # Take Profits
        risk = abs(entry_price - stop_loss)

        if direction == 'buy':
            # TP1: Premiere zone au-dessus ou 2R
            tp1_zone = self.zones.get_nearest_zone(zones, entry_price, 'above')
            take_profit_1 = tp1_zone.midpoint if tp1_zone else entry_price + (risk * 2)

            # TP Final: 3R minimum ou zone majeure
            take_profit_final = entry_price + (risk * MIN_RISK_REWARD)
            if tp1_zone:
                # Chercher zone plus haute
                higher_zones = [z for z in zones if z.midpoint > tp1_zone.midpoint]
                if higher_zones:
                    take_profit_final = max(higher_zones, key=lambda z: z.midpoint).midpoint
        else:
            tp1_zone = self.zones.get_nearest_zone(zones, entry_price, 'below')
            take_profit_1 = tp1_zone.midpoint if tp1_zone else entry_price - (risk * 2)

            take_profit_final = entry_price - (risk * MIN_RISK_REWARD)
            if tp1_zone:
                lower_zones = [z for z in zones if z.midpoint < tp1_zone.midpoint]
                if lower_zones:
                    take_profit_final = min(lower_zones, key=lambda z: z.midpoint).midpoint

        # TP2 (intermediaire)
        take_profit_2 = (take_profit_1 + take_profit_final) / 2

        # Calculer R:R
        reward = abs(take_profit_final - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < MIN_RISK_REWARD:
            logger.debug(f"R:R too low for {symbol}: {risk_reward:.2f}")
            return None

        # Position size
        sizing = self.position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        position_size = sizing['shares']

        if position_size <= 0:
            logger.debug(f"Position size too small for {symbol}")
            return None

        # Construire les raisons
        reasons = []
        if daily_analysis.get('breakout_price'):
            reasons.append(f"Cassure @ {daily_analysis['breakout_price']:.2f}")
        if daily_analysis.get('zone_in'):
            reasons.append(f"Dans zone {daily_analysis['zone_in'].zone_type}")
        if daily_analysis.get('has_rejection'):
            reasons.append("Bougie rejection")
        if h1_confirmed:
            reasons.append("Confirme H1")

        return TradeSetup(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_final=take_profit_final,
            position_size=position_size,
            risk_reward=risk_reward,
            signal_strength=daily_analysis['score'],
            reasons=reasons,
            daily_confirmation=True,
            h1_confirmation=h1_confirmed
        )

    # =========================================================================
    # SCAN MULTIPLE SYMBOLES
    # =========================================================================

    def scan_watchlist(self, symbols: List[str]) -> List[TradeSetup]:
        """
        Scanne une liste de symboles pour trouver des setups

        Returns:
            Liste de setups tries par force du signal
        """
        setups = []

        for symbol in symbols:
            try:
                setup = self.analyze_symbol(symbol)
                if setup:
                    setups.append(setup)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        # Trier par force du signal
        setups.sort(key=lambda x: x.signal_strength, reverse=True)

        logger.info(f"Found {len(setups)} valid setups out of {len(symbols)} symbols")
        return setups

    # =========================================================================
    # GESTION DES POSITIONS
    # =========================================================================

    def manage_position(
        self,
        symbol: str,
        current_price: float,
        position: Dict
    ) -> Optional[Dict]:
        """
        Gere une position existante

        Actions:
        - Verifier SL/TP
        - Passer en BE apres TP1
        - Sortie partielle
        """
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        side = position['side']

        # Recuperer setup original si existe
        setup = self.active_setups.get(symbol)
        if not setup:
            return None

        actions = {
            'action': None,
            'reason': None,
            'new_stop': None,
            'exit_percent': None
        }

        # Verifier Stop Loss
        if side == 'long' and current_price <= stop_loss:
            actions['action'] = 'close'
            actions['reason'] = 'Stop Loss'
            return actions

        if side == 'short' and current_price >= stop_loss:
            actions['action'] = 'close'
            actions['reason'] = 'Stop Loss'
            return actions

        # Verifier TP1
        if side == 'long' and current_price >= setup.take_profit_1:
            if position.get('tp1_hit', False):
                # Deja passe TP1, verifier TP final
                if current_price >= setup.take_profit_final:
                    actions['action'] = 'close'
                    actions['reason'] = 'Take Profit Final'
            else:
                # Premier TP1
                actions['action'] = 'partial_exit'
                actions['exit_percent'] = TAKE_PROFIT_CONFIG['tp1_percent']
                actions['reason'] = 'TP1 atteint'
                actions['new_stop'] = entry_price  # Move to BE
                position['tp1_hit'] = True

        if side == 'short' and current_price <= setup.take_profit_1:
            if position.get('tp1_hit', False):
                if current_price <= setup.take_profit_final:
                    actions['action'] = 'close'
                    actions['reason'] = 'Take Profit Final'
            else:
                actions['action'] = 'partial_exit'
                actions['exit_percent'] = TAKE_PROFIT_CONFIG['tp1_percent']
                actions['reason'] = 'TP1 atteint'
                actions['new_stop'] = entry_price
                position['tp1_hit'] = True

        return actions if actions['action'] else None

    # =========================================================================
    # HELPERS
    # =========================================================================

    def add_active_setup(self, symbol: str, setup: TradeSetup):
        """Ajoute un setup aux setups actifs"""
        self.active_setups[symbol] = setup

    def remove_active_setup(self, symbol: str):
        """Retire un setup"""
        if symbol in self.active_setups:
            del self.active_setups[symbol]

    def get_daily_summary(self) -> Dict:
        """Resume quotidien"""
        metrics = self.risk_manager.get_risk_metrics()
        return {
            'capital': metrics.current_capital,
            'daily_pnl': metrics.daily_pnl,
            'daily_pnl_percent': metrics.daily_pnl_percent,
            'open_positions': metrics.open_positions,
            'trades_today': metrics.trades_today,
            'can_trade': metrics.can_trade,
            'active_setups': len(self.active_setups)
        }


# =============================================================================
# SINGLETON
# =============================================================================
_strategy = None

def get_swing_strategy() -> SwingTradingStrategy:
    """Retourne l'instance singleton"""
    global _strategy
    if _strategy is None:
        _strategy = SwingTradingStrategy()
    return _strategy
