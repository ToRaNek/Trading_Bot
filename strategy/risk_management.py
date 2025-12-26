"""
Risk Management
Base sur MASTER_TRADING_SKILL PARTIE XI

Gestion du risque selon les principes:
- Max 2% par trade
- Daily max loss 5%
- Position limits
- Diversification
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from config.settings import (
    RISK_PER_TRADE, MIN_RISK_REWARD, DAILY_MAX_LOSS,
    MAX_OPEN_POSITIONS, MAX_POSITION_PERCENT, MAX_SECTOR_EXPOSURE,
    MAX_TRADES_PER_DAY, MAX_CONSECUTIVE_LOSSES, COOLDOWN_AFTER_LOSSES_MINUTES,
    DRAWDOWN_STOP_TRADING, INITIAL_CAPITAL
)

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Metriques de risque actuelles"""
    current_capital: float
    daily_pnl: float
    daily_pnl_percent: float
    open_positions: int
    total_exposure: float
    exposure_percent: float
    sector_exposures: Dict[str, float]
    trades_today: int
    consecutive_losses: int
    max_drawdown: float
    current_drawdown: float
    can_trade: bool
    block_reason: Optional[str] = None


class RiskManager:
    """
    Gestionnaire de risque selon MASTER_TRADING_SKILL

    PARTIE XI - Risk Management Avance:
    - Position limits
    - Daily loss limits
    - Diversification
    - VAR simplifie

    PARTIE X - Psychologie:
    - Max trades par jour (anti-CATS)
    - Cooldown apres pertes
    - Stop si drawdown important
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital

        # Daily tracking
        self.daily_start_capital = initial_capital
        self.daily_pnl = 0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.cooldown_until = None

        # Position tracking
        self.open_positions: Dict[str, Dict] = {}
        self.sector_exposure: Dict[str, float] = {}

    # =========================================================================
    # VALIDATION DES TRADES
    # =========================================================================

    def can_open_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        quantity: float,
        sector: str = "unknown"
    ) -> Tuple[bool, str]:
        """
        Verifie si un trade peut etre ouvert selon les regles de risk management

        Returns:
            (can_trade, reason)
        """
        # 1. Verifier cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds // 60
            return False, f"En cooldown ({remaining} min restantes)"

        # 2. Verifier daily max loss
        daily_loss_percent = abs(self.daily_pnl / self.daily_start_capital) if self.daily_pnl < 0 else 0
        if daily_loss_percent >= DAILY_MAX_LOSS:
            return False, f"Daily max loss atteint ({daily_loss_percent:.1%})"

        # 3. Verifier drawdown global
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= DRAWDOWN_STOP_TRADING:
            return False, f"Drawdown trop important ({current_drawdown:.1%})"

        # 4. Verifier max positions
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"Max positions atteint ({MAX_OPEN_POSITIONS})"

        # 5. Verifier si deja en position sur ce symbole
        if symbol in self.open_positions:
            return False, f"Position deja ouverte sur {symbol}"

        # 6. Verifier max trades par jour (anti-CATS)
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Max trades jour atteint ({MAX_TRADES_PER_DAY})"

        # 7. Verifier consecutive losses
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._start_cooldown()
            return False, f"Cooldown apres {MAX_CONSECUTIVE_LOSSES} pertes consecutives"

        # 8. Verifier risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < MIN_RISK_REWARD:
            return False, f"R:R insuffisant ({risk_reward:.2f} < {MIN_RISK_REWARD})"

        # 9. Verifier taille de position
        position_value = entry_price * quantity
        position_percent = position_value / self.current_capital

        if position_percent > MAX_POSITION_PERCENT:
            return False, f"Position trop grande ({position_percent:.1%} > {MAX_POSITION_PERCENT:.1%})"

        # 10. Verifier exposition sectorielle
        current_sector_exposure = self.sector_exposure.get(sector, 0)
        new_sector_exposure = current_sector_exposure + position_value

        if new_sector_exposure / self.current_capital > MAX_SECTOR_EXPOSURE:
            return False, f"Exposition secteur {sector} trop elevee"

        # 11. Verifier risque par trade
        risk_amount = risk * quantity
        risk_percent = risk_amount / self.current_capital

        if risk_percent > RISK_PER_TRADE:
            return False, f"Risque trop eleve ({risk_percent:.1%} > {RISK_PER_TRADE:.1%})"

        return True, "OK"

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = None
    ) -> int:
        """
        Calcule la taille de position optimale

        Formule: position_size = (capital * risk%) / (entry - stop)

        PARTIE XI - Section 87 (Position Sizing)
        """
        if risk_percent is None:
            risk_percent = RISK_PER_TRADE

        risk_amount = self.current_capital * risk_percent
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return 0

        position_size = risk_amount / risk_per_share

        # Arrondir vers le bas
        position_size = int(position_size)

        # Verifier max position
        max_position_value = self.current_capital * MAX_POSITION_PERCENT
        max_shares = int(max_position_value / entry_price)

        return min(position_size, max_shares)

    # =========================================================================
    # GESTION DES POSITIONS
    # =========================================================================

    def register_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        sector: str = "unknown"
    ):
        """Enregistre une nouvelle position"""
        position_value = entry_price * quantity

        self.open_positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sector': sector,
            'value': position_value,
            'entry_time': datetime.now()
        }

        # Mettre a jour exposition sectorielle
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_value

        self.trades_today += 1
        self.last_trade_time = datetime.now()

        logger.info(f"Position registered: {side} {quantity} {symbol} @ {entry_price}")

    def close_position(self, symbol: str, exit_price: float) -> Dict:
        """
        Ferme une position et calcule le P&L

        Returns:
            Dict avec pnl, pnl_percent, etc.
        """
        if symbol not in self.open_positions:
            raise ValueError(f"No position for {symbol}")

        position = self.open_positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        sector = position['sector']

        # Calculer P&L
        if side == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        pnl_percent = pnl / (entry_price * quantity) * 100

        # Mettre a jour capital
        self.current_capital += pnl
        self.daily_pnl += pnl

        # Mettre a jour peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Mettre a jour consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Mettre a jour exposition sectorielle
        position_value = position['value']
        self.sector_exposure[sector] = max(0, self.sector_exposure.get(sector, 0) - position_value)

        # Supprimer position
        del self.open_positions[symbol]

        result = {
            'symbol': symbol,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'side': side
        }

        logger.info(f"Position closed: {symbol} P&L: {pnl:.2f} ({pnl_percent:.2f}%)")

        return result

    def update_stop_loss(self, symbol: str, new_stop_loss: float):
        """
        Met a jour le stop loss (pour trailing stop ou break-even)
        """
        if symbol in self.open_positions:
            self.open_positions[symbol]['stop_loss'] = new_stop_loss
            logger.info(f"Stop loss updated for {symbol}: {new_stop_loss}")

    def move_to_breakeven(self, symbol: str):
        """
        Deplace le stop loss au break-even (PARTIE XII - TP1 + BE)
        """
        if symbol in self.open_positions:
            entry_price = self.open_positions[symbol]['entry_price']
            self.update_stop_loss(symbol, entry_price)
            logger.info(f"Moved to break-even: {symbol}")

    # =========================================================================
    # METRIQUES
    # =========================================================================

    def get_risk_metrics(self) -> RiskMetrics:
        """
        Retourne les metriques de risque actuelles
        """
        # Calculer exposition totale
        total_exposure = sum(p['value'] for p in self.open_positions.values())
        exposure_percent = total_exposure / self.current_capital if self.current_capital > 0 else 0

        # Calculer drawdown
        max_drawdown = (self.peak_capital - min(self.current_capital, self.peak_capital)) / self.peak_capital
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0

        # Verifier si on peut trader
        can_trade = True
        block_reason = None

        daily_loss_percent = abs(self.daily_pnl / self.daily_start_capital) if self.daily_pnl < 0 else 0

        if daily_loss_percent >= DAILY_MAX_LOSS:
            can_trade = False
            block_reason = "Daily max loss atteint"
        elif current_drawdown >= DRAWDOWN_STOP_TRADING:
            can_trade = False
            block_reason = "Drawdown trop important"
        elif len(self.open_positions) >= MAX_OPEN_POSITIONS:
            can_trade = False
            block_reason = "Max positions atteint"
        elif self.trades_today >= MAX_TRADES_PER_DAY:
            can_trade = False
            block_reason = "Max trades jour atteint"
        elif self.cooldown_until and datetime.now() < self.cooldown_until:
            can_trade = False
            block_reason = "En cooldown"

        return RiskMetrics(
            current_capital=self.current_capital,
            daily_pnl=self.daily_pnl,
            daily_pnl_percent=self.daily_pnl / self.daily_start_capital * 100 if self.daily_start_capital > 0 else 0,
            open_positions=len(self.open_positions),
            total_exposure=total_exposure,
            exposure_percent=exposure_percent,
            sector_exposures=self.sector_exposure.copy(),
            trades_today=self.trades_today,
            consecutive_losses=self.consecutive_losses,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            can_trade=can_trade,
            block_reason=block_reason
        )

    def get_var(self, confidence: float = 0.95) -> float:
        """
        Calcule un VAR simplifie base sur les positions actuelles

        PARTIE XI - Section 87 (VAR)
        """
        if not self.open_positions:
            return 0

        # VAR simplifie: somme des risques max de chaque position
        total_var = 0

        for symbol, position in self.open_positions.items():
            entry = position['entry_price']
            stop = position['stop_loss']
            quantity = position['quantity']

            max_loss = abs(entry - stop) * quantity
            total_var += max_loss

        return total_var

    # =========================================================================
    # GESTION QUOTIDIENNE
    # =========================================================================

    def reset_daily(self):
        """Reset les compteurs quotidiens (a appeler chaque jour)"""
        self.daily_start_capital = self.current_capital
        self.daily_pnl = 0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_until = None

        logger.info(f"Daily reset. Capital: {self.current_capital:.2f}")

    def _start_cooldown(self):
        """Demarre un cooldown apres pertes consecutives"""
        self.cooldown_until = datetime.now() + timedelta(minutes=COOLDOWN_AFTER_LOSSES_MINUTES)
        logger.warning(f"Cooldown started until {self.cooldown_until}")

    def check_positions_for_stops(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Verifie si des positions ont atteint leur stop loss

        Returns:
            Liste des symboles a fermer
        """
        to_close = []

        for symbol, position in self.open_positions.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            stop = position['stop_loss']
            side = position['side']

            if side == 'long' and price <= stop:
                to_close.append(symbol)
                logger.warning(f"Stop loss hit for {symbol} (long): {price} <= {stop}")

            elif side == 'short' and price >= stop:
                to_close.append(symbol)
                logger.warning(f"Stop loss hit for {symbol} (short): {price} >= {stop}")

        return to_close

    def check_positions_for_tp(self, current_prices: Dict[str, float]) -> List[Tuple[str, str]]:
        """
        Verifie si des positions ont atteint leur take profit

        Returns:
            Liste de tuples (symbol, 'tp1' ou 'full')
        """
        hits = []

        for symbol, position in self.open_positions.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            tp = position['take_profit']
            side = position['side']

            if side == 'long' and price >= tp:
                hits.append((symbol, 'full'))
                logger.info(f"Take profit hit for {symbol} (long): {price} >= {tp}")

            elif side == 'short' and price <= tp:
                hits.append((symbol, 'full'))
                logger.info(f"Take profit hit for {symbol} (short): {price} <= {tp}")

        return hits


# =============================================================================
# SINGLETON
# =============================================================================
_risk_manager = None

def get_risk_manager() -> RiskManager:
    """Retourne l'instance singleton"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
