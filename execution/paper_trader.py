"""
Paper Trading - Simulation sans argent reel
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging

from config.settings import INITIAL_CAPITAL, TAKE_PROFIT_CONFIG
from data.database import get_database
from execution.orders import Order, OrderManager, get_order_manager, OrderStatus
from strategy.risk_management import RiskManager, get_risk_manager
from strategy.swing_trading import TradeSetup

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Position en paper trading"""
    symbol: str
    side: str  # 'long' ou 'short'
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_time: datetime = field(default_factory=datetime.now)

    # TP management
    tp1_hit: bool = False
    current_stop: float = None
    partial_exits: List[Dict] = field(default_factory=list)

    @property
    def current_quantity(self) -> int:
        exited = sum(e['quantity'] for e in self.partial_exits)
        return self.quantity - exited


class PaperTrader:
    """
    Simulateur de trading (Paper Trading)

    Simule l'execution des ordres et la gestion des positions
    sans utiliser d'argent reel.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float = 0.001  # 0.1%
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission

        self.positions: Dict[str, PaperPosition] = {}
        self.trade_history: List[Dict] = []
        self.order_manager = get_order_manager()
        self.risk_manager = get_risk_manager()
        self.db = get_database()

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def execute_buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: float,
        take_profit: float,
        setup: TradeSetup = None
    ) -> Dict:
        """
        Execute un achat en paper trading
        """
        # Commission
        commission = price * quantity * self.commission
        total_cost = (price * quantity) + commission

        if total_cost > self.capital:
            logger.warning(f"Insufficient capital for {symbol}")
            return {'success': False, 'reason': 'Insufficient capital'}

        # Deduire du capital
        self.capital -= total_cost

        # Creer position
        position = PaperPosition(
            symbol=symbol,
            side='long',
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_stop=stop_loss
        )
        self.positions[symbol] = position

        # Enregistrer dans DB
        trade_id = self.db.add_trade(
            symbol=symbol,
            side='buy',
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy='swing_trading',
            signal_strength=setup.signal_strength if setup else None
        )

        # Enregistrer position
        self.db.update_position(
            symbol=symbol,
            side='long',
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Risk manager
        from config.symbols import get_sector
        sector = get_sector(symbol)
        self.risk_manager.register_position(
            symbol=symbol,
            side='long',
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sector=sector
        )

        logger.info(f"PAPER BUY: {quantity} {symbol} @ {price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        return {
            'success': True,
            'trade_id': trade_id,
            'symbol': symbol,
            'side': 'buy',
            'quantity': quantity,
            'price': price,
            'commission': commission
        }

    def execute_sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: float,
        take_profit: float,
        setup: TradeSetup = None
    ) -> Dict:
        """
        Execute une vente a decouvert en paper trading
        """
        commission = price * quantity * self.commission
        margin_required = price * quantity * 0.5  # 50% margin

        if margin_required > self.capital:
            logger.warning(f"Insufficient margin for {symbol}")
            return {'success': False, 'reason': 'Insufficient margin'}

        # Creer position short
        position = PaperPosition(
            symbol=symbol,
            side='short',
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_stop=stop_loss
        )
        self.positions[symbol] = position

        # Reserve margin
        self.capital -= margin_required

        # Enregistrer
        trade_id = self.db.add_trade(
            symbol=symbol,
            side='sell',
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy='swing_trading'
        )

        self.db.update_position(
            symbol=symbol,
            side='short',
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        from config.symbols import get_sector
        self.risk_manager.register_position(
            symbol=symbol,
            side='short',
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sector=get_sector(symbol)
        )

        logger.info(f"PAPER SELL: {quantity} {symbol} @ {price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        return {
            'success': True,
            'trade_id': trade_id,
            'symbol': symbol,
            'side': 'sell',
            'quantity': quantity,
            'price': price,
            'commission': commission
        }

    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "manual"
    ) -> Dict:
        """
        Ferme une position
        """
        if symbol not in self.positions:
            return {'success': False, 'reason': 'No position'}

        position = self.positions[symbol]
        quantity = position.current_quantity
        commission = price * quantity * self.commission

        # Calculer P&L
        if position.side == 'long':
            pnl = (price - position.entry_price) * quantity - commission
            self.capital += (price * quantity) - commission
        else:
            pnl = (position.entry_price - price) * quantity - commission
            # Liberer margin + profit/perte
            margin_returned = position.entry_price * position.quantity * 0.5
            self.capital += margin_returned + pnl

        pnl_percent = (pnl / (position.entry_price * position.quantity)) * 100

        # Stats
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Historique
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'reason': reason,
            'entry_time': position.entry_time,
            'exit_time': datetime.now()
        }
        self.trade_history.append(trade_record)

        # DB
        self.risk_manager.close_position(symbol, price)
        self.db.remove_position(symbol)

        # Supprimer position
        del self.positions[symbol]

        logger.info(f"PAPER CLOSE: {symbol} @ {price:.2f} | P&L: {pnl:.2f} ({pnl_percent:.2f}%) | Reason: {reason}")

        return {
            'success': True,
            'symbol': symbol,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'reason': reason
        }

    def partial_exit(
        self,
        symbol: str,
        price: float,
        exit_percent: float = 0.25
    ) -> Dict:
        """
        Sortie partielle d'une position (ex: TP1)
        """
        if symbol not in self.positions:
            return {'success': False, 'reason': 'No position'}

        position = self.positions[symbol]
        exit_quantity = int(position.current_quantity * exit_percent)

        if exit_quantity <= 0:
            return {'success': False, 'reason': 'Quantity too small'}

        commission = price * exit_quantity * self.commission

        # P&L partiel
        if position.side == 'long':
            pnl = (price - position.entry_price) * exit_quantity - commission
            self.capital += (price * exit_quantity) - commission
        else:
            pnl = (position.entry_price - price) * exit_quantity - commission
            margin_returned = position.entry_price * exit_quantity * 0.5
            self.capital += margin_returned + pnl

        # Enregistrer sortie partielle
        position.partial_exits.append({
            'quantity': exit_quantity,
            'price': price,
            'pnl': pnl,
            'time': datetime.now()
        })

        self.total_pnl += pnl

        logger.info(f"PAPER PARTIAL EXIT: {exit_quantity} {symbol} @ {price:.2f} | P&L: {pnl:.2f}")

        return {
            'success': True,
            'symbol': symbol,
            'quantity_exited': exit_quantity,
            'remaining': position.current_quantity,
            'pnl': pnl
        }

    # =========================================================================
    # UPDATE & MONITORING
    # =========================================================================

    def update_prices(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Met a jour les prix et verifie SL/TP

        Returns:
            Liste des actions effectuees
        """
        actions = []

        for symbol, position in list(self.positions.items()):
            if symbol not in prices:
                continue

            price = prices[symbol]

            # Verifier Stop Loss
            if position.side == 'long' and price <= position.current_stop:
                result = self.close_position(symbol, price, "Stop Loss")
                actions.append(result)
                continue

            if position.side == 'short' and price >= position.current_stop:
                result = self.close_position(symbol, price, "Stop Loss")
                actions.append(result)
                continue

            # Verifier TP1 (si pas deja touche)
            if not position.tp1_hit:
                if position.side == 'long' and price >= position.take_profit:
                    # Sortie partielle + BE
                    result = self.partial_exit(symbol, price, TAKE_PROFIT_CONFIG['tp1_percent'])
                    if result['success']:
                        position.tp1_hit = True
                        position.current_stop = position.entry_price  # Move to BE
                        result['action'] = 'tp1_partial_exit'
                        actions.append(result)

                if position.side == 'short' and price <= position.take_profit:
                    result = self.partial_exit(symbol, price, TAKE_PROFIT_CONFIG['tp1_percent'])
                    if result['success']:
                        position.tp1_hit = True
                        position.current_stop = position.entry_price
                        result['action'] = 'tp1_partial_exit'
                        actions.append(result)

        return actions

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calcule la valeur totale du portefeuille"""
        total = self.capital

        for symbol, position in self.positions.items():
            if symbol in prices:
                price = prices[symbol]
                if position.side == 'long':
                    value = price * position.current_quantity
                else:
                    # Short: profit si prix baisse
                    unrealized = (position.entry_price - price) * position.current_quantity
                    margin = position.entry_price * position.current_quantity * 0.5
                    value = margin + unrealized
                total += value

        return total

    def get_statistics(self) -> Dict:
        """Retourne les statistiques de performance"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'total_pnl': self.total_pnl,
            'total_return_percent': (self.total_pnl / self.initial_capital) * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'open_positions': len(self.positions)
        }


# =============================================================================
# SINGLETON
# =============================================================================
_paper_trader = None

def get_paper_trader() -> PaperTrader:
    """Retourne l'instance singleton"""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader()
    return _paper_trader
