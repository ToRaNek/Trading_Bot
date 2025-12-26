"""
Live Trading - Trading reel avec Alpaca API (gratuit)
"""
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

logger = logging.getLogger(__name__)

# Import conditionnel d'Alpaca
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-trade-api not installed. Live trading disabled.")


class LiveTrader:
    """
    Trading reel via Alpaca API

    Alpaca offre:
    - Trading US stocks gratuit
    - API REST simple
    - Paper trading integre
    - Pas de minimum de compte
    """

    def __init__(self):
        self.connected = False
        self.api = None

        if not ALPACA_AVAILABLE:
            logger.error("Alpaca API not available")
            return

        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.warning("Alpaca credentials not configured")
            return

        try:
            self.api = tradeapi.REST(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                ALPACA_BASE_URL,
                api_version='v2'
            )
            # Test connection
            account = self.api.get_account()
            self.connected = True
            logger.info(f"Connected to Alpaca. Account status: {account.status}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")

    # =========================================================================
    # ACCOUNT INFO
    # =========================================================================

    def get_account(self) -> Optional[Dict]:
        """Recupere les infos du compte"""
        if not self.connected:
            return None

        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'status': account.status,
                'trading_blocked': account.trading_blocked,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Recupere les positions ouvertes"""
        if not self.connected:
            return []

        try:
            positions = self.api.list_positions()
            return [{
                'symbol': p.symbol,
                'quantity': int(p.qty),
                'side': 'long' if int(p.qty) > 0 else 'short',
                'entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pnl': float(p.unrealized_pl),
                'unrealized_pnl_percent': float(p.unrealized_plpc) * 100
            } for p in positions]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    # =========================================================================
    # ORDERS
    # =========================================================================

    def submit_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str  # 'buy' ou 'sell'
    ) -> Optional[Dict]:
        """
        Soumet un ordre au marche
        """
        if not self.connected:
            logger.error("Not connected to Alpaca")
            return None

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )

            logger.info(f"LIVE ORDER: {side} {quantity} {symbol} (market)")

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.type,
                'quantity': int(order.qty),
                'status': order.status
            }
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return None

    def submit_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        price: float
    ) -> Optional[Dict]:
        """
        Soumet un ordre limite
        """
        if not self.connected:
            return None

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=price
            )

            logger.info(f"LIVE ORDER: {side} {quantity} {symbol} @ {price} (limit)")

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.type,
                'quantity': int(order.qty),
                'price': float(order.limit_price),
                'status': order.status
            }
        except Exception as e:
            logger.error(f"Error submitting limit order: {e}")
            return None

    def submit_bracket_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        entry_price: float = None,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[Dict]:
        """
        Soumet un ordre bracket (entry + SL + TP)
        """
        if not self.connected:
            return None

        try:
            order_type = 'limit' if entry_price else 'market'

            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force='day',
                limit_price=entry_price if entry_price else None,
                order_class='bracket',
                stop_loss={'stop_price': stop_loss} if stop_loss else None,
                take_profit={'limit_price': take_profit} if take_profit else None
            )

            logger.info(f"LIVE BRACKET: {side} {quantity} {symbol} | SL: {stop_loss} | TP: {take_profit}")

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': int(order.qty),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': order.status
            }
        except Exception as e:
            logger.error(f"Error submitting bracket order: {e}")
            return None

    def submit_stop_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        stop_price: float
    ) -> Optional[Dict]:
        """
        Soumet un ordre stop
        """
        if not self.connected:
            return None

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='stop',
                time_in_force='day',
                stop_price=stop_price
            )

            logger.info(f"LIVE STOP: {side} {quantity} {symbol} @ stop {stop_price}")

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'stop_price': float(order.stop_price),
                'status': order.status
            }
        except Exception as e:
            logger.error(f"Error submitting stop order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre"""
        if not self.connected:
            return False

        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Annule tous les ordres"""
        if not self.connected:
            return False

        try:
            self.api.cancel_all_orders()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Recupere un ordre par ID"""
        if not self.connected:
            return None

        try:
            order = self.api.get_order(order_id)
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.type,
                'quantity': int(order.qty),
                'filled_quantity': int(order.filled_qty),
                'filled_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'status': order.status
            }
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """Recupere les ordres ouverts"""
        if not self.connected:
            return []

        try:
            orders = self.api.list_orders(status='open')
            return [{
                'order_id': o.id,
                'symbol': o.symbol,
                'side': o.side,
                'type': o.type,
                'quantity': int(o.qty),
                'status': o.status
            } for o in orders]
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def close_position(self, symbol: str) -> Optional[Dict]:
        """Ferme une position"""
        if not self.connected:
            return None

        try:
            order = self.api.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return {
                'symbol': symbol,
                'order_id': order.id,
                'status': order.status
            }
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    def close_all_positions(self) -> bool:
        """Ferme toutes les positions"""
        if not self.connected:
            return False

        try:
            self.api.close_all_positions()
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Recupere la derniere cotation"""
        if not self.connected:
            return None

        try:
            quote = self.api.get_latest_quote(symbol)
            return {
                'symbol': symbol,
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size)
            }
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None

    def get_latest_trade(self, symbol: str) -> Optional[Dict]:
        """Recupere le dernier trade"""
        if not self.connected:
            return None

        try:
            trade = self.api.get_latest_trade(symbol)
            return {
                'symbol': symbol,
                'price': float(trade.price),
                'size': int(trade.size),
                'timestamp': trade.timestamp
            }
        except Exception as e:
            logger.error(f"Error getting trade: {e}")
            return None

    def is_market_open(self) -> bool:
        """Verifie si le marche est ouvert"""
        if not self.connected:
            return False

        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False


# =============================================================================
# SINGLETON
# =============================================================================
_live_trader = None

def get_live_trader() -> LiveTrader:
    """Retourne l'instance singleton"""
    global _live_trader
    if _live_trader is None:
        _live_trader = LiveTrader()
    return _live_trader
