"""
Gestion des Ordres
Types d'ordres et leur execution
"""
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict
import logging
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represente un ordre de trading"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None

    # Tracking
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    broker_order_id: Optional[str] = None

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
        }


class OrderManager:
    """Gere la creation et le suivi des ordres"""

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}

    def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int
    ) -> Order:
        """Cree un ordre au marche"""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        logger.info(f"Market order created: {order.order_id} - {side} {quantity} {symbol}")
        return order

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float
    ) -> Order:
        """Cree un ordre limite"""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        logger.info(f"Limit order created: {order.order_id} - {side} {quantity} {symbol} @ {price}")
        return order

    def create_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        stop_price: float
    ) -> Order:
        """Cree un ordre stop"""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=stop_price
        )
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        logger.info(f"Stop order created: {order.order_id} - {side} {quantity} {symbol} @ stop {stop_price}")
        return order

    def fill_order(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: int = None
    ) -> Order:
        """Marque un ordre comme execute"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")

        order = self.orders[order_id]

        if fill_quantity is None:
            fill_quantity = order.remaining_quantity

        order.filled_quantity += fill_quantity
        order.filled_price = fill_price
        order.updated_at = datetime.now()

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
        else:
            order.status = OrderStatus.PARTIAL

        logger.info(f"Order filled: {order_id} @ {fill_price} (qty: {fill_quantity})")
        return order

    def cancel_order(self, order_id: str) -> Order:
        """Annule un ordre"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")

        order = self.orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()

        if order_id in self.pending_orders:
            del self.pending_orders[order_id]

        logger.info(f"Order cancelled: {order_id}")
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Recupere un ordre par ID"""
        return self.orders.get(order_id)

    def get_pending_orders(self, symbol: str = None) -> list:
        """Recupere les ordres en attente"""
        orders = list(self.pending_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def cancel_all_orders(self, symbol: str = None):
        """Annule tous les ordres (ou pour un symbole)"""
        to_cancel = list(self.pending_orders.keys())
        for order_id in to_cancel:
            order = self.orders[order_id]
            if symbol is None or order.symbol == symbol:
                self.cancel_order(order_id)


# =============================================================================
# SINGLETON
# =============================================================================
_order_manager = None

def get_order_manager() -> OrderManager:
    """Retourne l'instance singleton"""
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager
