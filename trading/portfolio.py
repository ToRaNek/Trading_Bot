"""Système de portefeuille simulé pour le dry-run"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger('TradingBot')


class Portfolio:
    """Gestion du portefeuille simulé avec positions et historique"""

    def __init__(self, initial_cash: float = 1000.0, save_file: str = 'portfolio.json'):
        """
        Initialise le portefeuille

        Args:
            initial_cash: Capital initial en dollars
            save_file: Fichier pour sauvegarder l'état du portefeuille
        """
        self.save_file = save_file
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.trades_history = []  # Historique de tous les trades
        self.daily_values = []  # Valeur du portfolio chaque jour
        self.start_date = datetime.now()

        # Charger l'état s'il existe
        self.load_state()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Récupère la position pour un symbole donné"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Vérifie si on a une position sur ce symbole"""
        return symbol in self.positions and self.positions[symbol]['shares'] > 0

    def buy(self, symbol: str, price: float, shares: float, timestamp: datetime) -> bool:
        """
        Achète des actions

        Args:
            symbol: Ticker de l'action
            price: Prix d'achat
            shares: Nombre d'actions
            timestamp: Date/heure de l'achat

        Returns:
            True si l'achat a réussi, False sinon
        """
        cost = price * shares

        if cost > self.cash:
            logger.warning(f"[Portfolio] ❌ Solde insuffisant pour acheter {shares} {symbol} à ${price:.2f}")
            return False

        # Débiter le cash
        self.cash -= cost

        # Ajouter/mettre à jour la position
        if symbol in self.positions:
            # Calculer le nouveau prix moyen
            old_shares = self.positions[symbol]['shares']
            old_avg_price = self.positions[symbol]['avg_price']
            new_shares = old_shares + shares
            new_avg_price = ((old_shares * old_avg_price) + (shares * price)) / new_shares

            self.positions[symbol] = {
                'shares': new_shares,
                'avg_price': new_avg_price,
                'last_update': timestamp.isoformat()
            }
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price,
                'last_update': timestamp.isoformat()
            }

        # Ajouter à l'historique
        self.trades_history.append({
            'type': 'BUY',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'cost': cost,
            'timestamp': timestamp.isoformat(),
            'cash_after': self.cash
        })

        logger.info(f"[Portfolio] ✅ ACHAT: {shares} {symbol} à ${price:.2f} | Coût: ${cost:.2f} | Cash restant: ${self.cash:.2f}")

        self.save_state()
        return True

    def sell(self, symbol: str, price: float, shares: Optional[float] = None, timestamp: datetime = None) -> bool:
        """
        Vend des actions

        Args:
            symbol: Ticker de l'action
            price: Prix de vente
            shares: Nombre d'actions (None = tout vendre)
            timestamp: Date/heure de la vente

        Returns:
            True si la vente a réussi, False sinon
        """
        if not self.has_position(symbol):
            logger.warning(f"[Portfolio] ❌ Pas de position sur {symbol}")
            return False

        position = self.positions[symbol]
        shares_to_sell = shares if shares is not None else position['shares']

        if shares_to_sell > position['shares']:
            logger.warning(f"[Portfolio] ❌ Pas assez d'actions {symbol} (demandé: {shares_to_sell}, disponible: {position['shares']})")
            return False

        # Calculer le profit/perte
        proceeds = price * shares_to_sell
        cost_basis = position['avg_price'] * shares_to_sell
        profit = proceeds - cost_basis
        profit_pct = (profit / cost_basis) * 100

        # Créditer le cash
        self.cash += proceeds

        # Mettre à jour la position
        position['shares'] -= shares_to_sell
        if position['shares'] <= 0:
            del self.positions[symbol]
        else:
            position['last_update'] = timestamp.isoformat()

        # Ajouter à l'historique
        self.trades_history.append({
            'type': 'SELL',
            'symbol': symbol,
            'price': price,
            'shares': shares_to_sell,
            'proceeds': proceeds,
            'profit': profit,
            'profit_pct': profit_pct,
            'timestamp': timestamp.isoformat(),
            'cash_after': self.cash
        })

        emoji = "🟢" if profit > 0 else "🔴"
        logger.info(f"[Portfolio] {emoji} VENTE: {shares_to_sell} {symbol} à ${price:.2f} | Profit: ${profit:.2f} ({profit_pct:+.2f}%) | Cash: ${self.cash:.2f}")

        self.save_state()
        return True

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calcule la valeur totale du portefeuille

        Args:
            current_prices: Dictionnaire {symbol: price}

        Returns:
            Valeur totale (cash + positions)
        """
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position['shares'] * current_prices[symbol]

        return self.cash + positions_value

    def get_performance(self, current_prices: Dict[str, float]) -> Dict:
        """Calcule les statistiques de performance"""
        total_value = self.get_total_value(current_prices)
        total_return = total_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100

        # Calculer le nombre de trades gagnants/perdants
        winning_trades = [t for t in self.trades_history if t['type'] == 'SELL' and t['profit'] > 0]
        losing_trades = [t for t in self.trades_history if t['type'] == 'SELL' and t['profit'] < 0]

        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'positions_value': total_value - self.cash,
            'total_value': total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': len(self.trades_history),
            'buy_orders': len([t for t in self.trades_history if t['type'] == 'BUY']),
            'sell_orders': len([t for t in self.trades_history if t['type'] == 'SELL']),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100 if (winning_trades or losing_trades) else 0,
            'avg_profit': sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'start_date': self.start_date.isoformat(),
            'days_running': (datetime.now() - self.start_date).days
        }

    def save_state(self):
        """Sauvegarde l'état du portefeuille dans un fichier JSON"""
        state = {
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'positions': self.positions,
            'trades_history': self.trades_history,
            'daily_values': self.daily_values,
            'start_date': self.start_date.isoformat()
        }

        try:
            with open(self.save_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"[Portfolio] ✅ État sauvegardé dans {self.save_file}")
        except Exception as e:
            logger.error(f"[Portfolio] ❌ Erreur lors de la sauvegarde: {e}")

    def load_state(self):
        """Charge l'état du portefeuille depuis un fichier JSON"""
        if not Path(self.save_file).exists():
            logger.info(f"[Portfolio] Nouveau portefeuille créé avec ${self.initial_cash}")
            return

        try:
            with open(self.save_file, 'r') as f:
                state = json.load(f)

            self.initial_cash = state.get('initial_cash', self.initial_cash)
            self.cash = state.get('cash', self.cash)
            self.positions = state.get('positions', {})
            self.trades_history = state.get('trades_history', [])
            self.daily_values = state.get('daily_values', [])

            start_date_str = state.get('start_date')
            if start_date_str:
                self.start_date = datetime.fromisoformat(start_date_str)

            logger.info(f"[Portfolio] ✅ État chargé depuis {self.save_file}")
            logger.info(f"[Portfolio] Cash: ${self.cash:.2f} | Positions: {len(self.positions)} | Trades: {len(self.trades_history)}")
        except Exception as e:
            logger.error(f"[Portfolio] ❌ Erreur lors du chargement: {e}")

    def reset(self):
        """Réinitialise le portefeuille"""
        self.cash = self.initial_cash
        self.positions = {}
        self.trades_history = []
        self.daily_values = []
        self.start_date = datetime.now()
        self.save_state()
        logger.info(f"[Portfolio] 🔄 Portefeuille réinitialisé avec ${self.initial_cash}")
