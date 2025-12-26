"""
Database - Stockage SQLite pour trades, positions, et historique
"""
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import json
import logging

from config.settings import DB_PATH

logger = logging.getLogger(__name__)


class Database:
    """
    Gestionnaire de base de donnees SQLite
    Stocke les trades, positions, signaux, et metriques
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Cree une connexion a la base de donnees"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialise les tables de la base de donnees"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Table des trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,  -- 'buy' ou 'sell'
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                status TEXT DEFAULT 'open',  -- 'open', 'closed', 'cancelled'
                pnl REAL,
                pnl_percent REAL,
                fees REAL DEFAULT 0,
                strategy TEXT,
                signal_strength REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table des positions actuelles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                unrealized_pnl REAL,
                entry_time TIMESTAMP NOT NULL,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table des signaux generes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,  -- 'buy', 'sell', 'close'
                strength REAL,
                price_at_signal REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_reward REAL,
                indicators TEXT,  -- JSON des indicateurs
                zones TEXT,  -- JSON des zones
                timeframe TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                executed INTEGER DEFAULT 0,
                executed_at TIMESTAMP
            )
        ''')

        # Table du journal de trading (pour psychologie)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                trades_count INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                daily_return_percent REAL,
                max_drawdown REAL,
                emotions TEXT,  -- JSON notes emotionnelles
                lessons TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table des metriques quotidiennes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                starting_capital REAL,
                ending_capital REAL,
                daily_pnl REAL,
                daily_return_percent REAL,
                trades_count INTEGER,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table de configuration
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table des ordres
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,  -- ID du broker
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,  -- 'market', 'limit', 'stop'
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                stop_price REAL,
                status TEXT DEFAULT 'pending',
                filled_quantity REAL DEFAULT 0,
                filled_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    # =========================================================================
    # TRADES
    # =========================================================================

    def add_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None,
        strategy: str = None,
        signal_strength: float = None
    ) -> int:
        """Ajoute un nouveau trade"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (symbol, side, entry_price, quantity, stop_loss,
                               take_profit, entry_time, strategy, signal_strength, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        ''', (symbol, side, entry_price, quantity, stop_loss, take_profit,
              datetime.now(), strategy, signal_strength))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Trade added: {trade_id} - {side} {quantity} {symbol} @ {entry_price}")
        return trade_id

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        fees: float = 0
    ) -> Dict:
        """Ferme un trade et calcule le P&L"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Recuperer le trade
        cursor.execute('SELECT * FROM trades WHERE id = ?', (trade_id,))
        trade = cursor.fetchone()

        if not trade:
            conn.close()
            raise ValueError(f"Trade {trade_id} not found")

        # Calculer P&L
        if trade['side'] == 'buy':
            pnl = (exit_price - trade['entry_price']) * trade['quantity'] - fees
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity'] - fees

        pnl_percent = (pnl / (trade['entry_price'] * trade['quantity'])) * 100

        # Mettre a jour
        cursor.execute('''
            UPDATE trades
            SET exit_price = ?, exit_time = ?, status = 'closed',
                pnl = ?, pnl_percent = ?, fees = ?
            WHERE id = ?
        ''', (exit_price, datetime.now(), pnl, pnl_percent, fees, trade_id))

        conn.commit()
        conn.close()

        logger.info(f"Trade closed: {trade_id} - P&L: {pnl:.2f} ({pnl_percent:.2f}%)")

        return {'trade_id': trade_id, 'pnl': pnl, 'pnl_percent': pnl_percent}

    def get_open_trades(self) -> List[Dict]:
        """Retourne tous les trades ouverts"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Retourne l'historique des trades"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM trades
            WHERE status = 'closed'
            ORDER BY exit_time DESC
            LIMIT ?
        ''', (limit,))
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades

    def get_trades_today(self) -> List[Dict]:
        """Retourne les trades du jour"""
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().date()
        cursor.execute('''
            SELECT * FROM trades
            WHERE DATE(entry_time) = ?
        ''', (today,))
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades

    # =========================================================================
    # POSITIONS
    # =========================================================================

    def update_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        current_price: float = None,
        stop_loss: float = None,
        take_profit: float = None
    ):
        """Met a jour ou cree une position"""
        conn = self._get_connection()
        cursor = conn.cursor()

        unrealized_pnl = None
        if current_price:
            if side == 'buy':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity

        cursor.execute('''
            INSERT OR REPLACE INTO positions
            (symbol, side, entry_price, current_price, quantity, stop_loss,
             take_profit, unrealized_pnl, entry_time, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, side, entry_price, current_price, quantity, stop_loss,
              take_profit, unrealized_pnl, datetime.now(), datetime.now()))

        conn.commit()
        conn.close()

    def remove_position(self, symbol: str):
        """Supprime une position"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
        conn.commit()
        conn.close()

    def get_positions(self) -> List[Dict]:
        """Retourne toutes les positions ouvertes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM positions')
        positions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return positions

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Retourne une position specifique"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM positions WHERE symbol = ?', (symbol,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # =========================================================================
    # SIGNALS
    # =========================================================================

    def add_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        risk_reward: float,
        indicators: Dict = None,
        zones: List = None,
        timeframe: str = "1d"
    ) -> int:
        """Enregistre un nouveau signal"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO signals (symbol, signal_type, strength, price_at_signal,
                                stop_loss, take_profit, risk_reward, indicators,
                                zones, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, signal_type, strength, price, stop_loss, take_profit,
              risk_reward, json.dumps(indicators), json.dumps(zones), timeframe))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Signal added: {signal_id} - {signal_type} {symbol} (strength: {strength})")
        return signal_id

    def get_recent_signals(self, hours: int = 24) -> List[Dict]:
        """Retourne les signaux recents"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM signals
            WHERE created_at >= datetime('now', '-' || ? || ' hours')
            ORDER BY created_at DESC
        ''', (hours,))
        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return signals

    def mark_signal_executed(self, signal_id: int):
        """Marque un signal comme execute"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE signals SET executed = 1, executed_at = ? WHERE id = ?
        ''', (datetime.now(), signal_id))
        conn.commit()
        conn.close()

    # =========================================================================
    # METRICS
    # =========================================================================

    def save_daily_metrics(self, metrics: Dict):
        """Sauvegarde les metriques quotidiennes"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO daily_metrics
            (date, starting_capital, ending_capital, daily_pnl, daily_return_percent,
             trades_count, win_rate, avg_win, avg_loss, profit_factor, max_drawdown, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.get('date', datetime.now().date()),
            metrics.get('starting_capital'),
            metrics.get('ending_capital'),
            metrics.get('daily_pnl'),
            metrics.get('daily_return_percent'),
            metrics.get('trades_count'),
            metrics.get('win_rate'),
            metrics.get('avg_win'),
            metrics.get('avg_loss'),
            metrics.get('profit_factor'),
            metrics.get('max_drawdown'),
            metrics.get('sharpe_ratio')
        ))

        conn.commit()
        conn.close()

    def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Retourne l'historique de performance"""
        conn = self._get_connection()
        df = pd.read_sql_query(f'''
            SELECT * FROM daily_metrics
            ORDER BY date DESC
            LIMIT {days}
        ''', conn)
        conn.close()
        return df

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Calcule les statistiques globales"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Trades fermes
        cursor.execute('''
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades
            WHERE status = 'closed'
        ''')
        stats = dict(cursor.fetchone())

        # Win rate
        if stats['total_trades'] and stats['total_trades'] > 0:
            stats['win_rate'] = (stats['wins'] / stats['total_trades']) * 100
        else:
            stats['win_rate'] = 0

        # Profit factor
        if stats['avg_loss'] and stats['avg_loss'] != 0:
            stats['profit_factor'] = abs(stats['avg_win'] or 0) / abs(stats['avg_loss'])
        else:
            stats['profit_factor'] = 0

        conn.close()
        return stats


# =============================================================================
# SINGLETON
# =============================================================================
_db_instance = None

def get_database() -> Database:
    """Retourne l'instance singleton de Database"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
