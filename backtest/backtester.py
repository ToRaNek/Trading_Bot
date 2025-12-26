"""
Backtester - Test des strategies sur donnees historiques
Base sur MASTER_TRADING_SKILL
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from config.settings import (
    INITIAL_CAPITAL, RISK_PER_TRADE, MIN_RISK_REWARD,
    MAX_OPEN_POSITIONS, TAKE_PROFIT_CONFIG
)
from data.fetcher import get_fetcher
from analysis.indicators import get_indicators
from analysis.zones import get_zone_detector, find_swing_points, detect_breakout
from analysis.patterns import get_pattern_detector
from strategy.position_sizing import PositionSizer

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BacktestTrade:
    """Trade en backtest"""
    symbol: str
    side: str
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    tp1_hit: bool = False
    max_favorable: float = 0.0  # MFE
    max_adverse: float = 0.0    # MAE


@dataclass
class BacktestResult:
    """Resultats de backtest"""
    # Performance
    total_return: float
    total_return_percent: float
    cagr: float  # Compound Annual Growth Rate

    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Risk
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Average
    avg_win: float
    avg_loss: float
    avg_trade: float
    profit_factor: float
    expectancy: float

    # Timing
    avg_holding_days: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Details
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.Series = None


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """
    Backtester pour tester les strategies

    Simule le trading historique avec:
    - Gestion des positions
    - Stop Loss / Take Profit
    - Partial exits (TP1 + BE)
    - Calcul des metriques
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005    # 0.05%
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.zones = get_zone_detector()
        self.patterns = get_pattern_detector()
        self.position_sizer = PositionSizer(initial_capital)

        # State
        self.reset()

    def reset(self):
        """Reset le backtester"""
        self.capital = self.initial_capital
        self.positions: Dict[str, BacktestTrade] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve = []
        self.dates = []

    # =========================================================================
    # MAIN BACKTEST
    # =========================================================================

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str = None,
        strategy: str = "swing_trading"
    ) -> BacktestResult:
        """
        Execute le backtest

        Args:
            symbols: Liste des symboles a tester
            start_date: Date de debut (YYYY-MM-DD)
            end_date: Date de fin (defaut: aujourd'hui)
            strategy: Strategie a utiliser

        Returns:
            BacktestResult avec toutes les metriques
        """
        logger.info(f"Starting backtest from {start_date} to {end_date or 'today'}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Strategy: {strategy}")

        self.reset()

        # Recuperer les donnees
        all_data = {}
        for symbol in symbols:
            df = self.fetcher.get_daily_data(symbol, days=730)  # ~2 ans
            if df is not None and len(df) > 50:
                all_data[symbol] = self._prepare_data(df)
                logger.info(f"Loaded {len(df)} bars for {symbol}")
            else:
                logger.warning(f"Insufficient data for {symbol}")

        if not all_data:
            logger.error("No data available for backtest")
            return self._empty_result()

        # Normaliser les index (enlever timezone)
        for symbol in all_data:
            if all_data[symbol].index.tz is not None:
                all_data[symbol].index = all_data[symbol].index.tz_localize(None)

        # Trouver les dates communes
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

        # Simuler jour par jour
        first_df = list(all_data.values())[0]
        dates = first_df.loc[start:end].index

        for date in dates:
            # Mettre a jour positions existantes
            self._update_positions(all_data, date)

            # Chercher nouveaux signaux
            if len(self.positions) < MAX_OPEN_POSITIONS:
                for symbol, df in all_data.items():
                    if symbol not in self.positions and date in df.index:
                        signal = self._check_signal(df.loc[:date], strategy)
                        if signal:
                            self._open_position(symbol, df.loc[date], signal)

            # Enregistrer equity
            self._record_equity(date, all_data)

        # Fermer positions restantes
        self._close_all_positions(dates[-1], all_data)

        # Calculer resultats
        return self._calculate_results()

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare les donnees avec indicateurs"""
        df = df.copy()
        df = self.indicators.add_all_indicators(df)
        return df

    def _check_signal(self, df: pd.DataFrame, strategy: str) -> Optional[Dict]:
        """
        Verifie si un signal est present

        Returns:
            Dict avec direction, entry, sl, tp si signal
        """
        if len(df) < 50:
            return None

        # Trouver swing points
        swing_highs, swing_lows = find_swing_points(df)

        # Detecter breakout
        breakout = detect_breakout(df, swing_highs, swing_lows)
        if not breakout:
            return None

        direction = breakout['direction']
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02

        # Verifier indicateurs
        latest = df.iloc[-1]
        rsi = latest.get('rsi', 50)

        if direction == 'buy':
            if rsi > 70:  # Surachat
                return None
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 2 * MIN_RISK_REWARD)
        else:
            if rsi < 30:  # Survente
                return None
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 2 * MIN_RISK_REWARD)

        # Verifier R:R
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        rr = reward / risk if risk > 0 else 0

        if rr < MIN_RISK_REWARD:
            return None

        return {
            'direction': direction,
            'entry': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': rr
        }

    def _open_position(self, symbol: str, bar: pd.Series, signal: Dict):
        """Ouvre une position"""
        entry_price = signal['entry']

        # Appliquer slippage
        if signal['direction'] == 'buy':
            entry_price *= (1 + self.slippage)
        else:
            entry_price *= (1 - self.slippage)

        # Calculer taille
        sizing = self.position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss=signal['stop_loss']
        )
        quantity = sizing['shares']

        if quantity <= 0:
            return

        # Verifier capital
        cost = entry_price * quantity * (1 + self.commission)
        if cost > self.capital:
            quantity = int(self.capital / (entry_price * (1 + self.commission)))
            if quantity <= 0:
                return

        # Deduire du capital
        cost = entry_price * quantity * (1 + self.commission)
        self.capital -= cost

        # Creer trade
        trade = BacktestTrade(
            symbol=symbol,
            side='long' if signal['direction'] == 'buy' else 'short',
            entry_date=bar.name,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit']
        )

        self.positions[symbol] = trade
        logger.debug(f"Opened {trade.side} {symbol} @ {entry_price:.2f}")

    def _update_positions(self, all_data: Dict, date: pd.Timestamp):
        """Met a jour les positions existantes"""
        to_close = []

        for symbol, trade in self.positions.items():
            if symbol not in all_data:
                continue

            df = all_data[symbol]
            if date not in df.index:
                continue

            bar = df.loc[date]
            high = bar['high']
            low = bar['low']
            close = bar['close']

            # Mettre a jour MFE/MAE
            if trade.side == 'long':
                favorable = (high - trade.entry_price) / trade.entry_price
                adverse = (trade.entry_price - low) / trade.entry_price
            else:
                favorable = (trade.entry_price - low) / trade.entry_price
                adverse = (high - trade.entry_price) / trade.entry_price

            trade.max_favorable = max(trade.max_favorable, favorable)
            trade.max_adverse = max(trade.max_adverse, adverse)

            # Verifier Stop Loss
            if trade.side == 'long' and low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "Stop Loss"
                to_close.append(symbol)
                continue

            if trade.side == 'short' and high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "Stop Loss"
                to_close.append(symbol)
                continue

            # Verifier Take Profit (TP1 puis move to BE)
            if not trade.tp1_hit:
                tp1_price = trade.entry_price + (trade.take_profit - trade.entry_price) * 0.5

                if trade.side == 'long' and high >= tp1_price:
                    trade.tp1_hit = True
                    trade.stop_loss = trade.entry_price  # Move to BE
                    logger.debug(f"{symbol} TP1 hit, moved to BE")

                if trade.side == 'short' and low <= tp1_price:
                    trade.tp1_hit = True
                    trade.stop_loss = trade.entry_price
                    logger.debug(f"{symbol} TP1 hit, moved to BE")

            # Verifier Take Profit final
            if trade.side == 'long' and high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_reason = "Take Profit"
                to_close.append(symbol)
                continue

            if trade.side == 'short' and low <= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_reason = "Take Profit"
                to_close.append(symbol)
                continue

        # Fermer les positions
        for symbol in to_close:
            self._close_position(symbol, date)

    def _close_position(self, symbol: str, date: pd.Timestamp):
        """Ferme une position"""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]
        trade.exit_date = date

        # Appliquer slippage
        exit_price = trade.exit_price
        if trade.side == 'long':
            exit_price *= (1 - self.slippage)
        else:
            exit_price *= (1 + self.slippage)

        trade.exit_price = exit_price

        # Calculer P&L
        if trade.side == 'long':
            gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.quantity

        commission_cost = (trade.entry_price + exit_price) * trade.quantity * self.commission
        trade.pnl = gross_pnl - commission_cost
        trade.pnl_percent = trade.pnl / (trade.entry_price * trade.quantity) * 100

        # Ajouter au capital
        self.capital += (exit_price * trade.quantity) - (exit_price * trade.quantity * self.commission)
        if trade.side == 'long':
            self.capital += trade.pnl

        # Enregistrer
        self.trades.append(trade)
        del self.positions[symbol]

        logger.debug(f"Closed {symbol} @ {exit_price:.2f} | P&L: {trade.pnl:.2f} ({trade.exit_reason})")

    def _close_all_positions(self, date: pd.Timestamp, all_data: Dict):
        """Ferme toutes les positions ouvertes"""
        for symbol in list(self.positions.keys()):
            trade = self.positions[symbol]
            if symbol in all_data and date in all_data[symbol].index:
                trade.exit_price = all_data[symbol].loc[date]['close']
            else:
                trade.exit_price = trade.entry_price  # Fallback
            trade.exit_reason = "End of Backtest"
            self._close_position(symbol, date)

    def _record_equity(self, date: pd.Timestamp, all_data: Dict):
        """Enregistre l'equity"""
        equity = self.capital

        for symbol, trade in self.positions.items():
            if symbol in all_data and date in all_data[symbol].index:
                current_price = all_data[symbol].loc[date]['close']
                if trade.side == 'long':
                    equity += current_price * trade.quantity
                else:
                    equity += trade.entry_price * trade.quantity + (trade.entry_price - current_price) * trade.quantity

        self.equity_curve.append(equity)
        self.dates.append(date)

    # =========================================================================
    # CALCUL DES RESULTATS
    # =========================================================================

    def _calculate_results(self) -> BacktestResult:
        """Calcule toutes les metriques"""
        if not self.trades:
            return self._empty_result()

        # Equity curve
        equity = pd.Series(self.equity_curve, index=self.dates)

        # Returns
        total_return = self.capital - self.initial_capital
        total_return_percent = (total_return / self.initial_capital) * 100

        # CAGR
        years = len(self.dates) / 252 if self.dates else 1
        cagr = ((self.capital / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Trades stats
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([abs(t.pnl) for t in losing]) if losing else 0
        avg_trade = np.mean([t.pnl for t in self.trades])

        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

        # Drawdown
        max_dd, max_dd_percent = self._calculate_drawdown(equity)

        # Ratios
        sharpe = self._calculate_sharpe(equity)
        sortino = self._calculate_sortino(equity)
        calmar = cagr / max_dd_percent if max_dd_percent > 0 else 0

        # Holding time
        holding_days = []
        for t in self.trades:
            if t.exit_date and t.entry_date:
                days = (t.exit_date - t.entry_date).days
                holding_days.append(days)
        avg_holding = np.mean(holding_days) if holding_days else 0

        # Consecutive
        max_wins, max_losses = self._calculate_consecutive()

        return BacktestResult(
            total_return=total_return,
            total_return_percent=total_return_percent,
            cagr=cagr,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_percent,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_holding_days=avg_holding,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            trades=self.trades,
            equity_curve=equity
        )

    def _calculate_drawdown(self, equity: pd.Series) -> Tuple[float, float]:
        """Calcule le max drawdown"""
        peak = equity.expanding().max()
        drawdown = equity - peak
        max_dd = abs(drawdown.min())
        max_dd_percent = (max_dd / peak.max()) * 100 if peak.max() > 0 else 0
        return max_dd, max_dd_percent

    def _calculate_sharpe(self, equity: pd.Series, risk_free: float = 0.02) -> float:
        """Calcule le Sharpe Ratio"""
        returns = equity.pct_change().dropna()
        if len(returns) < 2:
            return 0

        excess_returns = returns - (risk_free / 252)
        if returns.std() == 0:
            return 0

        return np.sqrt(252) * (excess_returns.mean() / returns.std())

    def _calculate_sortino(self, equity: pd.Series, risk_free: float = 0.02) -> float:
        """Calcule le Sortino Ratio"""
        returns = equity.pct_change().dropna()
        if len(returns) < 2:
            return 0

        excess_returns = returns - (risk_free / 252)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        return np.sqrt(252) * (excess_returns.mean() / downside_returns.std())

    def _calculate_consecutive(self) -> Tuple[int, int]:
        """Calcule les series consecutives"""
        if not self.trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _empty_result(self) -> BacktestResult:
        """Retourne un resultat vide"""
        return BacktestResult(
            total_return=0,
            total_return_percent=0,
            cagr=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            max_drawdown=0,
            max_drawdown_percent=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            avg_win=0,
            avg_loss=0,
            avg_trade=0,
            profit_factor=0,
            expectancy=0,
            avg_holding_days=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_results(self, result: BacktestResult):
        """Affiche les resultats"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\n{'PERFORMANCE':=^40}")
        print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_percent:.2f}%)")
        print(f"CAGR: {result.cagr:.2f}%")

        print(f"\n{'TRADES':=^40}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning: {result.winning_trades} | Losing: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Avg Win: ${result.avg_win:.2f} | Avg Loss: ${result.avg_loss:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Expectancy: ${result.expectancy:.2f}")

        print(f"\n{'RISK':=^40}")
        print(f"Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_percent:.2f}%)")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {result.calmar_ratio:.2f}")

        print(f"\n{'TIMING':=^40}")
        print(f"Avg Holding: {result.avg_holding_days:.1f} days")
        print(f"Max Consecutive Wins: {result.max_consecutive_wins}")
        print(f"Max Consecutive Losses: {result.max_consecutive_losses}")

        print("\n" + "=" * 60)


# =============================================================================
# OPTIMIZATION
# =============================================================================

class ParameterOptimizer:
    """
    Optimisation des parametres de strategie

    Walk-forward optimization pour eviter l'overfitting
    """

    def __init__(self, backtester: Backtester):
        self.backtester = backtester

    def grid_search(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        param_grid: Dict[str, List]
    ) -> List[Dict]:
        """
        Grid search sur les parametres

        Args:
            symbols: Symboles a tester
            start_date: Date debut
            end_date: Date fin
            param_grid: Dictionnaire de parametres a tester

        Returns:
            Liste des resultats tries par performance
        """
        import itertools

        # Generer toutes les combinaisons
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        results = []
        total = len(combinations)

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            logger.info(f"Testing combination {i}/{total}: {params}")

            # Appliquer parametres (simplified - would need to pass to strategy)
            result = self.backtester.run(symbols, start_date, end_date)

            results.append({
                'params': params,
                'sharpe': result.sharpe_ratio,
                'return': result.total_return_percent,
                'win_rate': result.win_rate,
                'max_dd': result.max_drawdown_percent,
                'profit_factor': result.profit_factor
            })

        # Trier par Sharpe
        results.sort(key=lambda x: x['sharpe'], reverse=True)

        return results


# =============================================================================
# SINGLETON
# =============================================================================

_backtester = None


def get_backtester() -> Backtester:
    """Retourne l'instance singleton"""
    global _backtester
    if _backtester is None:
        _backtester = Backtester()
    return _backtester
