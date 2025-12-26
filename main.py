"""
Trading Bot - Point d'Entree Principal
Base sur MASTER_TRADING_SKILL

Usage:
    python main.py                  # Mode interactif
    python main.py scan             # Scanner les signaux
    python main.py trade            # Mode trading automatique
    python main.py backtest         # Lancer un backtest
    python main.py dashboard        # Lancer le dashboard Streamlit
"""
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import List
import schedule

from config.settings import (
    INITIAL_CAPITAL, RISK_PER_TRADE, MIN_RISK_REWARD,
    MAX_OPEN_POSITIONS, MAX_TRADES_PER_DAY,
    TRADING_MODE, LOG_LEVEL
)
from config.symbols import US_STOCKS, EU_STOCKS, WATCHLIST
from data.fetcher import get_fetcher
from data.database import get_database
from analysis.indicators import get_indicators
from analysis.signals import get_signal_generator
from strategy.swing_trading import get_swing_strategy, TradeSetup
from strategy.risk_management import get_risk_manager
from execution.paper_trader import get_paper_trader
from execution.live_trader import get_live_trader
from backtest.backtester import get_backtester
from utils.logger import init_all_loggers, get_logger, log_trade
from utils.notifications import get_notification_manager

# Initialize logging
init_all_loggers(LOG_LEVEL)
logger = get_logger("main")


# =============================================================================
# TRADING BOT CLASS
# =============================================================================

class TradingBot:
    """
    Bot de Trading Principal

    Orchestre tous les composants:
    - Analyse technique
    - Generation de signaux
    - Gestion des positions
    - Notifications
    """

    def __init__(self, mode: str = "paper"):
        """
        Initialise le bot

        Args:
            mode: "paper" ou "live"
        """
        self.mode = mode

        # Components
        self.fetcher = get_fetcher()
        self.strategy = get_swing_strategy()
        self.risk_manager = get_risk_manager()
        self.notifications = get_notification_manager()
        self.db = get_database()

        # Trader
        if mode == "live":
            self.trader = get_live_trader()
        else:
            self.trader = get_paper_trader()

        # State
        self.running = False
        self.trades_today = 0
        self.last_scan = None

        logger.info(f"TradingBot initialized in {mode.upper()} mode")

    # =========================================================================
    # SCANNING
    # =========================================================================

    def scan_signals(self, symbols: List[str] = None) -> List[TradeSetup]:
        """
        Scanne les symboles pour trouver des signaux

        Args:
            symbols: Liste des symboles a scanner

        Returns:
            Liste des setups valides
        """
        if symbols is None:
            symbols = WATCHLIST

        logger.info(f"Scanning {len(symbols)} symbols...")

        setups = self.strategy.scan_watchlist(symbols)

        logger.info(f"Found {len(setups)} valid setups")

        # Notifier
        for setup in setups:
            self.notifications.notify_signal(
                symbol=setup.symbol,
                direction=setup.direction,
                strength=setup.signal_strength,
                entry_price=setup.entry_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit_final,
                risk_reward=setup.risk_reward,
                reasons=setup.reasons
            )

        self.last_scan = datetime.now()
        return setups

    # =========================================================================
    # TRADING
    # =========================================================================

    def execute_setup(self, setup: TradeSetup) -> bool:
        """
        Execute un setup de trade

        Args:
            setup: TradeSetup a executer

        Returns:
            True si execute avec succes
        """
        logger.info(f"Executing setup: {setup.direction} {setup.symbol}")

        # Verifier limites
        if self.trades_today >= MAX_TRADES_PER_DAY:
            logger.warning("Max trades per day reached")
            return False

        # Verifier positions
        if self.mode == "paper":
            positions = self.trader.positions
        else:
            positions = self.trader.get_positions()

        if len(positions) >= MAX_OPEN_POSITIONS:
            logger.warning("Max positions reached")
            return False

        # Executer
        if self.mode == "paper":
            if setup.direction == "buy":
                result = self.trader.execute_buy(
                    symbol=setup.symbol,
                    quantity=setup.position_size,
                    price=setup.entry_price,
                    stop_loss=setup.stop_loss,
                    take_profit=setup.take_profit_final,
                    setup=setup
                )
            else:
                result = self.trader.execute_sell(
                    symbol=setup.symbol,
                    quantity=setup.position_size,
                    price=setup.entry_price,
                    stop_loss=setup.stop_loss,
                    take_profit=setup.take_profit_final,
                    setup=setup
                )
        else:
            # Live trading
            result = self.trader.submit_bracket_order(
                symbol=setup.symbol,
                quantity=setup.position_size,
                side="buy" if setup.direction == "buy" else "sell",
                entry_price=setup.entry_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit_final
            )

        if result and result.get('success', False):
            self.trades_today += 1
            self.strategy.add_active_setup(setup.symbol, setup)

            # Log et notifier
            log_trade(
                action=setup.direction.upper(),
                symbol=setup.symbol,
                quantity=setup.position_size,
                price=setup.entry_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit_final
            )

            self.notifications.notify_trade(
                symbol=setup.symbol,
                action=setup.direction.upper(),
                price=setup.entry_price,
                quantity=setup.position_size,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit_final
            )

            logger.info(f"Trade executed: {setup.symbol}")
            return True

        logger.error(f"Trade failed: {result.get('reason', 'Unknown')}")
        return False

    def update_positions(self):
        """
        Met a jour les positions (verifier SL/TP)
        """
        if self.mode != "paper":
            return  # Live gere par le broker

        # Recuperer prix actuels
        prices = {}
        for symbol in self.trader.positions.keys():
            df = self.fetcher.get_daily_data(symbol)
            if df is not None:
                prices[symbol] = df['close'].iloc[-1]

        # Mettre a jour
        actions = self.trader.update_prices(prices)

        for action in actions:
            if action.get('success'):
                log_trade(
                    action="CLOSE",
                    symbol=action['symbol'],
                    quantity=action.get('quantity', 0),
                    price=action.get('exit_price', 0),
                    pnl=action.get('pnl', 0),
                    reason=action.get('action', 'Unknown')
                )

                self.notifications.notify_trade(
                    symbol=action['symbol'],
                    action="CLOSE",
                    price=action.get('exit_price', 0),
                    quantity=action.get('quantity', 0),
                    pnl=action.get('pnl', 0),
                    reason=action.get('action', 'Unknown')
                )

    # =========================================================================
    # AUTO TRADING LOOP
    # =========================================================================

    def run_trading_loop(self):
        """
        Boucle de trading automatique
        """
        logger.info("Starting trading loop...")
        self.notifications.notify_startup()
        self.running = True

        # Reset quotidien
        def daily_reset():
            self.trades_today = 0
            metrics = self.risk_manager.get_risk_metrics()
            self.notifications.notify_daily_summary(
                date=datetime.now().strftime("%Y-%m-%d"),
                total_trades=self.trader.total_trades if hasattr(self.trader, 'total_trades') else 0,
                winning_trades=self.trader.winning_trades if hasattr(self.trader, 'winning_trades') else 0,
                losing_trades=self.trader.losing_trades if hasattr(self.trader, 'losing_trades') else 0,
                total_pnl=self.trader.total_pnl if hasattr(self.trader, 'total_pnl') else 0,
                portfolio_value=metrics.current_capital,
                open_positions=len(self.trader.positions) if hasattr(self.trader, 'positions') else 0
            )

        # Scanner toutes les heures pendant marche ouvert
        def hourly_scan():
            if self.risk_manager.get_risk_metrics().can_trade:
                setups = self.scan_signals()
                # Executer le meilleur setup
                if setups:
                    best = setups[0]
                    self.execute_setup(best)

        # Update positions toutes les 5 minutes
        def position_update():
            self.update_positions()

        # Scheduler
        schedule.every().day.at("09:00").do(daily_reset)
        schedule.every().hour.do(hourly_scan)
        schedule.every(5).minutes.do(position_update)

        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Trading loop stopped by user")
            self.running = False

    def stop(self):
        """Arrete le bot"""
        self.running = False
        logger.info("TradingBot stopped")


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_scan(args):
    """Commande: Scanner les signaux"""
    print("=" * 60)
    print("SCANNING FOR SIGNALS")
    print("=" * 60)

    bot = TradingBot(mode="paper")

    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = WATCHLIST[:20]  # Top 20

    setups = bot.scan_signals(symbols)

    if not setups:
        print("\nNo signals found.")
        return

    print(f"\nFound {len(setups)} signals:\n")

    for i, setup in enumerate(setups, 1):
        direction_emoji = "🟢" if setup.direction == "buy" else "🔴"
        print(f"{i}. {direction_emoji} {setup.symbol}")
        print(f"   Entry: ${setup.entry_price:.2f} | SL: ${setup.stop_loss:.2f} | TP: ${setup.take_profit_final:.2f}")
        print(f"   R:R: 1:{setup.risk_reward:.1f} | Signal: {setup.signal_strength:.0%}")
        print(f"   Size: {setup.position_size} shares | Risk: ${setup.risk_amount:.2f}")
        print(f"   Reasons: {', '.join(setup.reasons)}")
        print()


def cmd_trade(args):
    """Commande: Mode trading"""
    mode = args.mode if args.mode else "paper"

    print("=" * 60)
    print(f"TRADING BOT - {mode.upper()} MODE")
    print("=" * 60)

    if mode == "live":
        confirm = input("WARNING: Live trading mode! Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    bot = TradingBot(mode=mode)

    if args.auto:
        print("\nStarting automatic trading loop...")
        print("Press Ctrl+C to stop.\n")
        bot.run_trading_loop()
    else:
        # Mode manuel
        print("\nManual mode. Commands:")
        print("  scan    - Scan for signals")
        print("  status  - Show status")
        print("  quit    - Exit")

        while True:
            try:
                cmd = input("\n> ").strip().lower()

                if cmd == "scan":
                    setups = bot.scan_signals()
                    for setup in setups[:5]:
                        print(f"  {setup.symbol}: {setup.direction} @ {setup.entry_price:.2f}")

                elif cmd == "status":
                    if hasattr(bot.trader, 'get_statistics'):
                        stats = bot.trader.get_statistics()
                        print(f"  Capital: ${stats['current_capital']:,.2f}")
                        print(f"  P&L: ${stats['total_pnl']:,.2f} ({stats['total_return_percent']:.2f}%)")
                        print(f"  Trades: {stats['total_trades']} (Win: {stats['win_rate']:.1f}%)")
                        print(f"  Positions: {stats['open_positions']}")

                elif cmd == "quit":
                    break

                else:
                    print("Unknown command")

            except KeyboardInterrupt:
                break

    print("\nGoodbye!")


def cmd_backtest(args):
    """Commande: Backtest"""
    print("=" * 60)
    print("BACKTEST")
    print("=" * 60)

    backtester = get_backtester()

    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    start_date = args.start if args.start else (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = args.end if args.end else datetime.now().strftime("%Y-%m-%d")

    print(f"\nSymbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${INITIAL_CAPITAL:,.2f}")
    print("\nRunning backtest...")

    result = backtester.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )

    backtester.print_results(result)


def cmd_dashboard(args):
    """Commande: Lancer dashboard"""
    import subprocess
    import os

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")

    print("=" * 60)
    print("LAUNCHING DASHBOARD")
    print("=" * 60)
    print(f"\nStarting Streamlit on port {args.port}...")
    print("Press Ctrl+C to stop.\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port", str(args.port),
        "--server.headless", "true"
    ])


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entree principal"""
    parser = argparse.ArgumentParser(
        description="Trading Bot - Based on MASTER_TRADING_SKILL",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for trading signals")
    scan_parser.add_argument("--symbols", "-s", help="Comma-separated symbols")

    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Start trading")
    trade_parser.add_argument("--mode", "-m", choices=["paper", "live"], default="paper")
    trade_parser.add_argument("--auto", "-a", action="store_true", help="Automatic mode")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--symbols", "-s", help="Comma-separated symbols")
    backtest_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", help="End date (YYYY-MM-DD)")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dashboard_parser.add_argument("--port", "-p", type=int, default=8501)

    args = parser.parse_args()

    # Execute command
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "trade":
        cmd_trade(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        # Mode interactif par defaut
        print("=" * 60)
        print("TRADING BOT")
        print("Based on MASTER_TRADING_SKILL")
        print("=" * 60)
        print("\nUsage:")
        print("  python main.py scan          # Scan for signals")
        print("  python main.py trade         # Start paper trading")
        print("  python main.py trade --live  # Start live trading")
        print("  python main.py backtest      # Run backtest")
        print("  python main.py dashboard     # Launch web interface")
        print("\nRun 'python main.py <command> --help' for more info.")


if __name__ == "__main__":
    main()
