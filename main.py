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
    TRADING_MODE, LOG_LEVEL,
    ACTIVE_STRATEGY, ENABLED_STRATEGIES, STRATEGY_WEIGHTS
)
from config.symbols import US_STOCKS, EU_STOCKS, WATCHLIST
from data.fetcher import get_fetcher
from data.database import get_database
from analysis.indicators import get_indicators
from analysis.signals import get_signal_generator
from strategy.swing_trading import get_swing_strategy, TradeSetup
from strategy.strategy_selector import (
    get_strategy_selector, StrategyType, UnifiedSignal, quick_analyze
)
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

    def __init__(self, mode: str = "paper", strategy: str = None):
        """
        Initialise le bot

        Args:
            mode: "paper" ou "live"
            strategy: Strategy to use (swing, wyckoff, elliott, ichimoku, volume_profile, combined)
        """
        self.mode = mode

        # Components
        self.fetcher = get_fetcher()
        self.swing_strategy = get_swing_strategy()
        self.strategy_selector = get_strategy_selector()
        self.risk_manager = get_risk_manager()
        self.notifications = get_notification_manager()
        self.db = get_database()

        # Configure strategy selector
        self._configure_strategies(strategy or ACTIVE_STRATEGY)

        # Trader
        if mode == "live":
            self.trader = get_live_trader()
        else:
            self.trader = get_paper_trader()

        # State
        self.running = False
        self.trades_today = 0
        self.last_scan = None

        logger.info(f"TradingBot initialized in {mode.upper()} mode with {self.active_strategy_name} strategy")

    def _configure_strategies(self, strategy_name: str):
        """Configure le selecteur de strategies"""
        # Map string to StrategyType
        strategy_map = {
            'swing': StrategyType.SWING,
            'wyckoff': StrategyType.WYCKOFF,
            'elliott': StrategyType.ELLIOTT,
            'ichimoku': StrategyType.ICHIMOKU,
            'volume_profile': StrategyType.VOLUME_PROFILE,
            'combined': StrategyType.COMBINED
        }

        strategy_type = strategy_map.get(strategy_name.lower(), StrategyType.COMBINED)
        self.strategy_selector.set_active_strategy(strategy_type)
        self.active_strategy_name = strategy_name

        # Configure enabled strategies from settings
        for name, enabled in ENABLED_STRATEGIES.items():
            if name in strategy_map:
                self.strategy_selector.enable_strategy(strategy_map[name], enabled)

        # Configure weights from settings
        for name, weight in STRATEGY_WEIGHTS.items():
            if name in strategy_map:
                self.strategy_selector.set_strategy_weight(strategy_map[name], weight)

        logger.info(f"Strategy configured: {strategy_name}")

    # =========================================================================
    # SCANNING
    # =========================================================================

    def scan_signals(self, symbols: List[str] = None) -> List[UnifiedSignal]:
        """
        Scanne les symboles pour trouver des signaux

        Args:
            symbols: Liste des symboles a scanner

        Returns:
            Liste des signaux unifies valides
        """
        if symbols is None:
            symbols = WATCHLIST

        logger.info(f"Scanning {len(symbols)} symbols with {self.active_strategy_name} strategy...")

        # Use strategy selector for scanning
        signals = self.strategy_selector.scan_watchlist(symbols, self.fetcher)

        logger.info(f"Found {len(signals)} valid signals")

        # Notifier
        for signal in signals:
            self.notifications.notify_signal(
                symbol=signal.symbol,
                direction=signal.direction,
                strength=signal.confidence,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_reward=signal.risk_reward,
                reasons=signal.reasons
            )

        self.last_scan = datetime.now()
        return signals

    def scan_with_swing(self, symbols: List[str] = None) -> List[TradeSetup]:
        """
        Scan avec la strategie Swing Trading originale (pour compatibilite)

        Args:
            symbols: Liste des symboles

        Returns:
            Liste des TradeSetup
        """
        if symbols is None:
            symbols = WATCHLIST

        return self.swing_strategy.scan_watchlist(symbols)

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
    strategy = args.strategy if hasattr(args, 'strategy') and args.strategy else ACTIVE_STRATEGY

    print("=" * 60)
    print(f"SCANNING FOR SIGNALS - {strategy.upper()} STRATEGY")
    print("=" * 60)

    bot = TradingBot(mode="paper", strategy=strategy)

    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = WATCHLIST[:20]  # Top 20

    signals = bot.scan_signals(symbols)

    if not signals:
        print("\nNo signals found.")
        return

    print(f"\nFound {len(signals)} signals:\n")

    for i, signal in enumerate(signals, 1):
        direction_emoji = "🟢" if signal.direction == "buy" else "🔴"
        strategy_tag = f"[{signal.strategy.value.upper()}]"
        print(f"{i}. {direction_emoji} {signal.symbol} {strategy_tag}")
        print(f"   Entry: ${signal.entry_price:.2f} | SL: ${signal.stop_loss:.2f} | TP: ${signal.take_profit:.2f}")
        print(f"   R:R: 1:{signal.risk_reward:.1f} | Confidence: {signal.confidence:.0%}")
        print(f"   Strength: {signal.strength.name}")
        print(f"   Reasons: {', '.join(signal.reasons[:3])}")  # Limit to 3 reasons
        if signal.metadata:
            meta_str = ", ".join(f"{k}: {v}" for k, v in list(signal.metadata.items())[:3] if v)
            print(f"   Details: {meta_str}")
        print()


def cmd_analyze(args):
    """Commande: Analyser un symbole avec toutes les strategies"""
    print("=" * 60)
    print(f"MULTI-STRATEGY ANALYSIS: {args.symbol}")
    print("=" * 60)

    fetcher = get_fetcher()
    df_daily = fetcher.get_daily_data(args.symbol)
    df_h1 = fetcher.get_hourly_data(args.symbol)

    if df_daily is None or len(df_daily) < 50:
        print(f"\nInsufficient data for {args.symbol}")
        return

    strategies = ['swing', 'wyckoff', 'elliott', 'ichimoku', 'volume_profile']

    print(f"\nAnalyzing with {len(strategies)} strategies...\n")

    results = []
    for strategy in strategies:
        try:
            signal = quick_analyze(args.symbol, strategy, df_daily, df_h1)
            if signal:
                results.append((strategy, signal))
                status = "🟢" if signal.direction == "buy" else "🔴" if signal.direction == "sell" else "⚪"
                print(f"{status} {strategy.upper():15} | {signal.direction:6} | Confidence: {signal.confidence:.0%}")
                if signal.reasons:
                    print(f"   └─ {signal.reasons[0]}")
            else:
                print(f"⚪ {strategy.upper():15} | No signal")
        except Exception as e:
            print(f"❌ {strategy.upper():15} | Error: {str(e)[:40]}")

    # Summary
    print("\n" + "-" * 60)
    if results:
        buy_count = sum(1 for _, s in results if s.direction == "buy")
        sell_count = sum(1 for _, s in results if s.direction == "sell")
        avg_confidence = sum(s.confidence for _, s in results) / len(results)

        print(f"SUMMARY: {buy_count} BUY, {sell_count} SELL signals")
        print(f"Average confidence: {avg_confidence:.0%}")

        if buy_count > sell_count and buy_count >= 2:
            print("\n✅ CONSENSUS: BUY")
        elif sell_count > buy_count and sell_count >= 2:
            print("\n✅ CONSENSUS: SELL")
        else:
            print("\n⚠️  NO CLEAR CONSENSUS")
    else:
        print("No valid signals from any strategy.")


def cmd_trade(args):
    """Commande: Mode trading"""
    mode = args.mode if args.mode else "paper"
    strategy = args.strategy if hasattr(args, 'strategy') and args.strategy else ACTIVE_STRATEGY

    print("=" * 60)
    print(f"TRADING BOT - {mode.upper()} MODE - {strategy.upper()} STRATEGY")
    print("=" * 60)

    if mode == "live":
        confirm = input("WARNING: Live trading mode! Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    bot = TradingBot(mode=mode, strategy=strategy)

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

    # Strategie et timeframe
    strategy = getattr(args, 'strategy', 'swing_trading')
    timeframe = getattr(args, 'timeframe', '1d')

    print(f"\nSymbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: {strategy}")
    print(f"Timeframe: {timeframe}")
    print(f"Capital: ${INITIAL_CAPITAL:,.2f}")
    print("\nRunning backtest...")

    result = backtester.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy,
        timeframe=timeframe
    )

    backtester.print_results(result)

    # Generer rapport visuel si demande
    if getattr(args, 'report', False):
        try:
            from backtest.visualizer import visualize_backtest
            report_path = visualize_backtest(result, "backtest_report.html", open_browser=True)
            print(f"\n📊 Rapport visuel genere: {report_path}")
        except ImportError as e:
            print(f"\n⚠️ Visualisation non disponible: {e}")
            print("   Installer plotly: pip install plotly")


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

    # Strategy choices
    strategy_choices = ["swing", "wyckoff", "elliott", "ichimoku", "volume_profile", "combined"]

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for trading signals")
    scan_parser.add_argument("--symbols", "-s", help="Comma-separated symbols")
    scan_parser.add_argument("--strategy", "-st", choices=strategy_choices, default=ACTIVE_STRATEGY,
                            help="Strategy to use (default: from settings)")

    # Analyze command (NEW)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze symbol with all strategies")
    analyze_parser.add_argument("symbol", help="Symbol to analyze (e.g., AAPL)")

    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Start trading")
    trade_parser.add_argument("--mode", "-m", choices=["paper", "live"], default="paper")
    trade_parser.add_argument("--strategy", "-st", choices=strategy_choices, default=ACTIVE_STRATEGY,
                             help="Strategy to use (default: from settings)")
    trade_parser.add_argument("--auto", "-a", action="store_true", help="Automatic mode")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--symbols", "-s", help="Comma-separated symbols")
    backtest_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--strategy", "-st", choices=strategy_choices, default=ACTIVE_STRATEGY,
                                help="Strategy to use (default: from settings)")
    backtest_parser.add_argument("--timeframe", "-tf", choices=["1d", "1h", "30m", "15m", "5m"], default="1d",
                                help="Timeframe (default: 1d)")
    backtest_parser.add_argument("--report", "-r", action="store_true",
                                help="Generate visual HTML report with trade charts")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dashboard_parser.add_argument("--port", "-p", type=int, default=8501)

    # Strategies command (NEW)
    strategies_parser = subparsers.add_parser("strategies", help="List available strategies")

    args = parser.parse_args()

    # Execute command
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "trade":
        cmd_trade(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "strategies":
        cmd_strategies()
    else:
        # Mode interactif par defaut
        print("=" * 60)
        print("TRADING BOT")
        print("Based on MASTER_TRADING_SKILL")
        print("=" * 60)
        print(f"\nActive Strategy: {ACTIVE_STRATEGY}")
        print("\nUsage:")
        print("  python main.py scan                    # Scan for signals")
        print("  python main.py scan -st wyckoff        # Scan with Wyckoff")
        print("  python main.py analyze AAPL            # Multi-strategy analysis")
        print("  python main.py trade                   # Start paper trading")
        print("  python main.py trade --live            # Start live trading")
        print("  python main.py backtest                # Run backtest")
        print("  python main.py dashboard               # Launch web interface")
        print("  python main.py strategies              # List all strategies")
        print("\nAvailable strategies: swing, wyckoff, elliott, ichimoku, volume_profile, combined")
        print("\nRun 'python main.py <command> --help' for more info.")


def cmd_strategies():
    """Commande: Lister les strategies disponibles"""
    print("=" * 60)
    print("AVAILABLE TRADING STRATEGIES")
    print("=" * 60)

    strategies = [
        ("swing", "Swing Trading Hybride", "PARTIE XII", "Daily + H1, zones, breakouts"),
        ("wyckoff", "Wyckoff Method", "PARTIE XIII", "Accumulation, Distribution, Spring/Upthrust"),
        ("elliott", "Elliott Wave", "PARTIE XIV", "5-3 waves, Fibonacci, wave counting"),
        ("ichimoku", "Ichimoku Kinko Hyo", "PARTIE XV", "Kumo, TK Cross, Chikou confirmation"),
        ("volume_profile", "Volume Profile", "PARTIE XVI", "POC, Value Area, HVN/LVN"),
        ("combined", "Combined (All)", "-", "Vote pondere de toutes les strategies"),
    ]

    print(f"\n{'Strategy':<15} {'Name':<25} {'Source':<12} {'Key Features'}")
    print("-" * 80)

    for code, name, source, features in strategies:
        enabled = ENABLED_STRATEGIES.get(code, True)
        weight = STRATEGY_WEIGHTS.get(code, 1.0)
        status = "✓" if enabled else "✗"
        print(f"{status} {code:<13} {name:<25} {source:<12} {features}")

    print(f"\nActive strategy: {ACTIVE_STRATEGY}")
    print("\nTo change strategy, use --strategy (-st) option or set ACTIVE_STRATEGY in settings.")


if __name__ == "__main__":
    main()
