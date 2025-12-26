"""
Dashboard Streamlit - Interface Web du Trading Bot
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import INITIAL_CAPITAL
from config.symbols import US_STOCKS, EU_STOCKS, WATCHLIST, get_all_us_stocks, get_all_eu_stocks
from data.fetcher import get_fetcher
from data.database import get_database
from analysis.indicators import get_indicators
from analysis.signals import get_signal_generator
from analysis.zones import get_zone_detector
from strategy.swing_trading import get_swing_strategy
from strategy.risk_management import get_risk_manager
from execution.paper_trader import get_paper_trader
from backtest.backtester import get_backtester
from dashboard.charts import (
    create_candlestick_chart,
    create_equity_curve,
    create_drawdown_chart,
    create_pnl_distribution,
    create_monthly_returns_heatmap,
    create_positions_pie,
    create_win_loss_chart,
    create_sector_exposure
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric label {
        color: #888888 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .positive {
        color: #00C853 !important;
    }
    .negative {
        color: #FF1744 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render la sidebar"""
    st.sidebar.title("📈 Trading Bot")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Analyse", "Signaux", "Positions", "Backtest", "Parametres"]
    )

    st.sidebar.markdown("---")

    # Status
    paper_trader = get_paper_trader()
    risk_manager = get_risk_manager()

    st.sidebar.subheader("Status")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Capital", f"${paper_trader.capital:,.0f}")
    with col2:
        pnl = paper_trader.total_pnl
        st.metric("P&L", f"${pnl:,.0f}", delta=f"{(pnl/INITIAL_CAPITAL)*100:.1f}%")

    st.sidebar.metric("Positions", len(paper_trader.positions))

    metrics = risk_manager.get_risk_metrics()
    if metrics.can_trade:
        st.sidebar.success("Trading actif")
    else:
        st.sidebar.error(f"Trading pause: {metrics.pause_reason}")

    return page


# =============================================================================
# DASHBOARD PAGE
# =============================================================================

def render_dashboard():
    """Page principale du dashboard"""
    st.title("Dashboard")

    paper_trader = get_paper_trader()
    db = get_database()

    # Metriques principales
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Valeur Portfolio",
            f"${paper_trader.capital:,.2f}",
            delta=f"{((paper_trader.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%"
        )

    with col2:
        st.metric(
            "P&L Total",
            f"${paper_trader.total_pnl:,.2f}",
            delta=f"{paper_trader.winning_trades}W / {paper_trader.losing_trades}L"
        )

    with col3:
        win_rate = (paper_trader.winning_trades / paper_trader.total_trades * 100) if paper_trader.total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")

    with col4:
        st.metric("Trades", paper_trader.total_trades)

    with col5:
        st.metric("Positions", len(paper_trader.positions))

    st.markdown("---")

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Positions Ouvertes")
        if paper_trader.positions:
            positions_data = []
            for symbol, pos in paper_trader.positions.items():
                positions_data.append({
                    'Symbol': symbol,
                    'Side': pos.side,
                    'Entry': f"${pos.entry_price:.2f}",
                    'Qty': pos.current_quantity,
                    'Stop': f"${pos.current_stop:.2f}",
                    'TP': f"${pos.take_profit:.2f}",
                    'TP1 Hit': "Yes" if pos.tp1_hit else "No"
                })
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else:
            st.info("Aucune position ouverte")

    with col2:
        st.subheader("Allocation")
        if paper_trader.positions:
            positions_list = [{
                'symbol': s,
                'market_value': p.entry_price * p.current_quantity,
                'unrealized_pnl': 0
            } for s, p in paper_trader.positions.items()]
            fig = create_positions_pie(positions_list)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune position")

    # Historique recent
    st.subheader("Trades Recents")
    if paper_trader.trade_history:
        recent_trades = paper_trader.trade_history[-10:]
        trades_data = []
        for t in reversed(recent_trades):
            trades_data.append({
                'Symbol': t['symbol'],
                'Side': t['side'],
                'Entry': f"${t['entry_price']:.2f}",
                'Exit': f"${t['exit_price']:.2f}",
                'P&L': f"${t['pnl']:.2f}",
                '%': f"{t['pnl_percent']:.2f}%",
                'Reason': t['reason']
            })
        st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
    else:
        st.info("Aucun trade")


# =============================================================================
# ANALYSE PAGE
# =============================================================================

def render_analyse():
    """Page d'analyse technique"""
    st.title("Analyse Technique")

    # Selection symbole
    col1, col2 = st.columns([1, 3])

    with col1:
        all_symbols = get_all_us_stocks() + get_all_eu_stocks()
        symbol = st.selectbox("Symbole", all_symbols, index=0)

        timeframe = st.selectbox("Timeframe", ["Daily", "Hourly"], index=0)

        show_zones = st.checkbox("Afficher Zones", value=True)
        show_indicators = st.checkbox("Afficher Indicateurs", value=True)

    # Charger donnees
    fetcher = get_fetcher()
    indicators = get_indicators()
    zone_detector = get_zone_detector()

    if timeframe == "Daily":
        df = fetcher.get_daily_data(symbol)
    else:
        df = fetcher.get_hourly_data(symbol)

    if df is None or len(df) < 20:
        st.error("Donnees insuffisantes")
        return

    # Ajouter indicateurs
    df = indicators.add_all_indicators(df)

    # Detecter zones
    zones = None
    if show_zones:
        zones = zone_detector.find_zones(df)

    # Creer graphique
    with col2:
        fig = create_candlestick_chart(
            df,
            symbol,
            zones=zones,
            show_indicators=show_indicators
        )
        st.plotly_chart(fig, use_container_width=True)

    # Indicateurs actuels
    st.subheader("Indicateurs")
    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rsi = latest.get('rsi', 0)
        rsi_color = "red" if rsi > 70 else ("green" if rsi < 30 else "gray")
        st.metric("RSI", f"{rsi:.1f}", delta="Surachat" if rsi > 70 else ("Survente" if rsi < 30 else "Neutre"))

    with col2:
        macd = latest.get('macd', 0)
        signal = latest.get('macd_signal', 0)
        st.metric("MACD", f"{macd:.4f}", delta=f"Signal: {signal:.4f}")

    with col3:
        atr = latest.get('atr', 0)
        st.metric("ATR", f"${atr:.2f}")

    with col4:
        trend = latest.get('trend_ema', 0)
        st.metric("Trend", "Haussier" if trend > 0 else "Baissier")


# =============================================================================
# SIGNAUX PAGE
# =============================================================================

def render_signaux():
    """Page des signaux"""
    st.title("Signaux de Trading")

    strategy = get_swing_strategy()

    # Scanner
    st.subheader("Scanner de Signaux")

    col1, col2 = st.columns([1, 3])

    with col1:
        watchlist_option = st.selectbox(
            "Watchlist",
            ["US Stocks", "EU Stocks", "Custom"]
        )

        if watchlist_option == "US Stocks":
            symbols = get_all_us_stocks()
        elif watchlist_option == "EU Stocks":
            symbols = get_all_eu_stocks()
        else:
            symbols = [s.strip() for s in st.text_area("Symboles (un par ligne)", "AAPL\nMSFT\nGOOGL").split("\n") if s.strip()]

        if st.button("Scanner", type="primary"):
            with st.spinner("Analyse en cours..."):
                setups = strategy.scan_watchlist(symbols[:10])  # Limite pour demo

                if setups:
                    st.session_state['setups'] = setups
                    st.success(f"{len(setups)} signaux trouves")
                else:
                    st.info("Aucun signal trouve")

    with col2:
        if 'setups' in st.session_state and st.session_state['setups']:
            setups = st.session_state['setups']

            for setup in setups:
                with st.expander(f"{'🟢' if setup.direction == 'buy' else '🔴'} {setup.symbol} - {setup.direction.upper()}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Entry", f"${setup.entry_price:.2f}")
                        st.metric("Stop Loss", f"${setup.stop_loss:.2f}")

                    with col2:
                        st.metric("TP1", f"${setup.take_profit_1:.2f}")
                        st.metric("TP Final", f"${setup.take_profit_final:.2f}")

                    with col3:
                        st.metric("R:R", f"1:{setup.risk_reward:.1f}")
                        st.metric("Signal", f"{setup.signal_strength:.0%}")

                    st.write("**Raisons:**")
                    for reason in setup.reasons:
                        st.write(f"- {reason}")

                    st.metric("Position Size", f"{setup.position_size} actions")

                    if st.button(f"Execute {setup.symbol}", key=f"exec_{setup.symbol}"):
                        paper_trader = get_paper_trader()
                        result = paper_trader.execute_buy(
                            symbol=setup.symbol,
                            quantity=setup.position_size,
                            price=setup.entry_price,
                            stop_loss=setup.stop_loss,
                            take_profit=setup.take_profit_final,
                            setup=setup
                        )
                        if result['success']:
                            strategy.add_active_setup(setup.symbol, setup)
                            st.success(f"Trade execute: {setup.symbol}")
                        else:
                            st.error(f"Erreur: {result.get('reason', 'Unknown')}")


# =============================================================================
# POSITIONS PAGE
# =============================================================================

def render_positions():
    """Page de gestion des positions"""
    st.title("Gestion des Positions")

    paper_trader = get_paper_trader()
    fetcher = get_fetcher()

    # Positions ouvertes
    st.subheader("Positions Ouvertes")

    if not paper_trader.positions:
        st.info("Aucune position ouverte")
    else:
        for symbol, pos in paper_trader.positions.items():
            with st.expander(f"{'🟢' if pos.side == 'long' else '🔴'} {symbol}", expanded=True):
                # Prix actuel
                df = fetcher.get_daily_data(symbol)
                current_price = df['close'].iloc[-1] if df is not None else pos.entry_price

                # Calcul P&L
                if pos.side == 'long':
                    unrealized_pnl = (current_price - pos.entry_price) * pos.current_quantity
                else:
                    unrealized_pnl = (pos.entry_price - current_price) * pos.current_quantity

                pnl_percent = (unrealized_pnl / (pos.entry_price * pos.quantity)) * 100

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Entry", f"${pos.entry_price:.2f}")
                    st.metric("Current", f"${current_price:.2f}")

                with col2:
                    st.metric("Quantity", pos.current_quantity)
                    st.metric("Side", pos.side.upper())

                with col3:
                    st.metric("Stop Loss", f"${pos.current_stop:.2f}")
                    st.metric("Take Profit", f"${pos.take_profit:.2f}")

                with col4:
                    delta_color = "normal" if unrealized_pnl >= 0 else "inverse"
                    st.metric("P&L", f"${unrealized_pnl:.2f}", delta=f"{pnl_percent:.2f}%")
                    st.metric("TP1", "Hit" if pos.tp1_hit else "Pending")

                # Actions
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button(f"Fermer {symbol}", key=f"close_{symbol}"):
                        result = paper_trader.close_position(symbol, current_price, "Manual Close")
                        if result['success']:
                            st.success(f"Position fermee: P&L ${result['pnl']:.2f}")
                            st.rerun()

                with col2:
                    new_sl = st.number_input(f"Nouveau SL {symbol}", value=pos.current_stop, key=f"sl_{symbol}")
                    if st.button(f"Modifier SL", key=f"mod_sl_{symbol}"):
                        pos.current_stop = new_sl
                        st.success("Stop Loss modifie")

    # Historique
    st.markdown("---")
    st.subheader("Historique des Trades")

    if paper_trader.trade_history:
        df_trades = pd.DataFrame(paper_trader.trade_history)
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])

        st.dataframe(df_trades, use_container_width=True)

        # Graphique P&L
        fig = create_win_loss_chart([type('Trade', (), t)() for t in paper_trader.trade_history])
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# BACKTEST PAGE
# =============================================================================

def render_backtest():
    """Page de backtest"""
    st.title("Backtest")

    backtester = get_backtester()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parametres")

        symbols_input = st.text_area(
            "Symboles (un par ligne)",
            "AAPL\nMSFT\nGOOGL\nAMZN\nMETA"
        )
        symbols = [s.strip() for s in symbols_input.split("\n") if s.strip()]

        start_date = st.date_input(
            "Date debut",
            datetime.now() - timedelta(days=365)
        )

        end_date = st.date_input(
            "Date fin",
            datetime.now()
        )

        initial_capital = st.number_input(
            "Capital initial",
            value=float(INITIAL_CAPITAL),
            step=1000.0
        )

        if st.button("Lancer Backtest", type="primary"):
            with st.spinner("Backtest en cours..."):
                backtester.initial_capital = initial_capital
                result = backtester.run(
                    symbols=symbols,
                    start_date=str(start_date),
                    end_date=str(end_date)
                )
                st.session_state['backtest_result'] = result
                st.success("Backtest termine!")

    with col2:
        if 'backtest_result' in st.session_state:
            result = st.session_state['backtest_result']

            # Metriques
            st.subheader("Resultats")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Return", f"{result.total_return_percent:.2f}%")
                st.metric("CAGR", f"{result.cagr:.2f}%")

            with col2:
                st.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
                st.metric("Sortino", f"{result.sortino_ratio:.2f}")

            with col3:
                st.metric("Win Rate", f"{result.win_rate:.1f}%")
                st.metric("Profit Factor", f"{result.profit_factor:.2f}")

            with col4:
                st.metric("Max DD", f"{result.max_drawdown_percent:.2f}%")
                st.metric("Trades", result.total_trades)

            # Equity curve
            if result.equity_curve is not None:
                st.subheader("Equity Curve")
                fig = create_equity_curve(result.equity_curve)
                st.plotly_chart(fig, use_container_width=True)

                # Drawdown
                st.subheader("Drawdown")
                fig = create_drawdown_chart(result.equity_curve)
                st.plotly_chart(fig, use_container_width=True)

            # Distribution
            if result.trades:
                st.subheader("Distribution P&L")
                fig = create_pnl_distribution(result.trades)
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PARAMETRES PAGE
# =============================================================================

def render_parametres():
    """Page des parametres"""
    st.title("Parametres")

    st.subheader("Configuration Trading")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Capital Initial ($)", value=float(INITIAL_CAPITAL), disabled=True)
        st.number_input("Risque par Trade (%)", value=2.0, disabled=True)
        st.number_input("R:R Minimum", value=3.0, disabled=True)

    with col2:
        st.number_input("Max Positions", value=5.0, disabled=True)
        st.number_input("Max Daily Loss (%)", value=5.0, disabled=True)
        st.number_input("Max Trades/Jour", value=5.0, disabled=True)

    st.info("Les parametres sont configures dans config/settings.py")

    st.markdown("---")

    st.subheader("Notifications")
    st.text_input("Discord Webhook URL", type="password", placeholder="https://discord.com/api/webhooks/...")
    st.checkbox("Activer notifications trades")
    st.checkbox("Activer notifications signaux")
    st.checkbox("Activer resume quotidien")

    st.markdown("---")

    st.subheader("API Alpaca (Live Trading)")
    st.text_input("API Key", type="password")
    st.text_input("Secret Key", type="password")
    st.selectbox("Environment", ["Paper", "Live"])

    if st.button("Sauvegarder"):
        st.success("Parametres sauvegardes (simulation)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entree principal"""
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Analyse":
        render_analyse()
    elif page == "Signaux":
        render_signaux()
    elif page == "Positions":
        render_positions()
    elif page == "Backtest":
        render_backtest()
    elif page == "Parametres":
        render_parametres()


if __name__ == "__main__":
    main()
