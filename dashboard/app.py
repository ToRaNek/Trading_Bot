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

from config.settings import (
    INITIAL_CAPITAL, ACTIVE_STRATEGY, ENABLED_STRATEGIES, STRATEGY_WEIGHTS
)
from config.symbols import US_STOCKS, EU_STOCKS, WATCHLIST, get_all_us_stocks, get_all_eu_stocks
from data.fetcher import get_fetcher
from data.database import get_database
from analysis.indicators import get_indicators
from analysis.signals import get_signal_generator
from analysis.zones import get_zone_detector
from strategy.swing_trading import get_swing_strategy
from strategy.strategy_selector import (
    get_strategy_selector, StrategyType, UnifiedSignal, CombinedSignal, quick_analyze
)
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
        ["Dashboard", "Analyse", "Multi-Analyse", "Signaux", "Positions", "Backtest", "Parametres"]
    )

    st.sidebar.markdown("---")

    # Strategie Active
    strategy_selector = get_strategy_selector()
    st.sidebar.subheader("Strategie")

    strategy_names = {
        StrategyType.SWING: "Swing Trading",
        StrategyType.WYCKOFF: "Wyckoff",
        StrategyType.ELLIOTT: "Elliott Wave",
        StrategyType.ICHIMOKU: "Ichimoku",
        StrategyType.VOLUME_PROFILE: "Volume Profile",
        StrategyType.COMBINED: "Combinee"
    }

    # Determiner la strategie affichee
    if strategy_selector.combination_mode or strategy_selector.active_strategy is None:
        strategy_display = "Combinee"
    else:
        current_strategy = strategy_selector.active_strategy
        strategy_display = strategy_names.get(current_strategy, current_strategy.value)

    st.sidebar.info(f"Active: **{strategy_display}**")

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
# MULTI-ANALYSE PAGE
# =============================================================================

def render_multi_analyse():
    """Page d'analyse multi-strategies"""
    st.title("Multi-Analyse")
    st.caption("Analysez un symbole avec toutes les strategies disponibles")

    strategy_selector = get_strategy_selector()
    fetcher = get_fetcher()

    col1, col2 = st.columns([1, 3])

    with col1:
        # Selection du symbole
        all_symbols = get_all_us_stocks() + get_all_eu_stocks()
        symbol = st.selectbox("Symbole", all_symbols, index=0)

        st.markdown("---")

        # Afficher les strategies actives
        st.subheader("Strategies Actives")
        for name, enabled in ENABLED_STRATEGIES.items():
            weight = STRATEGY_WEIGHTS.get(name, 1.0)
            status = "✅" if enabled else "❌"
            st.write(f"{status} **{name.title()}** (poids: {weight})")

        st.markdown("---")

        if st.button("Analyser", type="primary"):
            with st.spinner(f"Analyse de {symbol} avec toutes les strategies..."):
                # Charger les donnees
                df = fetcher.get_daily_data(symbol)

                if df is None or len(df) < 50:
                    st.error("Donnees insuffisantes pour l'analyse")
                else:
                    # Analyser avec toutes les strategies
                    results = strategy_selector.analyze_all(symbol, df)
                    st.session_state['multi_results'] = results
                    st.session_state['multi_symbol'] = symbol

                    # Obtenir le consensus
                    combined = strategy_selector.get_combined_signal(symbol, df)
                    st.session_state['consensus'] = combined

                    st.success("Analyse terminee!")

    with col2:
        if 'multi_results' in st.session_state:
            symbol = st.session_state.get('multi_symbol', 'N/A')
            results = st.session_state['multi_results']
            consensus = st.session_state.get('consensus')

            st.subheader(f"Resultats pour {symbol}")

            # Consensus
            if consensus:
                st.markdown("### Consensus")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    icon = "🟢" if consensus.direction == 'buy' else ("🔴" if consensus.direction == 'sell' else "⚪")
                    st.metric("Signal", f"{icon} {consensus.direction.upper()}")

                with col2:
                    st.metric("Score", f"{consensus.consensus_score:.0%}")

                with col3:
                    st.metric("R:R", f"1:{consensus.risk_reward:.1f}")

                with col4:
                    st.metric("Confiance", f"{consensus.final_confidence:.0%}")

                # Votes par strategie
                st.write("**Strategies contribuantes:**")
                votes_df = []
                for signal in consensus.signals:
                    votes_df.append({
                        'Strategie': signal.strategy.value.title(),
                        'Signal': signal.direction.upper(),
                        'Confiance': f"{signal.confidence:.0%}",
                        'Poids': f"{strategy_selector.configs[signal.strategy].weight:.1f}"
                    })
                st.dataframe(pd.DataFrame(votes_df), use_container_width=True)

            st.markdown("---")

            # Detail par strategie
            st.markdown("### Detail par Strategie")

            for strat_type, signal in results.items():
                strat_name = strat_type.value.replace('_', ' ').title()
                if signal is None:
                    with st.expander(f"⚪ {strat_name} - Pas de signal"):
                        st.info("Aucun signal genere par cette strategie")
                    continue

                icon = "🟢" if signal.direction == 'buy' else ("🔴" if signal.direction == 'sell' else "⚪")
                with st.expander(f"{icon} {strat_name} - {signal.direction.upper()}", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Entry", f"${signal.entry_price:.2f}")
                        st.metric("Stop Loss", f"${signal.stop_loss:.2f}")

                    with col2:
                        if signal.take_profit:
                            st.metric("Take Profit", f"${signal.take_profit:.2f}")
                        if signal.risk_reward:
                            st.metric("R:R", f"1:{signal.risk_reward:.1f}")

                    with col3:
                        st.metric("Force", signal.strength.name)
                        st.metric("Confiance", f"{signal.confidence:.0%}")

                    # Raisons
                    if signal.reasons:
                        st.write("**Raisons:**")
                        for reason in signal.reasons:
                            st.write(f"- {reason}")

                    # Metadata specifique
                    if signal.metadata:
                        with st.expander("Details techniques"):
                            for key, value in signal.metadata.items():
                                if isinstance(value, float):
                                    st.write(f"**{key}:** {value:.4f}")
                                else:
                                    st.write(f"**{key}:** {value}")


# =============================================================================
# SIGNAUX PAGE
# =============================================================================

def render_signaux():
    """Page des signaux"""
    st.title("Signaux de Trading")

    strategy_selector = get_strategy_selector()

    # Scanner
    st.subheader("Scanner de Signaux")

    col1, col2 = st.columns([1, 3])

    with col1:
        # Selection de la strategie
        strategy_options = {
            "Swing Trading": StrategyType.SWING,
            "Wyckoff": StrategyType.WYCKOFF,
            "Elliott Wave": StrategyType.ELLIOTT,
            "Ichimoku": StrategyType.ICHIMOKU,
            "Volume Profile": StrategyType.VOLUME_PROFILE,
            "Combinee (toutes)": StrategyType.COMBINED
        }

        selected_strategy = st.selectbox(
            "Strategie",
            list(strategy_options.keys()),
            index=5  # Combinee par defaut
        )

        # Changer la strategie active si necessaire
        strategy_type = strategy_options[selected_strategy]
        current_is_combined = strategy_selector.combination_mode or strategy_selector.active_strategy is None
        new_is_combined = strategy_type == StrategyType.COMBINED

        if new_is_combined != current_is_combined or (not new_is_combined and strategy_type != strategy_selector.active_strategy):
            strategy_selector.set_active_strategy(strategy_type)

        st.markdown("---")

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
            with st.spinner(f"Analyse avec {selected_strategy}..."):
                signals = []
                for symbol in symbols[:10]:  # Limite pour demo
                    try:
                        df = fetcher.get_daily_data(symbol)
                        if df is not None and len(df) >= 50:
                            signal = strategy_selector.analyze(symbol, df)
                            if signal and signal.direction != 'neutral' and signal.confidence >= 0.6:
                                signals.append(signal)
                    except Exception as e:
                        pass  # Continuer avec les autres symboles

                if signals:
                    st.session_state['signals'] = signals
                    st.session_state['selected_strategy'] = selected_strategy
                    st.success(f"{len(signals)} signaux trouves")
                else:
                    st.info("Aucun signal trouve")

    with col2:
        # Afficher les nouveaux signaux (UnifiedSignal)
        if 'signals' in st.session_state and st.session_state['signals']:
            signals = st.session_state['signals']
            strategy_name = st.session_state.get('selected_strategy', 'N/A')

            st.caption(f"Resultats avec: **{strategy_name}**")

            for signal in signals:
                icon = "🟢" if signal.direction == 'buy' else "🔴"
                with st.expander(f"{icon} {signal.symbol} - {signal.direction.upper()} ({signal.strategy.value})"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Entry", f"${signal.entry_price:.2f}")
                        st.metric("Stop Loss", f"${signal.stop_loss:.2f}")

                    with col2:
                        if signal.take_profit:
                            st.metric("Take Profit", f"${signal.take_profit:.2f}")
                        if signal.risk_reward:
                            st.metric("R:R", f"1:{signal.risk_reward:.1f}")

                    with col3:
                        st.metric("Force", signal.strength.name)
                        st.metric("Confiance", f"{signal.confidence:.0%}")

                    st.write("**Raisons:**")
                    for reason in signal.reasons:
                        st.write(f"- {reason}")

                    # Bouton d'execution
                    if st.button(f"Execute {signal.symbol}", key=f"exec_{signal.symbol}_{signal.strategy.value}"):
                        paper_trader = get_paper_trader()
                        # Calculer la taille de position
                        from strategy.position_sizing import get_position_sizer
                        sizer = get_position_sizer()
                        sizing = sizer.calculate_position_size(
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss
                        )
                        quantity = sizing['shares']

                        result = paper_trader.execute_buy(
                            symbol=signal.symbol,
                            quantity=quantity,
                            price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit or (signal.entry_price * 1.06),
                            setup=None
                        )
                        if result['success']:
                            st.success(f"Trade execute: {signal.symbol} ({quantity} actions)")
                        else:
                            st.error(f"Erreur: {result.get('reason', 'Unknown')}")

        # Support ancien format (setups) pour compatibilite
        elif 'setups' in st.session_state and st.session_state['setups']:
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

                    if st.button(f"Execute {setup.symbol}", key=f"exec_old_{setup.symbol}"):
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

        # Selection de strategie pour le backtest
        strategy_options = {
            "Swing Trading": "swing_trading",
            "Wyckoff": "wyckoff",
            "Elliott Wave": "elliott",
            "Ichimoku": "ichimoku",
            "Volume Profile": "volume_profile",
            "Combinee (toutes)": "combined"
        }
        selected_strategy = st.selectbox(
            "Strategie",
            list(strategy_options.keys()),
            index=0,
            help="Choisissez la strategie a utiliser pour le backtest"
        )
        strategy_value = strategy_options[selected_strategy]

        # Selection du timeframe
        timeframe_options = {
            "Daily (1d)": "1d",
            "Hourly (1h)": "1h",
            "30 minutes": "30m",
            "15 minutes": "15m",
            "5 minutes": "5m",
        }
        selected_timeframe = st.selectbox(
            "Timeframe",
            list(timeframe_options.keys()),
            index=0,
            help="Timeframe des donnees. Note: 5m/15m/30m limites a 60 jours max"
        )
        timeframe_value = timeframe_options[selected_timeframe]

        # Avertissement pour timeframes courts
        if timeframe_value in ['5m', '15m', '30m']:
            st.warning("⚠️ Les donnees intraday sont limitees a ~60 jours par Yahoo Finance")

        if st.button("Lancer Backtest", type="primary"):
            with st.spinner(f"Backtest en cours avec {selected_strategy} ({selected_timeframe})..."):
                backtester.initial_capital = initial_capital
                result = backtester.run(
                    symbols=symbols,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    strategy=strategy_value,
                    timeframe=timeframe_value
                )
                st.session_state['backtest_result'] = result
                st.session_state['backtest_strategy'] = selected_strategy
                st.session_state['backtest_timeframe'] = selected_timeframe
                st.success(f"Backtest termine avec {selected_strategy} ({selected_timeframe})!")

    with col2:
        if 'backtest_result' in st.session_state:
            result = st.session_state['backtest_result']

            # Metriques
            strategy_used = st.session_state.get('backtest_strategy', 'Swing Trading')
            timeframe_used = st.session_state.get('backtest_timeframe', 'Daily (1d)')
            st.subheader(f"Resultats - {strategy_used} ({timeframe_used})")

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

            # =========================================================
            # TABLEAU DES TRADES CLIQUABLE
            # =========================================================
            if result.trades:
                st.markdown("---")
                st.subheader("📋 Historique des Trades")

                # Creer DataFrame des trades
                trades_data = []
                for i, trade in enumerate(result.trades, 1):
                    trades_data.append({
                        '#': i,
                        'Symbol': trade.symbol,
                        'Side': trade.side.upper(),
                        'Entry Date': trade.entry_date.strftime("%Y-%m-%d %H:%M") if trade.entry_date else "N/A",
                        'Entry Price': f"${trade.entry_price:.2f}",
                        'Exit Date': trade.exit_date.strftime("%Y-%m-%d %H:%M") if trade.exit_date else "N/A",
                        'Exit Price': f"${trade.exit_price:.2f}" if trade.exit_price else "N/A",
                        'P&L': trade.pnl,
                        'P&L %': trade.pnl_percent,
                        'Exit Reason': trade.exit_reason
                    })

                df_trades = pd.DataFrame(trades_data)

                # Afficher le tableau avec style
                def highlight_pnl(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return 'color: #00C853'
                        elif val < 0:
                            return 'color: #FF1744'
                    return ''

                styled_df = df_trades.style.applymap(highlight_pnl, subset=['P&L', 'P&L %'])
                styled_df = styled_df.format({'P&L': '${:.2f}', 'P&L %': '{:.1f}%'})

                st.dataframe(styled_df, use_container_width=True, height=300)

                # Selection du trade pour voir le graphique
                st.markdown("---")
                st.subheader("🔍 Detail du Trade")

                trade_numbers = list(range(1, len(result.trades) + 1))
                selected_trade_num = st.selectbox(
                    "Selectionner un trade pour voir le graphique",
                    trade_numbers,
                    format_func=lambda x: f"Trade #{x} - {result.trades[x-1].symbol} ({result.trades[x-1].side}) - P&L: ${result.trades[x-1].pnl:.2f}"
                )

                if selected_trade_num:
                    selected_trade = result.trades[selected_trade_num - 1]

                    # Infos du trade
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Symbol", selected_trade.symbol)
                        st.metric("Side", selected_trade.side.upper())
                    with col2:
                        st.metric("Entry", f"${selected_trade.entry_price:.2f}")
                        st.metric("Exit", f"${selected_trade.exit_price:.2f}" if selected_trade.exit_price else "N/A")
                    with col3:
                        st.metric("Stop Loss", f"${selected_trade.stop_loss:.2f}")
                        st.metric("Take Profit", f"${selected_trade.take_profit:.2f}")
                    with col4:
                        pnl_delta = selected_trade.pnl
                        st.metric("P&L", f"${selected_trade.pnl:.2f}", delta=f"{selected_trade.pnl_percent:.1f}%")
                        st.metric("Exit Reason", selected_trade.exit_reason)

                    # Graphique du trade
                    st.markdown("##### Graphique du Trade")
                    try:
                        fetcher = get_fetcher()
                        df_symbol = fetcher.get_daily_data(selected_trade.symbol)

                        if df_symbol is not None and len(df_symbol) > 0:
                            # Filtrer autour du trade
                            if selected_trade.entry_date and selected_trade.exit_date:
                                # Trouver les index
                                try:
                                    mask = (df_symbol.index >= selected_trade.entry_date - timedelta(days=30)) & \
                                           (df_symbol.index <= selected_trade.exit_date + timedelta(days=10))
                                    df_plot = df_symbol[mask]
                                except:
                                    df_plot = df_symbol.tail(60)
                            else:
                                df_plot = df_symbol.tail(60)

                            if len(df_plot) > 0:
                                # Creer graphique avec Plotly
                                import plotly.graph_objects as go

                                fig = go.Figure()

                                # Candlesticks
                                fig.add_trace(go.Candlestick(
                                    x=df_plot.index,
                                    open=df_plot['open'],
                                    high=df_plot['high'],
                                    low=df_plot['low'],
                                    close=df_plot['close'],
                                    name='Price',
                                    increasing_line_color='#00C853',
                                    decreasing_line_color='#FF1744'
                                ))

                                # Marker entree
                                entry_color = '#00C853' if selected_trade.side == 'long' else '#FF1744'
                                entry_symbol = 'triangle-up' if selected_trade.side == 'long' else 'triangle-down'
                                fig.add_trace(go.Scatter(
                                    x=[selected_trade.entry_date],
                                    y=[selected_trade.entry_price],
                                    mode='markers',
                                    marker=dict(symbol=entry_symbol, size=18, color=entry_color, line=dict(width=2, color='white')),
                                    name=f'Entry ({selected_trade.side})',
                                    hovertemplate=f"ENTRY<br>{selected_trade.side.upper()}<br>${selected_trade.entry_price:.2f}<extra></extra>"
                                ))

                                # Marker sortie
                                if selected_trade.exit_date and selected_trade.exit_price:
                                    exit_color = '#00C853' if selected_trade.pnl >= 0 else '#FF1744'
                                    fig.add_trace(go.Scatter(
                                        x=[selected_trade.exit_date],
                                        y=[selected_trade.exit_price],
                                        mode='markers',
                                        marker=dict(symbol='x', size=15, color=exit_color, line=dict(width=3, color=exit_color)),
                                        name=f'Exit ({selected_trade.exit_reason})',
                                        hovertemplate=f"EXIT<br>{selected_trade.exit_reason}<br>${selected_trade.exit_price:.2f}<extra></extra>"
                                    ))

                                # Ligne Stop Loss
                                fig.add_hline(y=selected_trade.stop_loss, line_dash="dash", line_color="#FF1744",
                                              annotation_text=f"SL: ${selected_trade.stop_loss:.2f}")

                                # Ligne Take Profit
                                fig.add_hline(y=selected_trade.take_profit, line_dash="dash", line_color="#00C853",
                                              annotation_text=f"TP: ${selected_trade.take_profit:.2f}")

                                # Layout
                                fig.update_layout(
                                    title=f"{selected_trade.symbol} - Trade #{selected_trade_num}",
                                    template='plotly_dark',
                                    height=500,
                                    xaxis_rangeslider_visible=False,
                                    showlegend=True,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Pas assez de donnees pour afficher le graphique")
                        else:
                            st.warning(f"Donnees non disponibles pour {selected_trade.symbol}")
                    except Exception as e:
                        st.error(f"Erreur lors du chargement du graphique: {e}")


# =============================================================================
# PARAMETRES PAGE
# =============================================================================

def render_parametres():
    """Page des parametres - EDITABLE"""
    from config.user_settings import load_user_settings, save_user_settings, reset_to_defaults

    st.title("⚙️ Parametres")

    # Charger les settings actuels
    settings = load_user_settings()

    # Tabs pour organiser
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Capital & Risk",
        "📊 Strategies",
        "📈 Indicateurs",
        "🔔 Notifications",
        "🔑 API"
    ])

    # =========================================================
    # TAB 1: CAPITAL & RISK
    # =========================================================
    with tab1:
        st.subheader("Gestion du Capital")

        col1, col2 = st.columns(2)

        with col1:
            settings["INITIAL_CAPITAL"] = st.number_input(
                "Capital Initial ($)",
                value=float(settings.get("INITIAL_CAPITAL", 10000)),
                min_value=100.0,
                step=1000.0,
                help="Capital de depart pour le trading"
            )

            settings["RISK_PER_TRADE"] = st.slider(
                "Risque par Trade (%)",
                min_value=0.5,
                max_value=5.0,
                value=float(settings.get("RISK_PER_TRADE", 0.02)) * 100,
                step=0.5,
                help="JAMAIS plus de 2% recommande"
            ) / 100

            settings["MIN_RISK_REWARD"] = st.slider(
                "Risk/Reward Minimum",
                min_value=1.0,
                max_value=5.0,
                value=float(settings.get("MIN_RISK_REWARD", 3.0)),
                step=0.5,
                help="Ratio R:R minimum pour entrer en position"
            )

        with col2:
            settings["MAX_OPEN_POSITIONS"] = st.number_input(
                "Max Positions Ouvertes",
                value=int(settings.get("MAX_OPEN_POSITIONS", 5)),
                min_value=1,
                max_value=20,
                help="Nombre maximum de positions simultanees"
            )

            settings["MAX_POSITION_PERCENT"] = st.slider(
                "Max % Capital par Position",
                min_value=5,
                max_value=50,
                value=int(settings.get("MAX_POSITION_PERCENT", 0.20) * 100),
                step=5,
                help="Pourcentage maximum du capital par position"
            ) / 100

            settings["MAX_SECTOR_EXPOSURE"] = st.slider(
                "Max % Exposition Secteur",
                min_value=10,
                max_value=100,
                value=int(settings.get("MAX_SECTOR_EXPOSURE", 0.30) * 100),
                step=10,
                help="Exposition maximum par secteur"
            ) / 100

        st.markdown("---")
        st.subheader("Gestion des Pertes")

        col1, col2 = st.columns(2)

        with col1:
            settings["DAILY_MAX_LOSS"] = st.slider(
                "Max Perte Journaliere (%)",
                min_value=1,
                max_value=20,
                value=int(settings.get("DAILY_MAX_LOSS", 0.05) * 100),
                step=1,
                help="Arrete le trading si atteint"
            ) / 100

            settings["MAX_TRADES_PER_DAY"] = st.number_input(
                "Max Trades par Jour",
                value=int(settings.get("MAX_TRADES_PER_DAY", 5)),
                min_value=1,
                max_value=20,
                help="Anti-overtrading"
            )

        with col2:
            settings["MAX_CONSECUTIVE_LOSSES"] = st.number_input(
                "Max Pertes Consecutives",
                value=int(settings.get("MAX_CONSECUTIVE_LOSSES", 3)),
                min_value=1,
                max_value=10,
                help="Pause apres X pertes consecutives"
            )

            settings["DRAWDOWN_STOP_TRADING"] = st.slider(
                "Drawdown Stop (%)",
                min_value=5,
                max_value=30,
                value=int(settings.get("DRAWDOWN_STOP_TRADING", 0.10) * 100),
                step=5,
                help="Arrete le trading si drawdown atteint"
            ) / 100

    # =========================================================
    # TAB 2: STRATEGIES
    # =========================================================
    with tab2:
        st.subheader("Strategie Active")

        strategy_options = ["combined", "swing", "wyckoff", "elliott", "ichimoku", "volume_profile"]
        strategy_labels = {
            "combined": "Combinee (toutes)",
            "swing": "Swing Trading",
            "wyckoff": "Wyckoff",
            "elliott": "Elliott Wave",
            "ichimoku": "Ichimoku",
            "volume_profile": "Volume Profile"
        }

        current_strategy = settings.get("ACTIVE_STRATEGY", "combined")
        selected_idx = strategy_options.index(current_strategy) if current_strategy in strategy_options else 0

        settings["ACTIVE_STRATEGY"] = st.selectbox(
            "Strategie",
            strategy_options,
            index=selected_idx,
            format_func=lambda x: strategy_labels.get(x, x)
        )

        st.markdown("---")
        st.subheader("Strategies Activees (Mode Combine)")

        enabled = settings.get("ENABLED_STRATEGIES", {})
        col1, col2 = st.columns(2)

        with col1:
            enabled["swing"] = st.checkbox("Swing Trading", value=enabled.get("swing", True))
            enabled["wyckoff"] = st.checkbox("Wyckoff", value=enabled.get("wyckoff", True))
            enabled["elliott"] = st.checkbox("Elliott Wave", value=enabled.get("elliott", True))

        with col2:
            enabled["ichimoku"] = st.checkbox("Ichimoku", value=enabled.get("ichimoku", True))
            enabled["volume_profile"] = st.checkbox("Volume Profile", value=enabled.get("volume_profile", True))

        settings["ENABLED_STRATEGIES"] = enabled

        st.markdown("---")
        st.subheader("Poids des Strategies")
        st.caption("Influence de chaque strategie dans le vote combine")

        weights = settings.get("STRATEGY_WEIGHTS", {})
        cols = st.columns(5)

        strategies = ["swing", "wyckoff", "elliott", "ichimoku", "volume_profile"]
        for i, strat in enumerate(strategies):
            with cols[i]:
                weights[strat] = st.slider(
                    strat.title(),
                    min_value=0.0,
                    max_value=2.0,
                    value=float(weights.get(strat, 1.0)),
                    step=0.1
                )

        settings["STRATEGY_WEIGHTS"] = weights

        st.markdown("---")
        st.subheader("Timeframes")

        col1, col2, col3 = st.columns(3)

        with col1:
            tf_options = ["1d", "1h", "30m", "15m", "5m"]
            settings["PRIMARY_TIMEFRAME"] = st.selectbox(
                "Timeframe Principal",
                tf_options,
                index=tf_options.index(settings.get("PRIMARY_TIMEFRAME", "1d"))
            )

        with col2:
            settings["SECONDARY_TIMEFRAME"] = st.selectbox(
                "Timeframe Secondaire",
                tf_options,
                index=tf_options.index(settings.get("SECONDARY_TIMEFRAME", "1h"))
            )

        with col3:
            settings["LOOKBACK_DAYS"] = st.number_input(
                "Lookback (jours)",
                value=int(settings.get("LOOKBACK_DAYS", 90)),
                min_value=30,
                max_value=365
            )

    # =========================================================
    # TAB 3: INDICATEURS
    # =========================================================
    with tab3:
        indicators = settings.get("INDICATORS", {})

        st.subheader("RSI")
        col1, col2, col3 = st.columns(3)
        with col1:
            indicators["rsi_period"] = st.number_input("Periode RSI", value=int(indicators.get("rsi_period", 14)), min_value=5, max_value=30)
        with col2:
            indicators["rsi_overbought"] = st.number_input("Surachat", value=int(indicators.get("rsi_overbought", 70)), min_value=60, max_value=90)
        with col3:
            indicators["rsi_oversold"] = st.number_input("Survente", value=int(indicators.get("rsi_oversold", 30)), min_value=10, max_value=40)

        st.markdown("---")
        st.subheader("MACD")
        col1, col2, col3 = st.columns(3)
        with col1:
            indicators["macd_fast"] = st.number_input("Fast", value=int(indicators.get("macd_fast", 12)), min_value=5, max_value=20)
        with col2:
            indicators["macd_slow"] = st.number_input("Slow", value=int(indicators.get("macd_slow", 26)), min_value=15, max_value=40)
        with col3:
            indicators["macd_signal"] = st.number_input("Signal", value=int(indicators.get("macd_signal", 9)), min_value=5, max_value=15)

        st.markdown("---")
        st.subheader("Moyennes Mobiles")
        col1, col2, col3 = st.columns(3)
        with col1:
            indicators["sma_short"] = st.number_input("SMA Court", value=int(indicators.get("sma_short", 20)), min_value=5, max_value=50)
            indicators["ema_short"] = st.number_input("EMA Court", value=int(indicators.get("ema_short", 9)), min_value=5, max_value=30)
        with col2:
            indicators["sma_medium"] = st.number_input("SMA Moyen", value=int(indicators.get("sma_medium", 50)), min_value=20, max_value=100)
            indicators["ema_medium"] = st.number_input("EMA Moyen", value=int(indicators.get("ema_medium", 21)), min_value=10, max_value=50)
        with col3:
            indicators["sma_long"] = st.number_input("SMA Long", value=int(indicators.get("sma_long", 200)), min_value=100, max_value=300)

        st.markdown("---")
        st.subheader("ATR & Bollinger")
        col1, col2, col3 = st.columns(3)
        with col1:
            indicators["atr_period"] = st.number_input("Periode ATR", value=int(indicators.get("atr_period", 14)), min_value=5, max_value=30)
            indicators["atr_multiplier"] = st.slider("Multiplicateur ATR", min_value=1.0, max_value=4.0, value=float(indicators.get("atr_multiplier", 2.0)), step=0.5)
        with col2:
            indicators["bb_period"] = st.number_input("Periode BB", value=int(indicators.get("bb_period", 20)), min_value=10, max_value=50)
            indicators["bb_std"] = st.slider("Ecart-type BB", min_value=1.0, max_value=3.0, value=float(indicators.get("bb_std", 2.0)), step=0.5)
        with col3:
            indicators["volume_sma"] = st.number_input("SMA Volume", value=int(indicators.get("volume_sma", 20)), min_value=5, max_value=50)

        settings["INDICATORS"] = indicators

    # =========================================================
    # TAB 4: NOTIFICATIONS
    # =========================================================
    with tab4:
        st.subheader("Discord")

        settings["DISCORD_WEBHOOK_URL"] = st.text_input(
            "Webhook URL Discord",
            value=settings.get("DISCORD_WEBHOOK_URL", ""),
            type="password",
            placeholder="https://discord.com/api/webhooks/..."
        )

        st.markdown("---")
        st.subheader("Types de Notifications")

        settings["ENABLE_TRADE_NOTIFICATIONS"] = st.checkbox(
            "Notifications Trades",
            value=settings.get("ENABLE_TRADE_NOTIFICATIONS", True),
            help="Notifier lors de l'ouverture/fermeture de positions"
        )

        settings["ENABLE_SIGNAL_NOTIFICATIONS"] = st.checkbox(
            "Notifications Signaux",
            value=settings.get("ENABLE_SIGNAL_NOTIFICATIONS", True),
            help="Notifier lors de nouveaux signaux detectes"
        )

        settings["ENABLE_DAILY_SUMMARY"] = st.checkbox(
            "Resume Quotidien",
            value=settings.get("ENABLE_DAILY_SUMMARY", True),
            help="Envoyer un resume en fin de journee"
        )

    # =========================================================
    # TAB 5: API
    # =========================================================
    with tab5:
        st.subheader("Alpaca API (Trading Live)")

        settings["ALPACA_API_KEY"] = st.text_input(
            "API Key",
            value=settings.get("ALPACA_API_KEY", ""),
            type="password"
        )

        settings["ALPACA_SECRET_KEY"] = st.text_input(
            "Secret Key",
            value=settings.get("ALPACA_SECRET_KEY", ""),
            type="password"
        )

        env_options = ["https://paper-api.alpaca.markets", "https://api.alpaca.markets"]
        env_labels = {
            "https://paper-api.alpaca.markets": "Paper Trading (Test)",
            "https://api.alpaca.markets": "Live Trading (Reel)"
        }
        current_env = settings.get("ALPACA_BASE_URL", env_options[0])
        env_idx = env_options.index(current_env) if current_env in env_options else 0

        settings["ALPACA_BASE_URL"] = st.selectbox(
            "Environnement",
            env_options,
            index=env_idx,
            format_func=lambda x: env_labels.get(x, x)
        )

        if settings["ALPACA_BASE_URL"] == "https://api.alpaca.markets":
            st.error("⚠️ ATTENTION: Mode LIVE - Argent reel!")

    # =========================================================
    # BOUTONS SAUVEGARDE
    # =========================================================
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("💾 Sauvegarder les Parametres", type="primary", use_container_width=True):
            if save_user_settings(settings):
                st.success("✅ Parametres sauvegardes!")
                st.balloons()
            else:
                st.error("❌ Erreur lors de la sauvegarde")

    with col2:
        if st.button("🔄 Recharger", use_container_width=True):
            st.rerun()

    with col3:
        if st.button("⚠️ Reset Defauts", use_container_width=True):
            if reset_to_defaults():
                st.success("Settings remis par defaut")
                st.rerun()
            else:
                st.error("Erreur lors du reset")


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
    elif page == "Multi-Analyse":
        render_multi_analyse()
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
