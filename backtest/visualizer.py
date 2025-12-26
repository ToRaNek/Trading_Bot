"""
Backtest Visualizer - Visualisation interactive des resultats de backtest
Genere un rapport HTML avec tableau des trades et graphiques
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly non installe. Installer avec: pip install plotly")

from backtest.backtester import BacktestResult, BacktestTrade
from data.fetcher import get_fetcher

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """
    Visualiseur interactif des resultats de backtest

    Fonctionnalites:
    - Tableau des trades avec stats
    - Graphique par trade avec entree/sortie
    - Equity curve
    - Export HTML
    """

    def __init__(self, result: BacktestResult, all_data: Dict[str, pd.DataFrame] = None):
        """
        Args:
            result: Resultat du backtest
            all_data: Donnees OHLCV par symbole (optionnel, sera telecharge si absent)
        """
        self.result = result
        self.all_data = all_data or {}
        self.fetcher = get_fetcher()

    def generate_html_report(self, output_path: str = "backtest_report.html") -> str:
        """
        Genere un rapport HTML complet et interactif

        Args:
            output_path: Chemin du fichier HTML

        Returns:
            Chemin du fichier genere
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly requis pour generer le rapport")
            return None

        # Generer les composants
        trades_table = self._generate_trades_table()
        equity_chart = self._generate_equity_chart()
        trade_charts = self._generate_trade_charts()
        summary_stats = self._generate_summary_stats()

        # Assembler le HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #00d4ff;
            margin: 20px 0 15px 0;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #0f3460;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .stat-card .label {{
            color: #888;
            margin-top: 5px;
        }}
        .stat-card.positive .value {{ color: #00ff88; }}
        .stat-card.negative .value {{ color: #ff4757; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #0f3460;
        }}
        th {{
            background: #0f3460;
            color: #00d4ff;
            font-weight: 600;
        }}
        tr:hover {{
            background: #1a2a4a;
            cursor: pointer;
        }}
        tr.winner {{ background: rgba(0, 255, 136, 0.1); }}
        tr.loser {{ background: rgba(255, 71, 87, 0.1); }}
        .pnl-positive {{ color: #00ff88; font-weight: bold; }}
        .pnl-negative {{ color: #ff4757; font-weight: bold; }}

        .chart-container {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid #0f3460;
        }}
        .trade-chart {{
            display: none;
            margin-top: 20px;
        }}
        .trade-chart.active {{
            display: block;
        }}
        .trade-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
            padding: 15px;
            background: #0f3460;
            border-radius: 8px;
        }}
        .trade-info div {{
            text-align: center;
        }}
        .trade-info .label {{
            color: #888;
            font-size: 0.9em;
        }}
        .trade-info .value {{
            font-weight: bold;
            font-size: 1.1em;
        }}

        .btn {{
            background: #00d4ff;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 5px;
        }}
        .btn:hover {{
            background: #00a8cc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Report</h1>

        {summary_stats}

        <h2>📈 Equity Curve</h2>
        <div class="chart-container">
            {equity_chart}
        </div>

        <h2>📋 Trades ({len(self.result.trades)} total)</h2>
        {trades_table}

        <h2>🔍 Detail du Trade</h2>
        <div id="trade-detail" class="chart-container">
            <p style="text-align: center; color: #888;">Cliquez sur un trade dans le tableau pour voir le graphique</p>
        </div>

        {trade_charts}
    </div>

    <script>
        function showTrade(tradeId) {{
            // Cacher tous les charts
            document.querySelectorAll('.trade-chart').forEach(el => el.classList.remove('active'));

            // Afficher le chart selectionne
            const chart = document.getElementById('trade-' + tradeId);
            if (chart) {{
                chart.classList.add('active');
                document.getElementById('trade-detail').innerHTML = '';
                document.getElementById('trade-detail').appendChild(chart);
                chart.style.display = 'block';
            }}

            // Highlight la ligne
            document.querySelectorAll('tr').forEach(el => el.style.outline = 'none');
            document.querySelector('tr[data-trade="' + tradeId + '"]').style.outline = '2px solid #00d4ff';
        }}
    </script>
</body>
</html>
"""

        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Rapport genere: {output_path}")
        return output_path

    def _generate_summary_stats(self) -> str:
        """Genere les stats resumees"""
        r = self.result

        return_class = "positive" if r.total_return >= 0 else "negative"
        wr_class = "positive" if r.win_rate >= 50 else "negative"

        return f"""
        <div class="stats-grid">
            <div class="stat-card {return_class}">
                <div class="value">${r.total_return:,.2f}</div>
                <div class="label">Total Return ({r.total_return_percent:.1f}%)</div>
            </div>
            <div class="stat-card {wr_class}">
                <div class="value">{r.win_rate:.1f}%</div>
                <div class="label">Win Rate ({r.winning_trades}W / {r.losing_trades}L)</div>
            </div>
            <div class="stat-card">
                <div class="value">{r.profit_factor:.2f}</div>
                <div class="label">Profit Factor</div>
            </div>
            <div class="stat-card">
                <div class="value">{r.sharpe_ratio:.2f}</div>
                <div class="label">Sharpe Ratio</div>
            </div>
            <div class="stat-card negative">
                <div class="value">{r.max_drawdown_percent:.1f}%</div>
                <div class="label">Max Drawdown</div>
            </div>
            <div class="stat-card">
                <div class="value">${r.expectancy:.2f}</div>
                <div class="label">Expectancy</div>
            </div>
            <div class="stat-card">
                <div class="value">{r.avg_holding_days:.1f}d</div>
                <div class="label">Avg Holding</div>
            </div>
            <div class="stat-card">
                <div class="value">{r.total_trades}</div>
                <div class="label">Total Trades</div>
            </div>
        </div>
        """

    def _generate_trades_table(self) -> str:
        """Genere le tableau HTML des trades"""
        if not self.result.trades:
            return "<p>Aucun trade</p>"

        rows = []
        for i, trade in enumerate(self.result.trades, 1):
            pnl_class = "pnl-positive" if trade.pnl >= 0 else "pnl-negative"
            row_class = "winner" if trade.pnl >= 0 else "loser"

            entry_date = trade.entry_date.strftime("%Y-%m-%d %H:%M") if trade.entry_date else "N/A"
            exit_date = trade.exit_date.strftime("%Y-%m-%d %H:%M") if trade.exit_date else "N/A"

            rows.append(f"""
            <tr class="{row_class}" data-trade="{i}" onclick="showTrade({i})">
                <td>#{i}</td>
                <td><strong>{trade.symbol}</strong></td>
                <td>{trade.side.upper()}</td>
                <td>{entry_date}</td>
                <td>${trade.entry_price:.2f}</td>
                <td>{exit_date}</td>
                <td>${trade.exit_price:.2f if trade.exit_price else 0:.2f}</td>
                <td class="{pnl_class}">${trade.pnl:.2f} ({trade.pnl_percent:.1f}%)</td>
                <td>{trade.exit_reason}</td>
            </tr>
            """)

        return f"""
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Entry Date</th>
                    <th>Entry Price</th>
                    <th>Exit Date</th>
                    <th>Exit Price</th>
                    <th>P&L</th>
                    <th>Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _generate_equity_chart(self) -> str:
        """Genere le graphique de l'equity curve"""
        if self.result.equity_curve is None or len(self.result.equity_curve) == 0:
            return "<p>Pas de donnees d'equity</p>"

        equity = self.result.equity_curve

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Equity',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=30, b=50),
            height=300,
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333', tickprefix='$'),
            showlegend=False
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _generate_trade_charts(self) -> str:
        """Genere les graphiques individuels pour chaque trade"""
        if not PLOTLY_AVAILABLE or not self.result.trades:
            return ""

        charts_html = []

        for i, trade in enumerate(self.result.trades, 1):
            chart_html = self._create_trade_chart(trade, i)
            charts_html.append(chart_html)

        return '\n'.join(charts_html)

    def _create_trade_chart(self, trade: BacktestTrade, trade_id: int) -> str:
        """Cree le graphique pour un trade specifique"""
        symbol = trade.symbol

        # Recuperer les donnees si pas deja en cache
        if symbol not in self.all_data:
            df = self.fetcher.get_daily_data(symbol)
            if df is not None:
                self.all_data[symbol] = df

        if symbol not in self.all_data:
            return f'<div id="trade-{trade_id}" class="trade-chart"><p>Donnees non disponibles pour {symbol}</p></div>'

        df = self.all_data[symbol]

        # Filtrer autour du trade (30 barres avant, 10 apres)
        if trade.entry_date and trade.exit_date:
            start_idx = df.index.get_indexer([trade.entry_date], method='nearest')[0]
            end_idx = df.index.get_indexer([trade.exit_date], method='nearest')[0]

            plot_start = max(0, start_idx - 30)
            plot_end = min(len(df), end_idx + 10)
            df_plot = df.iloc[plot_start:plot_end]
        else:
            df_plot = df.tail(100)

        # Creer le graphique
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )

        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index,
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4757'
            ),
            row=1, col=1
        )

        # Signal d'entree
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_date],
                y=[trade.entry_price],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if trade.side == 'long' else 'triangle-down',
                    size=20,
                    color='#00ff88' if trade.side == 'long' else '#ff4757',
                    line=dict(width=2, color='white')
                ),
                name='Entry',
                hovertemplate=f"ENTRY<br>{trade.side.upper()}<br>${trade.entry_price:.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Signal de sortie
        if trade.exit_date and trade.exit_price:
            exit_color = '#00ff88' if trade.pnl >= 0 else '#ff4757'
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_date],
                    y=[trade.exit_price],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color=exit_color,
                        line=dict(width=3, color=exit_color)
                    ),
                    name='Exit',
                    hovertemplate=f"EXIT<br>{trade.exit_reason}<br>${trade.exit_price:.2f}<extra></extra>"
                ),
                row=1, col=1
            )

        # Stop Loss line
        fig.add_hline(
            y=trade.stop_loss,
            line_dash="dash",
            line_color="#ff4757",
            annotation_text=f"SL: ${trade.stop_loss:.2f}",
            row=1, col=1
        )

        # Take Profit line
        fig.add_hline(
            y=trade.take_profit,
            line_dash="dash",
            line_color="#00ff88",
            annotation_text=f"TP: ${trade.take_profit:.2f}",
            row=1, col=1
        )

        # Volume
        colors = ['#00ff88' if c >= o else '#ff4757' for c, o in zip(df_plot['close'], df_plot['open'])]
        fig.add_trace(
            go.Bar(
                x=df_plot.index,
                y=df_plot['volume'],
                marker_color=colors,
                opacity=0.5,
                name='Volume'
            ),
            row=2, col=1
        )

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=30, b=50),
            height=500,
            xaxis_rangeslider_visible=False,
            showlegend=False
        )

        fig.update_xaxes(gridcolor='#333')
        fig.update_yaxes(gridcolor='#333')

        # Info du trade
        pnl_color = "#00ff88" if trade.pnl >= 0 else "#ff4757"
        trade_info = f"""
        <div class="trade-info">
            <div><span class="label">Symbol</span><br><span class="value">{trade.symbol}</span></div>
            <div><span class="label">Side</span><br><span class="value">{trade.side.upper()}</span></div>
            <div><span class="label">Entry</span><br><span class="value">${trade.entry_price:.2f}</span></div>
            <div><span class="label">Exit</span><br><span class="value">${trade.exit_price:.2f if trade.exit_price else 0:.2f}</span></div>
            <div><span class="label">Stop Loss</span><br><span class="value" style="color:#ff4757">${trade.stop_loss:.2f}</span></div>
            <div><span class="label">Take Profit</span><br><span class="value" style="color:#00ff88">${trade.take_profit:.2f}</span></div>
            <div><span class="label">P&L</span><br><span class="value" style="color:{pnl_color}">${trade.pnl:.2f} ({trade.pnl_percent:.1f}%)</span></div>
            <div><span class="label">Exit Reason</span><br><span class="value">{trade.exit_reason}</span></div>
        </div>
        """

        chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

        return f'<div id="trade-{trade_id}" class="trade-chart">{trade_info}{chart_html}</div>'

    def show_in_browser(self, output_path: str = "backtest_report.html"):
        """Genere et ouvre le rapport dans le navigateur"""
        import webbrowser
        import os

        path = self.generate_html_report(output_path)
        if path:
            webbrowser.open('file://' + os.path.realpath(path))


def visualize_backtest(result: BacktestResult, output_path: str = "backtest_report.html", open_browser: bool = True):
    """
    Fonction utilitaire pour visualiser un backtest

    Args:
        result: Resultat du backtest
        output_path: Chemin du fichier HTML
        open_browser: Ouvrir automatiquement dans le navigateur
    """
    viz = BacktestVisualizer(result)

    if open_browser:
        viz.show_in_browser(output_path)
    else:
        viz.generate_html_report(output_path)

    return output_path
