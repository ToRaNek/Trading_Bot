"""
Charts - Graphiques Interactifs avec Plotly
Pour le dashboard Streamlit
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from datetime import datetime


# =============================================================================
# COULEURS ET STYLE
# =============================================================================

COLORS = {
    'green': '#00C853',
    'red': '#FF1744',
    'blue': '#2196F3',
    'orange': '#FF9800',
    'purple': '#9C27B0',
    'cyan': '#00BCD4',
    'background': '#1E1E1E',
    'grid': '#333333',
    'text': '#FFFFFF'
}

LAYOUT_TEMPLATE = {
    'paper_bgcolor': COLORS['background'],
    'plot_bgcolor': COLORS['background'],
    'font': {'color': COLORS['text']},
    'xaxis': {
        'gridcolor': COLORS['grid'],
        'showgrid': True
    },
    'yaxis': {
        'gridcolor': COLORS['grid'],
        'showgrid': True
    }
}


# =============================================================================
# CANDLESTICK CHART
# =============================================================================

def create_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    zones: List = None,
    signals: List = None,
    show_volume: bool = True,
    show_indicators: bool = True
) -> go.Figure:
    """
    Cree un graphique candlestick complet

    Args:
        df: DataFrame OHLCV avec indicateurs
        symbol: Symbole
        zones: Liste des zones a afficher
        signals: Liste des signaux a afficher
        show_volume: Afficher le volume
        show_indicators: Afficher les indicateurs

    Returns:
        Figure Plotly
    """
    # Nombre de rows
    rows = 1
    row_heights = [0.7]

    if show_volume:
        rows += 1
        row_heights.append(0.15)

    if show_indicators and 'rsi' in df.columns:
        rows += 1
        row_heights.append(0.15)

    # Creer subplots
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color=COLORS['green'],
            decreasing_line_color=COLORS['red'],
            name='Price'
        ),
        row=1, col=1
    )

    # Moving Averages
    if show_indicators:
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color=COLORS['orange'], width=1)
                ),
                row=1, col=1
            )

        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color=COLORS['blue'], width=1)
                ),
                row=1, col=1
            )

        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color=COLORS['purple'], width=1, dash='dot')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color=COLORS['purple'], width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(156, 39, 176, 0.1)'
                ),
                row=1, col=1
            )

    # Zones
    if zones:
        for zone in zones:
            color = 'rgba(0, 200, 83, 0.2)' if zone.zone_type == 'support' else 'rgba(255, 23, 68, 0.2)'
            fig.add_hrect(
                y0=zone.price_low,
                y1=zone.price_high,
                fillcolor=color,
                line_width=0,
                row=1, col=1
            )

    # Signaux
    if signals:
        buy_signals = [s for s in signals if s.signal_type == 'buy']
        sell_signals = [s for s in signals if s.signal_type == 'sell']

        if buy_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.timestamp for s in buy_signals],
                    y=[s.price for s in buy_signals],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color=COLORS['green']
                    )
                ),
                row=1, col=1
            )

        if sell_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.timestamp for s in sell_signals],
                    y=[s.price for s in sell_signals],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color=COLORS['red']
                    )
                ),
                row=1, col=1
            )

    # Volume
    current_row = 2
    if show_volume:
        colors = [COLORS['green'] if c >= o else COLORS['red']
                  for o, c in zip(df['open'], df['close'])]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.7
            ),
            row=current_row, col=1
        )
        current_row += 1

    # RSI
    if show_indicators and 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color=COLORS['cyan'], width=1)
            ),
            row=current_row, col=1
        )

        # Niveaux RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      opacity=0.5, row=current_row, col=1)

    # Layout
    fig.update_layout(
        title=f'{symbol} - Trading Chart',
        xaxis_rangeslider_visible=False,
        height=800,
        **LAYOUT_TEMPLATE
    )

    return fig


# =============================================================================
# EQUITY CURVE
# =============================================================================

def create_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series = None,
    trades: List = None
) -> go.Figure:
    """
    Cree la courbe d'equity

    Args:
        equity: Serie de l'equity
        benchmark: Serie benchmark (optionnel)
        trades: Liste des trades pour markers

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Portfolio',
            line=dict(color=COLORS['blue'], width=2),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        )
    )

    # Benchmark
    if benchmark is not None:
        # Normaliser au meme point de depart
        normalized = benchmark / benchmark.iloc[0] * equity.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=normalized.values,
                mode='lines',
                name='Benchmark',
                line=dict(color=COLORS['orange'], width=1, dash='dash')
            )
        )

    # Trades markers
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        if wins:
            fig.add_trace(
                go.Scatter(
                    x=[t.exit_date for t in wins],
                    y=[equity.loc[t.exit_date] if t.exit_date in equity.index else equity.iloc[-1] for t in wins],
                    mode='markers',
                    name='Winning Trades',
                    marker=dict(symbol='circle', size=8, color=COLORS['green'])
                )
            )

        if losses:
            fig.add_trace(
                go.Scatter(
                    x=[t.exit_date for t in losses],
                    y=[equity.loc[t.exit_date] if t.exit_date in equity.index else equity.iloc[-1] for t in losses],
                    mode='markers',
                    name='Losing Trades',
                    marker=dict(symbol='circle', size=8, color=COLORS['red'])
                )
            )

    # Drawdown shading
    peak = equity.expanding().max()
    drawdown = equity - peak

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=peak.values,
            mode='lines',
            name='Peak',
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5
        )
    )

    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=500,
        **LAYOUT_TEMPLATE
    )

    return fig


# =============================================================================
# DRAWDOWN CHART
# =============================================================================

def create_drawdown_chart(equity: pd.Series) -> go.Figure:
    """
    Cree le graphique de drawdown

    Args:
        equity: Serie de l'equity

    Returns:
        Figure Plotly
    """
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            line=dict(color=COLORS['red'], width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 23, 68, 0.3)'
        )
    )

    # Max drawdown point
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.min()

    fig.add_trace(
        go.Scatter(
            x=[max_dd_idx],
            y=[max_dd],
            mode='markers+text',
            name=f'Max DD: {max_dd:.2f}%',
            marker=dict(size=12, color=COLORS['red']),
            text=[f'{max_dd:.2f}%'],
            textposition='bottom center'
        )
    )

    fig.update_layout(
        title='Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        height=300,
        **LAYOUT_TEMPLATE
    )

    return fig


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def create_metrics_cards(metrics: Dict) -> str:
    """
    Cree les cartes de metriques en HTML

    Args:
        metrics: Dictionnaire de metriques

    Returns:
        HTML string
    """
    cards = []

    metric_config = [
        ('Total Return', 'total_return_percent', '%', 'green' if metrics.get('total_return_percent', 0) >= 0 else 'red'),
        ('Win Rate', 'win_rate', '%', 'blue'),
        ('Sharpe Ratio', 'sharpe_ratio', '', 'purple'),
        ('Max Drawdown', 'max_drawdown_percent', '%', 'red'),
        ('Profit Factor', 'profit_factor', 'x', 'orange'),
        ('Total Trades', 'total_trades', '', 'cyan')
    ]

    for name, key, suffix, color in metric_config:
        value = metrics.get(key, 0)
        if isinstance(value, float):
            value_str = f"{value:.2f}{suffix}"
        else:
            value_str = f"{value}{suffix}"

        cards.append(f"""
        <div style="background: {COLORS['grid']}; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: {COLORS[color]}; margin: 0;">{value_str}</h3>
            <p style="color: {COLORS['text']}; margin: 5px 0 0 0;">{name}</p>
        </div>
        """)

    return f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
        {''.join(cards)}
    </div>
    """


# =============================================================================
# TRADE DISTRIBUTION
# =============================================================================

def create_pnl_distribution(trades: List) -> go.Figure:
    """
    Cree l'histogramme de distribution des P&L

    Args:
        trades: Liste des trades

    Returns:
        Figure Plotly
    """
    pnl_values = [t.pnl for t in trades]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=pnl_values,
            nbinsx=30,
            marker_color=[COLORS['green'] if p > 0 else COLORS['red'] for p in pnl_values],
            name='P&L Distribution'
        )
    )

    # Ligne zero
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)

    # Moyenne
    avg_pnl = np.mean(pnl_values)
    fig.add_vline(x=avg_pnl, line_dash="dot", line_color=COLORS['blue'],
                  annotation_text=f"Avg: ${avg_pnl:.2f}")

    fig.update_layout(
        title='P&L Distribution',
        xaxis_title='P&L ($)',
        yaxis_title='Frequency',
        height=400,
        **LAYOUT_TEMPLATE
    )

    return fig


# =============================================================================
# MONTHLY RETURNS HEATMAP
# =============================================================================

def create_monthly_returns_heatmap(equity: pd.Series) -> go.Figure:
    """
    Cree la heatmap des returns mensuels

    Args:
        equity: Serie de l'equity

    Returns:
        Figure Plotly
    """
    # Calculer les returns mensuels
    monthly = equity.resample('M').last()
    monthly_returns = monthly.pct_change() * 100

    # Creer matrice annee/mois
    df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })

    pivot = df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title='Monthly Returns (%)',
        height=400,
        **LAYOUT_TEMPLATE
    )

    return fig


# =============================================================================
# POSITIONS PIE CHART
# =============================================================================

def create_positions_pie(positions: List[Dict]) -> go.Figure:
    """
    Cree le pie chart des positions

    Args:
        positions: Liste des positions

    Returns:
        Figure Plotly
    """
    if not positions:
        fig = go.Figure()
        fig.add_annotation(text="No open positions", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=20, color='gray'))
        fig.update_layout(height=400, **LAYOUT_TEMPLATE)
        return fig

    symbols = [p['symbol'] for p in positions]
    values = [abs(p.get('market_value', 0)) for p in positions]
    colors = [COLORS['green'] if p.get('unrealized_pnl', 0) >= 0 else COLORS['red']
              for p in positions]

    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent'
    )])

    fig.update_layout(
        title='Portfolio Allocation',
        height=400,
        **LAYOUT_TEMPLATE
    )

    return fig


# =============================================================================
# WIN/LOSS CHART
# =============================================================================

def create_win_loss_chart(trades: List) -> go.Figure:
    """
    Cree le graphique win/loss par trade

    Args:
        trades: Liste des trades

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    pnl_values = [t.pnl for t in trades]
    colors = [COLORS['green'] if p > 0 else COLORS['red'] for p in pnl_values]

    fig.add_trace(
        go.Bar(
            x=list(range(1, len(trades) + 1)),
            y=pnl_values,
            marker_color=colors,
            name='P&L'
        )
    )

    # Ligne cumulative
    cumulative = np.cumsum(pnl_values)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(trades) + 1)),
            y=cumulative,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color=COLORS['blue'], width=2),
            yaxis='y2'
        )
    )

    fig.update_layout(
        title='Trade Results',
        xaxis_title='Trade #',
        yaxis_title='P&L ($)',
        yaxis2=dict(
            title='Cumulative P&L ($)',
            overlaying='y',
            side='right'
        ),
        height=400,
        **LAYOUT_TEMPLATE
    )

    return fig


# =============================================================================
# SECTOR EXPOSURE
# =============================================================================

def create_sector_exposure(positions: List[Dict]) -> go.Figure:
    """
    Cree le graphique d'exposition par secteur

    Args:
        positions: Liste des positions avec secteur

    Returns:
        Figure Plotly
    """
    sectors = {}
    for p in positions:
        sector = p.get('sector', 'Unknown')
        value = abs(p.get('market_value', 0))
        sectors[sector] = sectors.get(sector, 0) + value

    fig = go.Figure(data=[go.Bar(
        x=list(sectors.keys()),
        y=list(sectors.values()),
        marker_color=COLORS['blue']
    )])

    fig.update_layout(
        title='Sector Exposure',
        xaxis_title='Sector',
        yaxis_title='Value ($)',
        height=400,
        **LAYOUT_TEMPLATE
    )

    return fig
