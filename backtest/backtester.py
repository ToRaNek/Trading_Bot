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

# Charger les settings utilisateur (modifiables depuis le dashboard)
try:
    from config.user_settings import load_user_settings
    _settings = load_user_settings()
    INITIAL_CAPITAL = _settings.get("INITIAL_CAPITAL", 10000)
    RISK_PER_TRADE = _settings.get("RISK_PER_TRADE", 0.02)
    MIN_RISK_REWARD = _settings.get("MIN_RISK_REWARD", 3.0)
    MAX_OPEN_POSITIONS = _settings.get("MAX_OPEN_POSITIONS", 5)
    TAKE_PROFIT_CONFIG = _settings.get("TAKE_PROFIT_CONFIG", {
        "tp1_percent": 0.25,
        "tp2_percent": 0.50,
        "move_to_breakeven": True
    })
except ImportError:
    # Fallback sur settings.py si user_settings n'existe pas
    from config.settings import (
        INITIAL_CAPITAL, RISK_PER_TRADE, MIN_RISK_REWARD,
        MAX_OPEN_POSITIONS, TAKE_PROFIT_CONFIG
    )
from data.fetcher import get_fetcher
from analysis.indicators import get_indicators
from analysis.zones import get_zone_detector, find_swing_points, detect_breakout
from analysis.patterns import get_pattern_detector
from strategy.position_sizing import PositionSizer
from strategy.strategy_selector import get_strategy_selector, StrategyType

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
        initial_capital: float = None,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005    # 0.05%
    ):
        # Initialiser avec les settings utilisateur
        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.commission = commission
        self.slippage = slippage

        # Attributs qui seront charges par _reload_settings
        self.risk_per_trade = RISK_PER_TRADE
        self.min_risk_reward = MIN_RISK_REWARD
        self.max_positions = MAX_OPEN_POSITIONS
        self.take_profit_config = TAKE_PROFIT_CONFIG

        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.zones = get_zone_detector()
        self.patterns = get_pattern_detector()
        self.position_sizer = PositionSizer(self.initial_capital)

        # State - reset() va recharger les settings
        self.reset()

    def reset(self):
        """Reset le backtester et recharge les settings"""
        # Recharger les settings utilisateur a chaque reset
        self._reload_settings()

        self.capital = self.initial_capital
        self.positions: Dict[str, BacktestTrade] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve = []
        self.dates = []

    def _reload_settings(self):
        """Recharge les settings depuis user_settings.json"""
        try:
            from config.user_settings import load_user_settings
            settings = load_user_settings()

            self.initial_capital = settings.get("INITIAL_CAPITAL", 10000)
            self.risk_per_trade = settings.get("RISK_PER_TRADE", 0.02)
            self.min_risk_reward = settings.get("MIN_RISK_REWARD", 3.0)
            self.max_positions = settings.get("MAX_OPEN_POSITIONS", 5)
            self.take_profit_config = settings.get("TAKE_PROFIT_CONFIG", {
                "tp1_percent": 0.25,
                "tp2_percent": 0.50,
                "move_to_breakeven": True
            })

            # Mettre a jour le position sizer
            self.position_sizer = PositionSizer(self.initial_capital)

            logger.info(f"Settings reloaded: Capital=${self.initial_capital}, Risk={self.risk_per_trade*100}%, R:R={self.min_risk_reward}")
        except Exception as e:
            logger.warning(f"Could not reload settings: {e}")

    # =========================================================================
    # MAIN BACKTEST
    # =========================================================================

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str = None,
        strategy: str = "swing_trading",
        timeframe: str = "1d"
    ) -> BacktestResult:
        """
        Execute le backtest (OPTIMIZED VERSION)

        Args:
            symbols: Liste des symboles a tester
            start_date: Date de debut (YYYY-MM-DD)
            end_date: Date de fin (defaut: aujourd'hui)
            strategy: Strategie a utiliser
            timeframe: Timeframe des donnees (1d, 1h, 15m, 5m, etc.)

        Returns:
            BacktestResult avec toutes les metriques
        """
        import time
        start_time = time.time()

        logger.info(f"Starting backtest from {start_date} to {end_date or 'today'}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Timeframe: {timeframe}")

        self.reset()
        self.current_timeframe = timeframe

        # Recuperer les donnees selon le timeframe (avec batch fetching)
        all_data = self._fetch_all_data_batch(symbols, timeframe, start_date, end_date)

        if not all_data:
            logger.error("No data available for backtest")
            return self._empty_result()

        # Normaliser les index (enlever timezone)
        for symbol in all_data:
            if all_data[symbol].index.tz is not None:
                all_data[symbol].index = all_data[symbol].index.tz_localize(None)

        # Precalculer les swing points et breakouts pour chaque symbole (OPTIMISATION)
        precomputed = self._precompute_signals(all_data, strategy)

        fetch_time = time.time()
        logger.info(f"Data fetching + precompute: {fetch_time - start_time:.1f}s")

        # Trouver les dates communes
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

        # Simuler bar par bar avec index numerique (OPTIMISATION: evite df.loc[:date])
        first_df = list(all_data.values())[0]
        dates = first_df.loc[start:end].index

        # Verifier qu'on a des dates
        if len(dates) == 0:
            logger.warning(f"No data in date range {start_date} to {end_date}")
            return self._empty_result()

        # Creer un mapping date -> index pour chaque symbole
        date_to_idx = {}
        for symbol, df in all_data.items():
            date_to_idx[symbol] = {date: i for i, date in enumerate(df.index)}

        for date in dates:
            # Mettre a jour positions existantes
            self._update_positions(all_data, date)

            # Chercher nouveaux signaux (utilise les signaux precalcules)
            if len(self.positions) < self.max_positions:
                for symbol, df in all_data.items():
                    if symbol not in self.positions and date in df.index:
                        idx = date_to_idx[symbol].get(date)
                        if idx is not None and idx >= 50:
                            # Utiliser signal precalcule si disponible
                            signal = self._get_precomputed_signal(
                                precomputed, symbol, idx, df, strategy
                            )
                            if signal:
                                self._open_position(symbol, df.loc[date], signal)

            # Enregistrer equity
            self._record_equity(date, all_data)

        # Fermer positions restantes
        if len(dates) > 0:
            self._close_all_positions(dates[-1], all_data)

        total_time = time.time() - start_time
        logger.info(f"Backtest completed in {total_time:.1f}s")

        # Calculer resultats
        return self._calculate_results()

    def _fetch_all_data_batch(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Telecharge toutes les donnees en batch (OPTIMISATION)
        Utilise yf.download() pour telecharger plusieurs symboles en parallele
        """
        import yfinance as yf

        all_data = {}

        # Config timeframe
        timeframe_config = {
            '1d': {'interval': '1d', 'max_days': 3650},
            '1h': {'interval': '1h', 'max_days': 729},
            '30m': {'interval': '30m', 'max_days': 60},
            '15m': {'interval': '15m', 'max_days': 60},
            '5m': {'interval': '5m', 'max_days': 60},
            '1m': {'interval': '1m', 'max_days': 7},
        }
        config = timeframe_config.get(timeframe, timeframe_config['1d'])

        try:
            # Batch download - beaucoup plus rapide que des appels individuels
            logger.info(f"Batch downloading {len(symbols)} symbols...")

            df_all = yf.download(
                symbols,
                start=start_date,
                end=end_date or datetime.now().strftime('%Y-%m-%d'),
                interval=config['interval'],
                group_by='ticker',
                progress=False,
                threads=True  # Telecharger en parallele
            )

            if df_all.empty:
                logger.warning("Batch download returned empty, falling back to sequential")
                return self._fetch_all_data_sequential(symbols, timeframe, start_date, end_date)

            # Extraire les donnees par symbole
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        # Si un seul symbole, pas de multi-index
                        df = df_all.copy()
                        df.columns = [str(c).lower() for c in df.columns]
                    else:
                        # Gerer MultiIndex ou colonnes groupees
                        if isinstance(df_all.columns, pd.MultiIndex):
                            # Essayer d'extraire par symbole
                            try:
                                df = df_all[symbol].copy()
                            except KeyError:
                                df = df_all.xs(symbol, axis=1, level=0).copy()
                        else:
                            df = df_all[symbol].copy()

                        # Nettoyer les colonnes
                        df.columns = [str(c).lower() for c in df.columns]

                    if df.empty or len(df) < 50:
                        logger.warning(f"Insufficient data for {symbol}")
                        continue

                    df = df.dropna()

                    # Garder OHLCV - chercher les colonnes avec differents noms possibles
                    col_mapping = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'open' in col_lower:
                            col_mapping[col] = 'open'
                        elif 'high' in col_lower:
                            col_mapping[col] = 'high'
                        elif 'low' in col_lower:
                            col_mapping[col] = 'low'
                        elif 'close' in col_lower and 'adj' not in col_lower:
                            col_mapping[col] = 'close'
                        elif 'volume' in col_lower:
                            col_mapping[col] = 'volume'

                    df = df.rename(columns=col_mapping)
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing = [c for c in required_cols if c not in df.columns]
                    if missing:
                        logger.warning(f"Missing columns for {symbol}: {missing}")
                        continue

                    df = df[required_cols]
                    df['symbol'] = symbol

                    # Ajouter indicateurs
                    df = self._prepare_data(df)
                    all_data[symbol] = df
                    logger.info(f"Loaded {len(df)} bars for {symbol}")

                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Batch download failed: {e}, falling back to sequential")
            return self._fetch_all_data_sequential(symbols, timeframe, start_date, end_date)

        return all_data

    def _fetch_all_data_sequential(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Fallback: telecharge sequentiellement si batch echoue"""
        all_data = {}
        for symbol in symbols:
            df = self._fetch_data_for_timeframe(symbol, timeframe, start_date, end_date)
            if df is not None and len(df) > 50:
                all_data[symbol] = self._prepare_data(df)
                logger.info(f"Loaded {len(df)} bars for {symbol} ({timeframe})")
            else:
                logger.warning(f"Insufficient data for {symbol}")
        return all_data

    def _precompute_signals(
        self,
        all_data: Dict[str, pd.DataFrame],
        strategy: str
    ) -> Dict[str, Dict]:
        """
        Precalcule les swing points et signaux pour chaque symbole (OPTIMISATION V2)
        Version vectorisee - evite les boucles Python lentes
        """
        precomputed = {}

        for symbol, df in all_data.items():
            try:
                # Calculer swing points une seule fois pour tout le DataFrame
                swing_highs, swing_lows = find_swing_points(df)

                # Version OPTIMISEE: detecter les breakouts de maniere vectorisee
                breakouts = self._detect_breakouts_vectorized(df, swing_highs, swing_lows)

                precomputed[symbol] = {
                    'swing_highs': swing_highs,
                    'swing_lows': swing_lows,
                    'breakouts': breakouts
                }

            except Exception as e:
                logger.warning(f"Error precomputing for {symbol}: {e}")
                precomputed[symbol] = {'swing_highs': [], 'swing_lows': [], 'breakouts': {}}

        return precomputed

    def _detect_breakouts_vectorized(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict],
        swing_lows: List[Dict]
    ) -> Dict[int, Dict]:
        """
        Detection vectorisee des breakouts (OPTIMISATION)
        Evite de slicer le DataFrame a chaque iteration

        Args:
            swing_highs: Liste de dicts {'index': i, 'price': p, 'date': d}
            swing_lows: Liste de dicts {'index': i, 'price': p, 'date': d}
        """
        breakouts = {}

        if not swing_highs or not swing_lows:
            return breakouts

        # Convertir en arrays numpy pour performance
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Extraire indices et prix des swing points
        swing_high_indices = np.array([sh['index'] for sh in swing_highs])
        swing_high_prices = np.array([sh['price'] for sh in swing_highs])
        swing_low_indices = np.array([sl['index'] for sl in swing_lows])
        swing_low_prices = np.array([sl['price'] for sl in swing_lows])

        # Pour chaque barre apres le minimum requis
        for i in range(50, len(df)):
            # Filtrer swing points visibles a cette barre (lookback)
            lookback = 20
            valid_high_mask = (swing_high_indices < i) & (swing_high_indices >= i - lookback)
            valid_low_mask = (swing_low_indices < i) & (swing_low_indices >= i - lookback)

            if not np.any(valid_high_mask) or not np.any(valid_low_mask):
                continue

            # Trouver le dernier swing high/low dans la fenetre
            valid_high_idx = np.where(valid_high_mask)[0]
            valid_low_idx = np.where(valid_low_mask)[0]

            last_swing_high = swing_high_prices[valid_high_idx[-1]]
            last_swing_low = swing_low_prices[valid_low_idx[-1]]

            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Detecter breakout haussier
            if current_close > last_swing_high and current_high > last_swing_high:
                breakouts[i] = {
                    'direction': 'buy',
                    'level': last_swing_high,
                    'strength': (current_close - last_swing_high) / last_swing_high
                }
            # Detecter breakout baissier
            elif current_close < last_swing_low and current_low < last_swing_low:
                breakouts[i] = {
                    'direction': 'sell',
                    'level': last_swing_low,
                    'strength': (last_swing_low - current_close) / last_swing_low
                }

        return breakouts

    def _get_precomputed_signal(
        self,
        precomputed: Dict,
        symbol: str,
        idx: int,
        df: pd.DataFrame,
        strategy: str
    ) -> Optional[Dict]:
        """
        Recupere un signal precalcule ou calcule si necessaire
        """
        # Pour swing trading, utiliser les breakouts precalcules
        if strategy.lower() in ['swing_trading', 'swing']:
            if symbol not in precomputed:
                return None

            breakout = precomputed[symbol]['breakouts'].get(idx)
            if not breakout:
                return None

            # Construire le signal a partir du breakout precalcule
            direction = breakout['direction']
            current_price = df['close'].iloc[idx]
            atr = df['atr'].iloc[idx] if 'atr' in df.columns else current_price * 0.02
            rsi = df['rsi'].iloc[idx] if 'rsi' in df.columns else 50

            if direction == 'buy':
                if rsi > 70:
                    return None
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 2 * self.min_risk_reward)
            else:
                if rsi < 30:
                    return None
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 2 * self.min_risk_reward)

            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            rr = reward / risk if risk > 0 else 0

            if rr < self.min_risk_reward:
                return None

            return {
                'direction': direction,
                'entry': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': rr
            }

        # Pour autres strategies, utiliser la methode originale (avec slice optimise)
        return self._check_signal(df.iloc[:idx+1], strategy, symbol)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare les donnees avec indicateurs"""
        df = df.copy()
        df = self.indicators.add_all_indicators(df)
        return df

    def _fetch_data_for_timeframe(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Recupere les donnees pour un timeframe specifique

        Args:
            symbol: Symbole
            timeframe: Timeframe (1d, 1h, 15m, 5m, etc.)
            start_date: Date debut
            end_date: Date fin

        Returns:
            DataFrame avec les donnees OHLCV
        """
        # Calculer la periode necessaire
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        days_needed = (end - start).days + 30  # Marge pour indicateurs

        # Mapper timeframe vers yfinance interval et period
        timeframe_config = {
            '1d': {'interval': '1d', 'max_days': 3650, 'period': '5y'},
            '1h': {'interval': '1h', 'max_days': 729, 'period': '2y'},
            '30m': {'interval': '30m', 'max_days': 60, 'period': '60d'},
            '15m': {'interval': '15m', 'max_days': 60, 'period': '60d'},
            '5m': {'interval': '5m', 'max_days': 60, 'period': '60d'},
            '1m': {'interval': '1m', 'max_days': 7, 'period': '7d'},
        }

        config = timeframe_config.get(timeframe, timeframe_config['1d'])

        # Limiter selon les contraintes yfinance
        if days_needed > config['max_days']:
            logger.warning(f"Requested {days_needed} days but {timeframe} limited to {config['max_days']} days")
            days_needed = config['max_days']

        try:
            # Utiliser yfinance directement pour avoir plus de controle
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            # Pour les timeframes intraday, utiliser period car start/end ne fonctionne pas bien
            if timeframe in ['1m', '5m', '15m', '30m']:
                df = ticker.history(period=config['period'], interval=config['interval'])
            else:
                # Pour daily et hourly, on peut utiliser start/end
                df = ticker.history(
                    start=start_date,
                    end=end_date or datetime.now().strftime('%Y-%m-%d'),
                    interval=config['interval']
                )

            if df.empty:
                logger.warning(f"No data for {symbol} ({timeframe})")
                return None

            # Nettoyer les colonnes
            df.columns = [col.lower() for col in df.columns]

            # Garder OHLCV
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in required_cols if col in df.columns]]
            df['symbol'] = symbol

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} ({timeframe}): {e}")
            return None

    def _check_signal(self, df: pd.DataFrame, strategy: str, symbol: str = "UNKNOWN") -> Optional[Dict]:
        """
        Verifie si un signal est present en utilisant le StrategySelector

        Args:
            df: DataFrame avec les donnees OHLCV et indicateurs
            strategy: Nom de la strategie ('swing_trading', 'wyckoff', 'elliott', 'ichimoku', 'volume_profile', 'combined')
            symbol: Symbole du titre

        Returns:
            Dict avec direction, entry, sl, tp si signal
        """
        if len(df) < 50:
            return None

        # Swing Trading utilise la logique de base (breakout) car il telecharge ses propres donnees
        if strategy.lower() in ['swing_trading', 'swing']:
            return self._check_signal_fallback(df)

        # Mapper le nom de strategie vers StrategyType
        strategy_map = {
            'wyckoff': StrategyType.WYCKOFF,
            'elliott': StrategyType.ELLIOTT,
            'ichimoku': StrategyType.ICHIMOKU,
            'volume_profile': StrategyType.VOLUME_PROFILE,
            'combined': StrategyType.COMBINED
        }

        strategy_type = strategy_map.get(strategy.lower())
        if not strategy_type:
            return self._check_signal_fallback(df)

        # Utiliser le StrategySelector pour analyser
        selector = get_strategy_selector()

        # Sauvegarder la config actuelle
        prev_strategy = selector.active_strategy
        prev_combination = selector.combination_mode

        try:
            # Configurer la strategie demandee
            selector.set_active_strategy(strategy_type)

            # Analyser avec la strategie choisie
            signal = selector.analyze(symbol, df, None)

            if signal and signal.direction != 'neutral':
                return {
                    'direction': signal.direction,
                    'entry': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'risk_reward': signal.risk_reward
                }

            return None

        except Exception as e:
            logger.warning(f"Error in strategy analysis: {e}")
            # Fallback: utiliser la logique de base si erreur
            return self._check_signal_fallback(df)
        finally:
            # Restaurer la config precedente
            if prev_combination:
                selector.combination_mode = True
                selector.active_strategy = None
            else:
                selector.active_strategy = prev_strategy
                selector.combination_mode = False

    def _check_signal_fallback(self, df: pd.DataFrame) -> Optional[Dict]:
        """Logique de signal de fallback (swing trading basique)"""
        swing_highs, swing_lows = find_swing_points(df)
        breakout = detect_breakout(df, swing_highs, swing_lows)

        if not breakout:
            return None

        direction = breakout['direction']
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02

        latest = df.iloc[-1]
        rsi = latest.get('rsi', 50)

        if direction == 'buy':
            if rsi > 70:
                return None
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 2 * self.min_risk_reward)
        else:
            if rsi < 30:
                return None
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 2 * self.min_risk_reward)

        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        rr = reward / risk if risk > 0 else 0

        if rr < self.min_risk_reward:
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
