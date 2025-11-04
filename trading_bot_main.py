"""
Bot Discord de Trading - VERSION CORRIGÉE
- Scraping des news via BeautifulSoup (Yahoo Finance)
- Backtest multi-intervalles (1m, 5m, 1h, 1d)
- Analyse sentiment avec HuggingFace
"""

import discord
from discord.ext import commands, tasks
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from textblob import TextBlob
import logging
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv
import warnings
import time
import re
warnings.filterwarnings('ignore')

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingBot')

load_dotenv()

WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'DIS', 'NFLX', 'ADBE', 
    'CRM', 'AMD', 'ORCL', 'INTC', 'CSCO', 'PEP', 'COST', 'AVGO'
]

BACKTEST_CONFIGS = {
    '1m': {'period': '7d', 'interval': '1m', 'name': '1 minute'},
    '5m': {'period': '60d', 'interval': '5m', 'name': '5 minutes'},
    '1h': {'period': '730d', 'interval': '1h', 'name': '1 heure'},
    '1d': {'period': '730d', 'interval': '1d', 'name': '1 jour'}
}


class EventDetector:
    
    def detect_events_in_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détecte les événements importants dans l'historique des prix"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df['price_change'] = df['Close'].pct_change()
        df['volume_sma'] = df['Volume'].rolling(window=min(20, len(df)//2)).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma'].replace(0, 1)
        
        prev_close = df['Close'].shift(1)
        df['gap'] = abs(df['Open'] - prev_close) / prev_close.replace(0, 1)
        
        df['volatility_5d'] = df['Close'].pct_change().rolling(window=min(5, len(df)//4)).std()
        
        df['has_event'] = (
            (df['volume_ratio'].fillna(0) > 2.0) |
            (abs(df['price_change'].fillna(0)) > 0.03) |
            (df['gap'].fillna(0) > 0.02) |
            (df['volatility_5d'].fillna(0) > 0.025)
        )
        
        df['event_score'] = 0.0
        df.loc[df['volume_ratio'].fillna(0) > 2.0, 'event_score'] += 30
        df.loc[abs(df['price_change'].fillna(0)) > 0.03, 'event_score'] += 25
        df.loc[abs(df['price_change'].fillna(0)) > 0.05, 'event_score'] += 15
        df.loc[df['gap'].fillna(0) > 0.02, 'event_score'] += 20
        df.loc[df['volatility_5d'].fillna(0) > 0.025, 'event_score'] += 10
        
        return df


class NewsAnalyzer:
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 1800
        
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def scrape_yahoo_news(self, symbol: str, hours: int = 48) -> Tuple[bool, List[Dict], float]:
        """Scrape les actualités de Yahoo Finance"""
        cache_key = f"{symbol}_{hours}"
        
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_duration:
                logger.info(f"📦 {symbol} - Cache utilisé")
                return cached_data
        
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            
            session = await self.get_session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            async with session.get(url, headers=headers, timeout=15, ssl=False) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ {symbol} - Status code: {response.status}")
                    return False, [], 0.0
                
                html = await response.text()
        
        except Exception as e:
            logger.error(f"❌ Erreur scraping {symbol}: {e}")
            return False, [], 0.0
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            news_items = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Chercher les articles dans les sections stream-item
            for item in soup.find_all('li', class_='stream-item'):
                try:
                    # Chercher le titre
                    h3 = item.find('h3')
                    if not h3:
                        continue
                    
                    title_elem = h3.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    if not title or len(title) < 5:
                        continue
                    
                    # Chercher le lien
                    link = title_elem.get('href', '')
                    if link and not link.startswith('http'):
                        if link.startswith('/'):
                            link = 'https://finance.yahoo.com' + link
                    
                    # Chercher la source/publisher
                    publisher = 'Yahoo Finance'
                    footer = item.find(class_='publishing')
                    if footer:
                        text = footer.get_text(strip=True)
                        source_match = re.search(r'([^•]+)', text)
                        if source_match:
                            publisher = source_match.group(1).strip()
                    
                    # Parser la date
                    pub_date = datetime.now()
                    if footer:
                        footer_text = footer.get_text(strip=True)
                        
                        # Regex pour extraire les temps
                        if 'm ago' in footer_text:
                            match = re.search(r'(\d+)m ago', footer_text)
                            if match:
                                mins = int(match.group(1))
                                pub_date = datetime.now() - timedelta(minutes=mins)
                        elif 'h ago' in footer_text:
                            match = re.search(r'(\d+)h ago', footer_text)
                            if match:
                                hours_ago = int(match.group(1))
                                pub_date = datetime.now() - timedelta(hours=hours_ago)
                        elif 'd ago' in footer_text:
                            match = re.search(r'(\d+)d ago', footer_text)
                            if match:
                                days_ago = int(match.group(1))
                                pub_date = datetime.now() - timedelta(days=days_ago)
                    
                    # Filtrer par date
                    if pub_date < cutoff_time:
                        continue
                    
                    # Calcul de l'importance
                    importance_keywords = {
                        'earnings': 3.0, 'revenue': 2.5, 'profit': 2.5, 'launch': 2.0,
                        'partnership': 2.0, 'acquisition': 3.0, 'merger': 3.0, 'FDA': 2.5,
                        'approval': 2.0, 'breakthrough': 2.0, 'record': 1.5, 'guidance': 2.0,
                        'upgrade': 2.0, 'downgrade': 2.0, 'analyst': 1.5, 'lawsuit': 1.5,
                        'investigation': 1.5, 'recall': 2.0, 'bankruptcy': 3.0, 'dividend': 1.5,
                        'split': 2.0, 'buyback': 1.5, 'expansion': 1.5, 'contract': 1.5, 'deal': 1.5
                    }
                    
                    title_lower = title.lower()
                    importance = 1.0
                    matched_keywords = []
                    
                    for keyword, weight in importance_keywords.items():
                        if keyword in title_lower:
                            importance += weight
                            matched_keywords.append(keyword)
                    
                    news_items.append({
                        'title': title,
                        'publisher': publisher,
                        'link': link,
                        'date': pub_date,
                        'importance': importance,
                        'keywords': matched_keywords
                    })
                
                except Exception as e:
                    logger.debug(f"Erreur parsing article: {e}")
                    continue
            
            has_news = len(news_items) > 0
            total_importance = sum(n['importance'] for n in news_items)
            news_score = min(100, total_importance * 10) if has_news else 0.0
            
            result = (has_news, news_items, news_score)
            self.cache[cache_key] = (datetime.now(), result)
            
            logger.info(f"✅ {symbol} - {len(news_items)} actualités récentes | Score: {news_score:.0f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur parsing HTML {symbol}: {e}")
            return False, [], 0.0
    
    async def close(self):
        if self.session:
            await self.session.close()


class TechnicalAnalyzer:
    
    def calculate_indicators_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcul des indicateurs techniques adaptés à l'intervalle"""
        data_size = len(df)
        
        if data_size > 200:
            sma_short, sma_mid, sma_long = 20, 50, 200
            rsi_window = 14
            bb_window = 20
        elif data_size > 100:
            sma_short, sma_mid, sma_long = 10, 25, 100
            rsi_window = 10
            bb_window = 10
        else:
            sma_short, sma_mid, sma_long = 5, 15, 50
            rsi_window = 7
            bb_window = 10
        
        df['sma_20'] = df['Close'].rolling(window=min(sma_short, data_size//2)).mean()
        if data_size > sma_mid:
            df['sma_50'] = df['Close'].rolling(window=sma_mid).mean()
        if data_size > sma_long:
            df['sma_200'] = df['Close'].rolling(window=sma_long).mean()
        
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        bb_sma = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)
        
        df['volume_sma'] = df['Volume'].rolling(window=min(20, data_size//2)).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        return df
    
    def get_technical_score(self, row: pd.Series) -> Tuple[float, List[str]]:
        score = 50.0
        reasons = []
        
        if pd.notna(row.get('sma_20')) and pd.notna(row.get('sma_50')):
            if row['sma_20'] > row['sma_50']:
                score += 15
                reasons.append(f"✅ Tendance haussière")
            else:
                score -= 15
                reasons.append(f"⚠️ Tendance baissière")
        
        if pd.notna(row.get('rsi')):
            if row['rsi'] < 30:
                score += 20
                reasons.append(f"🔥 RSI survendu ({row['rsi']:.1f})")
            elif row['rsi'] > 70:
                score -= 20
                reasons.append(f"❄️ RSI suracheté ({row['rsi']:.1f})")
            elif 30 <= row['rsi'] <= 40:
                score += 10
                reasons.append(f"✅ RSI favorable ({row['rsi']:.1f})")
        
        if pd.notna(row.get('macd')) and pd.notna(row.get('macd_signal')):
            if row['macd'] > row['macd_signal']:
                score += 10
                reasons.append(f"✅ MACD bullish")
            else:
                score -= 10
                reasons.append(f"⚠️ MACD bearish")
        
        if pd.notna(row.get('bb_lower')) and pd.notna(row.get('bb_upper')):
            bb_position = (row['Close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
            if bb_position < 0.2:
                score += 15
                reasons.append(f"🎯 Prix près BB inférieure")
            elif bb_position > 0.8:
                score -= 15
                reasons.append(f"⚠️ Prix près BB supérieure")
        
        if pd.notna(row.get('volume_ratio')) and row['volume_ratio'] > 1.5:
            score += 5
            reasons.append(f"📈 Volume élevé ({row['volume_ratio']:.1f}x)")
        
        score = max(0, min(100, score))
        return score, reasons


class SentimentAnalyzer:
    
    def __init__(self):
        self.session = None
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
        
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def analyze_with_ai(self, text: str) -> float:
        try:
            session = await self.get_session()
            url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
            headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            payload = {"inputs": text[:512]}
            
            async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        scores = {item['label'].lower(): item['score'] for item in result[0]}
                        sentiment_score = scores.get('positive', 0) - scores.get('negative', 0)
                        return sentiment_score
            
            return TextBlob(text).sentiment.polarity
            
        except Exception as e:
            return TextBlob(text).sentiment.polarity
    
    async def get_sentiment_score(self, symbol: str, news_data: List[Dict] = None) -> Tuple[float, List[str]]:
        reasons = []
        
        if news_data and len(news_data) > 0:
            news_sentiments = []
            for article in news_data[:5]:
                sentiment = await self.analyze_with_ai(article['title'])
                news_sentiments.append(sentiment * article['importance'])
            
            news_score = 50.0
            if news_sentiments:
                avg_sentiment = np.mean(news_sentiments)
                news_score = (avg_sentiment + 1) * 50
            
            reasons.append(f"📰 News: {news_score:.0f}/100 ({len(news_data)} articles)")
        else:
            news_score = 50.0
            reasons.append(f"📰 News: Aucune actualité")
        
        return news_score, reasons
    
    async def close(self):
        if self.session:
            await self.session.close()


class SmartBacktestEngine:
    
    def __init__(self):
        self.results = []
        
    async def backtest_with_interval(self, symbol: str, interval: str = '1d') -> Optional[Dict]:
        """Backtest sur un intervalle spécifique"""
        start_time = time.time()
        
        config = BACKTEST_CONFIGS.get(interval, BACKTEST_CONFIGS['1d'])
        logger.info(f"[>>] Backtest {symbol} - {config['name']}")
        
        try:
            event_detector = EventDetector()
            tech_analyzer = TechnicalAnalyzer()
            sent_analyzer = SentimentAnalyzer()
            
            stock = yf.Ticker(symbol)
            df = stock.history(period=config['period'], interval=config['interval'])
            
            if df.empty or len(df) < 50:
                logger.warning(f"   [X] Données insuffisantes")
                return None
            
            logger.info(f"   [✓] {len(df)} bougies téléchargées")
            
            df = event_detector.detect_events_in_history(df)
            events_count = df['has_event'].sum()
            
            df = tech_analyzer.calculate_indicators_batch(df)
            
            sentiment_score, _ = await sent_analyzer.get_sentiment_score(symbol)
            
            trades = []
            position = 0
            entry_price = 0
            entry_date = None
            signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            min_index = 50
            
            for i in range(min_index, len(df)):
                row = df.iloc[i]
                current_price = row['Close']
                
                tech_score, tech_reasons = tech_analyzer.get_technical_score(row)
                
                if row['has_event']:
                    combined_score = tech_score * 0.6 + sentiment_score * 0.2 + row['event_score'] * 0.2
                else:
                    combined_score = tech_score * 0.8 + sentiment_score * 0.2
                
                if combined_score > 60:
                    action = "BUY"
                    signals['BUY'] += 1
                elif combined_score < 40:
                    action = "SELL"
                    signals['SELL'] += 1
                else:
                    action = "HOLD"
                    signals['HOLD'] += 1
                
                if action == "BUY" and position == 0:
                    position = 1
                    entry_price = current_price
                    entry_date = df.index[i]
                    
                elif action == "SELL" and position == 1:
                    position = 0
                    profit = (current_price - entry_price) / entry_price * 100
                    
                    if interval in ['1m', '5m']:
                        hold_minutes = (df.index[i] - entry_date).total_seconds() / 60
                        hold_display = f"{hold_minutes:.0f}min"
                    else:
                        hold_days = (df.index[i] - entry_date).days
                        hold_display = f"{hold_days}j"
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit': profit,
                        'hold_time': hold_display
                    })
            
            if position == 1:
                final_price = df['Close'].iloc[-1]
                profit = (final_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[-1],
                    'entry_price': entry_price,
                    'exit_price': final_price,
                    'profit': profit,
                    'hold_time': 'En cours'
                })
            
            if trades:
                profitable = [t for t in trades if t['profit'] > 0]
                total_profit = sum(t['profit'] for t in trades)
                win_rate = len(profitable) / len(trades) * 100
                avg_profit = np.mean([t['profit'] for t in trades])
                max_profit = max(t['profit'] for t in trades)
                max_loss = min(t['profit'] for t in trades)
            else:
                total_profit = win_rate = avg_profit = max_profit = max_loss = 0
            
            first_valid_idx = min_index
            buy_hold = (df['Close'].iloc[-1] - df['Close'].iloc[first_valid_idx]) / df['Close'].iloc[first_valid_idx] * 100
            
            strategy_score = 50
            if total_profit > buy_hold:
                strategy_score += 20
            if win_rate > 50:
                strategy_score += 15
            if total_profit > 5:
                strategy_score += 15
            
            elapsed = time.time() - start_time
            logger.info(f"   [OK] {elapsed:.2f}s | Score: {strategy_score:.0f} | Profit: {total_profit:+.2f}% | Trades: {len(trades)}")
            
            await sent_analyzer.close()
            
            return {
                'symbol': symbol,
                'interval': config['name'],
                'interval_code': interval,
                'events_detected': int(events_count),
                'total_trades': len(trades),
                'profitable_trades': len(profitable) if trades else 0,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'buy_hold_return': buy_hold,
                'strategy_vs_hold': total_profit - buy_hold,
                'strategy_score': strategy_score,
                'signals': signals,
                'processing_time': elapsed,
                'trades': trades[-5:],
                'candles': len(df)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur backtest {symbol} - {interval}: {e}")
            return None
    
    async def compare_intervals(self, symbol: str) -> Dict:
        """Compare les performances sur différents intervalles"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 COMPARAISON INTERVALLES: {symbol}")
        logger.info(f"{'='*80}")
        
        results = {}
        
        for interval_code in ['1m', '5m', '1h', '1d']:
            result = await self.backtest_with_interval(symbol, interval_code)
            if result:
                results[interval_code] = result
            await asyncio.sleep(1)
        
        if not results:
            logger.error(f"❌ Aucun résultat pour {symbol}")
            return {
                'symbol': symbol,
                'results': {},
                'best_interval': None,
                'best_score': 0,
                'error': 'Aucune donnée disponible'
            }
        
        best_interval = None
        best_score = 0
        
        for interval_code, result in results.items():
            if result['strategy_score'] > best_score:
                best_score = result['strategy_score']
                best_interval = interval_code
        
        logger.info(f"\n🏆 MEILLEUR: {BACKTEST_CONFIGS[best_interval]['name']} (Score: {best_score:.0f})")
        
        return {
            'symbol': symbol,
            'results': results,
            'best_interval': best_interval,
            'best_score': best_score
        }
    
    async def backtest_watchlist(self, watchlist: List[str], interval: str = "1d") -> List[Dict]:
        """Backtest sur liste de surveillance"""
        results = []
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[>>] BACKTEST WATCHLIST: {len(watchlist)} actions - {BACKTEST_CONFIGS[interval]['name']}")
        logger.info(f"{'='*80}")
        
        for i, symbol in enumerate(watchlist):
            result = await self.backtest_with_interval(symbol, interval)
            if result:
                results.append(result)
            
            if i > 0 and len(results) > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(watchlist) - i - 1)
                logger.info(f"[~] {i+1}/{len(watchlist)} | OK: {len(results)} | Restant: ~{remaining:.0f}s")
            
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"[DONE] {total_time:.0f}s ({total_time/60:.1f}min)")
        logger.info(f"[*] Analysées: {len(results)}/{len(watchlist)}")
        
        results.sort(key=lambda x: x['strategy_score'], reverse=True)
        return results


class TradingBot(commands.Bot):
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        
        super().__init__(command_prefix='!', intents=intents, help_command=None)
        
        self.news_analyzer = NewsAnalyzer()
        self.tech_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.backtest_engine = SmartBacktestEngine()
        
        self.alert_channel_id = None
        self.monitoring_active = False
        self.backtest_results = {}
        
    async def setup_hook(self):
        self.monitor_markets.start()
        logger.info("Bot initialisé")
    
    async def on_ready(self):
        logger.info(f'{self.user} connecté!')
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="événements & actualités 📰"
            )
        )
    
    @tasks.loop(minutes=1)
    async def monitor_markets(self):
        if not self.monitoring_active or not self.alert_channel_id:
            return
        
        channel = self.get_channel(self.alert_channel_id)
        if not channel:
            return
        
        logger.info("🔍 Surveillance...")
        
        for symbol in WATCHLIST[:5]:
            try:
                has_news, news_data, news_score = await self.news_analyzer.scrape_yahoo_news(symbol, hours=24)
                
                if not has_news:
                    continue
                
                stock = yf.Ticker(symbol)
                df = stock.history(period="3mo")
                
                if df.empty:
                    continue
                
                df = self.tech_analyzer.calculate_indicators_batch(df)
                row = df.iloc[-1]
                tech_score, tech_reasons = self.tech_analyzer.get_technical_score(row)
                
                sentiment_score, sentiment_reasons = await self.sentiment_analyzer.get_sentiment_score(symbol, news_data)
                
                combined_score = tech_score * 0.6 + sentiment_score * 0.4
                
                if combined_score > 60:
                    action = "BUY"
                    color = 0x00ff00
                elif combined_score < 40:
                    action = "SELL"
                    color = 0xff0000
                else:
                    continue
                
                embed = discord.Embed(
                    title=f"{'🟢' if action == 'BUY' else '🔴'} {action} - {symbol}",
                    color=color,
                    timestamp=datetime.now()
                )
                
                embed.add_field(name="💰 Prix", value=f"${row['Close']:.2f}", inline=True)
                embed.add_field(name="📊 Score", value=f"{combined_score:.0f}/100", inline=True)
                embed.add_field(name="📈 Tech", value=f"{tech_score:.0f}", inline=True)
                
                news_text = "\n".join(f"• {n['title'][:60]}..." for n in news_data[:3])
                embed.add_field(name="📰 Actualités", value=news_text, inline=False)
                
                all_reasons = tech_reasons[:2] + sentiment_reasons[:2]
                embed.add_field(
                    name="📝 Analyse",
                    value="\n".join(f"• {r}" for r in all_reasons[:4]),
                    inline=False
                )
                
                await channel.send(embed=embed)
                logger.info(f"✅ Signal: {symbol} - {action}")
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Erreur {symbol}: {e}")
    
    @monitor_markets.before_loop
    async def before_monitor(self):
        await self.wait_until_ready()


bot = TradingBot()

@bot.command(name='start')
async def start_monitoring(ctx):
    """Active la surveillance des marchés"""
    bot.alert_channel_id = ctx.channel.id
    bot.monitoring_active = True
    
    embed = discord.Embed(
        title="🚀 Surveillance Activée",
        description="Analyse en temps réel avec actualités",
        color=0x00ff00
    )
    embed.add_field(name="📋 Watchlist", value=f"{len(WATCHLIST)} actions", inline=True)
    embed.add_field(name="⏱️ Fréquence", value="1 minute", inline=True)
    await ctx.send(embed=embed)

@bot.command(name='backtest')
async def backtest(ctx, interval: str = "1d"):
    """
    Backtest sur un intervalle spécifique
    Intervalles: 1m, 5m, 1h, 1d
    Exemple: !backtest 5m
    """
    if interval not in BACKTEST_CONFIGS:
        await ctx.send(f"❌ Intervalle invalide. Utilisez: {', '.join(BACKTEST_CONFIGS.keys())}")
        return
    
    config = BACKTEST_CONFIGS[interval]
    
    embed = discord.Embed(
        title="⏳ Backtest en cours...",
        description=f"Analyse sur {config['name']} - {config['period']}",
        color=0xffff00
    )
    message = await ctx.send(embed=embed)
    
    start_time = time.time()
    
    try:
        results = await bot.backtest_engine.backtest_watchlist(WATCHLIST, interval)
        elapsed = time.time() - start_time
        
        if not results:
            embed = discord.Embed(title="❌ Aucun résultat", color=0xff0000)
            await message.edit(embed=embed)
            return
        
        embed = discord.Embed(
            title=f"📊 Backtest Terminé - {config['name']}",
            description=f"{len(results)} actions analysées en {elapsed:.1f}s",
            color=0x00ff00
        )
        
        total_candles = sum(r['candles'] for r in results)
        total_trades = sum(r['total_trades'] for r in results)
        
        embed.add_field(name="⏱️ Temps", value=f"{elapsed:.1f}s", inline=True)
        embed.add_field(name="✅ Actions", value=f"{len(results)}", inline=True)
        embed.add_field(name="📊 Bougies", value=f"{total_candles:,}", inline=True)
        embed.add_field(name="💼 Trades", value=f"{total_trades}", inline=True)
        
        for i, r in enumerate(results[:5], 1):
            perf = f"**{r['symbol']}** - {r['interval']}\n"
            perf += f"Profit: {r['total_profit']:+.2f}% | Win: {r['win_rate']:.0f}%\n"
            perf += f"Trades: {r['total_trades']} | vs Hold: {r['strategy_vs_hold']:+.2f}%\n"
            perf += f"Score: {r['strategy_score']:.0f}/100"
            
            embed.add_field(name=f"#{i}", value=perf, inline=True)
            
            if i % 3 == 0:
                embed.add_field(name="\u200b", value="\u200b", inline=False)
        
        bot.backtest_results[interval] = {r['symbol']: r for r in results}
        
        await message.edit(embed=embed)
        
    except Exception as e:
        logger.error(f"Erreur backtest: {e}")
        await message.edit(content=f"❌ Erreur: {str(e)}")

@bot.command(name='compare')
async def compare_intervals(ctx, symbol: str):
    """
    Compare les performances sur tous les intervalles
    Exemple: !compare AAPL
    """
    symbol = symbol.upper()
    
    embed = discord.Embed(
        title=f"⏳ Comparaison {symbol}...",
        description="Test sur 1m, 5m, 1h, 1d",
        color=0xffff00
    )
    message = await ctx.send(embed=embed)
    
    try:
        comparison = await bot.backtest_engine.compare_intervals(symbol)
        results = comparison['results']
        
        if not results or 'error' in comparison:
            embed = discord.Embed(
                title=f"❌ Erreur - {symbol}",
                description="Impossible de récupérer les données",
                color=0xff0000
            )
            embed.add_field(
                name="💡 Suggestions",
                value="• Vérifiez le symbole (ex: NVDA, pas NVDIA)\n• Certains symboles peuvent être indisponibles\n• Réessayez dans quelques instants",
                inline=False
            )
            await message.edit(embed=embed)
            return
        
        embed = discord.Embed(
            title=f"📊 Comparaison Intervalles - {symbol}",
            description=f"🏆 Meilleur: **{BACKTEST_CONFIGS[comparison['best_interval']]['name']}** (Score: {comparison['best_score']:.0f}/100)",
            color=0x00ff00
        )
        
        for interval_code in ['1m', '5m', '1h', '1d']:
            if interval_code in results:
                r = results[interval_code]
                
                is_best = interval_code == comparison['best_interval']
                emoji = "🏆" if is_best else "📈"
                
                value = f"{emoji} **Score: {r['strategy_score']:.0f}/100**\n"
                value += f"Profit: {r['total_profit']:+.2f}%\n"
                value += f"Win Rate: {r['win_rate']:.0f}%\n"
                value += f"Trades: {r['total_trades']}\n"
                value += f"vs Hold: {r['strategy_vs_hold']:+.2f}%\n"
                value += f"Bougies: {r['candles']:,}"
                
                embed.add_field(
                    name=f"{r['interval']}",
                    value=value,
                    inline=True
                )
            else:
                embed.add_field(
                    name=f"{BACKTEST_CONFIGS[interval_code]['name']}",
                    value="❌ Pas de données",
                    inline=True
                )
        
        best = results[comparison['best_interval']]
        recommendation = "\n**Recommandation:**\n"
        if best['total_profit'] > 10 and best['win_rate'] > 60:
            recommendation += "✅ Stratégie très performante"
        elif best['total_profit'] > 5 and best['win_rate'] > 50:
            recommendation += "⚠️ Stratégie moyennement performante"
        else:
            recommendation += "❌ Stratégie peu performante"
        
        embed.add_field(name="\u200b", value=recommendation, inline=False)
        embed.set_footer(text=f"Données analysées: {sum(r.get('candles', 0) for r in results.values()):,} bougies")
        
        await message.edit(embed=embed)
        
    except Exception as e:
        logger.error(f"Erreur comparaison: {e}")
        embed = discord.Embed(
            title=f"❌ Erreur - {symbol}",
            description=f"```{str(e)[:200]}```",
            color=0xff0000
        )
        embed.add_field(
            name="💡 Conseil",
            value="Vérifiez que le symbole est correct (ex: NVDA, AAPL, TSLA)",
            inline=False
        )
        await message.edit(embed=embed)

@bot.command(name='analyze')
async def analyze(ctx, symbol: str):
    """Analyse complète d'une action"""
    symbol = symbol.upper()
    
    embed = discord.Embed(title=f"⏳ Analyse {symbol}...", color=0xffff00)
    message = await ctx.send(embed=embed)
    
    try:
        has_news, news_data, news_score = await bot.news_analyzer.scrape_yahoo_news(symbol, hours=48)
        
        stock = yf.Ticker(symbol)
        df = stock.history(period="3mo")
        
        if df.empty:
            await message.edit(content=f"❌ Impossible de récupérer les données pour {symbol}")
            return
        
        df = bot.tech_analyzer.calculate_indicators_batch(df)
        row = df.iloc[-1]
        tech_score, tech_reasons = bot.tech_analyzer.get_technical_score(row)
        
        sentiment_score, sentiment_reasons = await bot.sentiment_analyzer.get_sentiment_score(symbol, news_data if has_news else None)
        
        combined_score = tech_score * 0.6 + sentiment_score * 0.4
        
        if combined_score > 60:
            action = "BUY 🟢"
            color = 0x00ff00
        elif combined_score < 40:
            action = "SELL 🔴"
            color = 0xff0000
        else:
            action = "HOLD 🟡"
            color = 0xffff00
        
        embed = discord.Embed(
            title=f"📊 Analyse Complète - {symbol}",
            description=f"**Action recommandée: {action}**",
            color=color,
            timestamp=datetime.now()
        )
        
        embed.add_field(name="💰 Prix Actuel", value=f"${row['Close']:.2f}", inline=True)
        embed.add_field(name="📊 Score Global", value=f"{combined_score:.0f}/100", inline=True)
        embed.add_field(name="📈 Score Tech", value=f"{tech_score:.0f}/100", inline=True)
        
        if pd.notna(row.get('rsi')):
            embed.add_field(name="📉 RSI", value=f"{row['rsi']:.1f}", inline=True)
        if pd.notna(row.get('macd')):
            embed.add_field(name="📊 MACD", value=f"{row['macd']:.4f}", inline=True)
        if pd.notna(row.get('volume_ratio')):
            embed.add_field(name="📦 Volume", value=f"{row['volume_ratio']:.2f}x", inline=True)
        
        if has_news and news_data:
            news_text = f"**{len(news_data)} actualités récentes** (Score: {news_score:.0f}/100)\n\n"
            for i, n in enumerate(news_data[:4], 1):
                news_text += f"{i}. {n['title'][:80]}...\n"
                if n['keywords']:
                    news_text += f"   🏷️ {', '.join(n['keywords'][:3])}\n"
            embed.add_field(name="📰 Actualités", value=news_text, inline=False)
        else:
            embed.add_field(name="📰 Actualités", value="Aucune actualité récente trouvée", inline=False)
        
        tech_analysis = "**Indicateurs Techniques:**\n"
        tech_analysis += "\n".join(f"• {r}" for r in tech_reasons[:4])
        embed.add_field(name="🔍 Analyse Technique", value=tech_analysis, inline=False)
        
        sentiment_analysis = "**Sentiment du Marché:**\n"
        sentiment_analysis += "\n".join(f"• {r}" for r in sentiment_reasons[:3])
        embed.add_field(name="💭 Sentiment", value=sentiment_analysis, inline=False)
        
        await message.edit(embed=embed)
        
    except Exception as e:
        logger.error(f"Erreur analyse {symbol}: {e}")
        await message.edit(content=f"❌ Erreur lors de l'analyse: {str(e)}")

@bot.command(name='news')
async def news(ctx, symbol: str):
    """Affiche les actualités récentes d'une action"""
    symbol = symbol.upper()
    
    embed = discord.Embed(title=f"⏳ Recherche actualités {symbol}...", color=0xffff00)
    message = await ctx.send(embed=embed)
    
    try:
        has_news, news_data, news_score = await bot.news_analyzer.scrape_yahoo_news(symbol, hours=48)
        
        if has_news and news_data:
            embed = discord.Embed(
                title=f"📰 Actualités - {symbol}",
                description=f"**{len(news_data)} actualités récentes** | Score: {news_score:.0f}/100",
                color=0x00ff00,
                timestamp=datetime.now()
            )
            
            for i, article in enumerate(news_data[:8], 1):
                date_str = article['date'].strftime("%d/%m %H:%M")
                
                value = f"📅 {date_str} | 📰 {article['publisher']}\n"
                value += f"**{article['title']}**\n"
                
                if article['keywords']:
                    value += f"🏷️ Tags: {', '.join(article['keywords'][:4])}\n"
                
                value += f"⭐ Importance: {article['importance']:.1f}/10"
                
                if article.get('link'):
                    value += f"\n🔗 [Lire l'article]({article['link']})"
                
                embed.add_field(
                    name=f"#{i}",
                    value=value,
                    inline=False
                )
                
                if i % 4 == 0 and i < len(news_data):
                    embed.add_field(name="\u200b", value="─" * 40, inline=False)
        else:
            embed = discord.Embed(
                title=f"📰 Actualités - {symbol}",
                description="❌ Aucune actualité récente trouvée (48h)",
                color=0xff6600
            )
            embed.add_field(
                name="💡 Suggestions",
                value="• Vérifiez le symbole\n• Essayez un autre symbole\n• Cette action peut avoir peu d'actualités",
                inline=False
            )
        
        await message.edit(embed=embed)
        
    except Exception as e:
        logger.error(f"Erreur news {symbol}: {e}")
        await message.edit(content=f"❌ Erreur: {str(e)}")

@bot.command(name='stop')
async def stop(ctx):
    """Désactive la surveillance"""
    bot.monitoring_active = False
    
    embed = discord.Embed(
        title="⏸️ Surveillance Désactivée",
        description="La surveillance des marchés est arrêtée",
        color=0xff9900
    )
    await ctx.send(embed=embed)

@bot.command(name='aide')
async def aide(ctx):
    """Affiche l'aide"""
    embed = discord.Embed(
        title="📚 Guide des Commandes",
        description="Bot de Trading avec analyse multi-intervalles",
        color=0x00ffff
    )
    
    embed.add_field(name="🔍 **Surveillance**", value="\u200b", inline=False)
    embed.add_field(name="!start", value="Active la surveillance temps réel", inline=False)
    embed.add_field(name="!stop", value="Désactive la surveillance", inline=False)
    
    embed.add_field(name="📊 **Analyse**", value="\u200b", inline=False)
    embed.add_field(name="!analyze [SYMBOL]", value="Analyse complète d'une action\nEx: `!analyze AAPL`", inline=False)
    embed.add_field(name="!news [SYMBOL]", value="Affiche les actualités récentes\nEx: `!news NVDA`", inline=False)
    
    embed.add_field(name="⏱️ **Backtest**", value="\u200b", inline=False)
    embed.add_field(name="!backtest [interval]", value="Backtest sur la watchlist\nIntervalles: 1m, 5m, 1h, 1d\nEx: `!backtest 5m`", inline=False)
    embed.add_field(name="!compare [SYMBOL]", value="Compare tous les intervalles\nEx: `!compare TSLA`", inline=False)
    
    embed.set_footer(text="💡 Astuce: Commencez par !backtest 1d puis !compare SYMBOL")
    
    await ctx.send(embed=embed)

@bot.command(name='status')
async def status(ctx):
    """Affiche le statut du bot"""
    embed = discord.Embed(
        title="🤖 Statut du Bot",
        color=0x00ff00 if bot.monitoring_active else 0xff9900
    )
    
    embed.add_field(
        name="📡 Surveillance",
        value="✅ Active" if bot.monitoring_active else "⏸️ Inactive",
        inline=True
    )
    
    embed.add_field(
        name="📋 Watchlist",
        value=f"{len(WATCHLIST)} actions",
        inline=True
    )
    
    backtest_count = len(bot.backtest_results)
    embed.add_field(
        name="💾 Backtests",
        value=f"{backtest_count} intervalles en cache",
        inline=True
    )
    
    if bot.backtest_results:
        bt_info = "\n".join([
            f"• {BACKTEST_CONFIGS[k]['name']}: {len(v)} actions"
            for k, v in bot.backtest_results.items()
        ])
        embed.add_field(name="📊 Données disponibles", value=bt_info, inline=False)
    
    await ctx.send(embed=embed)


if __name__ == "__main__":
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("❌ Token Discord manquant!")
        print("\n⚠️ Configuration requise:")
        print("1. Créez un fichier .env")
        print("2. Ajoutez: DISCORD_BOT_TOKEN=votre_token")
        print("3. (Optionnel)")