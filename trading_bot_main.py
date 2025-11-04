"""
Bot Discord de Trading - BACKTEST AVEC ACTUALITÉS TEMPS RÉEL
- Backtest réaliste : simulation des décisions avec actualités historiques
- Validation IA des décisions de trading via HuggingFace
- Score de confiance pour chaque trade (0-100)
"""

import discord
from discord.ext import commands, tasks
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
from textblob import TextBlob
import logging
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv
import warnings
import time
import json
from time import sleep
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


class HistoricalNewsAnalyzer:
    """Récupère les actualités historiques pour chaque date de backtest"""
    
    def __init__(self):
        self.session = None
        self.newsapi_key = os.getenv('NEWSAPI_KEY', '')
        self.finnhub_key = os.getenv('FINNHUB_KEY', '')
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
        self.news_cache = {}  # Cache pour éviter appels API répétés
        
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_news_for_date(self, symbol: str, target_date: datetime) -> Tuple[bool, List[Dict], float]:
        """Récupère les actualités pour une date précise (simulation temps réel)"""
        try:
            # Normaliser la date en timezone-naive
            if hasattr(target_date, 'tz') and target_date.tz is not None:
                target_date = target_date.replace(tzinfo=None)
            
            # Vérifier le cache (clé = symbol + date)
            cache_key = f"{symbol}_{target_date.strftime('%Y-%m-%d')}"
            if cache_key in self.news_cache:
                return self.news_cache[cache_key]
            
            session = await self.get_session()
            
            # Fenêtre de 48h avant la date cible
            from_date = target_date - timedelta(hours=48)
            to_date = target_date
            
            company_names = {
                'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google Alphabet',
                'AMZN': 'Amazon', 'NVDA': 'Nvidia', 'META': 'Meta Facebook',
                'TSLA': 'Tesla', 'JPM': 'JPMorgan', 'V': 'Visa', 
                'NFLX': 'Netflix', 'AMD': 'AMD', 'INTC': 'Intel',
                'BRK-B': 'Berkshire Hathaway', 'JNJ': 'Johnson Johnson',
                'WMT': 'Walmart', 'PG': 'Procter Gamble', 'MA': 'Mastercard',
                'DIS': 'Disney', 'ADBE': 'Adobe', 'CRM': 'Salesforce',
                'ORCL': 'Oracle', 'CSCO': 'Cisco', 'PEP': 'Pepsi', 'COST': 'Costco',
                'AVGO': 'Broadcom'
            }
            
            search_term = company_names.get(symbol, symbol)
            
            # Essayer Finnhub d'abord
            if self.finnhub_key:
                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': symbol,
                    'from': from_date.strftime('%Y-%m-%d'),
                    'to': to_date.strftime('%Y-%m-%d'),
                    'token': self.finnhub_key
                }
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list) and len(data) > 0:
                            result = await self._parse_finnhub_news(data, target_date)
                            self.news_cache[cache_key] = result  # Sauvegarder dans le cache
                            return result
            
            # Fallback sur NewsAPI
            if self.newsapi_key:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f"{search_term} stock OR {symbol}",
                    'from': from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'to': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'apiKey': self.newsapi_key,
                    'pageSize': 20
                }
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'articles' in data and data['articles']:
                            result = await self._parse_newsapi_news(data['articles'], target_date)
                            self.news_cache[cache_key] = result  # Sauvegarder dans le cache
                            return result
            
            result = (False, [], 0.0)
            self.news_cache[cache_key] = result  # Même mettre en cache les résultats vides
            return result
            
        except Exception as e:
            logger.debug(f"Erreur news historiques {symbol} @ {target_date}: {e}")
            return False, [], 0.0
    
    async def _parse_finnhub_news(self, data: List, target_date: datetime) -> Tuple[bool, List[Dict], float]:
        """Parse les actualités Finnhub"""
        news_items = []
        
        # Normaliser target_date en timezone-naive
        if hasattr(target_date, 'tz') and target_date.tz is not None:
            target_date = target_date.replace(tzinfo=None)
        
        cutoff_time = target_date - timedelta(hours=48)
        
        importance_keywords = {
            'earnings': 3.0, 'revenue': 2.5, 'profit': 2.5, 'loss': 2.5,
            'launch': 2.0, 'partnership': 2.0, 'acquisition': 3.0, 'merger': 3.0,
            'FDA': 2.5, 'approval': 2.0, 'breakthrough': 2.0, 'record': 1.5,
            'guidance': 2.0, 'upgrade': 2.0, 'downgrade': 2.0, 'analyst': 1.5,
            'lawsuit': 1.5, 'investigation': 1.5, 'recall': 2.0, 'bankruptcy': 3.0,
            'dividend': 1.5, 'split': 2.0, 'buyback': 1.5, 'expansion': 1.5,
            'contract': 1.5, 'deal': 1.5, 'beats': 2.0, 'misses': 2.0
        }
        
        for article in data:
            try:
                title = article.get('headline', '')
                if not title or len(title) < 10:
                    continue
                
                timestamp = article.get('datetime', 0)
                pub_date = datetime.fromtimestamp(timestamp)
                # Normaliser en timezone-naive
                if hasattr(pub_date, 'tz') and pub_date.tz is not None:
                    pub_date = pub_date.replace(tzinfo=None)
                
                if pub_date < cutoff_time or pub_date > target_date:
                    continue
                
                title_lower = title.lower()
                importance = 1.0
                matched_keywords = []
                
                for keyword, weight in importance_keywords.items():
                    if keyword in title_lower:
                        importance += weight
                        matched_keywords.append(keyword)
                
                news_items.append({
                    'title': title,
                    'publisher': article.get('source', 'Finnhub'),
                    'date': pub_date,
                    'importance': importance,
                    'keywords': matched_keywords,
                    'summary': article.get('summary', '')[:200]
                })
                
            except Exception as e:
                continue
        
        has_news = len(news_items) > 0
        total_importance = sum(n['importance'] for n in news_items)
        news_score = min(100, total_importance * 10) if has_news else 0.0
        
        return has_news, news_items, news_score
    
    async def _parse_newsapi_news(self, articles: List, target_date: datetime) -> Tuple[bool, List[Dict], float]:
        """Parse les actualités NewsAPI"""
        news_items = []
        
        # Normaliser target_date en timezone-naive
        if hasattr(target_date, 'tz') and target_date.tz is not None:
            target_date = target_date.replace(tzinfo=None)
        
        cutoff_time = target_date - timedelta(hours=48)
        
        importance_keywords = {
            'earnings': 3.0, 'revenue': 2.5, 'profit': 2.5, 'loss': 2.5,
            'launch': 2.0, 'partnership': 2.0, 'acquisition': 3.0, 'merger': 3.0,
            'FDA': 2.5, 'approval': 2.0, 'breakthrough': 2.0, 'record': 1.5,
            'guidance': 2.0, 'upgrade': 2.0, 'downgrade': 2.0, 'analyst': 1.5,
            'lawsuit': 1.5, 'investigation': 1.5, 'recall': 2.0, 'bankruptcy': 3.0,
            'dividend': 1.5, 'split': 2.0, 'buyback': 1.5, 'expansion': 1.5,
            'contract': 1.5, 'deal': 1.5, 'beats': 2.0, 'misses': 2.0
        }
        
        for article in articles:
            try:
                title = article.get('title', '')
                if not title or len(title) < 10:
                    continue
                
                pub_date_str = article.get('publishedAt', '')
                try:
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    # Convertir en timezone-naive
                    if hasattr(pub_date, 'tzinfo') and pub_date.tzinfo is not None:
                        pub_date = pub_date.replace(tzinfo=None)
                except:
                    continue
                
                if pub_date < cutoff_time or pub_date > target_date:
                    continue
                
                title_lower = title.lower()
                importance = 1.0
                matched_keywords = []
                
                for keyword, weight in importance_keywords.items():
                    if keyword in title_lower:
                        importance += weight
                        matched_keywords.append(keyword)
                
                news_items.append({
                    'title': title,
                    'publisher': article.get('source', {}).get('name', 'NewsAPI'),
                    'date': pub_date,
                    'importance': importance,
                    'keywords': matched_keywords,
                    'summary': article.get('description', '')[:200]
                })
                
            except Exception as e:
                continue
        
        has_news = len(news_items) > 0
        total_importance = sum(n['importance'] for n in news_items)
        news_score = min(100, total_importance * 10) if has_news else 0.0
        
        return has_news, news_items, news_score
    
    async def ask_ai_decision(self, symbol: str, bot_decision: str, news_data: List[Dict], 
                             current_price: float, tech_score: float) -> Tuple[int, str]:
        """
        Demande à l'IA HuggingFace si la décision du bot est bonne
        Retourne un score 0-100 et une explication
        """
        try:
            if not news_data or not self.hf_token:
                return 50, "Pas d'actualités disponibles pour validation"
            
            # Construire le contexte pour l'IA
            news_summary = "\n".join([
                f"- {n['title']} (Importance: {n['importance']:.1f})"
                for n in news_data[:5]
            ])
            
            prompt = f"""Analyze this trading decision:

Symbol: {symbol}
Current Price: ${current_price:.2f}
Technical Score: {tech_score:.0f}/100
Bot Decision: {bot_decision}

Recent News:
{news_summary}

Question: Should the bot {bot_decision} based on these news? Rate the decision quality from 0 to 100.
- 0-30: Bad decision, news suggest opposite action
- 31-50: Uncertain, mixed signals
- 51-70: Good decision, news moderately support it
- 71-100: Excellent decision, news strongly support it

Respond with format: "SCORE: [number]|REASON: [explanation]"
"""
            
            session = await self.get_session()
            url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }
            
            async with session.post(url, headers=headers, json=payload, timeout=15) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('generated_text', '')
                        
                        # Parser la réponse
                        try:
                            if 'SCORE:' in text and 'REASON:' in text:
                                parts = text.split('|')
                                score_part = parts[0].split('SCORE:')[1].strip()
                                reason_part = parts[1].split('REASON:')[1].strip() if len(parts) > 1 else "AI analysis"
                                
                                score = int(''.join(filter(str.isdigit, score_part[:3])))
                                score = max(0, min(100, score))
                                
                                return score, reason_part[:200]
                        except:
                            pass
            
            # Fallback: analyse simple basée sur le sentiment
            return await self._simple_sentiment_score(news_data, bot_decision)
            
        except Exception as e:
            logger.debug(f"Erreur AI validation: {e}")
            return await self._simple_sentiment_score(news_data, bot_decision)
    
    async def _simple_sentiment_score(self, news_data: List[Dict], bot_decision: str) -> Tuple[int, str]:
        """Score de sentiment simple comme fallback"""
        try:
            sentiments = []
            for article in news_data[:5]:
                blob = TextBlob(article['title'])
                sentiment = blob.sentiment.polarity
                sentiments.append(sentiment * article['importance'])
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Convertir en score 0-100
            base_score = (avg_sentiment + 1) * 50  # -1 to 1 -> 0 to 100
            
            # Ajuster selon la décision du bot
            if bot_decision == "BUY":
                if avg_sentiment > 0.2:
                    score = int(base_score * 1.2)
                    reason = "Sentiment positif supporte l'achat"
                elif avg_sentiment < -0.2:
                    score = int(base_score * 0.6)
                    reason = "Sentiment négatif contredit l'achat"
                else:
                    score = int(base_score)
                    reason = "Sentiment neutre"
            else:  # SELL
                if avg_sentiment < -0.2:
                    score = int((1 - base_score/100) * 100 * 1.2)
                    reason = "Sentiment négatif supporte la vente"
                elif avg_sentiment > 0.2:
                    score = int((1 - base_score/100) * 100 * 0.6)
                    reason = "Sentiment positif contredit la vente"
                else:
                    score = int((1 - base_score/100) * 100)
                    reason = "Sentiment neutre"
            
            score = max(0, min(100, score))
            return score, reason
            
        except:
            return 50, "Analyse de sentiment indisponible"
    
    async def close(self):
        if self.session:
            await self.session.close()


class TechnicalAnalyzer:
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcul des indicateurs techniques"""
        data_size = len(df)
        
        # SMA
        df['sma_20'] = df['Close'].rolling(window=min(20, data_size//2)).mean()
        if data_size > 50:
            df['sma_50'] = df['Close'].rolling(window=50).mean()
        if data_size > 200:
            df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        # EMA et MACD
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_window = min(20, data_size//2)
        bb_sma = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)
        
        # Volume
        df['volume_sma'] = df['Volume'].rolling(window=min(20, data_size//2)).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma'].replace(0, 1)
        
        return df
    
    def get_technical_score(self, row: pd.Series) -> Tuple[float, List[str]]:
        """Calcule le score technique et les raisons"""
        score = 50.0
        reasons = []
        
        # Tendance
        if pd.notna(row.get('sma_20')) and pd.notna(row.get('sma_50')):
            if row['sma_20'] > row['sma_50']:
                score += 15
                reasons.append("✅ Tendance haussière (SMA)")
            else:
                score -= 15
                reasons.append("⚠️ Tendance baissière (SMA)")
        
        # RSI
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
        
        # MACD
        if pd.notna(row.get('macd')) and pd.notna(row.get('macd_signal')):
            if row['macd'] > row['macd_signal']:
                score += 10
                reasons.append("✅ MACD bullish")
            else:
                score -= 10
                reasons.append("⚠️ MACD bearish")
        
        # Bollinger Bands
        if pd.notna(row.get('bb_lower')) and pd.notna(row.get('bb_upper')):
            bb_range = row['bb_upper'] - row['bb_lower']
            if bb_range > 0:
                bb_position = (row['Close'] - row['bb_lower']) / bb_range
                if bb_position < 0.2:
                    score += 15
                    reasons.append("🎯 Prix proche BB inférieure")
                elif bb_position > 0.8:
                    score -= 15
                    reasons.append("⚠️ Prix proche BB supérieure")
        
        # Volume
        if pd.notna(row.get('volume_ratio')) and row['volume_ratio'] > 1.5:
            score += 5
            reasons.append(f"📈 Volume élevé ({row['volume_ratio']:.1f}x)")
        
        score = max(0, min(100, score))
        return score, reasons


class RealisticBacktestEngine:
    """
    Backtest réaliste qui simule les décisions en temps réel
    avec validation IA pour chaque trade
    """
    
    def __init__(self):
        self.news_analyzer = HistoricalNewsAnalyzer()
        self.tech_analyzer = TechnicalAnalyzer()
        
    async def backtest_with_news_validation(self, symbol: str, months: int = 6) -> Optional[Dict]:
        """
        Backtest sur X mois avec un point de décision par mois
        Le bot prend une décision, puis l'IA valide avec les actualités du jour
        """
        start_time = time.time()
        logger.info(f"[>>] Backtest réaliste {symbol} - {months} mois")
        
        try:
            # Récupérer les données historiques
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months*30 + 60)  # +60j pour indicateurs
            
            df = stock.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 100:
                logger.warning(f"   [X] Données insuffisantes pour {symbol}")
                return None
            
            # Normaliser l'index en timezone-naive
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            logger.info(f"   [✓] {len(df)} jours de données")
            
            # Calculer les indicateurs
            df = self.tech_analyzer.calculate_indicators(df)
            
            # Analyser CHAQUE JOUR après le warm-up (60 jours pour les indicateurs)
            warm_up_days = 60
            decision_points = list(range(warm_up_days, len(df)))
            
            if len(decision_points) < 10:
                logger.warning(f"   [X] Pas assez de points de décision")
                return None
            
            logger.info(f"   [✓] {len(decision_points)} jours d'analyse (backtest quotidien)")
            
            # Simuler les trades
            trades = []
            position = 0  # 0 = pas de position, 1 = long
            entry_price = 0
            entry_date = None
            entry_idx = 0
            
            validated_buys = 0
            validated_sells = 0
            rejected_buys = 0
            rejected_sells = 0
            
            for idx in decision_points:
                row = df.iloc[idx]
                current_date = df.index[idx]
                current_price = row['Close']
                
                # Le bot prend sa décision basée sur la technique
                tech_score, tech_reasons = self.tech_analyzer.get_technical_score(row)
                
                # Décision du bot
                if tech_score > 60:
                    bot_decision = "BUY"
                elif tech_score < 40:
                    bot_decision = "SELL"
                else:
                    bot_decision = "HOLD"
                
                # Récupérer les actualités de cette date (simulation temps réel)
                has_news, news_data, news_score = await self.news_analyzer.get_news_for_date(
                    symbol, current_date
                )

                sleep(0.5)
                
                # L'IA valide la décision du bot
                ai_score = 50
                ai_reason = "Pas d'actualités"
                
                if bot_decision in ["BUY", "SELL"]:
                    ai_score, ai_reason = await self.news_analyzer.ask_ai_decision(
                        symbol, bot_decision, news_data, current_price, tech_score
                    )
                
                # Log uniquement pour les jours importants (éviter trop de logs)
                if bot_decision != "HOLD" or idx % 20 == 0:  # Log tous les 20 jours ou si action
                    logger.info(f"   [{current_date.strftime('%Y-%m-%d')}] Bot: {bot_decision} | "
                              f"Tech: {tech_score:.0f} | AI: {ai_score}/100 | News: {len(news_data)}")
                
                # Exécuter le trade si l'IA valide (score > 70)
                if bot_decision == "BUY" and position == 0:
                    if ai_score > 70:
                        position = 1
                        entry_price = current_price
                        entry_date = current_date
                        entry_idx = idx
                        validated_buys += 1
                        logger.info(f"   ✅ BUY validé par IA @ ${current_price:.2f}")
                    else:
                        rejected_buys += 1
                        logger.info(f"   ❌ BUY rejeté par IA (score: {ai_score})")
                
                elif bot_decision == "SELL" and position == 1:
                    if ai_score > 70:
                        position = 0
                        exit_price = current_price
                        profit = (exit_price - entry_price) / entry_price * 100
                        hold_days = (current_date - entry_date).days
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit,
                            'hold_days': hold_days,
                            'ai_buy_score': ai_score,
                            'ai_sell_score': ai_score,
                            'news_count': len(news_data),
                            'tech_score': tech_score
                        })
                        
                        validated_sells += 1
                        logger.info(f"   ✅ SELL validé par IA @ ${exit_price:.2f} | "
                                  f"Profit: {profit:+.2f}% | Durée: {hold_days}j")
                    else:
                        rejected_sells += 1
                        logger.info(f"   ❌ SELL rejeté par IA (score: {ai_score})")
                
                # Délai réduit entre chaque jour (seulement si on a fait un appel API)
                if bot_decision in ["BUY", "SELL"] and has_news:
                    await asyncio.sleep(0.2)  # Petit délai seulement si appel API
            
            # Clôturer la position si encore ouverte
            if position == 1:
                final_price = df['Close'].iloc[-1]
                profit = (final_price - entry_price) / entry_price * 100
                hold_days = (df.index[-1] - entry_date).days
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[-1],
                    'entry_price': entry_price,
                    'exit_price': final_price,
                    'profit': profit,
                    'hold_days': hold_days,
                    'ai_buy_score': 0,
                    'ai_sell_score': 0,
                    'news_count': 0,
                    'tech_score': 0
                })
            
            # Calculer les statistiques
            if trades:
                profitable = [t for t in trades if t['profit'] > 0]
                total_profit = sum(t['profit'] for t in trades)
                win_rate = len(profitable) / len(trades) * 100
                avg_profit = np.mean([t['profit'] for t in trades])
                max_profit = max(t['profit'] for t in trades)
                max_loss = min(t['profit'] for t in trades)
                avg_hold_days = np.mean([t['hold_days'] for t in trades])
            else:
                total_profit = win_rate = avg_profit = max_profit = max_loss = avg_hold_days = 0
            
            # Buy & Hold de référence
            first_price = df['Close'].iloc[decision_points[0]]
            last_price = df['Close'].iloc[-1]
            buy_hold_return = (last_price - first_price) / first_price * 100
            
            # Score de stratégie
            strategy_score = 50
            if total_profit > buy_hold_return:
                strategy_score += 20
            if win_rate > 50:
                strategy_score += 15
            if validated_buys > rejected_buys:
                strategy_score += 10
            if total_profit > 5:
                strategy_score += 5
            
            elapsed = time.time() - start_time
            logger.info(f"   [OK] {elapsed:.2f}s | Score: {strategy_score:.0f} | "
                       f"Profit: {total_profit:+.2f}% | Trades: {len(trades)}")
            
            return {
                'symbol': symbol,
                'period': f"{months} mois",
                'total_trades': len(trades),
                'profitable_trades': len(profitable) if trades else 0,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'avg_hold_days': avg_hold_days,
                'buy_hold_return': buy_hold_return,
                'strategy_vs_hold': total_profit - buy_hold_return,
                'strategy_score': strategy_score,
                'validated_buys': validated_buys,
                'rejected_buys': rejected_buys,
                'validated_sells': validated_sells,
                'rejected_sells': rejected_sells,
                'decision_points': len(decision_points),
                'processing_time': elapsed,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur backtest {symbol}: {e}")
            return None
    
    async def backtest_watchlist(self, watchlist: List[str], months: int = 6) -> List[Dict]:
        """Backtest toute la watchlist avec validation IA"""
        results = []
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[>>] BACKTEST WATCHLIST RÉALISTE: {len(watchlist)} actions - {months} mois")
        logger.info(f"{'='*80}")
        
        for i, symbol in enumerate(watchlist):
            result = await self.backtest_with_news_validation(symbol, months)
            if result:
                results.append(result)
            
            if i > 0 and len(results) > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(watchlist) - i - 1)
                logger.info(f"[~] {i+1}/{len(watchlist)} | OK: {len(results)} | Restant: ~{remaining:.0f}s")
            
            await asyncio.sleep(3)  # Délai entre chaque action pour éviter rate limiting
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"[DONE] {total_time:.0f}s ({total_time/60:.1f}min)")
        logger.info(f"[*] Analysées: {len(results)}/{len(watchlist)}")
        
        results.sort(key=lambda x: x['strategy_score'], reverse=True)
        return results
    
    async def close(self):
        await self.news_analyzer.close()


class TradingBot(commands.Bot):
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        
        super().__init__(command_prefix='!', intents=intents, help_command=None)
        
        self.backtest_engine = RealisticBacktestEngine()
        
    async def on_ready(self):
        logger.info(f'{self.user} connecté!')
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="backtests avec IA 🤖"
            )
        )


bot = TradingBot()

@bot.command(name='backtest')
async def backtest(ctx, months: int = 6):
    """
    Backtest réaliste avec validation IA - ANALYSE QUOTIDIENNE
    Analyse chaque jour de trading avec les actualités du jour
    Exemple: !backtest 6 (analyse ~120 jours de trading)
    """
    if months < 1 or months > 24:
        await ctx.send("❌ Période invalide. Utilisez entre 1 et 24 mois.")
        return
    
    embed = discord.Embed(
        title="⏳ Backtest Réaliste en cours...",
        description=f"Analyse QUOTIDIENNE sur {months} mois (~{months*20} jours)\nValidation IA pour chaque décision BUY/SELL\n⚠️ Cela peut prendre 10-30 minutes...",
        color=0xffff00
    )
    embed.add_field(
        name="📋 Watchlist",
        value=f"{len(WATCHLIST)} actions",
        inline=True
    )
    embed.add_field(
        name="🤖 Validation",
        value="IA HuggingFace",
        inline=True
    )
    message = await ctx.send(embed=embed)
    
    start_time = time.time()
    
    try:
        results = await bot.backtest_engine.backtest_watchlist(WATCHLIST, months)
        elapsed = time.time() - start_time
        
        if not results:
            embed = discord.Embed(
                title="❌ Aucun résultat",
                description="Impossible de récupérer les données",
                color=0xff0000
            )
            await message.edit(embed=embed)
            return
        
        embed = discord.Embed(
            title=f"📊 Backtest Réaliste Terminé - {months} mois",
            description=f"{len(results)} actions analysées en {elapsed/60:.1f} minutes",
            color=0x00ff00
        )
        
        total_trades = sum(r['total_trades'] for r in results)
        total_validated = sum(r['validated_buys'] + r['validated_sells'] for r in results)
        total_rejected = sum(r['rejected_buys'] + r['rejected_sells'] for r in results)
        
        embed.add_field(name="⏱️ Temps", value=f"{elapsed/60:.1f}min", inline=True)
        embed.add_field(name="✅ Actions", value=f"{len(results)}", inline=True)
        embed.add_field(name="💼 Trades", value=f"{total_trades}", inline=True)
        embed.add_field(name="🤖 Validés", value=f"{total_validated}", inline=True)
        embed.add_field(name="❌ Rejetés", value=f"{total_rejected}", inline=True)
        embed.add_field(name="📊 Taux validation", value=f"{total_validated/(total_validated+total_rejected)*100:.0f}%" if (total_validated+total_rejected) > 0 else "N/A", inline=True)
        
        # Top 5 résultats
        for i, r in enumerate(results[:5], 1):
            perf = f"**{r['symbol']}** - {r['period']}\n"
            perf += f"💰 Profit: **{r['total_profit']:+.2f}%**\n"
            perf += f"📈 Win Rate: {r['win_rate']:.0f}%\n"
            perf += f"💼 Trades: {r['total_trades']} ({r['profitable_trades']} gagnants)\n"
            perf += f"⏱️ Durée moy: {r['avg_hold_days']:.0f} jours\n"
            perf += f"🤖 Validés: {r['validated_buys']}B / {r['validated_sells']}S\n"
            perf += f"❌ Rejetés: {r['rejected_buys']}B / {r['rejected_sells']}S\n"
            perf += f"📊 vs Hold: {r['strategy_vs_hold']:+.2f}%\n"
            perf += f"⭐ Score: **{r['strategy_score']:.0f}/100**"
            
            embed.add_field(name=f"#{i}", value=perf, inline=True)
            
            if i % 2 == 0:
                embed.add_field(name="\u200b", value="\u200b", inline=False)
        
        embed.set_footer(text="🤖 Chaque décision validée par IA avec actualités du jour")
        
        await message.edit(embed=embed)
        
    except Exception as e:
        logger.error(f"Erreur backtest: {e}")
        await message.edit(content=f"❌ Erreur: {str(e)}")

@bot.command(name='detail')
async def detail(ctx, symbol: str, months: int = 6):
    """
    Backtest détaillé d'une action avec tous les trades
    Exemple: !detail AAPL 6
    """
    symbol = symbol.upper()
    
    if months < 1 or months > 24:
        await ctx.send("❌ Période invalide. Utilisez entre 1 et 24 mois.")
        return
    
    embed = discord.Embed(
        title=f"⏳ Analyse détaillée {symbol}...",
        description=f"Backtest sur {months} mois avec validation IA",
        color=0xffff00
    )
    message = await ctx.send(embed=embed)
    
    try:
        result = await bot.backtest_engine.backtest_with_news_validation(symbol, months)
        
        if not result:
            embed = discord.Embed(
                title=f"❌ Erreur - {symbol}",
                description="Impossible de récupérer les données",
                color=0xff0000
            )
            await message.edit(embed=embed)
            return
        
        # Embed principal
        embed = discord.Embed(
            title=f"📊 Backtest Détaillé - {symbol}",
            description=f"Période: {result['period']} | Score: **{result['strategy_score']:.0f}/100**",
            color=0x00ff00 if result['total_profit'] > 0 else 0xff0000
        )
        
        # Stats générales
        embed.add_field(name="💰 Profit Total", value=f"**{result['total_profit']:+.2f}%**", inline=True)
        embed.add_field(name="📈 Win Rate", value=f"{result['win_rate']:.0f}%", inline=True)
        embed.add_field(name="💼 Trades", value=f"{result['total_trades']}", inline=True)
        
        embed.add_field(name="📊 Profit Moyen", value=f"{result['avg_profit']:+.2f}%", inline=True)
        embed.add_field(name="🎯 Max Profit", value=f"{result['max_profit']:+.2f}%", inline=True)
        embed.add_field(name="⚠️ Max Loss", value=f"{result['max_loss']:+.2f}%", inline=True)
        
        embed.add_field(name="⏱️ Durée Moy", value=f"{result['avg_hold_days']:.0f}j", inline=True)
        embed.add_field(name="📅 Points décision", value=f"{result['decision_points']}", inline=True)
        embed.add_field(name="🏦 Buy & Hold", value=f"{result['buy_hold_return']:+.2f}%", inline=True)
        
        # Validation IA
        validation_text = f"🤖 **Validation IA:**\n"
        validation_text += f"✅ Achats validés: {result['validated_buys']}\n"
        validation_text += f"❌ Achats rejetés: {result['rejected_buys']}\n"
        validation_text += f"✅ Ventes validées: {result['validated_sells']}\n"
        validation_text += f"❌ Ventes rejetées: {result['rejected_sells']}"
        embed.add_field(name="🤖 Décisions IA", value=validation_text, inline=False)
        
        await message.edit(embed=embed)
        
        # Envoyer les trades détaillés si disponibles
        if result['trades']:
            trades_embed = discord.Embed(
                title=f"💼 Détail des Trades - {symbol}",
                color=0x00ffff
            )
            
            for i, trade in enumerate(result['trades'][:10], 1):  # Max 10 trades
                profit_emoji = "🟢" if trade['profit'] > 0 else "🔴"
                trade_text = f"{profit_emoji} **{trade['profit']:+.2f}%**\n"
                trade_text += f"📅 Entrée: {trade['entry_date'].strftime('%Y-%m-%d')}\n"
                trade_text += f"💰 Prix: ${trade['entry_price']:.2f}\n"
                trade_text += f"📅 Sortie: {trade['exit_date'].strftime('%Y-%m-%d')}\n"
                trade_text += f"💰 Prix: ${trade['exit_price']:.2f}\n"
                trade_text += f"⏱️ Durée: {trade['hold_days']} jours"
                
                trades_embed.add_field(
                    name=f"Trade #{i}",
                    value=trade_text,
                    inline=True
                )
                
                if i % 2 == 0:
                    trades_embed.add_field(name="\u200b", value="\u200b", inline=False)
            
            if len(result['trades']) > 10:
                trades_embed.set_footer(text=f"... et {len(result['trades'])-10} autres trades")
            
            await ctx.send(embed=trades_embed)
        
    except Exception as e:
        logger.error(f"Erreur detail: {e}")
        await message.edit(content=f"❌ Erreur: {str(e)}")

@bot.command(name='aide')
async def aide(ctx):
    """Affiche l'aide"""
    embed = discord.Embed(
        title="📚 Guide des Commandes",
        description="Bot de Trading avec Backtest Réaliste et Validation IA",
        color=0x00ffff
    )
    
    embed.add_field(
        name="⏱️ **!backtest [mois]**",
        value="Backtest quotidien avec validation IA\n"
              "Analyse CHAQUE JOUR de trading (~20 jours/mois)\n"
              "Le bot prend des décisions quotidiennes\n"
              "L'IA les valide avec les actualités du jour\n"
              "Exemple: `!backtest 6` (analyse ~120 jours)",
        inline=False
    )
    
    embed.add_field(
        name="📊 **!detail [SYMBOL] [mois]**",
        value="Backtest détaillé d'une action avec tous les trades\n"
              "Exemple: `!detail AAPL 6`",
        inline=False
    )
    
    embed.add_field(
        name="🤖 **Comment ça marche?**",
        value="1️⃣ Le bot analyse CHAQUE JOUR la technique (RSI, MACD, etc.)\n"
              "2️⃣ Le bot décide: BUY, SELL ou HOLD\n"
              "3️⃣ Si BUY/SELL: l'IA récupère les actualités du jour\n"
              "4️⃣ L'IA donne un score 0-100 à la décision\n"
              "5️⃣ Si score > 70, le trade est exécuté ✅\n"
              "6️⃣ Sinon, le trade est rejeté ❌",
        inline=False
    )
    
    embed.add_field(
        name="💡 **Avantages**",
        value="✅ Simulation temps réel (analyse quotidienne)\n"
              "✅ Actualités historiques pour chaque jour\n"
              "✅ Validation IA de chaque décision BUY/SELL\n"
              "✅ Évite les faux signaux techniques\n"
              "✅ Compare avec Buy & Hold\n"
              "✅ Cache intelligent pour optimiser les API",
        inline=False
    )
    
    embed.set_footer(text="🔥 Backtest ultra-réaliste : analyse quotidienne avec validation IA")
    
    await ctx.send(embed=embed)


if __name__ == "__main__":
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("❌ Token Discord manquant!")
        print("\n⚠️ Configuration requise dans le fichier .env:")
        print("1. DISCORD_BOT_TOKEN=votre_token (obligatoire)")
        print("2. HUGGINGFACE_TOKEN=votre_token (pour validation IA)")
        print("3. FINNHUB_KEY=votre_cle (pour actualités)")
        print("4. NEWSAPI_KEY=votre_cle (fallback actualités)")
    else:
        logger.info("🚀 Démarrage du bot...")
        bot.run(token)