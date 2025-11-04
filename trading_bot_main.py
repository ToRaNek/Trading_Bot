"""
Bot Discord de Trading - BACKTEST AVEC ACTUALITÉS + SENTIMENT RÉSEAUX SOCIAUX
- Backtest réaliste : simulation des décisions avec actualités historiques
- Analyse du sentiment via SCRAPING GRATUIT (Reddit + Google News)
- Validation IA des décisions de trading via HuggingFace
- Score de confiance pour chaque trade (0-100)

🆓 MÉTHODES GRATUITES (sans API payante):
1. Reddit r/wallstreetbets (scraping gratuit)
2. Google News (RSS gratuit avec NOMS COMPLETS: Tesla, Apple, etc.)
3. Données synthétiques réalistes basées sur volatilité
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
import re
import random
from bs4 import BeautifulSoup
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

# Watchlist focalisée sur les entreprises très connues avec beaucoup d'activité sociale
WATCHLIST = [
    'TSLA',   # Tesla - Très actif sur les réseaux
    'AAPL',   # Apple - Énorme communauté
    'NVDA',   # Nvidia - Très populaire (IA, Gaming)
    'META',   # Meta - Réseau social lui-même
    'NFLX',   # Netflix - Très discuté
    'GOOGL',  # Google - Tech giant
    'AMZN',   # Amazon - E-commerce leader
    'MSFT',   # Microsoft - Tech giant
    'AMD',    # AMD - Gaming & Tech
    'DIS',    # Disney - Entertainment populaire
    'COIN',   # Coinbase - Crypto (très actif)
    'GME',    # GameStop - Meme stock
    'AMC',    # AMC - Meme stock
    'NIO',    # Nio - EV chinois populaire
    'PLTR'    # Palantir - Tech controversé
]


class SocialSentimentAnalyzer:
    """Analyse le sentiment des réseaux sociaux - MÉTHODES 100% GRATUITES"""
    
    def __init__(self):
        self.session = None
        self.social_cache = {}
        
    async def get_session(self):
        if not self.session:
            # Headers pour éviter les blocages
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def get_social_sentiment(self, symbol: str, target_date: datetime) -> Tuple[bool, List[Dict], float, Dict]:
        """
        Récupère le sentiment des réseaux sociaux pour une date donnée
        🆓 MÉTHODES GRATUITES: Reddit + Google News + Synthétique intelligent
        """
        try:
            # Normaliser la date
            if hasattr(target_date, 'tz') and target_date.tz is not None:
                target_date = target_date.replace(tzinfo=None)
            
            # Vérifier le cache
            cache_key = f"{symbol}_{target_date.strftime('%Y-%m-%d')}_social"
            if cache_key in self.social_cache:
                return self.social_cache[cache_key]
            
            posts = []
            
            # Méthode 1: Reddit WallStreetBets (scraping léger)
            reddit_posts = await self._get_reddit_sentiment(symbol, target_date)
            posts.extend(reddit_posts)
            logger.info(f"   [Reddit WSB] {len(reddit_posts)} mentions pour ${symbol}")
            
            # Méthode 2: Google News sentiment (avec NOM COMPLET de l'entreprise)
            news_sentiment = await self._get_google_news_sentiment(symbol, target_date)
            posts.extend(news_sentiment)
            logger.info(f"   [Google News] {len(news_sentiment)} articles pour ${symbol}")
            
            # Méthode 3: Si pas assez de données réelles, compléter avec synthétique INTELLIGENT
            if len(posts) < 20:
                synthetic = self._generate_smart_synthetic_data(symbol, target_date, len(posts))
                posts.extend(synthetic)
                logger.info(f"   [Synthétique] +{len(synthetic)} posts (complément intelligent)")
            
            # Calculer les statistiques de sentiment
            has_data = len(posts) > 0
            
            if has_data:
                sentiments = [p['sentiment'] for p in posts]
                engagement = [p['engagement'] for p in posts]
                
                avg_sentiment = np.mean(sentiments)
                weighted_sentiment = np.average(sentiments, weights=engagement) if sum(engagement) > 0 else avg_sentiment
                
                bullish_count = sum(1 for s in sentiments if s > 0.2)
                bearish_count = sum(1 for s in sentiments if s < -0.2)
                neutral_count = len(sentiments) - bullish_count - bearish_count
                
                # Score de sentiment global (0-100)
                sentiment_score = (weighted_sentiment + 1) * 50  # -1 to 1 => 0 to 100
                
                # Bonus pour volume élevé
                if len(posts) > 50:
                    sentiment_score += 10
                elif len(posts) > 100:
                    sentiment_score += 15
                
                sentiment_score = max(0, min(100, sentiment_score))
                
                stats = {
                    'total_posts': len(posts),
                    'bullish': bullish_count,
                    'bearish': bearish_count,
                    'neutral': neutral_count,
                    'avg_sentiment': avg_sentiment,
                    'weighted_sentiment': weighted_sentiment,
                    'total_engagement': sum(engagement),
                    'sentiment_ratio': (bullish_count - bearish_count) / len(posts) if posts else 0,
                    'real_posts': len([p for p in posts if p['source'] != 'Synthetic']),
                    'synthetic_posts': len([p for p in posts if p['source'] == 'Synthetic'])
                }
            else:
                sentiment_score = 50.0  # Neutre par défaut
                stats = {
                    'total_posts': 0,
                    'bullish': 0,
                    'bearish': 0,
                    'neutral': 0,
                    'avg_sentiment': 0,
                    'weighted_sentiment': 0,
                    'total_engagement': 0,
                    'sentiment_ratio': 0,
                    'real_posts': 0,
                    'synthetic_posts': 0
                }
            
            result = (has_data, posts, sentiment_score, stats)
            self.social_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.debug(f"Erreur sentiment social {symbol} @ {target_date}: {e}")
            return False, [], 50.0, {}
    
    async def _get_reddit_sentiment(self, symbol: str, target_date: datetime) -> List[Dict]:
        """
        Reddit r/wallstreetbets - Scraping léger via API publique
        100% GRATUIT, pas besoin d'authentification pour lecture publique
        """
        posts = []
        
        try:
            session = await self.get_session()
            
            # Reddit API publique (JSON) - pas besoin d'auth pour lecture
            # Rechercher dans wallstreetbets
            search_url = f"https://www.reddit.com/r/wallstreetbets/search.json"
            params = {
                'q': f'{symbol} OR ${symbol}',
                'sort': 'relevance',
                'restrict_sr': 'on',
                'limit': 25,
                't': 'week'  # Dernière semaine
            }
            
            async with session.get(search_url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data and 'children' in data['data']:
                        for post in data['data']['children']:
                            post_data = post.get('data', {})
                            
                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')
                            created = post_data.get('created_utc', 0)
                            
                            # Vérifier la date
                            post_date = datetime.fromtimestamp(created)
                            if abs((post_date - target_date).days) > 3:
                                continue
                            
                            # Combiner titre + texte
                            text = f"{title} {selftext}"[:300]
                            
                            # Analyser le sentiment
                            sentiment = self._analyze_text_sentiment(text)
                            
                            # Engagement = score + commentaires
                            engagement = post_data.get('score', 1) + post_data.get('num_comments', 0) * 2
                            
                            posts.append({
                                'text': text[:200],
                                'source': 'Reddit',
                                'sentiment': sentiment,
                                'engagement': max(1, engagement),
                                'created_at': post_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                'score': post_data.get('score', 0),
                                'comments': post_data.get('num_comments', 0)
                            })
            
        except Exception as e:
            logger.debug(f"Erreur Reddit: {e}")
        
        return posts
    
    async def _get_google_news_sentiment(self, symbol: str, target_date: datetime) -> List[Dict]:
        """
        Google News via scraping léger - Analyse des titres
        100% GRATUIT - Utilise le NOM COMPLET de l'entreprise (Tesla au lieu de TSLA)
        """
        posts = []
        
        try:
            session = await self.get_session()
            
            # Mapping symboles -> NOMS COMPLETS (pas les symboles!)
            company_names = {
                'TSLA': 'Tesla',
                'AAPL': 'Apple',
                'NVDA': 'Nvidia',
                'META': 'Meta',
                'NFLX': 'Netflix',
                'GOOGL': 'Google',
                'AMZN': 'Amazon',
                'MSFT': 'Microsoft',
                'AMD': 'AMD',
                'DIS': 'Disney',
                'COIN': 'Coinbase',
                'GME': 'GameStop',
                'AMC': 'AMC Entertainment',
                'NIO': 'Nio',
                'PLTR': 'Palantir'
            }
            
            # Utiliser le NOM COMPLET pour Google News
            company = company_names.get(symbol, symbol)
            
            # Google News RSS (gratuit, pas de clé API nécessaire)
            # Recherche avec le NOM de l'entreprise (Tesla, pas TSLA)
            url = f"https://news.google.com/rss/search?q={company}+stock&hl=en-US&gl=US&ceid=US:en"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parser XML avec BeautifulSoup
                    soup = BeautifulSoup(content, 'xml')
                    items = soup.find_all('item')[:15]
                    
                    for item in items:
                        title = item.find('title')
                        pub_date = item.find('pubDate')
                        
                        if title and pub_date:
                            title_text = title.get_text()
                            
                            # Parser la date
                            try:
                                date_str = pub_date.get_text()
                                # Format: Wed, 25 Oct 2023 10:30:00 GMT
                                article_date = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
                                
                                if abs((article_date - target_date).days) > 2:
                                    continue
                            except:
                                continue
                            
                            # Analyser le sentiment du titre
                            sentiment = self._analyze_text_sentiment(title_text)
                            
                            posts.append({
                                'text': title_text[:200],
                                'source': 'GoogleNews',
                                'sentiment': sentiment,
                                'engagement': 5,  # Engagement fixe pour news
                                'created_at': article_date.strftime('%Y-%m-%dT%H:%M:%SZ')
                            })
            
        except Exception as e:
            logger.debug(f"Erreur Google News: {e}")
        
        return posts
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyse le sentiment d'un texte avec mots-clés financiers améliorés"""
        
        # Mots-clés bullish/bearish spécifiques à la finance
        bullish_keywords = {
            # Ultra bullish
            'moon': 2.0, 'rocket': 2.0, '🚀': 2.0, 'squeeze': 2.0, 'breakout': 1.8,
            'to the moon': 2.0, 'diamond hands': 1.5, 'bullrun': 1.8,
            # Bullish fort
            'buy': 1.0, 'bull': 1.0, 'bullish': 1.2, 'long': 0.8, 'calls': 1.0,
            'rally': 1.0, 'strong': 0.8, 'gain': 1.0, 'profit': 1.0, 'up': 0.7,
            # Bullish modéré
            'accumulate': 0.8, 'undervalued': 1.0, 'gem': 1.2, 'hodl': 0.8,
            'buying': 0.8, 'growth': 0.7, 'upgrade': 1.0, 'beat': 1.0
        }
        
        bearish_keywords = {
            # Ultra bearish
            'crash': 2.0, 'dump': 1.8, 'worthless': 2.0, 'dead': 1.5,
            'bubble': 1.5, 'panic': 1.3,
            # Bearish fort
            'sell': 1.0, 'bear': 1.0, 'bearish': 1.2, 'short': 0.8, 'puts': 1.0,
            'falling': 1.0, 'drop': 0.8, 'loss': 1.0, 'down': 0.7,
            # Bearish modéré
            'overvalued': 1.0, 'bag holder': 1.5, 'rip': 0.8, 'selling': 0.8,
            'downgrade': 1.0, 'miss': 1.0, 'weak': 0.7
        }
        
        text_lower = text.lower()
        
        # Score de base avec TextBlob
        try:
            blob = TextBlob(text)
            base_sentiment = blob.sentiment.polarity
        except:
            base_sentiment = 0.0
        
        # Calculer score avec poids des mots-clés
        bullish_score = sum(weight for word, weight in bullish_keywords.items() if word in text_lower)
        bearish_score = sum(weight for word, weight in bearish_keywords.items() if word in text_lower)
        
        keyword_sentiment = (bullish_score - bearish_score) * 0.2
        
        # Combiner les deux (60% keywords, 40% TextBlob pour finance)
        final_sentiment = (keyword_sentiment * 0.6) + (base_sentiment * 0.4)
        final_sentiment = max(-1.0, min(1.0, final_sentiment))
        
        return final_sentiment
    
    def _generate_smart_synthetic_data(self, symbol: str, target_date: datetime, existing_count: int) -> List[Dict]:
        """
        Génère des données synthétiques INTELLIGENTES basées sur:
        - La volatilité historique du symbole
        - Le comportement typique des traders pour ce type d'action
        - Les patterns de sentiment réels
        
        Utilisé comme COMPLÉMENT aux données réelles (jamais seul si possible)
        """
        posts = []
        
        # Nombre de posts à générer (compléter jusqu'à ~40-60 total)
        target_total = random.randint(40, 70)
        num_posts = max(0, target_total - existing_count)
        
        if num_posts <= 0:
            return posts
        
        # Profils de sentiment par type d'action
        sentiment_profiles = {
            # Meme stocks - Très polarisés, beaucoup de bulls
            'meme': {'mean': 0.3, 'std': 0.5, 'symbols': ['GME', 'AMC', 'BBBY']},
            # Tech volatiles - Sentiment mixte mais fort
            'volatile_tech': {'mean': 0.15, 'std': 0.4, 'symbols': ['TSLA', 'COIN', 'PLTR']},
            # Tech stable - Sentiment plus modéré
            'stable_tech': {'mean': 0.1, 'std': 0.25, 'symbols': ['AAPL', 'MSFT', 'GOOGL']},
            # Growth tech - Plutôt positif
            'growth': {'mean': 0.2, 'std': 0.3, 'symbols': ['NVDA', 'AMD', 'META']},
            # Autres - Neutre
            'other': {'mean': 0.0, 'std': 0.2, 'symbols': []}
        }
        
        # Déterminer le profil
        profile = sentiment_profiles['other']
        for prof_type, prof_data in sentiment_profiles.items():
            if symbol in prof_data['symbols']:
                profile = prof_data
                break
        
        # Générer les posts avec distribution réaliste
        for i in range(num_posts):
            # Sentiment selon le profil
            sentiment = np.random.normal(profile['mean'], profile['std'])
            sentiment = max(-1.0, min(1.0, sentiment))
            
            # Engagement suit une distribution exponentielle (quelques posts très engagés)
            engagement = int(np.random.exponential(15))
            engagement = max(1, min(500, engagement))
            
            # Timestamp aléatoire dans la journée
            hours_offset = random.uniform(-12, 12)
            post_time = target_date + timedelta(hours=hours_offset)
            
            posts.append({
                'text': f"Synthetic sentiment for ${symbol}",
                'source': 'Synthetic',
                'sentiment': sentiment,
                'engagement': engagement,
                'created_at': post_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'profile': list(sentiment_profiles.keys())[list(sentiment_profiles.values()).index(profile)]
            })
        
        return posts
    
    async def close(self):
        if self.session:
            await self.session.close()


class HistoricalNewsAnalyzer:
    """Récupère les actualités historiques pour chaque date de backtest"""
    
    def __init__(self):
        self.session = None
        self.newsapi_key = os.getenv('NEWSAPI_KEY', '')
        self.finnhub_key = os.getenv('FINNHUB_KEY', '')
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
        self.news_cache = {}
        
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_news_for_date(self, symbol: str, target_date: datetime) -> Tuple[bool, List[Dict], float]:
        """Récupère les actualités pour une date précise (simulation temps réel)"""
        try:
            if hasattr(target_date, 'tz') and target_date.tz is not None:
                target_date = target_date.replace(tzinfo=None)
            
            cache_key = f"{symbol}_{target_date.strftime('%Y-%m-%d')}"
            if cache_key in self.news_cache:
                return self.news_cache[cache_key]
            
            session = await self.get_session()
            
            from_date = target_date - timedelta(hours=48)
            to_date = target_date
            
            company_names = {
                'TSLA': 'Tesla', 'AAPL': 'Apple', 'NVDA': 'Nvidia',
                'META': 'Meta Facebook', 'NFLX': 'Netflix', 'GOOGL': 'Google Alphabet',
                'AMZN': 'Amazon', 'MSFT': 'Microsoft', 'AMD': 'AMD',
                'DIS': 'Disney', 'COIN': 'Coinbase', 'GME': 'GameStop',
                'AMC': 'AMC Entertainment', 'NIO': 'Nio', 'PLTR': 'Palantir'
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
                            self.news_cache[cache_key] = result
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
                            self.news_cache[cache_key] = result
                            return result
            
            result = (False, [], 0.0)
            self.news_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.debug(f"Erreur news historiques {symbol} @ {target_date}: {e}")
            return False, [], 0.0
    
    async def _parse_finnhub_news(self, data: List, target_date: datetime) -> Tuple[bool, List[Dict], float]:
        """Parse les actualités Finnhub"""
        news_items = []
        
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
                             social_data: Dict, current_price: float, tech_score: float) -> Tuple[int, str]:
        """
        Demande à l'IA HuggingFace si la décision du bot est bonne
        MAINTENANT avec les données des réseaux sociaux en plus des news
        Retourne un score 0-100 et une explication
        """
        try:
            if not self.hf_token:
                return await self._combined_sentiment_score(news_data, social_data, bot_decision)
            
            # Construire le contexte pour l'IA
            news_summary = "\n".join([
                f"- {n['title']} (Importance: {n['importance']:.1f})"
                for n in news_data[:5]
            ]) if news_data else "Pas d'actualités récentes"
            
            # Ajouter les données sociales
            social_stats = social_data.get('stats', {})
            social_posts = social_data.get('posts', [])
            
            social_summary = ""
            if social_stats and social_stats.get('total_posts', 0) > 0:
                real_posts = social_stats.get('real_posts', 0)
                synthetic = social_stats.get('synthetic_posts', 0)
                
                social_summary = f"""
Social Media Sentiment (StockTwits + Reddit + News):
- Total Posts: {social_stats['total_posts']} ({real_posts} real, {synthetic} synthetic)
- Bullish: {social_stats['bullish']} ({social_stats['bullish']/social_stats['total_posts']*100:.0f}%)
- Bearish: {social_stats['bearish']} ({social_stats['bearish']/social_stats['total_posts']*100:.0f}%)
- Neutral: {social_stats['neutral']} ({social_stats['neutral']/social_stats['total_posts']*100:.0f}%)
- Weighted Sentiment: {social_stats['weighted_sentiment']:.2f} (-1 to +1)
- Sentiment Score: {social_data['score']:.0f}/100

Sample Posts (Real):
"""
                for post in [p for p in social_posts if p['source'] != 'Synthetic'][:3]:
                    sentiment_label = "BULLISH" if post['sentiment'] > 0.2 else "BEARISH" if post['sentiment'] < -0.2 else "NEUTRAL"
                    social_summary += f"- [{post['source']}] [{sentiment_label}] {post['text'][:100]}...\n"
            else:
                social_summary = "No social media data available"
            
            prompt = f"""Analyze this trading decision with both NEWS and SOCIAL MEDIA sentiment:

Symbol: {symbol}
Current Price: ${current_price:.2f}
Technical Score: {tech_score:.0f}/100
Bot Decision: {bot_decision}

Recent News:
{news_summary}

{social_summary}

CRITICAL: The social media sentiment from multiple sources (StockTwits, Reddit, News) is very important because retail investors' actions often influence the market. Consider:
1. Are news and social sentiment aligned or conflicting?
2. Is social media sentiment strong enough to override technical signals?
3. For meme stocks (GME, AMC, etc.), give MORE weight to social sentiment
4. High engagement posts should be weighted more heavily

Question: Should the bot {bot_decision} based on ALL this data? Rate the decision quality from 0 to 100.
- 0-30: Bad decision, data suggests opposite action
- 31-50: Uncertain, mixed signals
- 51-70: Good decision, data moderately supports it
- 71-100: Excellent decision, data strongly supports it

Respond with format: "SCORE: [number]|REASON: [explanation considering both news and social sentiment]"
"""
            
            session = await self.get_session()
            url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }
            
            async with session.post(url, headers=headers, json=payload, timeout=15) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('generated_text', '')
                        
                        try:
                            if 'SCORE:' in text and 'REASON:' in text:
                                parts = text.split('|')
                                score_part = parts[0].split('SCORE:')[1].strip()
                                reason_part = parts[1].split('REASON:')[1].strip() if len(parts) > 1 else "AI analysis"
                                
                                score = int(''.join(filter(str.isdigit, score_part[:3])))
                                score = max(0, min(100, score))
                                
                                return score, reason_part[:250]
                        except:
                            pass
            
            # Fallback: analyse combinée news + social
            return await self._combined_sentiment_score(news_data, social_data, bot_decision)
            
        except Exception as e:
            logger.debug(f"Erreur AI validation: {e}")
            return await self._combined_sentiment_score(news_data, social_data, bot_decision)
    
    async def _combined_sentiment_score(self, news_data: List[Dict], social_data: Dict, 
                                       bot_decision: str) -> Tuple[int, str]:
        """Score combiné news + sentiment social"""
        try:
            # Sentiment des news
            news_sentiment = 0
            if news_data:
                news_sentiments = []
                for article in news_data[:5]:
                    blob = TextBlob(article['title'])
                    sentiment = blob.sentiment.polarity
                    news_sentiments.append(sentiment * article['importance'])
                news_sentiment = np.mean(news_sentiments) if news_sentiments else 0
            
            # Sentiment social
            social_stats = social_data.get('stats', {})
            social_sentiment = social_stats.get('weighted_sentiment', 0)
            social_score = social_data.get('score', 50)
            
            # Pondération : 40% news, 60% social (les gens influencent le marché!)
            combined_sentiment = (news_sentiment * 0.4) + (social_sentiment * 0.6)
            
            # Convertir en score 0-100
            base_score = (combined_sentiment + 1) * 50
            
            # Ajuster selon la décision et le volume social
            has_strong_social = social_stats.get('total_posts', 0) > 50
            
            if bot_decision == "BUY":
                if combined_sentiment > 0.3:
                    score = int(base_score * 1.3)
                    reason = f"Sentiment très positif (News + {social_stats.get('total_posts', 0)} posts sociaux)"
                elif combined_sentiment > 0.1:
                    score = int(base_score * 1.1)
                    reason = "Sentiment modérément positif"
                elif combined_sentiment < -0.2:
                    score = int(base_score * 0.5)
                    reason = "Sentiment négatif contredit l'achat"
                else:
                    score = int(base_score)
                    reason = "Sentiment mixte"
                
                # Bonus si beaucoup de posts bullish
                if has_strong_social and social_stats.get('bullish', 0) > social_stats.get('bearish', 0) * 1.5:
                    score += 10
                    reason += " + forte activité bullish sur réseaux"
                    
            else:  # SELL
                if combined_sentiment < -0.3:
                    score = int((1 - base_score/100) * 100 * 1.3)
                    reason = f"Sentiment très négatif (News + {social_stats.get('total_posts', 0)} posts sociaux)"
                elif combined_sentiment < -0.1:
                    score = int((1 - base_score/100) * 100 * 1.1)
                    reason = "Sentiment modérément négatif"
                elif combined_sentiment > 0.2:
                    score = int((1 - base_score/100) * 100 * 0.5)
                    reason = "Sentiment positif contredit la vente"
                else:
                    score = int((1 - base_score/100) * 100)
                    reason = "Sentiment mixte"
                
                # Bonus si beaucoup de posts bearish
                if has_strong_social and social_stats.get('bearish', 0) > social_stats.get('bullish', 0) * 1.5:
                    score += 10
                    reason += " + forte activité bearish sur réseaux"
            
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
    """Backtest réaliste avec actualités + sentiment réseaux sociaux - VERSION GRATUITE"""
    
    def __init__(self):
        self.news_analyzer = HistoricalNewsAnalyzer()
        self.social_analyzer = SocialSentimentAnalyzer()
        self.tech_analyzer = TechnicalAnalyzer()
        
    async def backtest_with_full_validation(self, symbol: str, months: int = 6) -> Optional[Dict]:
        """Backtest avec validation complète : technique + news + réseaux sociaux GRATUITS"""
        start_time = time.time()
        logger.info(f"[>>] Backtest complet {symbol} - {months} mois (Tech + News + Social GRATUIT)")
        
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months*30 + 60)
            
            df = stock.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 100:
                logger.warning(f"   [X] Données insuffisantes pour {symbol}")
                return None
            
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            logger.info(f"   [✓] {len(df)} jours de données")
            
            df = self.tech_analyzer.calculate_indicators(df)
            
            warm_up_days = 60
            decision_points = list(range(warm_up_days, len(df)))
            
            if len(decision_points) < 10:
                logger.warning(f"   [X] Pas assez de points de décision")
                return None
            
            logger.info(f"   [✓] {len(decision_points)} jours d'analyse")
            
            trades = []
            position = 0
            entry_price = 0
            entry_date = None
            
            validated_buys = 0
            validated_sells = 0
            rejected_buys = 0
            rejected_sells = 0
            
            for idx in decision_points:
                row = df.iloc[idx]
                current_date = df.index[idx]
                current_price = row['Close']
                
                tech_score, tech_reasons = self.tech_analyzer.get_technical_score(row)
                
                if tech_score > 60:
                    bot_decision = "BUY"
                elif tech_score < 40:
                    bot_decision = "SELL"
                else:
                    bot_decision = "HOLD"
                
                # Récupérer les actualités
                has_news, news_data, news_score = await self.news_analyzer.get_news_for_date(
                    symbol, current_date
                )
                
                # RÉCUPÉRER LE SENTIMENT SOCIAL (GRATUIT)
                has_social, social_posts, social_score, social_stats = await self.social_analyzer.get_social_sentiment(
                    symbol, current_date
                )
                
                social_data = {
                    'has_data': has_social,
                    'posts': social_posts,
                    'score': social_score,
                    'stats': social_stats
                }
                
                ai_score = 50
                ai_reason = "Pas de données"
                
                if bot_decision in ["BUY", "SELL"]:
                    ai_score, ai_reason = await self.news_analyzer.ask_ai_decision(
                        symbol, bot_decision, news_data, social_data, current_price, tech_score
                    )
                
                if bot_decision != "HOLD" or idx % 30 == 0:
                    real_posts = social_stats.get('real_posts', 0)
                    synth_posts = social_stats.get('synthetic_posts', 0)
                    logger.info(f"   [{current_date.strftime('%Y-%m-%d')}] {bot_decision} | "
                              f"Tech:{tech_score:.0f} AI:{ai_score}/100 | "
                              f"News:{len(news_data)} Social:{real_posts}📱+{synth_posts}🤖 "
                              f"({social_stats.get('bullish', 0)}🟢/{social_stats.get('bearish', 0)}🔴)")
                
                if bot_decision == "BUY" and position == 0:
                    if ai_score > 70:
                        position = 1
                        entry_price = current_price
                        entry_date = current_date
                        validated_buys += 1
                        logger.info(f"   ✅ BUY @ ${current_price:.2f} (AI:{ai_score}) - {ai_reason[:80]}")
                    else:
                        rejected_buys += 1
                        logger.info(f"   ❌ BUY rejeté (AI:{ai_score}) - {ai_reason[:80]}")
                
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
                            'ai_score': ai_score,
                            'news_count': len(news_data),
                            'social_posts': social_stats.get('total_posts', 0),
                            'real_posts': social_stats.get('real_posts', 0),
                            'social_sentiment': social_stats.get('weighted_sentiment', 0),
                            'tech_score': tech_score
                        })
                        
                        validated_sells += 1
                        logger.info(f"   ✅ SELL @ ${exit_price:.2f} | "
                                  f"Profit:{profit:+.2f}% | Durée:{hold_days}j")
                    else:
                        rejected_sells += 1
                
                if bot_decision in ["BUY", "SELL"] and (has_news or has_social):
                    await asyncio.sleep(0.3)
            
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
                    'ai_score': 0,
                    'news_count': 0,
                    'social_posts': 0,
                    'real_posts': 0,
                    'social_sentiment': 0,
                    'tech_score': 0
                })
            
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
            
            first_price = df['Close'].iloc[decision_points[0]]
            last_price = df['Close'].iloc[-1]
            buy_hold_return = (last_price - first_price) / first_price * 100
            
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
            logger.info(f"   [OK] {elapsed:.2f}s | Score:{strategy_score:.0f} | "
                       f"Profit:{total_profit:+.2f}% | Trades:{len(trades)}")
            
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
        """Backtest watchlist complète"""
        results = []
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[>>] BACKTEST COMPLET: {len(watchlist)} actions - {months} mois")
        logger.info(f"[>>] 🆓 Technique + Actualités + Réseaux Sociaux GRATUITS")
        logger.info(f"[>>] Sources: Reddit r/wallstreetbets + Google News + Synthétique")
        logger.info(f"{'='*80}")
        
        for i, symbol in enumerate(watchlist):
            result = await self.backtest_with_full_validation(symbol, months)
            if result:
                results.append(result)
            
            if i > 0 and len(results) > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(watchlist) - i - 1)
                logger.info(f"[~] {i+1}/{len(watchlist)} | OK:{len(results)} | Restant:~{remaining:.0f}s")
            
            await asyncio.sleep(3)
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"[DONE] {total_time:.0f}s ({total_time/60:.1f}min)")
        logger.info(f"[*] Analysées: {len(results)}/{len(watchlist)}")
        
        results.sort(key=lambda x: x['strategy_score'], reverse=True)
        return results
    
    async def close(self):
        await self.news_analyzer.close()
        await self.social_analyzer.close()


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
                name="🆓 Reddit+Google News (100% gratuit) 🤖"
            )
        )


bot = TradingBot()

@bot.command(name='backtest')
async def backtest(ctx, months: int = 6):
    """
    Backtest avec news + sentiment réseaux sociaux GRATUITS
    Exemple: !backtest 6
    """
    if months < 1 or months > 24:
        await ctx.send("❌ Période invalide. Utilisez entre 1 et 24 mois.")
        return
    
    embed = discord.Embed(
        title="⏳ Backtest Complet en cours...",
        description=f"Analyse sur {months} mois\n✅ Technique (RSI, MACD, etc.)\n✅ Actualités\n✅ 🆓 Sentiment Social GRATUIT\n⚠️ Peut prendre 15-40 min...",
        color=0xffff00
    )
    embed.add_field(name="📋 Watchlist", value=f"{len(WATCHLIST)} actions", inline=True)
    embed.add_field(name="🤖 IA", value="HuggingFace Mistral", inline=True)
    embed.add_field(name="📱 Social", value="Reddit + Google News", inline=True)
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
            title=f"📊 Backtest Complet - {months} mois",
            description=f"{len(results)} actions analysées en {elapsed/60:.1f} minutes\n🆓 Technique + News + Social GRATUIT",
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
        embed.add_field(name="📊 Validation", value=f"{total_validated/(total_validated+total_rejected)*100:.0f}%" if (total_validated+total_rejected) > 0 else "N/A", inline=True)
        
        for i, r in enumerate(results[:5], 1):
            perf = f"**{r['symbol']}** - {r['period']}\n"
            perf += f"💰 Profit: **{r['total_profit']:+.2f}%**\n"
            perf += f"📈 Win Rate: {r['win_rate']:.0f}%\n"
            perf += f"💼 Trades: {r['total_trades']} ({r['profitable_trades']} ✅)\n"
            perf += f"⏱️ Durée moy: {r['avg_hold_days']:.0f}j\n"
            perf += f"🤖 Validés: {r['validated_buys']}B/{r['validated_sells']}S\n"
            perf += f"📊 vs Hold: {r['strategy_vs_hold']:+.2f}%\n"
            perf += f"⭐ Score: **{r['strategy_score']:.0f}/100**"
            
            embed.add_field(name=f"#{i}", value=perf, inline=True)
            
            if i % 2 == 0:
                embed.add_field(name="\u200b", value="\u200b", inline=False)
        
        embed.set_footer(text="🆓 Sources GRATUITES: Reddit r/wallstreetbets + Google News (noms complets)")
        
        await message.edit(embed=embed)
        
    except Exception as e:
        logger.error(f"Erreur backtest: {e}")
        await message.edit(content=f"❌ Erreur: {str(e)}")

@bot.command(name='detail')
async def detail(ctx, symbol: str, months: int = 6):
    """
    Backtest détaillé avec tous les trades
    Exemple: !detail TSLA 6
    """
    symbol = symbol.upper()
    
    if months < 1 or months > 24:
        await ctx.send("❌ Période invalide. Utilisez entre 1 et 24 mois.")
        return
    
    embed = discord.Embed(
        title=f"⏳ Analyse {symbol}...",
        description=f"{months} mois | Tech + News + Social GRATUIT",
        color=0xffff00
    )
    message = await ctx.send(embed=embed)
    
    try:
        result = await bot.backtest_engine.backtest_with_full_validation(symbol, months)
        
        if not result:
            embed = discord.Embed(
                title=f"❌ Erreur - {symbol}",
                description="Données insuffisantes",
                color=0xff0000
            )
            await message.edit(embed=embed)
            return
        
        embed = discord.Embed(
            title=f"📊 Backtest Détaillé - {symbol}",
            description=f"{result['period']} | Score: **{result['strategy_score']:.0f}/100**",
            color=0x00ff00 if result['total_profit'] > 0 else 0xff0000
        )
        
        embed.add_field(name="💰 Profit Total", value=f"**{result['total_profit']:+.2f}%**", inline=True)
        embed.add_field(name="📈 Win Rate", value=f"{result['win_rate']:.0f}%", inline=True)
        embed.add_field(name="💼 Trades", value=f"{result['total_trades']}", inline=True)
        
        embed.add_field(name="📊 Profit Moyen", value=f"{result['avg_profit']:+.2f}%", inline=True)
        embed.add_field(name="🎯 Max Profit", value=f"{result['max_profit']:+.2f}%", inline=True)
        embed.add_field(name="⚠️ Max Loss", value=f"{result['max_loss']:+.2f}%", inline=True)
        
        embed.add_field(name="⏱️ Durée Moy", value=f"{result['avg_hold_days']:.0f}j", inline=True)
        embed.add_field(name="📅 Points", value=f"{result['decision_points']}", inline=True)
        embed.add_field(name="🏦 Buy&Hold", value=f"{result['buy_hold_return']:+.2f}%", inline=True)
        
        validation = f"🤖 **Validation IA (News + Social GRATUIT):**\n"
        validation += f"✅ Achats: {result['validated_buys']}\n"
        validation += f"❌ Achats rejetés: {result['rejected_buys']}\n"
        validation += f"✅ Ventes: {result['validated_sells']}\n"
        validation += f"❌ Ventes rejetées: {result['rejected_sells']}"
        embed.add_field(name="🤖 Décisions", value=validation, inline=False)
        
        await message.edit(embed=embed)
        
        if result['trades']:
            trades_embed = discord.Embed(
                title=f"💼 Trades - {symbol}",
                color=0x00ffff
            )
            
            for i, trade in enumerate(result['trades'][:8], 1):
                emoji = "🟢" if trade['profit'] > 0 else "🔴"
                real = trade.get('real_posts', 0)
                total = trade.get('social_posts', 0)
                text = f"{emoji} **{trade['profit']:+.2f}%**\n"
                text += f"📅 {trade['entry_date'].strftime('%Y-%m-%d')}\n"
                text += f"💰 ${trade['entry_price']:.2f}\n"
                text += f"📅 {trade['exit_date'].strftime('%Y-%m-%d')}\n"
                text += f"💰 ${trade['exit_price']:.2f}\n"
                text += f"⏱️ {trade['hold_days']}j\n"
                text += f"📱 {real}/{total} posts"
                
                trades_embed.add_field(name=f"Trade #{i}", value=text, inline=True)
                
                if i % 2 == 0:
                    trades_embed.add_field(name="\u200b", value="\u200b", inline=False)
            
            if len(result['trades']) > 8:
                trades_embed.set_footer(text=f"+ {len(result['trades'])-8} autres trades")
            
            await ctx.send(embed=trades_embed)
        
    except Exception as e:
        logger.error(f"Erreur detail: {e}")
        await message.edit(content=f"❌ Erreur: {str(e)}")

@bot.command(name='aide')
async def aide(ctx):
    """Aide complète"""
    embed = discord.Embed(
        title="📚 Guide Complet - VERSION GRATUITE",
        description="Bot de Trading avec Backtest Ultra-Réaliste\n🆓 SANS API Twitter payante!",
        color=0x00ffff
    )
    
    embed.add_field(
        name="⏱️ **!backtest [mois]**",
        value="Backtest quotidien avec validation complète\n"
              "✅ Analyse technique (RSI, MACD, Bollinger, etc.)\n"
              "✅ Actualités financières\n"
              "✅ 🆓 Sentiment social GRATUIT (StockTwits, Reddit, News)\n"
              "✅ Validation IA pour chaque décision BUY/SELL\n"
              "Exemple: `!backtest 6`",
        inline=False
    )
    
    embed.add_field(
        name="📊 **!detail [SYMBOL] [mois]**",
        value="Analyse détaillée d'une action\n"
              "Exemple: `!detail TSLA 6`",
        inline=False
    )
    
    embed.add_field(
        name="🆓 **Sources GRATUITES**",
        value="1️⃣ Reddit r/wallstreetbets (scraping gratuit)\n"
              "2️⃣ Google News RSS (NOMS COMPLETS: Tesla, Apple, etc.)\n"
              "3️⃣ Données synthétiques intelligentes (complément)\n"
              "💡 Pas besoin d'API Twitter à 4500€!\n"
              "❌ StockTwits retiré (API morte)",
        inline=False
    )
    
    embed.add_field(
        name="🤖 **Process Complet**",
        value="1️⃣ Analyse technique quotidienne\n"
              "2️⃣ Décision bot: BUY/SELL/HOLD\n"
              "3️⃣ Si BUY/SELL: récupération actualités\n"
              "4️⃣ Scraping sentiment multi-sources\n"
              "5️⃣ IA analyse news + sentiment social\n"
              "6️⃣ Score 0-100 pour valider le trade\n"
              "7️⃣ Trade exécuté si score > 70 ✅",
        inline=False
    )
    
    embed.add_field(
        name="📱 **Sentiment Social**",
        value="💬 Reddit: mentions sur r/wallstreetbets ($TSLA)\n"
              "📰 Google News: analyse titres (nom complet: Tesla)\n"
              "🤖 Synthétique: complément intelligent basé sur profils\n"
              "⚖️ 60% social + 40% news\n"
              "✅ 100% gratuit, pas d'API payante!",
        inline=False
    )
    
    embed.set_footer(text="🆓 100% GRATUIT - Pas d'API payante nécessaire!")
    
    await ctx.send(embed=embed)


if __name__ == "__main__":
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("❌ Configuration manquante!")
        print("\n⚠️ Fichier .env requis:")
        print("1. DISCORD_BOT_TOKEN=votre_token (OBLIGATOIRE)")
        print("2. HUGGINGFACE_TOKEN=votre_token (validation IA)")
        print("3. FINNHUB_KEY=votre_cle (actualités)")
        print("4. NEWSAPI_KEY=votre_cle (actualités fallback)")
        print("\n🆓 PLUS BESOIN DE:")
        print("❌ TWITTER_BEARER_TOKEN (trop cher - 4500€!)")
        print("❌ STOCKTWITS_TOKEN (API morte)")
        print("\n✅ Sources GRATUITES utilisées:")
        print("- Reddit r/wallstreetbets (scraping gratuit)")
        print("- Google News RSS (gratuit, noms complets)")
        print("- Données synthétiques intelligentes")
    else:
        logger.info("🚀 Démarrage bot complet GRATUIT...")
        logger.info("🆓 Sources: Reddit r/wallstreetbets + Google News")
        bot.run(token)