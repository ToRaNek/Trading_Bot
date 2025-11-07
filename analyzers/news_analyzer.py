"""Analyseur de news historiques avec validation IA via HuggingFace"""

import aiohttp
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import Dict, List, Tuple, Optional
import os
import logging
from .ai_scorer import AIScorer

logger = logging.getLogger('TradingBot')


class HistoricalNewsAnalyzer:
    """Récupère les actualités historiques pour chaque date de backtest"""

    def __init__(self):
        self.session = None
        self.newsapi_key = os.getenv('NEWSAPI_KEY', '')
        self.finnhub_key = os.getenv('FINNHUB_KEY', '')
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
        self.news_cache = {}  # Cache pour éviter appels API répétés
        self.ai_scorer = AIScorer(self.hf_token)  # Scorer pour Reddit et News

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

            except Exception:
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

                pub_str = article.get('publishedAt', '')
                if not pub_str:
                    continue

                pub_date = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
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
                    'publisher': article.get('source', {}).get('name', 'Unknown'),
                    'date': pub_date,
                    'importance': importance,
                    'keywords': matched_keywords,
                    'summary': article.get('description', '')[:200] if article.get('description') else ''
                })

            except Exception:
                continue

        has_news = len(news_items) > 0
        total_importance = sum(n['importance'] for n in news_items)
        news_score = min(100, total_importance * 10) if has_news else 0.0

        return has_news, news_items, news_score

    async def ask_ai_decision(self, symbol: str, bot_decision: str, news_data: List[Dict],
                             current_price: float, tech_confidence: float, reddit_posts: List[Dict] = None,
                             target_date: datetime = None, last_5_prices: List[float] = None,
                             buy_price: float = None, sell_price: float = None) -> Tuple[int, str]:
        """
        Demande à l'IA HuggingFace de valider la décision du bot

        NOUVEAU SYSTÈME:
        1. Calcule d'abord les scores Reddit et News via AIScorer
        2. Si pas de Reddit → score réduit (personne n'en parle)
        3. Si pas de News → score réduit
        4. Si aucun des 2 → score final = 0
        5. Utilise les 5 derniers prix pour contexte de tendance
        6. Inclut buy_price ou sell_price selon la décision

        Retourne: SCORE FINAL (0-100) et explication
        """
        try:
            if not self.hf_token:
                return int(tech_confidence), "Token HuggingFace manquant - utilisation tech seul"

            # ÉTAPE 1: Calculer les scores Reddit et News via AIScorer
            reddit_score_ai = 0.0
            news_score_ai = 0.0

            if reddit_posts and len(reddit_posts) > 0:
                reddit_score_ai = await self.ai_scorer.score_reddit_posts(symbol, reddit_posts, target_date)
            else:
                logger.info(f"   [AI Decision] {symbol}: Pas de Reddit → score 0 (personne n'en parle)")

            if news_data and len(news_data) > 0:
                news_score_ai = await self.ai_scorer.score_news(symbol, news_data, target_date)
            else:
                logger.info(f"   [AI Decision] {symbol}: Pas de News → score 0 (pas d'actualité)")

            # ÉTAPE 2: Si aucune donnée, score final = 0
            if reddit_score_ai == 0 and news_score_ai == 0:
                logger.warning(f"   [AI Decision] {symbol}: Aucune donnée Reddit/News → SCORE FINAL = 0")
                return 0, "Aucune donnée disponible (ni Reddit ni News)"

            # ÉTAPE 3: Construire le contexte des 5 derniers prix
            price_context = ""
            if last_5_prices and len(last_5_prices) >= 5:
                price_context = f"\n📈 LAST 5 PRICES (trend context):\n"
                for i, price in enumerate(last_5_prices[-5:], 1):
                    price_context += f"   {i}. ${price:.2f}\n"

                # Calculer la tendance
                price_change = (last_5_prices[-1] - last_5_prices[-5]) / last_5_prices[-5] * 100
                trend = "📈 UPTREND" if price_change > 2 else "📉 DOWNTREND" if price_change < -2 else "➡️ SIDEWAYS"
                price_context += f"   Trend: {trend} ({price_change:+.2f}%)\n"

            # ÉTAPE 4: Construire le prompt optimisé
            price_info = f"💰 Current Price: ${current_price:.2f}\n"
            if buy_price:
                price_info += f"🎯 Target Buy Price: ${buy_price:.2f}\n"
            elif sell_price:
                price_info += f"🎯 Target Sell Price: ${sell_price:.2f}\n"

            prompt = f"""TRADING DECISION VALIDATION - OPTIMIZED SYSTEM

🎯 STOCK: {symbol}
{price_info}{price_context}

🤖 TECHNICAL DECISION: {bot_decision}
📊 Technical Confidence: {tech_confidence:.0f}/100

📊 PRE-CALCULATED SCORES (by AI Scorer):
💬 Reddit Community Score: {reddit_score_ai:.0f}/100 ({len(reddit_posts) if reddit_posts else 0} posts analyzed)
📰 News Sentiment Score: {news_score_ai:.0f}/100 ({len(news_data) if news_data else 0} news analyzed)

⚠️  IMPORTANT RULES:
- If no Reddit data (score 0) → Reduce final score (nobody talking about it)
- If no News data (score 0) → Reduce final score (no market events)
- If BOTH are 0 → FINAL SCORE = 0 (no information available)

TASK: Provide the FINAL TRADING SCORE (0-100)

Combine:
1. Technical Confidence: {tech_confidence:.0f}/100
2. Reddit Score: {reddit_score_ai:.0f}/100
3. News Score: {news_score_ai:.0f}/100
4. Price trend context
5. Decision type ({bot_decision})

FINAL SCORE SCALE:
- 0-30: Bad decision or insufficient data
- 31-50: Weak decision, mixed signals
- 51-70: Good decision, moderately supported
- 71-100: Excellent decision, strongly supported

Respond EXACTLY: "SCORE: [number]|REASON: [short explanation]"
"""

            session = await self.get_session()
            url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "return_full_text": False
                }
            }

            async with session.post(url, json=payload, headers=headers, timeout=20) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('generated_text', '')
                        try:
                            if 'SCORE:' in text and 'REASON:' in text:
                                parts = text.split('|')
                                score_part = parts[0].split('SCORE:')[1].strip()
                                reason_part = parts[1].split('REASON:')[1].strip() if len(parts) > 1 else "Analyse IA"

                                score = int(''.join(filter(str.isdigit, score_part))[:3])
                                score = max(0, min(100, score))

                                # Réduction supplémentaire si pas de données
                                if reddit_score_ai == 0 and news_score_ai > 0:
                                    score = int(score * 0.7)  # Réduction 30% si pas de Reddit
                                    reason_part += " (réduit: pas de Reddit)"
                                elif news_score_ai == 0 and reddit_score_ai > 0:
                                    score = int(score * 0.7)  # Réduction 30% si pas de News
                                    reason_part += " (réduit: pas de News)"

                                return score, reason_part[:200]
                        except Exception as parse_err:
                            logger.error(f"   [AI Decision] Erreur parsing: {parse_err}")
                            pass

            # Fallback: analyse simple basée sur le sentiment
            return await self._simple_sentiment_score(news_data, bot_decision, reddit_posts, tech_confidence)

        except Exception as e:
            logger.error(f"Erreur AI validation: {e}")
            return await self._simple_sentiment_score(news_data, bot_decision, reddit_posts, tech_confidence)

    async def _simple_sentiment_score(self, news_data: List[Dict], bot_decision: str, reddit_posts: List[Dict] = None, tech_confidence: float = 50) -> Tuple[int, str]:
        """Score de sentiment simple comme fallback, basé sur tech_confidence + ajustement news/reddit"""
        try:
            sentiments = []

            # Sentiment des news
            for article in news_data[:5] if news_data else []:
                blob = TextBlob(article['title'])
                sentiment = blob.sentiment.polarity
                sentiments.append(sentiment * article['importance'])

            # Sentiment des posts Reddit
            if reddit_posts:
                reddit_sentiments = []
                for post in reddit_posts[:10]:
                    text = post.get('title', '') + ' ' + post.get('body', '')
                    if len(text) > 10:
                        blob = TextBlob(text)
                        sentiment = blob.sentiment.polarity
                        upvotes = post.get('upvotes', 0)
                        weight = min(upvotes / 10, 3)
                        reddit_sentiments.append(sentiment * weight)

                if reddit_sentiments:
                    reddit_sentiment = np.mean(reddit_sentiments)
                    sentiments.append(reddit_sentiment * 2)

            avg_sentiment = np.mean(sentiments) if sentiments else 0

            # Partir de la confiance technique et ajuster selon sentiment
            score = tech_confidence

            # Ajustement selon la décision et le sentiment
            if bot_decision == "BUY":
                if avg_sentiment > 0.3:
                    score = min(100, score + 15)
                    reason = f"Tech {tech_confidence:.0f} + Sentiment très positif → BOOST"
                elif avg_sentiment > 0.1:
                    score = min(100, score + 8)
                    reason = f"Tech {tech_confidence:.0f} + Sentiment positif → boost"
                elif avg_sentiment < -0.2:
                    score = max(0, score - 15)
                    reason = f"Tech {tech_confidence:.0f} + Sentiment négatif → PÉNALITÉ"
                else:
                    reason = f"Tech {tech_confidence:.0f} + Sentiment neutre"

            else:  # SELL
                if avg_sentiment < -0.3:
                    score = min(100, score + 15)
                    reason = f"Tech {tech_confidence:.0f} + Sentiment très négatif → BOOST SELL"
                elif avg_sentiment < -0.1:
                    score = min(100, score + 8)
                    reason = f"Tech {tech_confidence:.0f} + Sentiment négatif → boost SELL"
                elif avg_sentiment > 0.2:
                    score = max(0, score - 15)
                    reason = f"Tech {tech_confidence:.0f} + Sentiment positif contredit SELL → PÉNALITÉ"
                else:
                    reason = f"Tech {tech_confidence:.0f} + Sentiment neutre"

            score = max(0, min(100, score))
            return int(score), reason

        except:
            return int(tech_confidence), "Fallback: confiance technique seule"

    async def close(self):
        if self.session:
            await self.session.close()
