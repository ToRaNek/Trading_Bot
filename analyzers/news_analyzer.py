"""Analyseur de news historiques avec validation IA via HuggingFace"""

import aiohttp
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import Dict, List, Tuple, Optional
import os
import sys
import logging
from .ai_scorer import AIScorer

# Importer le rotateur de clés API
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_key_rotator import APIKeyRotator

logger = logging.getLogger('TradingBot')


class HistoricalNewsAnalyzer:
    """Récupère les actualités historiques pour chaque date de backtest"""

    def __init__(self, api_keys_csv_path: str = None):
        self.session = None
        # Initialiser le système de rotation des clés NewsAPI
        self.newsapi_rotator = APIKeyRotator(csv_path=api_keys_csv_path)
        self.finnhub_key = os.getenv('FINNHUB_KEY', '')
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
        self.news_cache = {}  # Cache pour éviter appels API répétés
        self.ai_scorer = AIScorer(self.hf_token)  # Scorer pour Reddit et News

        # Log des stats du rotateur
        stats = self.newsapi_rotator.get_stats()
        logger.info(f"[NewsAPI] Rotateur initialisé: {stats['active_keys']}/{stats['total_keys']} clés actives")

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

            # Collecter les news des deux sources
            all_news_items = []
            all_sentiments = []

            # 1. Essayer Finnhub
            if self.finnhub_key:
                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': symbol,
                    'from': from_date.strftime('%Y-%m-%d'),
                    'to': to_date.strftime('%Y-%m-%d'),
                    'token': self.finnhub_key
                }

                try:
                    async with session.get(url, params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, list) and len(data) > 0:
                                logger.info(f"[News] {symbol}: Finnhub → {len(data)} articles bruts")
                                has_news, news_items, score = await self._parse_finnhub_news(data, target_date)
                                all_news_items.extend(news_items)
                                logger.info(f"[News] {symbol}: Finnhub → {len(news_items)} news gardées après filtre")
                            else:
                                logger.warning(f"[News] {symbol}: Finnhub OK mais 0 news")
                        else:
                            logger.warning(f"[News] {symbol}: Finnhub status {response.status}")
                except Exception as e:
                    logger.warning(f"[News] {symbol}: Finnhub erreur {e}")

            # 2. Essayer NewsAPI avec rotation des clés (seulement si date < 30 jours)
            days_ago = (datetime.now() - target_date).days

            if days_ago <= 30:
                newsapi_key = self.newsapi_rotator.get_current_key()
                max_retries = self.newsapi_rotator.get_stats()['total_keys']

                for retry in range(max_retries):
                    if not newsapi_key:
                        logger.debug(f"[News] {symbol}: Aucune clé NewsAPI disponible")
                        break

                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': f"{search_term} OR {symbol}",
                        'from': from_date.strftime('%Y-%m-%d'),
                        'to': to_date.strftime('%Y-%m-%d'),
                        'sortBy': 'publishedAt',
                        'language': 'en',
                        'apiKey': newsapi_key,
                        'pageSize': 100
                    }

                    try:
                        async with session.get(url, params=params, timeout=8) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'articles' in data and data['articles']:
                                    self.newsapi_rotator.mark_current_as_success()
                                    has_news, news_items, score = await self._parse_newsapi_news(data['articles'], target_date)
                                    all_news_items.extend(news_items)
                                    logger.info(f"[News] {symbol}: NewsAPI → {len(news_items)} news (clé {self.newsapi_rotator.current_index + 1})")
                                    break
                                else:
                                    logger.warning(f"[News] {symbol}: NewsAPI OK mais 0 news")
                                    break
                            elif response.status == 429:
                                logger.warning(f"[News] {symbol}: NewsAPI clé {self.newsapi_rotator.current_index + 1} limite atteinte (429)")
                                self.newsapi_rotator.mark_current_as_failed()
                                newsapi_key = self.newsapi_rotator.get_current_key()
                                if retry < max_retries - 1:
                                    logger.info(f"[News] {symbol}: Rotation vers clé {retry + 2}/{max_retries}")
                                    continue
                            else:
                                logger.warning(f"[News] {symbol}: NewsAPI status {response.status}")
                                break
                    except Exception as e:
                        logger.warning(f"[News] {symbol}: NewsAPI erreur {e}")
                        break
            else:
                logger.debug(f"[News] {symbol}: NewsAPI skip (date trop ancienne: {days_ago} jours)")

            # Combiner tous les résultats
            if all_news_items:
                # Recalculer le score basé sur tous les articles
                total_sentiments = []
                for item in all_news_items:
                    if 'sentiment' in item:
                        importance = item.get('importance', 1.0)
                        weighted_sentiment = item['sentiment'] * importance
                        total_sentiments.append(weighted_sentiment)

                if total_sentiments:
                    avg_sentiment = np.mean(total_sentiments)
                    final_score = ((avg_sentiment + 3) / 6) * 100
                    final_score = max(0, min(100, final_score))
                else:
                    final_score = 50.0

                result = (True, all_news_items, final_score)
                logger.info(f"[News] {symbol}: ✓ TOTAL {len(all_news_items)} news, score={final_score:.0f}/100")
                self.news_cache[cache_key] = result
                return result

            # Aucune news trouvée
            stats = self.newsapi_rotator.get_stats()
            logger.info(f"[News] {symbol} @ {target_date.strftime('%Y-%m-%d')}: ⚠️ Aucune news trouvée (Finnhub={'✓' if self.finnhub_key else '✗'}, NewsAPI={stats['active_keys']}/{stats['total_keys']} clés)")
            result = (False, [], 0.0)
            self.news_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"[News] Erreur news historiques {symbol} @ {target_date}: {e}")
            import traceback
            traceback.print_exc()
            return False, [], 0.0

    async def _parse_finnhub_news(self, data: List, target_date: datetime) -> Tuple[bool, List[Dict], float]:
        """Parse les actualités Finnhub avec analyse de sentiment"""
        news_items = []

        if hasattr(target_date, 'tz') and target_date.tz is not None:
            target_date = target_date.replace(tzinfo=None)

        cutoff_time = target_date - timedelta(hours=48)

        # Keywords pour pondération
        positive_keywords = {
            'earnings beat': 3.0, 'beats': 2.5, 'profit': 2.0, 'surge': 2.5,
            'breakthrough': 2.5, 'record': 2.0, 'growth': 1.5, 'upgrade': 2.0,
            'partnership': 1.5, 'acquisition': 2.0, 'approval': 2.0,
            'bullish': 2.0, 'gains': 1.5, 'jumps': 2.0, 'soars': 2.5
        }

        negative_keywords = {
            'earnings miss': 3.0, 'misses': 2.5, 'loss': 2.0, 'plunge': 2.5,
            'crash': 3.0, 'downgrade': 2.0, 'lawsuit': 2.0, 'investigation': 2.0,
            'recall': 2.0, 'bankruptcy': 3.0, 'bearish': 2.0, 'falls': 1.5,
            'drops': 1.5, 'slides': 1.5, 'tumbles': 2.0, 'slumps': 2.0
        }

        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in data:
            try:
                title = article.get('headline', '')
                summary = article.get('summary', '')

                if not title or len(title) < 10:
                    continue

                timestamp = article.get('datetime', 0)
                pub_date = datetime.fromtimestamp(timestamp)
                if hasattr(pub_date, 'tz') and pub_date.tz is not None:
                    pub_date = pub_date.replace(tzinfo=None)

                if pub_date < cutoff_time or pub_date > target_date:
                    continue

                # Analyse de sentiment avec TextBlob
                text = f"{title} {summary}"
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 (négatif) à +1 (positif)

                # Détecter les keywords importants
                text_lower = text.lower()
                importance = 1.0

                for keyword, weight in positive_keywords.items():
                    if keyword in text_lower:
                        importance += weight

                for keyword, weight in negative_keywords.items():
                    if keyword in text_lower:
                        importance += weight

                # Pondérer le sentiment par l'importance
                weighted_sentiment = polarity * importance
                sentiments.append(weighted_sentiment)

                # Compter les sentiments
                if polarity > 0.1:
                    positive_count += 1
                elif polarity < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1

                news_items.append({
                    'title': title,
                    'publisher': article.get('source', 'Finnhub'),
                    'date': pub_date,
                    'importance': importance,
                    'sentiment': polarity,
                    'summary': summary[:200]
                })

            except Exception:
                continue

        has_news = len(news_items) > 0

        # Calculer le score basé sur le sentiment (AMPLIFIE pour être plus tranché)
        if sentiments and has_news:
            avg_sentiment = np.mean(sentiments)

            # NOUVEAU: Amplifier le sentiment pour être plus tranché
            # Multiplier par 3 pour avoir des scores plus extrêmes
            amplified_sentiment = avg_sentiment * 3.0

            # Convertir en score 0-100 avec amplification
            score = ((amplified_sentiment + 3) / 6) * 100
            score = max(0, min(100, score))

            logger.debug(f"[News] Sentiment: avg={avg_sentiment:.2f}, amplified={amplified_sentiment:.2f}, score={score:.0f}, pos={positive_count}, neg={negative_count}, neu={neutral_count}")
        else:
            score = 0.0

        return has_news, news_items, score

    async def _parse_newsapi_news(self, articles: List, target_date: datetime) -> Tuple[bool, List[Dict], float]:
        """Parse les actualités NewsAPI avec analyse de sentiment"""
        news_items = []

        if hasattr(target_date, 'tz') and target_date.tz is not None:
            target_date = target_date.replace(tzinfo=None)

        cutoff_time = target_date - timedelta(hours=48)

        # Keywords pour pondération
        positive_keywords = {
            'earnings beat': 3.0, 'beats': 2.5, 'profit': 2.0, 'surge': 2.5,
            'breakthrough': 2.5, 'record': 2.0, 'growth': 1.5, 'upgrade': 2.0,
            'partnership': 1.5, 'acquisition': 2.0, 'approval': 2.0,
            'bullish': 2.0, 'gains': 1.5, 'jumps': 2.0, 'soars': 2.5
        }

        negative_keywords = {
            'earnings miss': 3.0, 'misses': 2.5, 'loss': 2.0, 'plunge': 2.5,
            'crash': 3.0, 'downgrade': 2.0, 'lawsuit': 2.0, 'investigation': 2.0,
            'recall': 2.0, 'bankruptcy': 3.0, 'bearish': 2.0, 'falls': 1.5,
            'drops': 1.5, 'slides': 1.5, 'tumbles': 2.0, 'slumps': 2.0
        }

        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')

                if not title or len(title) < 10:
                    continue

                pub_str = article.get('publishedAt', '')
                if not pub_str:
                    continue

                pub_date = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
                if hasattr(pub_date, 'tz') and pub_date.tz is not None:
                    pub_date = pub_date.replace(tzinfo=None)

                # Pas de filtrage supplémentaire, NewsAPI a déjà filtré par date
                # (pour éviter le problème de la fenêtre 48h)

                # Analyse de sentiment avec TextBlob
                text = f"{title} {description or content}"
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 (négatif) à +1 (positif)

                # Détecter les keywords importants
                text_lower = text.lower()
                importance = 1.0

                for keyword, weight in positive_keywords.items():
                    if keyword in text_lower:
                        importance += weight

                for keyword, weight in negative_keywords.items():
                    if keyword in text_lower:
                        importance += weight

                # Pondérer le sentiment par l'importance
                weighted_sentiment = polarity * importance
                sentiments.append(weighted_sentiment)

                # Compter les sentiments
                if polarity > 0.1:
                    positive_count += 1
                elif polarity < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1

                news_items.append({
                    'title': title,
                    'publisher': article.get('source', {}).get('name', 'Unknown'),
                    'date': pub_date,
                    'importance': importance,
                    'sentiment': polarity,
                    'summary': (description or content or '')[:200]
                })

            except Exception:
                continue

        has_news = len(news_items) > 0

        # Calculer le score basé sur le sentiment (AMPLIFIE pour être plus tranché)
        if sentiments and has_news:
            avg_sentiment = np.mean(sentiments)

            # NOUVEAU: Amplifier le sentiment pour être plus tranché
            # Multiplier par 3 pour avoir des scores plus extrêmes
            amplified_sentiment = avg_sentiment * 3.0

            # Convertir en score 0-100 avec amplification
            score = ((amplified_sentiment + 3) / 6) * 100
            score = max(0, min(100, score))

            logger.debug(f"[News] Sentiment: avg={avg_sentiment:.2f}, amplified={amplified_sentiment:.2f}, score={score:.0f}, pos={positive_count}, neg={negative_count}, neu={neutral_count}")
        else:
            score = 0.0

        return has_news, news_items, score

    async def ask_ai_decision(self, symbol: str, bot_decision: str, news_data: List[Dict],
                             current_price: float, tech_confidence: float, reddit_posts: List[Dict] = None,
                             target_date: datetime = None, last_5_prices: List[float] = None,
                             buy_price: float = None, sell_price: float = None) -> Tuple[int, str, float, float]:
        """
        Demande à l'IA HuggingFace de valider la décision du bot

        NOUVEAU SYSTÈME:
        1. Calcule d'abord les scores Reddit et News via AIScorer
        2. Si pas de Reddit → score réduit (personne n'en parle)
        3. Si pas de News → score réduit
        4. Si aucun des 2 → score final = 0
        5. Utilise les 5 derniers prix pour contexte de tendance
        6. Inclut buy_price ou sell_price selon la décision

        Retourne: (SCORE FINAL, explication, reddit_score_ai, news_score_ai)
        """
        try:
            if not self.hf_token:
                return int(tech_confidence), "Token HuggingFace manquant - utilisation tech seul", 0.0, 0.0

            # ÉTAPE 1: Calculer les scores Reddit et News via AIScorer
            reddit_score_ai = 0.0
            news_score_ai = 0.0

            # Calculer le score textblob pour fallback
            reddit_score_textblob = 0.0
            if reddit_posts and len(reddit_posts) > 0:
                # Calculer sentiment textblob
                from textblob import TextBlob
                sentiments = []
                for post in reddit_posts[:50]:
                    text = post.get('title', '') + ' ' + post.get('body', '')
                    if len(text) > 10:
                        blob = TextBlob(text)
                        sentiment = blob.sentiment.polarity
                        score = post.get('score', 1)
                        weight = min(score / 10, 3)
                        sentiments.append(sentiment * weight)

                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    reddit_score_textblob = (avg_sentiment + 1) * 50
                    reddit_score_textblob = max(0, min(100, reddit_score_textblob))

                # Essayer AI scorer avec fallback textblob
                reddit_score_ai = await self.ai_scorer.score_reddit_posts(symbol, reddit_posts, target_date, fallback_score=reddit_score_textblob)
            else:
                logger.debug(f"   [AI Decision] {symbol}: Pas de Reddit → score 0")

            if news_data and len(news_data) > 0:
                news_score_ai = await self.ai_scorer.score_news(symbol, news_data, target_date)
            else:
                logger.debug(f"   [AI Decision] {symbol}: Pas de News → score 0")

            # ÉTAPE 2: Calculer le score final avec moyenne pondérée intelligente
            # Au lieu de mettre 0, on fait une moyenne selon les données disponibles
            scores_available = []
            weights = []

            # Toujours inclure le score technique
            scores_available.append(tech_confidence)
            weights.append(0.4)  # 40% technique

            # Ajouter Reddit si disponible
            if reddit_score_ai > 0:
                scores_available.append(reddit_score_ai)
                weights.append(0.3)  # 30% Reddit

            # Ajouter News si disponible
            if news_score_ai > 0:
                scores_available.append(news_score_ai)
                weights.append(0.3)  # 30% News

            # Normaliser les poids
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # Calculer le score final pondéré
            final_score = sum(s * w for s, w in zip(scores_available, normalized_weights))

            # Si on n'a NI Reddit NI News, pénaliser un peu le score
            if reddit_score_ai == 0 and news_score_ai == 0:
                final_score = final_score * 0.7  # Réduction de 30%
                reason = f"Tech seul ({tech_confidence:.0f}) - Pas de Reddit/News → réduit à {final_score:.0f}"
                logger.debug(f"   [AI Decision] {symbol}: {reason}")
                return int(final_score), reason, 0.0, 0.0

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

                                return score, reason_part[:200], reddit_score_ai, news_score_ai
                        except Exception as parse_err:
                            logger.error(f"   [AI Decision] Erreur parsing: {parse_err}")
                            pass

            # Fallback: analyse simple basée sur le sentiment
            return await self._simple_sentiment_score(news_data, bot_decision, reddit_posts, tech_confidence, reddit_score_ai, news_score_ai)

        except Exception as e:
            logger.error(f"Erreur AI validation: {e}")
            return await self._simple_sentiment_score(news_data, bot_decision, reddit_posts, tech_confidence, reddit_score_ai, news_score_ai)

    async def _simple_sentiment_score(self, news_data: List[Dict], bot_decision: str, reddit_posts: List[Dict] = None, tech_confidence: float = 50, reddit_score_ai: float = 0.0, news_score_ai: float = 0.0) -> Tuple[int, str, float, float]:
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
            return int(score), reason, reddit_score_ai, news_score_ai

        except:
            return int(tech_confidence), "Fallback: confiance technique seule", 0.0, 0.0

    async def close(self):
        if self.session:
            await self.session.close()
