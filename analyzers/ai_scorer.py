"""
AI Scorer - Utilise Hugging Face pour scorer les données Reddit et News
avant la décision finale de trading
"""

import logging
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import aiohttp

logger = logging.getLogger('TradingBot')


class AIScorer:
    """
    Classe pour scorer les données Reddit et News via Hugging Face
    Génère des scores intermédiaires qui seront utilisés dans la décision finale
    """

    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_TOKEN')
        self.api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"
        self.session = None

        # Cache des scores pour éviter les requêtes répétées
        self.reddit_scores_cache = {}
        self.news_scores_cache = {}

    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def score_reddit_posts(self, symbol: str, posts: List[Dict], target_date: datetime = None) -> float:
        """
        Demande à Hugging Face de scorer les posts Reddit

        Args:
            symbol: Ticker de l'action
            posts: Liste des posts Reddit (titre, body, upvotes, downvotes)
            target_date: Date cible pour le cache

        Returns:
            float: Score entre 0-100 (0 = très négatif, 50 = neutre, 100 = très positif)
        """
        if not posts:
            logger.info(f"   [AI Scorer] Reddit {symbol}: 0 posts → score 0")
            return 0.0

        # Vérifier le cache
        if target_date:
            cache_key = f"{symbol}_{target_date.strftime('%Y-%m-%d')}"
            if cache_key in self.reddit_scores_cache:
                logger.debug(f"   [AI Scorer] Reddit {symbol}: Utilisation cache")
                return self.reddit_scores_cache[cache_key]

        try:
            # Limiter à 10 posts max pour ne pas surcharger le prompt
            top_posts = sorted(posts[:50], key=lambda x: x.get('upvotes', 0), reverse=True)[:10]

            # Construire le résumé des posts
            posts_summary = []
            for i, post in enumerate(top_posts, 1):
                title = post.get('title', '')[:100]  # Limiter la longueur
                body = post.get('body', '')[:200]
                upvotes = post.get('upvotes', 0)
                downvotes = post.get('downvotes', 0)

                posts_summary.append(
                    f"{i}. \"{title}\"\n"
                    f"   Contenu: {body}\n"
                    f"   Votes: {upvotes} ⬆️ / {downvotes} ⬇️"
                )

            posts_text = "\n\n".join(posts_summary)

            # Prompt pour Hugging Face
            prompt = f"""Analyse le sentiment des discussions Reddit sur {symbol}.

POSTS REDDIT ({len(top_posts)} posts les plus votés):
{posts_text}

TOTAL: {len(posts)} posts analysés

Ta mission: Donne un score de sentiment entre 0-100
- 0-30: Sentiment très négatif (vendre, problèmes, peur)
- 30-50: Sentiment négatif à neutre
- 50-70: Sentiment neutre à positif
- 70-100: Sentiment très positif (acheter, optimisme, confiance)

Prends en compte:
1. Le ton général des posts (positif/négatif)
2. Le nombre d'upvotes (plus = plus important)
3. Le volume de discussions (beaucoup de posts = important)

Réponds UNIQUEMENT avec un nombre entre 0-100, sans explication.
Score:"""

            # Appeler Hugging Face
            session = await self.get_session()
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 10,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }

            async with session.post(self.api_url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extraire le score
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('generated_text', '').strip()

                        # Extraire le nombre
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)', text)
                        if match:
                            score = float(match.group(1))
                            score = max(0, min(100, score))  # Limiter entre 0-100

                            logger.info(f"   [AI Scorer] Reddit {symbol}: Score {score}/100 ({len(posts)} posts)")

                            # Mettre en cache
                            if target_date:
                                self.reddit_scores_cache[cache_key] = score

                            return score
                        else:
                            logger.warning(f"   [AI Scorer] Reddit {symbol}: Impossible d'extraire le score")
                            return 50.0  # Score neutre par défaut
                    else:
                        logger.warning(f"   [AI Scorer] Reddit {symbol}: Réponse HF invalide")
                        return 50.0
                else:
                    logger.error(f"   [AI Scorer] Reddit {symbol}: Erreur HF {response.status}")
                    return 50.0

        except Exception as e:
            logger.error(f"   [AI Scorer] Reddit {symbol}: Erreur {e}")
            return 50.0

    async def score_news(self, symbol: str, news_items: List[Dict], target_date: datetime = None) -> float:
        """
        Demande à Hugging Face de scorer les news

        Args:
            symbol: Ticker de l'action
            news_items: Liste des news (titre, description, sentiment)
            target_date: Date cible pour le cache

        Returns:
            float: Score entre 0-100
        """
        if not news_items:
            logger.info(f"   [AI Scorer] News {symbol}: 0 news → score 0")
            return 0.0

        # Vérifier le cache
        if target_date:
            cache_key = f"{symbol}_{target_date.strftime('%Y-%m-%d')}"
            if cache_key in self.news_scores_cache:
                logger.debug(f"   [AI Scorer] News {symbol}: Utilisation cache")
                return self.news_scores_cache[cache_key]

        try:
            # Limiter à 10 news max
            top_news = news_items[:10]

            # Construire le résumé des news
            news_summary = []
            for i, news in enumerate(top_news, 1):
                title = news.get('title', '')[:150]
                description = news.get('description', '')[:200]

                news_summary.append(
                    f"{i}. {title}\n"
                    f"   {description}"
                )

            news_text = "\n\n".join(news_summary)

            # Prompt pour Hugging Face
            prompt = f"""Analyse le sentiment des actualités sur {symbol}.

ACTUALITÉS ({len(top_news)} news récentes):
{news_text}

TOTAL: {len(news_items)} news analysées

Ta mission: Donne un score de sentiment entre 0-100
- 0-30: News très négatives (scandales, pertes, problèmes majeurs)
- 30-50: News négatives à neutres
- 50-70: News neutres à positives
- 70-100: News très positives (croissance, innovation, succès)

Prends en compte:
1. Le ton des titres et descriptions
2. L'importance des événements mentionnés
3. Le volume de news (beaucoup de news = événement important)

Réponds UNIQUEMENT avec un nombre entre 0-100, sans explication.
Score:"""

            # Appeler Hugging Face
            session = await self.get_session()
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 10,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }

            async with session.post(self.api_url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extraire le score
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('generated_text', '').strip()

                        # Extraire le nombre
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)', text)
                        if match:
                            score = float(match.group(1))
                            score = max(0, min(100, score))

                            logger.info(f"   [AI Scorer] News {symbol}: Score {score}/100 ({len(news_items)} news)")

                            # Mettre en cache
                            if target_date:
                                self.news_scores_cache[cache_key] = score

                            return score
                        else:
                            logger.warning(f"   [AI Scorer] News {symbol}: Impossible d'extraire le score")
                            return 50.0
                    else:
                        logger.warning(f"   [AI Scorer] News {symbol}: Réponse HF invalide")
                        return 50.0
                else:
                    logger.error(f"   [AI Scorer] News {symbol}: Erreur HF {response.status}")
                    return 50.0

        except Exception as e:
            logger.error(f"   [AI Scorer] News {symbol}: Erreur {e}")
            return 50.0

    async def close(self):
        if self.session:
            await self.session.close()
