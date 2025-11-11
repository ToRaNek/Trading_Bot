"""Analyseur de sentiment Reddit avec support CSV et APIs (Reddit + Pushshift)"""

import aiohttp
import asyncio
import numpy as np
import csv
import json
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger('TradingBot')


class RedditSentimentAnalyzer:
    """
    Analyse le sentiment des discussions Reddit pour chaque action
    Utilise l'API REST de Reddit (pas besoin de PRAW)
    """

    def __init__(self, csv_file: str = None, data_dir: str = 'data', reddit_client_id: str = None, reddit_client_secret: str = None):
        self.session = None
        self.reddit_token = None  # Token OAuth pour l'API Reddit
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.sentiment_cache = {}  # Cache pour éviter appels répétés
        self.csv_file = csv_file
        self.csv_data = None  # Données chargées depuis le CSV
        self.data_dir = data_dir  # Dossier contenant les CSV par action
        self.csv_data_by_symbol = {}  # Cache des CSV chargés par ticker

        # Configuration des subreddits par ticker
        self.ticker_subreddits = {
            'NVDA': ['NVDA_Stock', 'stocks'],
            'AAPL': ['AAPL', 'stocks'],
            'GOOGL': ['GOOG_Stock', 'stocks'],
            'GOOG': ['GOOG_Stock', 'stocks'],
            'AMZN': ['amzn', 'stocks'],
            'META': ['stocks'],  # Recherche sur r/stocks avec $meta
            'TSLA': ['TSLA', 'stocks'],
            'BRK-B': ['BerkshireHathaway', 'stocks'],
            'JPM': ['JPMorganChase', 'stocks'],
            'V': ['stocks'],  # Recherche avec $visa
            'JNJ': ['ValueInvesting', 'stocks'],
            'WMT': ['stocks'],
            'MSFT': ['stocks'],
            'MA': ['stocks'],
            'PG': ['stocks'],
            'DIS': ['stocks'],
            'NFLX': ['stocks'],
            'ADBE': ['stocks'],
            'CRM': ['stocks'],
            'AMD': ['AMD_Stock', 'stocks'],
            'ORCL': ['stocks'],
            'INTC': ['intel', 'stocks'],
            'CSCO': ['stocks'],
            'PEP': ['stocks'],
            'COST': ['stocks'],
            'AVGO': ['stocks']
        }

        # Tickers qui nécessitent une recherche spéciale
        self.special_search_tickers = {
            'META': 'meta',
            'V': 'visa',
            'JNJ': 'JNJ',
            'WMT': 'wmt',
            'BRK-B': 'berkshire'
        }

    async def get_reddit_oauth_token(self):
        """Obtient un token OAuth pour l'API Reddit (fonctionne depuis Azure/VPS)"""
        if not self.reddit_client_id or not self.reddit_client_secret:
            logger.warning("[Reddit] Pas de credentials OAuth - utilisation sans authentification")
            return None

        try:
            import aiohttp
            auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_client_secret)

            async with aiohttp.ClientSession() as temp_session:
                data = {
                    'grant_type': 'client_credentials'
                }
                headers = {
                    'User-Agent': 'TradingBot/1.0'
                }

                async with temp_session.post(
                    'https://www.reddit.com/api/v1/access_token',
                    auth=auth,
                    data=data,
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.reddit_token = token_data.get('access_token')
                        logger.info("[Reddit] ✅ Token OAuth obtenu avec succès")
                        return self.reddit_token
                    else:
                        logger.error(f"[Reddit] ❌ Erreur OAuth: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"[Reddit] ❌ Erreur obtention token OAuth: {e}")
            return None

    async def get_session(self):
        if not self.session or self.session.closed:
            # Si on a des credentials OAuth, obtenir un token
            if self.reddit_client_id and self.reddit_client_secret and not self.reddit_token:
                await self.get_reddit_oauth_token()

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

            # Ajouter le token OAuth si disponible
            if self.reddit_token:
                headers['Authorization'] = f'Bearer {self.reddit_token}'
                logger.info("[Reddit] Session avec authentification OAuth")

            self.session = aiohttp.ClientSession(headers=headers)
            logger.info("[Reddit] Nouvelle session HTTP créée")
        return self.session

    def load_csv_data(self, symbol: str = None):
        """Charge les données Reddit depuis le CSV

        Args:
            symbol: Si fourni, charge data/Sentiment_[SYMBOL].csv au lieu du csv_file global
        """
        try:
            import pandas as pd
            from pathlib import Path

            # Déterminer quel fichier charger
            if symbol:
                # Charger le fichier spécifique à l'action
                csv_path = Path(self.data_dir) / f"Sentiment_{symbol}.csv"

                # Vérifier si déjà en cache
                if symbol in self.csv_data_by_symbol:
                    logger.debug(f"[Reddit] ✅ Utilisation cache pour {symbol}")
                    return True

                if not csv_path.exists():
                    logger.warning(f"[Reddit] ⚠️ Fichier {csv_path} introuvable")
                    return False

                # Charger et mettre en cache
                data = pd.read_csv(csv_path)
                data['created'] = pd.to_datetime(data['created'])
                self.csv_data_by_symbol[symbol] = data
                logger.info(f"[Reddit] ✅ {len(data)} posts chargés pour {symbol} depuis {csv_path}")
                return True
            else:
                # Charger le fichier global (ancien comportement)
                if self.csv_file is None:
                    logger.warning("[Reddit] Pas de fichier CSV spécifié")
                    return False

                self.csv_data = pd.read_csv(self.csv_file)
                self.csv_data['created'] = pd.to_datetime(self.csv_data['created'])
                logger.info(f"[Reddit] ✅ {len(self.csv_data)} posts chargés depuis {self.csv_file}")
                return True

        except Exception as e:
            logger.error(f"[Reddit] ❌ Erreur chargement CSV: {e}")
            return False

    def get_posts_from_csv(self, symbol: str, target_date: datetime, lookback_hours: int = 48) -> List[Dict]:
        """Récupère les posts Reddit depuis le CSV pour une date donnée"""
        # Essayer de charger le CSV spécifique à l'action
        csv_data = None

        if symbol in self.csv_data_by_symbol:
            csv_data = self.csv_data_by_symbol[symbol]
        else:
            # Tenter de charger le fichier spécifique
            if not self.load_csv_data(symbol=symbol):
                # Fallback sur le CSV global si disponible
                if self.csv_data is None:
                    if not self.load_csv_data():
                        return []
                csv_data = self.csv_data
            else:
                csv_data = self.csv_data_by_symbol[symbol]

        try:
            # Normaliser la date
            if hasattr(target_date, 'tz') and target_date.tz is not None:
                target_date = target_date.replace(tzinfo=None)

            cutoff_time = target_date - timedelta(hours=lookback_hours)

            # Filtrer les posts par date
            mask = (csv_data['created'] >= cutoff_time) & (csv_data['created'] <= target_date)
            filtered_posts = csv_data[mask]

            # Convertir en liste de dictionnaires
            posts = []
            for _, row in filtered_posts.iterrows():
                posts.append({
                    'title': row.get('title', ''),
                    'body': row.get('body', ''),
                    'score': row.get('upvotes', 0) - row.get('downvotes', 0),  # Recalculer score
                    'upvotes': row.get('upvotes', 0),
                    'downvotes': row.get('downvotes', 0),
                    'created': row['created'],
                    'author': row.get('author', ''),
                    'source': row.get('source', '')
                })

            logger.debug(f"[Reddit CSV] {symbol}: {len(posts)} posts trouvés pour {target_date.strftime('%Y-%m-%d')}")
            return posts

        except Exception as e:
            logger.error(f"[Reddit CSV] Erreur filtrage: {e}")
            return []

    async def get_reddit_sentiment(self, symbol: str, target_date: datetime = None,
                                   lookback_hours: int = 168, save_csv: bool = False) -> Tuple[float, int, List[str], List[Dict]]:
        """
        Récupère et analyse le sentiment Reddit pour un ticker (OPTIMISÉ POUR LIVE TRADING)

        Args:
            symbol: Ticker de l'action
            target_date: Date cible (None = maintenant)
            lookback_hours: Fenêtre de temps (défaut: 168h = 7 jours)
            save_csv: Si True, sauvegarde tous les posts en CSV

        Returns:
            Tuple[sentiment_score (0-100), post_count, sample_posts, all_posts_details]
        """
        try:
            # Normaliser la date
            if target_date is None:
                target_date = datetime.now()
            if hasattr(target_date, 'tz') and target_date.tz is not None:
                target_date = target_date.replace(tzinfo=None)

            # Vérifier le cache
            cache_key = f"{symbol}_{target_date.strftime('%Y-%m-%d')}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]

            all_posts = []
            session = await self.get_session()

            # LIVE TRADING: Utiliser UNIQUEMENT l'API Reddit (7 derniers jours)
            # Pas de CSV, pas de Pushshift → rapide et récent
            logger.info(f"[Reddit] {symbol}: Récupération via API Reddit (7 derniers jours)")

            # Récupérer les subreddits configurés pour ce ticker
            subreddits = self.ticker_subreddits.get(symbol, ['stocks'])

            for subreddit in subreddits:
                # Utiliser API REST Reddit pour données récentes
                if subreddit == 'stocks' or symbol in self.special_search_tickers:
                    search_term = self.special_search_tickers.get(symbol, symbol)
                    posts = await self._search_reddit_comments(
                        session, subreddit, search_term, target_date, lookback_hours
                    )
                else:
                    posts = await self._get_subreddit_posts(
                        session, subreddit, target_date, lookback_hours
                    )

                all_posts.extend(posts)

                # Délai augmenté pour éviter le rate limiting Reddit (403)
                await asyncio.sleep(0.5)

            # Analyser le sentiment
            if not all_posts:
                result = (0.0, 0, [], [])  # Score 0 si pas de posts (personne n'en parle)
                self.sentiment_cache[cache_key] = result
                logger.info(f"[Reddit] {symbol}: ⚠️ Score 0/100 (aucun post récupéré)")
                return result

            sentiments = []
            sample_posts = []

            for post in all_posts[:50]:  # Limiter à 50 posts max
                text = post.get('title', '') + ' ' + post.get('body', '')
                if len(text) > 10:
                    blob = TextBlob(text)
                    sentiment = blob.sentiment.polarity  # -1 à +1

                    # Pondérer par score (upvotes)
                    score = post.get('score', 1)
                    weight = min(score / 10, 3)  # Max 3x weight
                    weighted_sentiment = sentiment * weight

                    sentiments.append(weighted_sentiment)

                    # Garder quelques exemples
                    if len(sample_posts) < 5:
                        emoji = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
                        sample_posts.append(f"{emoji} {text[:100]}... (sentiment: {sentiment:.2f})")

            # Calculer le score moyen
            avg_sentiment = np.mean(sentiments) if sentiments else 0

            # Convertir en score 0-100
            # -1 (très négatif) -> 0
            #  0 (neutre) -> 50
            # +1 (très positif) -> 100
            sentiment_score = (avg_sentiment + 1) * 50
            sentiment_score = max(0, min(100, sentiment_score))

            result = (sentiment_score, len(all_posts), sample_posts, all_posts)
            self.sentiment_cache[cache_key] = result

            return result

        except Exception as e:
            logger.debug(f"Erreur Reddit sentiment {symbol}: {e}")
            return 0.0, 0, [], []  # Score 0 en cas d'erreur (pas de données valides)

    async def _get_subreddit_posts(self, session: aiohttp.ClientSession, subreddit: str,
                                   target_date: datetime, lookback_hours: int) -> List[Dict]:
        """Récupère les posts récents d'un subreddit"""
        try:
            # Utiliser oauth.reddit.com si on a un token, sinon old.reddit.com
            if self.reddit_token:
                url = f"https://oauth.reddit.com/r/{subreddit}/new"
            else:
                url = f"https://old.reddit.com/r/{subreddit}/new.json"
            params = {'limit': 100}

            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    children = data.get('data', {}).get('children', [])

                    posts = []
                    cutoff_time = target_date - timedelta(hours=lookback_hours)
                    # Permettre une marge de 1 heure dans le futur pour les décalages d'horloge
                    future_margin = target_date + timedelta(hours=1)

                    for post in children:
                        post_data = post.get('data', {})
                        created_utc = post_data.get('created_utc', 0)
                        post_date = datetime.fromtimestamp(created_utc)

                        if cutoff_time <= post_date <= future_margin:
                            # Calculer upvotes/downvotes à partir de score et upvote_ratio
                            score = post_data.get('score', 0)
                            upvote_ratio = post_data.get('upvote_ratio', 0.5)

                            if upvote_ratio > 0 and upvote_ratio != 0.5:
                                upvotes = int(score / (2 * upvote_ratio - 1))
                                downvotes = upvotes - score
                            else:
                                upvotes = max(0, score)
                                downvotes = 0

                            posts.append({
                                'title': post_data.get('title', ''),
                                'body': post_data.get('selftext', ''),
                                'score': score,  # Gardé pour la logique de pondération
                                'upvotes': upvotes,
                                'downvotes': downvotes,
                                'created': post_date
                            })

                    return posts
                else:
                    logger.warning(f"[Reddit] r/{subreddit}: Status {response.status}")

        except Exception as e:
            logger.warning(f"[Reddit] Erreur récupération r/{subreddit}: {e}")

        return []

    async def _search_reddit_comments(self, session: aiohttp.ClientSession, subreddit: str,
                                     search_term: str, target_date: datetime,
                                     lookback_hours: int) -> List[Dict]:
        """Recherche des commentaires sur r/stocks avec le ticker"""
        try:
            # Utiliser oauth.reddit.com si on a un token, sinon old.reddit.com
            if self.reddit_token:
                url = f"https://oauth.reddit.com/r/{subreddit}/search"
            else:
                url = f"https://old.reddit.com/r/{subreddit}/search.json"

            params = {
                'q': f'${search_term}' if subreddit == 'stocks' else search_term,
                'restrict_sr': 'on',
                'sort': 'new',
                'limit': 100,
                't': 'week'  # Dernière semaine
            }

            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    children = data.get('data', {}).get('children', [])

                    posts = []
                    cutoff_time = target_date - timedelta(hours=lookback_hours)
                    # Permettre une marge de 1 heure dans le futur pour les décalages d'horloge
                    future_margin = target_date + timedelta(hours=1)

                    for post in children:
                        post_data = post.get('data', {})
                        created_utc = post_data.get('created_utc', 0)
                        post_date = datetime.fromtimestamp(created_utc)

                        if cutoff_time <= post_date <= future_margin:
                            # Calculer upvotes/downvotes à partir de score et upvote_ratio
                            score = post_data.get('score', 0)
                            upvote_ratio = post_data.get('upvote_ratio', 0.5)

                            if upvote_ratio > 0 and upvote_ratio != 0.5:
                                upvotes = int(score / (2 * upvote_ratio - 1))
                                downvotes = upvotes - score
                            else:
                                upvotes = max(0, score)
                                downvotes = 0

                            posts.append({
                                'title': post_data.get('title', ''),
                                'body': post_data.get('selftext', ''),
                                'score': score,  # Gardé pour la logique de pondération
                                'upvotes': upvotes,
                                'downvotes': downvotes,
                                'created': post_date
                            })

                    return posts
                else:
                    logger.warning(f"[Reddit] r/{subreddit} (recherche '{search_term}'): Status {response.status}")

        except Exception as e:
            logger.warning(f"[Reddit] Erreur recherche '{search_term}' sur r/{subreddit}: {e}")

        return []

    async def _get_pushshift_posts(self, session: aiohttp.ClientSession, subreddit: str,
                                   target_date: datetime, lookback_hours: int) -> List[Dict]:
        """Récupère TOUS les posts historiques via PullPush/Pushshift avec pagination ILLIMITÉE"""
        try:
            logger.info(f"   [Pushshift] Récupération COMPLÈTE r/{subreddit} - TOUS LES POSTS")

            url = "https://api.pullpush.io/reddit/search/submission"
            all_posts = []
            before = int(target_date.timestamp())
            iteration = 0

            # Boucle INFINIE jusqu'à épuisement des posts
            while True:
                iteration += 1
                params = {
                    'subreddit': subreddit,
                    'before': before,
                    'size': 500,  # Max par requête
                    'sort': 'desc',
                    'sort_type': 'created_utc'
                }

                async with session.get(url, params=params, timeout=20) as response:
                    if response.status == 200:
                        text = await response.text()
                        data = json.loads(text)
                        data_posts = data.get('data', [])

                        if not data_posts or len(data_posts) == 0:
                            logger.info(f"   [Pushshift] 🏁 Fin de pagination (plus de posts)")
                            break

                        # Ajouter tous les posts
                        for post in data_posts:
                            try:
                                created_utc = post.get('created_utc', 0)
                                post_date = datetime.fromtimestamp(created_utc)

                                # Calculer upvotes/downvotes (Pushshift ne fournit pas toujours upvote_ratio)
                                score = post.get('score', 0)
                                upvote_ratio = post.get('upvote_ratio', None)

                                if upvote_ratio is not None and upvote_ratio > 0 and upvote_ratio != 0.5:
                                    upvotes = int(score / (2 * upvote_ratio - 1))
                                    downvotes = upvotes - score
                                else:
                                    upvotes = max(0, score)
                                    downvotes = 0

                                all_posts.append({
                                    'title': post.get('title', ''),
                                    'body': post.get('selftext', ''),
                                    'score': score,
                                    'upvotes': upvotes,
                                    'downvotes': downvotes,
                                    'created': post_date
                                })
                            except Exception:
                                continue

                        # Mettre à jour 'before' pour la prochaine page
                        last_post_time = data_posts[-1].get('created_utc', 0)

                        # Vérifier qu'on avance (éviter boucle infinie)
                        if last_post_time >= before:
                            logger.warning(f"   [Pushshift] ⚠️ Timestamp ne diminue pas, arrêt")
                            break

                        before = last_post_time
                        logger.info(f"   [Pushshift] Page {iteration}: +{len(data_posts)} posts | Total: {len(all_posts)}")

                        # Délai pour rate limiting
                        await asyncio.sleep(0.5)
                    else:
                        logger.error(f"   [Pushshift] Status {response.status}, arrêt")
                        break

            # Trier par date
            all_posts.sort(key=lambda x: x['created'], reverse=True)

            logger.info(f"   [Pushshift] ✅ {len(all_posts)} posts TOTAUX récupérés pour r/{subreddit}")
            return all_posts

        except Exception as e:
            logger.error(f"   [Pushshift] ❌ Erreur subreddit {subreddit}: {e}")
            import traceback
            traceback.print_exc()

        return []

    async def _search_pushshift(self, session: aiohttp.ClientSession, subreddit: str,
                                search_term: str, target_date: datetime,
                                lookback_hours: int) -> List[Dict]:
        """Recherche TOUS les posts historiques avec un terme via Pushshift - pagination ILLIMITÉE"""
        try:
            query = f'${search_term}' if subreddit == 'stocks' else search_term
            logger.info(f"   [Pushshift] Récupération COMPLÈTE '{query}' r/{subreddit} - TOUS LES POSTS")

            url = "https://api.pullpush.io/reddit/search/submission"
            all_posts = []
            before = int(target_date.timestamp())
            iteration = 0

            # Boucle INFINIE jusqu'à épuisement des posts
            while True:
                iteration += 1
                params = {
                    'subreddit': subreddit,
                    'q': query,
                    'before': before,
                    'size': 500,
                    'sort': 'desc',
                    'sort_type': 'created_utc'
                }

                async with session.get(url, params=params, timeout=20) as response:
                    if response.status == 200:
                        text = await response.text()
                        data = json.loads(text)
                        data_posts = data.get('data', [])

                        if not data_posts or len(data_posts) == 0:
                            logger.info(f"   [Pushshift] 🏁 Fin de pagination pour '{query}'")
                            break

                        # Ajouter tous les posts
                        for post in data_posts:
                            try:
                                created_utc = post.get('created_utc', 0)
                                post_date = datetime.fromtimestamp(created_utc)

                                # Calculer upvotes/downvotes
                                score = post.get('score', 0)
                                upvote_ratio = post.get('upvote_ratio', None)

                                if upvote_ratio is not None and upvote_ratio > 0 and upvote_ratio != 0.5:
                                    upvotes = int(score / (2 * upvote_ratio - 1))
                                    downvotes = upvotes - score
                                else:
                                    upvotes = max(0, score)
                                    downvotes = 0

                                all_posts.append({
                                    'title': post.get('title', ''),
                                    'body': post.get('selftext', ''),
                                    'score': score,
                                    'upvotes': upvotes,
                                    'downvotes': downvotes,
                                    'created': post_date
                                })
                            except Exception:
                                continue

                        # Mettre à jour 'before' pour la prochaine page
                        last_post_time = data_posts[-1].get('created_utc', 0)

                        # Vérifier qu'on avance
                        if last_post_time >= before:
                            logger.warning(f"   [Pushshift] ⚠️ Timestamp ne diminue pas, arrêt")
                            break

                        before = last_post_time
                        logger.info(f"   [Pushshift] Page {iteration}: +{len(data_posts)} posts | Total: {len(all_posts)}")

                        # Délai pour rate limiting
                        await asyncio.sleep(1.5)
                    else:
                        logger.error(f"   [Pushshift] Status {response.status}, arrêt")
                        break

            # Trier par date
            all_posts.sort(key=lambda x: x['created'], reverse=True)

            logger.info(f"   [Pushshift] ✅ {len(all_posts)} posts TOTAUX récupérés pour '{query}'")
            return all_posts

        except Exception as e:
            logger.error(f"   [Pushshift] ❌ Erreur search {search_term}: {e}")
            import traceback
            traceback.print_exc()

        return []

    def save_posts_to_csv(self, symbol: str, posts: List[Dict], source: str):
        """Sauvegarde les posts dans un fichier CSV"""
        try:
            filename = f"reddit_posts_{symbol}_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['created', 'title', 'body', 'upvotes', 'downvotes'])
                writer.writeheader()

                for post in posts:
                    writer.writerow({
                        'created': post['created'].strftime('%Y-%m-%d %H:%M:%S'),
                        'title': post['title'],
                        'body': post['body'],
                        'upvotes': post.get('upvotes', 0),
                        'downvotes': post.get('downvotes', 0)
                    })

            logger.info(f"   [CSV] ✅ {len(posts)} posts sauvegardés dans {filename}")
            return filename
        except Exception as e:
            logger.error(f"   [CSV] ❌ Erreur sauvegarde: {e}")
            return None

    async def reset_session(self):
        """Force la recréation de la session HTTP avec les nouveaux headers"""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
        logger.info("[Reddit] Session HTTP fermée - sera recréée au prochain appel")

    async def close(self):
        if self.session:
            await self.session.close()
