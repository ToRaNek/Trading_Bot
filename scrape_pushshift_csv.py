#!/usr/bin/env python3
"""
Script de scraping Pushshift UNIQUEMENT
Récupère TOUS les posts historiques et sauvegarde en CSV
Pas de calcul de sentiment - juste extraction brute
"""

import asyncio
import aiohttp
from datetime import datetime
import json
import csv


class PushshiftScraper:
    """Scraper Pushshift + Reddit API - récupération complète"""

    def __init__(self):
        self.session = None

    async def get_session(self):
        if not self.session:
            headers = {'User-Agent': 'TradingBot/1.0 (by /u/TradingBotUser)'}
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def scrape_reddit_api(self, subreddit: str, query: str = None, limit: int = 1000):
        """Récupère les posts RÉCENTS via Reddit API (jusqu'à 1000 posts max)"""
        print(f"\n{'='*80}")
        if query:
            print(f"🔍 SCRAPING REDDIT API: r/{subreddit} (recherche: {query})")
        else:
            print(f"🔍 SCRAPING REDDIT API: r/{subreddit}")
        print(f"{'='*80}")

        session = await self.get_session()
        all_posts = []
        after = None
        iteration = 0
        max_iterations = 10  # 10 pages × 100 posts = 1000 posts max (limite Reddit)

        # Pagination Reddit API (limite 1000 posts)
        while len(all_posts) < limit and iteration < max_iterations:
            iteration += 1

            if query:
                # Recherche avec query
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    'q': query,
                    'restrict_sr': 'true',
                    'sort': 'new',
                    'limit': 100,
                    't': 'all'  # Période: all time
                }
                if after:
                    params['after'] = after
            else:
                # Tous les posts du subreddit
                url = f"https://www.reddit.com/r/{subreddit}/new.json"
                params = {
                    'limit': 100,
                    't': 'all'  # Période: all time
                }
                if after:
                    params['after'] = after

            print(f"📥 Page {iteration}... ", end='', flush=True)

            try:
                async with session.get(url, params=params, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])

                        if not posts or len(posts) == 0:
                            print("🏁 FIN (aucun post)")
                            break

                        for post_data in posts:
                            post = post_data.get('data', {})
                            all_posts.append({
                                'created': datetime.fromtimestamp(post.get('created_utc', 0)),
                                'title': post.get('title', ''),
                                'body': post.get('selftext', ''),
                                'score': post.get('score', 0),
                                'author': post.get('author', ''),
                                'url': post.get('url', ''),
                                'id': post.get('id', ''),
                                'source': f"r/{subreddit}"
                            })

                        after = data.get('data', {}).get('after')
                        if not after:
                            print(f"+{len(posts)} posts | Total: {len(all_posts)} | FIN")
                            break

                        print(f"+{len(posts)} posts | Total: {len(all_posts)}")
                        await asyncio.sleep(2.0)  # Rate limiting Reddit API

                    else:
                        print(f"❌ Status {response.status}")
                        break

            except Exception as e:
                print(f"❌ Erreur: {e}")
                break

        print(f"✅ {len(all_posts)} posts Reddit API récupérés (limite: 1000)")
        return all_posts

    async def scrape_subreddit(self, subreddit: str, before_date: datetime = None):
        """Récupère TOUS les posts d'un subreddit via Pushshift"""
        print(f"\n{'='*80}")
        print(f"🔍 SCRAPING: r/{subreddit}")
        print(f"{'='*80}")

        if before_date is None:
            before_date = datetime.now()

        session = await self.get_session()
        url = "https://api.pullpush.io/reddit/search/submission"

        all_posts = []
        before = int(before_date.timestamp())
        iteration = 0

        # BOUCLE INFINIE jusqu'à épuisement
        while True:
            iteration += 1
            params = {
                'subreddit': subreddit,
                'before': before,
                'size': 100,  # Réduit à 100 pour éviter surcharge
                'sort': 'desc',
                'sort_type': 'created_utc'
            }

            print(f"📥 Page {iteration}... ", end='', flush=True)

            # Système de retry pour erreurs 500
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    async with session.get(url, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = data.get('data', [])

                            if not posts or len(posts) == 0:
                                print("🏁 FIN")
                                return all_posts

                            # Ajouter tous les posts avec source
                            for post in posts:
                                all_posts.append({
                                    'created': datetime.fromtimestamp(post.get('created_utc', 0)),
                                    'title': post.get('title', ''),
                                    'body': post.get('selftext', ''),
                                    'score': post.get('score', 0),
                                    'author': post.get('author', ''),
                                    'url': post.get('url', ''),
                                    'id': post.get('id', ''),
                                    'source': f"r/{subreddit}"
                                })

                            # Mettre à jour before
                            last_time = posts[-1].get('created_utc', 0)
                            if last_time >= before:
                                print("⚠️ Timestamp bloqué, STOP")
                                return all_posts

                            before = last_time

                            print(f"+{len(posts)} posts | Total: {len(all_posts)}")
                            success = True

                            # Rate limiting augmenté
                            await asyncio.sleep(3.0)

                        elif response.status == 500:
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = retry_count * 5  # 5s, 10s, 15s
                                print(f"⚠️ 500 | Retry {retry_count}/{max_retries} dans {wait_time}s... ", end='', flush=True)
                                await asyncio.sleep(wait_time)
                            else:
                                print(f"❌ 500 après {max_retries} tentatives, STOP")
                                return all_posts
                        else:
                            print(f"❌ Status {response.status}")
                            return all_posts

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_count * 5
                        print(f"⚠️ Erreur | Retry {retry_count}/{max_retries} dans {wait_time}s... ", end='', flush=True)
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"❌ Erreur après {max_retries} tentatives: {e}")
                        return all_posts

        print(f"\n✅ {len(all_posts)} posts récupérés pour r/{subreddit}")
        return all_posts

    async def scrape_search(self, subreddit: str, query: str, before_date: datetime = None):
        """Recherche TOUS les posts avec un terme via Pushshift"""
        print(f"\n{'='*80}")
        print(f"🔍 SCRAPING: r/{subreddit} (recherche: {query})")
        print(f"{'='*80}")

        if before_date is None:
            before_date = datetime.now()

        session = await self.get_session()
        url = "https://api.pullpush.io/reddit/search/submission"

        all_posts = []
        before = int(before_date.timestamp())
        iteration = 0

        # BOUCLE INFINIE jusqu'à épuisement
        while True:
            iteration += 1
            params = {
                'subreddit': subreddit,
                'q': query,
                'before': before,
                'size': 100,  # Réduit à 100 pour éviter surcharge
                'sort': 'desc',
                'sort_type': 'created_utc'
            }

            print(f"📥 Page {iteration}... ", end='', flush=True)

            # Système de retry pour erreurs 500
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    async with session.get(url, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = data.get('data', [])

                            if not posts or len(posts) == 0:
                                print("🏁 FIN")
                                return all_posts

                            # Ajouter tous les posts avec source
                            for post in posts:
                                all_posts.append({
                                    'created': datetime.fromtimestamp(post.get('created_utc', 0)),
                                    'title': post.get('title', ''),
                                    'body': post.get('selftext', ''),
                                    'score': post.get('score', 0),
                                    'author': post.get('author', ''),
                                    'url': post.get('url', ''),
                                    'id': post.get('id', ''),
                                    'source': f"r/{subreddit}"
                                })

                            # Mettre à jour before
                            last_time = posts[-1].get('created_utc', 0)
                            if last_time >= before:
                                print("⚠️ Timestamp bloqué, STOP")
                                return all_posts

                            before = last_time

                            print(f"+{len(posts)} posts | Total: {len(all_posts)}")
                            success = True

                            # Rate limiting augmenté
                            await asyncio.sleep(3.0)

                        elif response.status == 500:
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = retry_count * 5  # 5s, 10s, 15s
                                print(f"⚠️ 500 | Retry {retry_count}/{max_retries} dans {wait_time}s... ", end='', flush=True)
                                await asyncio.sleep(wait_time)
                            else:
                                print(f"❌ 500 après {max_retries} tentatives, STOP")
                                return all_posts
                        else:
                            print(f"❌ Status {response.status}")
                            return all_posts

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_count * 5
                        print(f"⚠️ Erreur | Retry {retry_count}/{max_retries} dans {wait_time}s... ", end='', flush=True)
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"❌ Erreur après {max_retries} tentatives: {e}")
                        return all_posts

        print(f"\n✅ {len(all_posts)} posts récupérés pour '{query}' sur r/{subreddit}")
        return all_posts

    def save_to_csv(self, posts, filename):
        """Sauvegarde les posts en CSV avec source"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'created', 'source', 'title', 'body', 'score', 'author', 'url', 'id'
                ])
                writer.writeheader()

                for post in posts:
                    writer.writerow({
                        'created': post['created'].strftime('%Y-%m-%d %H:%M:%S'),
                        'source': post.get('source', 'unknown'),
                        'title': post['title'],
                        'body': post['body'],
                        'score': post['score'],
                        'author': post['author'],
                        'url': post['url'],
                        'id': post['id']
                    })

            print(f"💾 ✅ Sauvegardé: {filename} ({len(posts)} posts)")
            return filename
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return None

    async def close(self):
        if self.session:
            await self.session.close()


async def main():
    """Script principal"""

    print("\n" + "="*80)
    print("🔥 SCRAPER PUSHSHIFT → CSV")
    print("="*80)
    print("\n📋 Configuration:")
    print("   Ticker: NVDA")
    print("   Source 1: r/NVDA_Stock (subreddit dédié)")
    print("   Source 2: r/stocks (recherche $NVDA)")
    print("   Méthode: Reddit API (1000 posts récents) + Pushshift (historique ancien)")
    print("   Output: UN SEUL fichier CSV combiné")
    print("\n⚠️ ATTENTION:")
    print("   - Reddit API: max 1000 posts les plus récents")
    print("   - Pushshift: historique ancien (mars-mai 2025)")
    print("   - Combiné = couverture maximale disponible")

    # input("\n▶ Appuyez sur ENTRÉE pour démarrer...")  # Commenté pour permettre exécution auto

    scraper = PushshiftScraper()
    all_posts_combined = []

    try:
        print("\n" + "="*80)
        print("🚀 PHASE 1: POSTS RÉCENTS (Reddit API - max 1000)")
        print("="*80)

        # Source 1 REDDIT API: r/NVDA_Stock
        posts_nvda_recent = await scraper.scrape_reddit_api('NVDA_Stock', limit=1000)
        if posts_nvda_recent:
            all_posts_combined.extend(posts_nvda_recent)

        # Source 2 REDDIT API: r/stocks avec $NVDA
        posts_stocks_recent = await scraper.scrape_reddit_api('stocks', query='$NVDA', limit=1000)
        if posts_stocks_recent:
            all_posts_combined.extend(posts_stocks_recent)

        print("\n" + "="*80)
        print("🚀 PHASE 2: HISTORIQUE ANCIEN (Pushshift)")
        print("="*80)

        # Source 1 PUSHSHIFT: r/NVDA_Stock (historique)
        posts_nvda_stock = await scraper.scrape_subreddit('NVDA_Stock', before_date=datetime.now())
        if posts_nvda_stock:
            all_posts_combined.extend(posts_nvda_stock)

        # Source 2 PUSHSHIFT: r/stocks avec $NVDA (historique)
        posts_stocks = await scraper.scrape_search('stocks', '$NVDA', before_date=datetime.now())
        if posts_stocks:
            all_posts_combined.extend(posts_stocks)

        # Dédupliquer par ID
        print("\n🔄 Déduplication des posts...")
        seen_ids = set()
        deduplicated_posts = []
        for post in all_posts_combined:
            if post['id'] not in seen_ids:
                seen_ids.add(post['id'])
                deduplicated_posts.append(post)

        duplicates_removed = len(all_posts_combined) - len(deduplicated_posts)
        print(f"✅ {duplicates_removed} doublons supprimés")

        # Sauvegarder TOUT dans UN SEUL fichier
        if deduplicated_posts:
            # Trier par date (plus récent en premier)
            deduplicated_posts.sort(key=lambda x: x['created'], reverse=True)

            filename = f"pushshift_NVDA_ALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            scraper.save_to_csv(deduplicated_posts, filename)

        # Résumé
        print("\n" + "="*80)
        print("📊 RÉSUMÉ")
        print("="*80)

        reddit_api_total = len(posts_nvda_recent) + len(posts_stocks_recent)
        pushshift_total = len(posts_nvda_stock) + len(posts_stocks)

        print(f"\n🌐 Reddit API (récent - max 1000 posts):")
        print(f"   r/NVDA_Stock: {len(posts_nvda_recent)} posts")
        print(f"   r/stocks ($NVDA): {len(posts_stocks_recent)} posts")
        print(f"   Sous-total: {reddit_api_total} posts")

        print(f"\n📜 Pushshift (historique ancien):")
        print(f"   r/NVDA_Stock: {len(posts_nvda_stock)} posts")
        print(f"   r/stocks ($NVDA): {len(posts_stocks)} posts")
        print(f"   Sous-total: {pushshift_total} posts")

        print(f"\n📊 TOTAL (après déduplication): {len(deduplicated_posts)} posts")

        if deduplicated_posts:
            oldest = min(p['created'] for p in deduplicated_posts)
            newest = max(p['created'] for p in deduplicated_posts)
            days = (newest - oldest).days
            print(f"\n📅 Période COMPLÈTE: {oldest.strftime('%Y-%m-%d')} → {newest.strftime('%Y-%m-%d')}")
            print(f"📊 Durée totale: {days} jours")

        print("\n" + "="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
