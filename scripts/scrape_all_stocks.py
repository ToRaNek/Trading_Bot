#!/usr/bin/env python3
"""
Script de scraping Reddit pour TOUTES les actions
Récupère les posts historiques via Reddit API + Pushshift
Sauvegarde chaque action dans data/Sentiment_[TICKER].csv
"""

import asyncio
import aiohttp
from datetime import datetime
import json
import csv
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer config_stocks
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_stocks import STOCK_CONFIGS


class MultiStockScraper:
    """Scraper Reddit pour plusieurs actions"""

    def __init__(self, output_dir: str = 'data'):
        self.session = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def get_session(self):
        if not self.session:
            headers = {'User-Agent': 'TradingBot/1.0 (by /u/TradingBotUser)'}
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def scrape_reddit_api_subreddit(self, subreddit: str, limit: int = 1000):
        """Récupère les posts récents d'un subreddit via Reddit API"""
        print(f"   📥 Reddit API: r/{subreddit}")

        session = await self.get_session()
        all_posts = []
        after = None
        iteration = 0
        max_iterations = 10

        while len(all_posts) < limit and iteration < max_iterations:
            iteration += 1
            url = f"https://www.reddit.com/r/{subreddit}/new.json"
            params = {'limit': 100, 't': 'all'}
            if after:
                params['after'] = after

            try:
                async with session.get(url, params=params, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])

                        if not posts:
                            break

                        for post_data in posts:
                            post = post_data.get('data', {})
                            score = post.get('score', 0)
                            upvote_ratio = post.get('upvote_ratio', 0.5)

                            if upvote_ratio > 0 and upvote_ratio != 0.5:
                                upvotes = int(score / (2 * upvote_ratio - 1))
                                downvotes = upvotes - score
                            else:
                                upvotes = max(0, score)
                                downvotes = 0

                            all_posts.append({
                                'created': datetime.fromtimestamp(post.get('created_utc', 0)),
                                'title': post.get('title', ''),
                                'body': post.get('selftext', ''),
                                'upvotes': upvotes,
                                'downvotes': downvotes,
                                'author': post.get('author', ''),
                                'url': post.get('url', ''),
                                'id': post.get('id', ''),
                                'source': f"r/{subreddit}"
                            })

                        after = data.get('data', {}).get('after')
                        if not after:
                            break

                        await asyncio.sleep(2.0)
                    else:
                        break

            except Exception as e:
                print(f"      ❌ Erreur: {e}")
                break

        print(f"      ✅ {len(all_posts)} posts")
        return all_posts

    async def scrape_reddit_api_search(self, subreddit: str, query: str, limit: int = 1000):
        """Recherche des posts via Reddit API"""
        print(f"   📥 Reddit API: r/{subreddit}/search?q={query}")

        session = await self.get_session()
        all_posts = []
        after = None
        iteration = 0
        max_iterations = 10

        while len(all_posts) < limit and iteration < max_iterations:
            iteration += 1
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': query,
                'restrict_sr': 'true',
                'sort': 'new',
                'limit': 100,
                't': 'all'
            }
            if after:
                params['after'] = after

            try:
                async with session.get(url, params=params, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])

                        if not posts:
                            break

                        for post_data in posts:
                            post = post_data.get('data', {})
                            score = post.get('score', 0)
                            upvote_ratio = post.get('upvote_ratio', 0.5)

                            if upvote_ratio > 0 and upvote_ratio != 0.5:
                                upvotes = int(score / (2 * upvote_ratio - 1))
                                downvotes = upvotes - score
                            else:
                                upvotes = max(0, score)
                                downvotes = 0

                            all_posts.append({
                                'created': datetime.fromtimestamp(post.get('created_utc', 0)),
                                'title': post.get('title', ''),
                                'body': post.get('selftext', ''),
                                'upvotes': upvotes,
                                'downvotes': downvotes,
                                'author': post.get('author', ''),
                                'url': post.get('url', ''),
                                'id': post.get('id', ''),
                                'source': f"r/{subreddit}"
                            })

                        after = data.get('data', {}).get('after')
                        if not after:
                            break

                        await asyncio.sleep(2.0)
                    else:
                        break

            except Exception as e:
                print(f"      ❌ Erreur: {e}")
                break

        print(f"      ✅ {len(all_posts)} posts")
        return all_posts

    async def scrape_pushshift_subreddit(self, subreddit: str, before_date: datetime = None):
        """Récupère TOUS les posts historiques d'un subreddit via Pushshift"""
        print(f"   📜 Pushshift: r/{subreddit}")

        if before_date is None:
            before_date = datetime.now()

        session = await self.get_session()
        url = "https://api.pullpush.io/reddit/search/submission"

        all_posts = []
        before = int(before_date.timestamp())
        iteration = 0

        while True:
            iteration += 1
            params = {
                'subreddit': subreddit,
                'before': before,
                'size': 100,
                'sort': 'desc',
                'sort_type': 'created_utc'
            }

            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    async with session.get(url, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = data.get('data', [])

                            if not posts:
                                return all_posts

                            for post in posts:
                                score = post.get('score', 0)
                                upvote_ratio = post.get('upvote_ratio', None)

                                if upvote_ratio is not None and upvote_ratio > 0 and upvote_ratio != 0.5:
                                    upvotes = int(score / (2 * upvote_ratio - 1))
                                    downvotes = upvotes - score
                                else:
                                    upvotes = max(0, score)
                                    downvotes = 0

                                all_posts.append({
                                    'created': datetime.fromtimestamp(post.get('created_utc', 0)),
                                    'title': post.get('title', ''),
                                    'body': post.get('selftext', ''),
                                    'upvotes': upvotes,
                                    'downvotes': downvotes,
                                    'author': post.get('author', ''),
                                    'url': post.get('url', ''),
                                    'id': post.get('id', ''),
                                    'source': f"r/{subreddit}"
                                })

                            last_time = posts[-1].get('created_utc', 0)
                            if last_time >= before:
                                return all_posts

                            before = last_time
                            success = True
                            await asyncio.sleep(3.0)

                        elif response.status == 500:
                            retry_count += 1
                            if retry_count < max_retries:
                                await asyncio.sleep(retry_count * 5)
                            else:
                                return all_posts
                        else:
                            return all_posts

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(retry_count * 5)
                    else:
                        return all_posts

        print(f"      ✅ {len(all_posts)} posts")
        return all_posts

    async def scrape_pushshift_search(self, subreddit: str, query: str, before_date: datetime = None):
        """Recherche TOUS les posts historiques via Pushshift"""
        print(f"   📜 Pushshift: r/{subreddit}/search?q={query}")

        if before_date is None:
            before_date = datetime.now()

        session = await self.get_session()
        url = "https://api.pullpush.io/reddit/search/submission"

        all_posts = []
        before = int(before_date.timestamp())
        iteration = 0

        while True:
            iteration += 1
            params = {
                'subreddit': subreddit,
                'q': query,
                'before': before,
                'size': 100,
                'sort': 'desc',
                'sort_type': 'created_utc'
            }

            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    async with session.get(url, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = data.get('data', [])

                            if not posts:
                                return all_posts

                            for post in posts:
                                score = post.get('score', 0)
                                upvote_ratio = post.get('upvote_ratio', None)

                                if upvote_ratio is not None and upvote_ratio > 0 and upvote_ratio != 0.5:
                                    upvotes = int(score / (2 * upvote_ratio - 1))
                                    downvotes = upvotes - score
                                else:
                                    upvotes = max(0, score)
                                    downvotes = 0

                                all_posts.append({
                                    'created': datetime.fromtimestamp(post.get('created_utc', 0)),
                                    'title': post.get('title', ''),
                                    'body': post.get('selftext', ''),
                                    'upvotes': upvotes,
                                    'downvotes': downvotes,
                                    'author': post.get('author', ''),
                                    'url': post.get('url', ''),
                                    'id': post.get('id', ''),
                                    'source': f"r/{subreddit}"
                                })

                            last_time = posts[-1].get('created_utc', 0)
                            if last_time >= before:
                                return all_posts

                            before = last_time
                            success = True
                            await asyncio.sleep(3.0)

                        elif response.status == 500:
                            retry_count += 1
                            if retry_count < max_retries:
                                await asyncio.sleep(retry_count * 5)
                            else:
                                return all_posts
                        else:
                            return all_posts

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(retry_count * 5)
                    else:
                        return all_posts

        print(f"      ✅ {len(all_posts)} posts")
        return all_posts

    async def scrape_stock(self, ticker: str, config: dict):
        """Scrape toutes les sources pour une action"""
        print(f"\n{'='*80}")
        print(f"🔍 SCRAPING: {ticker}")
        print(f"{'='*80}")

        all_posts = []

        for source in config['sources']:
            try:
                if source['type'] == 'subreddit':
                    # Scraper un subreddit dédié
                    subreddit = source['name']

                    # Reddit API (posts récents)
                    posts_recent = await self.scrape_reddit_api_subreddit(subreddit)
                    all_posts.extend(posts_recent)

                    # Pushshift (historique)
                    posts_history = await self.scrape_pushshift_subreddit(subreddit)
                    all_posts.extend(posts_history)

                elif source['type'] == 'search':
                    # Recherche dans un subreddit
                    subreddit = source['subreddit']
                    query = source['query']

                    # Reddit API (posts récents)
                    posts_recent = await self.scrape_reddit_api_search(subreddit, query)
                    all_posts.extend(posts_recent)

                    # Pushshift (historique)
                    posts_history = await self.scrape_pushshift_search(subreddit, query)
                    all_posts.extend(posts_history)

            except Exception as e:
                print(f"   ❌ Erreur source: {e}")
                continue

        # Dédupliquer par ID
        print(f"\n🔄 Déduplication...")
        seen_ids = set()
        deduplicated_posts = []
        for post in all_posts:
            if post['id'] not in seen_ids:
                seen_ids.add(post['id'])
                deduplicated_posts.append(post)

        duplicates_removed = len(all_posts) - len(deduplicated_posts)
        print(f"   ✅ {duplicates_removed} doublons supprimés")

        # Trier par date
        if deduplicated_posts:
            deduplicated_posts.sort(key=lambda x: x['created'], reverse=True)

        return deduplicated_posts

    def save_to_csv(self, ticker: str, posts: list):
        """Sauvegarde les posts dans data/Sentiment_[TICKER].csv"""
        try:
            filename = self.output_dir / f"Sentiment_{ticker}.csv"

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'created', 'source', 'title', 'body', 'upvotes', 'downvotes', 'author', 'url', 'id'
                ])
                writer.writeheader()

                for post in posts:
                    writer.writerow({
                        'created': post['created'].strftime('%Y-%m-%d %H:%M:%S'),
                        'source': post.get('source', 'unknown'),
                        'title': post['title'],
                        'body': post['body'],
                        'upvotes': post['upvotes'],
                        'downvotes': post['downvotes'],
                        'author': post['author'],
                        'url': post['url'],
                        'id': post['id']
                    })

            print(f"💾 ✅ Sauvegardé: {filename} ({len(posts)} posts)")

            if posts:
                oldest = min(p['created'] for p in posts)
                newest = max(p['created'] for p in posts)
                days = (newest - oldest).days
                print(f"📅 Période: {oldest.strftime('%Y-%m-%d')} → {newest.strftime('%Y-%m-%d')} ({days} jours)")

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
    print("🔥 SCRAPER REDDIT - TOUTES LES ACTIONS")
    print("="*80)
    print(f"\n📋 {len(STOCK_CONFIGS)} actions configurées:")
    for ticker in STOCK_CONFIGS:
        print(f"   - {ticker}")

    print("\n⚠️  Mode: Reddit API (récent) + Pushshift (historique)")
    print(f"📁 Output: data/Sentiment_[TICKER].csv")

    scraper = MultiStockScraper(output_dir='data')

    try:
        results = {}

        for ticker, config in STOCK_CONFIGS.items():
            posts = await scraper.scrape_stock(ticker, config)

            if posts:
                scraper.save_to_csv(ticker, posts)
                results[ticker] = len(posts)
            else:
                print(f"⚠️  Aucun post trouvé pour {ticker}")
                results[ticker] = 0

        # Résumé
        print("\n" + "="*80)
        print("📊 RÉSUMÉ FINAL")
        print("="*80)

        for ticker, count in results.items():
            print(f"   {ticker}: {count} posts")

        print(f"\n✅ Total: {sum(results.values())} posts")
        print(f"📁 Fichiers créés dans: data/")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
