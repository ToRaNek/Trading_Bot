#!/usr/bin/env python3
"""
Test rapide de l'API Pushshift/PullPush
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
import json


async def test_pushshift():
    """Test simulation backtest réaliste"""

    print("\n🔍 SIMULATION BACKTEST - NVDA le 2025-06-18\n")
    print("=" * 80)

    headers = {'User-Agent': 'TradingBot/1.0 (by /u/TradingBotUser)'}
    url = "https://api.pullpush.io/reddit/search/submission"

    # Date du backtest : 18 juin 2025
    # On veut les posts des 48h précédentes (16-18 juin)
    target_date = datetime(2025, 6, 18)
    lookback_hours = 48
    cutoff_date = target_date - timedelta(hours=lookback_hours)

    print(f"📅 Date du backtest: {target_date.strftime('%Y-%m-%d')}")
    print(f"📅 Période recherche: {cutoff_date.strftime('%Y-%m-%d')} → {target_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # Source 1: r/NVDA_Stock
    print("\n🔍 SOURCE 1: r/NVDA_Stock")
    print("-" * 80)

    params1 = {
        'subreddit': 'NVDA_Stock',
        'before': int(target_date.timestamp()),
        'size': 500,
        'sort': 'desc',
        'sort_type': 'created_utc'
    }

    print(f"📦 Params: {json.dumps(params1, indent=2)}")

    posts_nvda_stock = []
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.get(url, params=params1, timeout=20) as response:
                print(f"✅ Status: {response.status}")

                if response.status == 200:
                    text = await response.text()
                    data = json.loads(text)
                    all_posts = data.get('data', [])

                    print(f"📥 {len(all_posts)} posts récupérés")

                    # Filtrer par période
                    for post in all_posts:
                        created = datetime.fromtimestamp(post.get('created_utc', 0))
                        if cutoff_date <= created <= target_date:
                            posts_nvda_stock.append({
                                'title': post.get('title', ''),
                                'score': post.get('score', 0),
                                'created': created
                            })

                    print(f"✅ {len(posts_nvda_stock)} posts dans la période 48h")

                    if posts_nvda_stock:
                        print("\n📋 Exemples:")
                        for i, post in enumerate(posts_nvda_stock[:3], 1):
                            print(f"   {i}. [{post['created'].strftime('%Y-%m-%d %H:%M')}] {post['title'][:60]}")
                else:
                    print(f"❌ Status {response.status}")

        except Exception as e:
            print(f"❌ Erreur: {e}")

    # Source 2: r/stocks avec recherche $NVDA
    print("\n🔍 SOURCE 2: r/stocks (recherche $NVDA)")
    print("-" * 80)

    await asyncio.sleep(1.5)  # Délai pour rate limiting

    params2 = {
        'subreddit': 'stocks',
        'q': '$NVDA',
        'before': int(target_date.timestamp()),
        'size': 500,
        'sort': 'desc',
        'sort_type': 'created_utc'
    }

    print(f"📦 Params: {json.dumps(params2, indent=2)}")

    posts_stocks = []
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.get(url, params=params2, timeout=20) as response:
                print(f"✅ Status: {response.status}")

                if response.status == 200:
                    text = await response.text()
                    data = json.loads(text)
                    all_posts = data.get('data', [])

                    print(f"📥 {len(all_posts)} posts récupérés")

                    # Filtrer par période
                    for post in all_posts:
                        created = datetime.fromtimestamp(post.get('created_utc', 0))
                        if cutoff_date <= created <= target_date:
                            posts_stocks.append({
                                'title': post.get('title', ''),
                                'score': post.get('score', 0),
                                'created': created
                            })

                    print(f"✅ {len(posts_stocks)} posts dans la période 48h")

                    if posts_stocks:
                        print("\n📋 Exemples:")
                        for i, post in enumerate(posts_stocks[:3], 1):
                            print(f"   {i}. [{post['created'].strftime('%Y-%m-%d %H:%M')}] {post['title'][:60]}")
                else:
                    print(f"❌ Status {response.status}")

        except Exception as e:
            print(f"❌ Erreur: {e}")

    # Résumé final
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ FINAL")
    print("=" * 80)

    total_posts = len(posts_nvda_stock) + len(posts_stocks)

    print(f"\n📅 Date backtest: {target_date.strftime('%Y-%m-%d')}")
    print(f"📅 Période recherche: {cutoff_date.strftime('%Y-%m-%d')} → {target_date.strftime('%Y-%m-%d')} (48h)")
    print(f"\n📱 r/NVDA_Stock: {len(posts_nvda_stock)} posts")
    print(f"📱 r/stocks ($NVDA): {len(posts_stocks)} posts")
    print(f"📊 TOTAL: {total_posts} posts")

    if total_posts > 0:
        print(f"\n✅ SUCCÈS ! On peut récupérer des posts Reddit historiques")
        print(f"\n💡 Méthode qui fonctionne:")
        print(f"   1. Utiliser 'before': {int(target_date.timestamp())}")
        print(f"   2. Récupérer 500 posts max")
        print(f"   3. Filtrer côté client selon période")
        print(f"\n🎯 Pour le backtest:")
        print(f"   - Sentiment Reddit sera calculé sur {total_posts} posts")
        print(f"   - Pondération par upvotes")
        print(f"   - Score 0-100 basé sur TextBlob")
    else:
        print(f"\n⚠️ Aucun post trouvé pour cette période")
        print(f"   Possible raisons:")
        print(f"   - Période trop ancienne (>6 mois)")
        print(f"   - Subreddit peu actif")
        print(f"   - Besoin d'augmenter la taille (size > 500)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_pushshift())
