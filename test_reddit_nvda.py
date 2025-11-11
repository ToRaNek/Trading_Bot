"""
Test de récupération des posts Reddit pour NVDA
"""

import asyncio
from datetime import datetime
from analyzers import RedditSentimentAnalyzer

async def test_nvda():
    analyzer = RedditSentimentAnalyzer()

    print("=" * 80)
    print("TEST: Récupération posts Reddit pour NVDA")
    print("=" * 80)

    # Tester avec différentes configurations
    now = datetime.now()

    # Test 1: Récupération avec lookback par défaut (168h = 7 jours)
    print("\n1. Test avec lookback_hours=168 (7 jours)")
    score, count, samples, all_posts = await analyzer.get_reddit_sentiment(
        symbol='NVDA',
        target_date=now,
        lookback_hours=168
    )

    print(f"   Score: {score:.0f}/100")
    print(f"   Nombre de posts: {count}")
    print(f"   Posts récupérés: {len(all_posts)}")

    if all_posts:
        print(f"\n   Exemple de posts:")
        for i, post in enumerate(all_posts[:5], 1):
            print(f"   {i}. {post.get('title', '')[:80]}...")
            print(f"      Score: {post.get('score', 0)}, Date: {post.get('created', 'N/A')}")

    # Test 2: Récupération directe du subreddit NVDA_Stock
    print("\n\n2. Test direct du subreddit NVDA_Stock")
    session = await analyzer.get_session()
    posts = await analyzer._get_subreddit_posts(session, 'NVDA_Stock', now, 168)
    print(f"   Posts récupérés: {len(posts)}")

    if posts:
        print(f"\n   Exemple de posts:")
        for i, post in enumerate(posts[:5], 1):
            print(f"   {i}. {post.get('title', '')[:80]}...")
            print(f"      Score: {post.get('score', 0)}, Date: {post.get('created', 'N/A')}")
    else:
        print("   ⚠️ Aucun post récupéré!")

    # Test 3: Recherche sur r/stocks avec $NVDA
    print("\n\n3. Test recherche sur r/stocks avec $NVDA")
    posts = await analyzer._search_reddit_comments(session, 'stocks', 'NVDA', now, 168)
    print(f"   Posts récupérés: {len(posts)}")

    if posts:
        print(f"\n   Exemple de posts:")
        for i, post in enumerate(posts[:5], 1):
            print(f"   {i}. {post.get('title', '')[:80]}...")
            print(f"      Score: {post.get('score', 0)}, Date: {post.get('created', 'N/A')}")

    # Test 4: Test direct de l'API Reddit
    print("\n\n4. Test direct de l'API Reddit (NVDA_Stock)")
    url = "https://www.reddit.com/r/NVDA_Stock/new.json"
    params = {'limit': 100}

    try:
        async with session.get(url, params=params, timeout=10) as response:
            print(f"   Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                children = data.get('data', {}).get('children', [])
                print(f"   Posts bruts récupérés: {len(children)}")

                if children:
                    print(f"\n   Exemple de posts (avec dates):")
                    for i, post in enumerate(children[:5], 1):
                        post_data = post.get('data', {})
                        created_utc = post_data.get('created_utc', 0)
                        post_date = datetime.fromtimestamp(created_utc)
                        print(f"   {i}. {post_data.get('title', '')[:80]}...")
                        print(f"      Score: {post_data.get('score', 0)}, Date: {post_date}")
                else:
                    print("   ⚠️ Aucun post dans la réponse JSON!")
            else:
                print(f"   ❌ Erreur: status {response.status}")
                text = await response.text()
                print(f"   Réponse: {text[:200]}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Fermer la session
    await analyzer.close()

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_nvda())
