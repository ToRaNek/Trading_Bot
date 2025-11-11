"""
Test d'authentification Reddit OAuth
"""

import asyncio
import os
from dotenv import load_dotenv
from analyzers import RedditSentimentAnalyzer
from datetime import datetime

async def test_oauth():
    print("=" * 80)
    print("TEST: Authentification Reddit OAuth")
    print("=" * 80)

    # Charger les variables d'environnement
    load_dotenv()

    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')

    print(f"\n1. Configuration:")
    print(f"   CLIENT_ID: {'[SET]' if client_id else '[VIDE]'}")
    print(f"   CLIENT_SECRET: {'[SET]' if client_secret else '[VIDE]'}")

    if not client_id or not client_secret:
        print("\n   ERREUR: Credentials manquants!")
        print("   Suivez les instructions dans REDDIT_OAUTH_SETUP.md")
        return

    # Créer l'analyzer avec OAuth
    analyzer = RedditSentimentAnalyzer(
        reddit_client_id=client_id,
        reddit_client_secret=client_secret
    )

    print("\n2. Test d'obtention du token OAuth...")
    token = await analyzer.get_reddit_oauth_token()

    if token:
        print(f"   OK Token obtenu: {token[:20]}...")
    else:
        print("   ERREUR: Impossible d'obtenir le token")
        return

    # Test de récupération de posts
    print("\n3. Test de récupération des posts (NVDA)...")
    score, count, samples, posts = await analyzer.get_reddit_sentiment(
        symbol='NVDA',
        target_date=datetime.now(),
        lookback_hours=168
    )

    print(f"   Score: {score:.0f}/100")
    print(f"   Posts: {count}")

    if count > 0:
        print(f"\n   SUCCESS! {count} posts recuperes")
        if samples:
            print(f"\n   Exemples:")
            for i, sample in enumerate(samples[:3], 1):
                # Nettoyer les emojis
                clean_sample = sample.replace('🟢', '[+]').replace('🔴', '[-]').replace('🟡', '[=]')
                print(f"   {i}. {clean_sample[:80]}...")
    else:
        print(f"\n   PROBLEME: Aucun post recupere")

    await analyzer.close()

    print("\n" + "=" * 80)
    print("Test termine!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_oauth())
