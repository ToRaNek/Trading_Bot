#!/usr/bin/env python3
"""Test rapide des APIs de news"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

async def test_finnhub():
    """Test Finnhub API"""
    print("\n" + "="*60)
    print("TEST FINNHUB API")
    print("="*60)

    key = os.getenv('FINNHUB_KEY')
    print(f"Cle Finnhub: {'OK Presente' if key else 'ERREUR Manquante'}")

    if not key:
        return

    # Test pour NVDA, derniers 2 jours
    from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': 'NVDA',
        'from': from_date,
        'to': to_date,
        'token': key
    }

    print(f"\nRequete: {url}")
    print(f"   Symbol: NVDA")
    print(f"   Date: {from_date} -> {to_date}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, timeout=10) as response:
                status = response.status
                print(f"\nStatus: {status}")

                if status == 200:
                    data = await response.json()
                    print(f"OK Succes! {len(data)} news recuperees")

                    if len(data) > 0:
                        print(f"\nExemple (premiere news):")
                        first = data[0]
                        print(f"   Titre: {first.get('headline', 'N/A')[:80]}")
                        print(f"   Date: {datetime.fromtimestamp(first.get('datetime', 0))}")
                        print(f"   Source: {first.get('source', 'N/A')}")
                    else:
                        print("ATTENTION  Aucune news dans la reponse (liste vide)")
                elif status == 403:
                    print("ERREUR 403: Cle API invalide ou expire")
                elif status == 429:
                    print("ERREUR 429: Rate limit depasse")
                else:
                    text = await response.text()
                    print(f"ERREUR: {text[:200]}")

        except Exception as e:
            print(f"EXCEPTION: {e}")


async def test_newsapi():
    """Test NewsAPI"""
    print("\n" + "="*60)
    print("TEST NEWSAPI")
    print("="*60)

    key = os.getenv('NEWSAPI_KEY')
    print(f"Cle NewsAPI: {'OK Presente' if key else 'ERREUR Manquante'}")

    if not key:
        return

    # Test pour NVDA, derniers 2 jours
    from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S')
    to_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'Nvidia stock OR NVDA',
        'from': from_date,
        'to': to_date,
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': key,
        'pageSize': 20
    }

    print(f"\nRequete: {url}")
    print(f"   Query: Nvidia stock OR NVDA")
    print(f"   Date: {from_date[:10]} -> {to_date[:10]}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, timeout=10) as response:
                status = response.status
                print(f"\nStatus: {status}")

                if status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    print(f"OK Succes! {len(articles)} news recuperees")

                    if len(articles) > 0:
                        print(f"\nExemple (premiere news):")
                        first = articles[0]
                        print(f"   Titre: {first.get('title', 'N/A')[:80]}")
                        print(f"   Date: {first.get('publishedAt', 'N/A')}")
                        print(f"   Source: {first.get('source', {}).get('name', 'N/A')}")
                    else:
                        print("ATTENTION  Aucune news dans la reponse (liste vide)")
                elif status == 401:
                    print("ERREUR 401: Cle API invalide")
                elif status == 426:
                    print("ERREUR 426: Plan gratuit limite (upgrade requis)")
                elif status == 429:
                    print("ERREUR 429: Rate limit depasse")
                else:
                    data = await response.json()
                    print(f"ERREUR: {data}")

        except Exception as e:
            print(f"EXCEPTION: {e}")


async def main():
    print("\nTEST DES APIs DE NEWS")
    print("="*60)

    await test_finnhub()
    await test_newsapi()

    print("\n" + "="*60)
    print("Tests termines")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
