"""
Test pour vérifier la connexion à Reddit avec les nouveaux headers
"""

import asyncio
import aiohttp
from datetime import datetime

async def test_reddit_access():
    # Test avec les nouveaux headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        print("=" * 80)
        print("TEST: Accès à Reddit avec nouveaux headers")
        print("=" * 80)

        # Test 1: old.reddit.com vs www.reddit.com
        for domain in ['old.reddit.com', 'www.reddit.com']:
            print(f"\n1. Test avec {domain}")
            url = f"https://{domain}/r/NVDA_Stock/new.json"
            params = {'limit': 10}

            try:
                async with session.get(url, params=params, timeout=10) as response:
                    print(f"   Status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        children = data.get('data', {}).get('children', [])
                        print(f"   OK Succes! {len(children)} posts recuperes")
                        if children:
                            print(f"   Premier post: {children[0]['data']['title'][:50]}...")
                    elif response.status == 403:
                        print(f"   ERREUR 403 - Acces refuse")
                    else:
                        print(f"   ATTENTION Status inattendu: {response.status}")
            except Exception as e:
                print(f"   ERREUR: {e}")

            await asyncio.sleep(2)

        # Test 2: Recherche sur r/stocks
        print(f"\n2. Test recherche sur r/stocks")
        url = "https://old.reddit.com/r/stocks/search.json"
        params = {
            'q': '$NVDA',
            'restrict_sr': 'on',
            'sort': 'new',
            'limit': 10,
            't': 'week'
        }

        try:
            async with session.get(url, params=params, timeout=10) as response:
                print(f"   Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    children = data.get('data', {}).get('children', [])
                    print(f"   OK Succes! {len(children)} posts recuperes")
                elif response.status == 403:
                    print(f"   ERREUR 403 - Acces refuse")
                else:
                    print(f"   ATTENTION Status inattendu: {response.status}")
        except Exception as e:
            print(f"   ERREUR: {e}")

        print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_reddit_access())
