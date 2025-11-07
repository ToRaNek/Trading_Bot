#!/usr/bin/env python3
"""Test news pour date historique du backtest"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

async def test_finnhub_historical():
    """Test Finnhub pour date historique"""
    print("\nTEST FINNHUB - DATE HISTORIQUE DU BACKTEST")
    print("="*60)

    key = os.getenv('FINNHUB_KEY')

    # Date du backtest: 2025-07-08 (comme dans vos logs)
    target_date = datetime(2025, 7, 8)
    from_date = (target_date - timedelta(days=2)).strftime('%Y-%m-%d')
    to_date = target_date.strftime('%Y-%m-%d')

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': 'NVDA',
        'from': from_date,
        'to': to_date,
        'token': key
    }

    print(f"\nRequete: {url}")
    print(f"   Symbol: NVDA")
    print(f"   Date: {from_date} -> {to_date} (BACKTEST DATE)")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, timeout=10) as response:
                status = response.status
                print(f"\nStatus: {status}")

                if status == 200:
                    data = await response.json()
                    print(f"\nResultat: {len(data)} news recuperees")

                    if len(data) > 0:
                        print(f"\nPremiere news:")
                        first = data[0]
                        print(f"   Titre: {first.get('headline', 'N/A')[:80]}")
                        print(f"   Date: {datetime.fromtimestamp(first.get('datetime', 0))}")
                    else:
                        print("\n>>> PROBLEME: 0 news pour cette date historique <<<")
                        print(">>> Finnhub ne fournit pas de news anciennes <<<")
                else:
                    text = await response.text()
                    print(f"ERREUR: {text[:200]}")

        except Exception as e:
            print(f"EXCEPTION: {e}")


async def main():
    await test_finnhub_historical()


if __name__ == "__main__":
    asyncio.run(main())
