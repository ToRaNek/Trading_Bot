"""
Test pour simuler ce que fait le LiveTrader avec Reddit
"""

import asyncio
from datetime import datetime
from analyzers import RedditSentimentAnalyzer

async def test_live_reddit():
    print("=" * 80)
    print("TEST: Simulation LiveTrader avec Reddit")
    print("=" * 80)

    analyzer = RedditSentimentAnalyzer()

    # Simuler ce que fait le LiveTrader au démarrage
    print("\n1. Reset de la session (comme au !start)")
    await analyzer.reset_session()

    # Test sur plusieurs actions comme dans le live trading
    symbols = ['NVDA', 'MSFT', 'AAPL']

    for symbol in symbols:
        print(f"\n2. Analyse de {symbol}")
        score, count, samples, posts = await analyzer.get_reddit_sentiment(
            symbol=symbol,
            target_date=datetime.now(),
            lookback_hours=168
        )

        print(f"   Score: {score:.0f}/100")
        print(f"   Posts: {count}")

        if count > 0:
            print(f"   OK - Posts recuperes!")
        else:
            print(f"   PROBLEME - Aucun post")

        # Petit délai comme dans le live trader
        await asyncio.sleep(3)

    await analyzer.close()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_live_reddit())
