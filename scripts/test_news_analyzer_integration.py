# -*- coding: utf-8 -*-
"""Test du nouveau systeme d'analyse de sentiment integre dans news_analyzer.py"""

import asyncio
import sys
import os
from datetime import datetime

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from analyzers.news_analyzer import HistoricalNewsAnalyzer

load_dotenv()


async def test_news_analyzer():
    """Test du news analyzer avec plusieurs symboles"""

    print("=" * 70)
    print("TEST DU NOUVEAU SYSTEME D'ANALYSE DE SENTIMENT")
    print("=" * 70)
    print()

    analyzer = HistoricalNewsAnalyzer()

    # Symboles a tester
    symbols = ['NVDA', 'AAPL', 'TSLA']

    # Date recente (NewsAPI + Finnhub)
    recent_date = datetime(2025, 11, 5)

    # Date ancienne (Finnhub seulement)
    old_date = datetime(2025, 7, 15)

    print("\n" + "=" * 70)
    print("TEST 1: Date recente (2025-11-05) - Finnhub + NewsAPI")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n>>> Analyse de {symbol}...")

        has_news, news_items, score = await analyzer.get_news_for_date(symbol, recent_date)

        print(f"    Resultat: {len(news_items)} articles, score={score:.0f}/100")

        if news_items:
            # Calculer les ratios
            positive = sum(1 for item in news_items if item.get('sentiment', 0) > 0.1)
            negative = sum(1 for item in news_items if item.get('sentiment', 0) < -0.1)
            neutral = len(news_items) - positive - negative

            total = len(news_items)
            pos_ratio = (positive / total * 100) if total > 0 else 0
            neg_ratio = (negative / total * 100) if total > 0 else 0
            neu_ratio = (neutral / total * 100) if total > 0 else 0

            print(f"    Sentiment: {pos_ratio:.0f}% positif, {neg_ratio:.0f}% negatif, {neu_ratio:.0f}% neutre")

            # Afficher quelques exemples
            print(f"    Exemples:")
            for i, item in enumerate(news_items[:3], 1):
                sentiment_str = "POS" if item.get('sentiment', 0) > 0.1 else "NEG" if item.get('sentiment', 0) < -0.1 else "NEU"
                print(f"      {i}. [{sentiment_str}] {item['title'][:60]}...")

    print("\n" + "=" * 70)
    print("TEST 2: Date ancienne (2025-07-15) - Finnhub seulement")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n>>> Analyse de {symbol}...")

        has_news, news_items, score = await analyzer.get_news_for_date(symbol, old_date)

        print(f"    Resultat: {len(news_items)} articles, score={score:.0f}/100")

        if news_items:
            # Calculer les ratios
            positive = sum(1 for item in news_items if item.get('sentiment', 0) > 0.1)
            negative = sum(1 for item in news_items if item.get('sentiment', 0) < -0.1)
            neutral = len(news_items) - positive - negative

            total = len(news_items)
            pos_ratio = (positive / total * 100) if total > 0 else 0
            neg_ratio = (negative / total * 100) if total > 0 else 0
            neu_ratio = (neutral / total * 100) if total > 0 else 0

            print(f"    Sentiment: {pos_ratio:.0f}% positif, {neg_ratio:.0f}% negatif, {neu_ratio:.0f}% neutre")

    await analyzer.close()

    print("\n" + "=" * 70)
    print("TESTS TERMINES")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_news_analyzer())
