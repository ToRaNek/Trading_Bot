"""
Analyse approfondie des 2 cas catastrophiques:
- BNP.PA: V9=74.6, Claude=25 (diff 49.6)
- SAF.PA: V9=65.0, Claude=15 (diff 50.0)

Objectif: Comprendre pourquoi V9 sur-estime autant
"""

import asyncio
import sys
sys.path.insert(0, '.')

from datetime import datetime
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from optimize_textblob_final import UltraTextBlob
from textblob_v9_refined import score_news_v9_refined
from textblob import TextBlob
import re


async def analyze_catastrophic_case(symbol: str, scorer, analyzer):
    """Analyse détaillée d'un cas catastrophique"""

    print(f"\n{'='*100}")
    print(f"ANALYSE CATASTROPHIQUE: {symbol}")
    print('='*100)

    has_news, news_items, _ = await analyzer.get_news_for_date(symbol, datetime.now())

    if not news_items:
        print("Pas de news!")
        return

    print(f"\nNombre de news: {len(news_items)}")

    # Scores
    v9_score = score_news_v9_refined(news_items)
    claude_score = await scorer.score_with_claude(symbol, news_items)

    print(f"\nV9 Score:     {v9_score:.1f}")
    print(f"Claude Score: {claude_score:.1f}")
    print(f"DIVERGENCE:   {abs(claude_score - v9_score):.1f} pts")

    # Analyser les 15 premières news
    print(f"\n{'='*100}")
    print("TOP 15 NEWS (ce que Claude voit)")
    print('='*100)

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for i, news in enumerate(news_items[:15], 1):
        title = news.get('title', '')
        summary = news.get('summary', '')[:200]

        # TextBlob sentiment
        blob = TextBlob(title)
        polarity = blob.sentiment.polarity

        if polarity > 0.12:
            positive_count += 1
            sentiment = 'POS'
        elif polarity < -0.12:
            negative_count += 1
            sentiment = 'NEG'
        else:
            neutral_count += 1
            sentiment = 'NEU'

        text_lower = (title + ' ' + summary).lower()

        # Détecter keywords négatifs forts
        strong_negative = ['bankruptcy', 'crash', 'scandal', 'plunge', 'earnings miss',
                          'downgrade', 'lawsuit', 'fraud', 'loss', 'losses', 'decline',
                          'falls', 'drops', 'tumbles', 'slump', 'weak']

        neg_keywords = [kw for kw in strong_negative if kw in text_lower]

        # Détecter keywords positifs
        strong_positive = ['beats expectations', 'record profit', 'surge', 'soars',
                          'breakthrough', 'invests', 'expansion', 'growth', 'profit']

        pos_keywords = [kw for kw in strong_positive if kw in text_lower]

        # Montants
        billions = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
        millions = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)

        print(f"\n{i:2}. [{sentiment}] Pol={polarity:.2f}")
        print(f"    {title}")
        if summary:
            print(f"    {summary[:150]}...")

        if neg_keywords:
            print(f"    NEG Keywords: {', '.join(neg_keywords)}")
        if pos_keywords:
            print(f"    POS Keywords: {', '.join(pos_keywords)}")
        if billions:
            print(f"    $ {billions[0]}B")
        if millions:
            print(f"    $ {millions[0]}M")

    print(f"\n{'='*100}")
    print("STATISTIQUES")
    print('='*100)
    print(f"Positive: {positive_count}/15")
    print(f"Negative: {negative_count}/15")
    print(f"Neutral:  {neutral_count}/15")

    ratio_pos = positive_count / 15
    ratio_neg = negative_count / 15
    ratio_neu = neutral_count / 15

    print(f"\nRatios:")
    print(f"  Positive: {ratio_pos:.2f}")
    print(f"  Negative: {ratio_neg:.2f}")
    print(f"  Neutral:  {ratio_neu:.2f}")

    print(f"\n{'='*100}")
    print("DIAGNOSTIC")
    print('='*100)

    if claude_score < 30:
        print("Claude voit cette action comme TRES NEGATIVE")
    elif claude_score < 45:
        print("Claude voit cette action comme NEGATIVE")
    elif claude_score < 55:
        print("Claude voit cette action comme NEUTRE")

    if v9_score > 60 and claude_score < 30:
        print("\nPROBLEME: V9 sur-estime massivement!")
        print("Hypotheses:")
        print("  1. V9 ne detecte pas les keywords negatifs")
        print("  2. V9 sur-pondere les news neutres")
        print("  3. V9 rate le contexte negatif global")


async def main():
    print("="*100)
    print("ANALYSE DES CAS CATASTROPHIQUES")
    print("="*100)

    analyzer = HistoricalNewsAnalyzer()
    scorer = UltraTextBlob()

    # Analyser les 2 pires cas
    await analyze_catastrophic_case('BNP.PA', scorer, analyzer)
    await asyncio.sleep(1)

    await analyze_catastrophic_case('SAF.PA', scorer, analyzer)
    await asyncio.sleep(1)

    # Analyser aussi un cas moyen pour comparaison
    print("\n\n" + "="*100)
    print("COMPARAISON AVEC UN CAS NORMAL")
    print("="*100)
    await analyze_catastrophic_case('AMZN', scorer, analyzer)

    await scorer.close()
    await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())
