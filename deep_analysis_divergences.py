"""
Analyse APPROFONDIE des divergences TextBlob vs Claude
Pour comprendre exactement ce qui fait diverger les scores
"""

import asyncio
import sys
import os
sys.path.insert(0, '.')

from datetime import datetime
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from optimize_textblob_final import UltraTextBlob
from textblob import TextBlob
import re


async def deep_analyze_divergence(symbol: str, scorer, analyzer):
    """Analyse en profondeur d'une action"""

    print(f"\n{'='*100}")
    print(f"ANALYSE APPROFONDIE: {symbol}")
    print('='*100)

    has_news, news_items, _ = await analyzer.get_news_for_date(symbol, datetime.now())

    if not news_items:
        print("Pas de news")
        return None

    print(f"\nNombre de news: {len(news_items)}")

    # Scorer avec Claude
    claude_score = await scorer.score_with_claude(symbol, news_items)

    # Analyser les 15 premières news (ce que Claude voit)
    print(f"\n--- TOP 15 NEWS (ce que Claude analyse) ---")

    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    big_money_news = []
    strong_keywords_news = []

    for i, news in enumerate(news_items[:15], 1):
        title = news.get('title', '')
        description = news.get('description', '')[:150]

        # TextBlob sentiment
        blob = TextBlob(title)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            sentiment_counts['positive'] += 1
            sentiment_label = 'POS'
        elif polarity < -0.1:
            sentiment_counts['negative'] += 1
            sentiment_label = 'NEG'
        else:
            sentiment_counts['neutral'] += 1
            sentiment_label = 'NEU'

        text_lower = (title + ' ' + description).lower()

        # Détecter montants
        billions = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
        millions = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)

        if billions:
            big_money_news.append({
                'index': i,
                'title': title,
                'amount': f"${billions[0]}B"
            })

        # Détecter keywords forts
        strong_pos = ['beats expectations', 'record profit', 'surge', 'soars', 'breakthrough', 'invests']
        strong_neg = ['bankruptcy', 'crash', 'scandal', 'plunge', 'earnings miss']

        for kw in strong_pos:
            if kw in text_lower:
                strong_keywords_news.append({
                    'index': i,
                    'keyword': kw,
                    'type': 'POSITIVE',
                    'title': title[:60]
                })

        for kw in strong_neg:
            if kw in text_lower:
                strong_keywords_news.append({
                    'index': i,
                    'keyword': kw,
                    'type': 'NEGATIVE',
                    'title': title[:60]
                })

        print(f"{i:2}. [{sentiment_label}] {title[:70]}")
        if billions:
            print(f"    $ ${billions[0]}B detected")
        if description:
            print(f"    {description[:100]}...")

    print(f"\n--- SENTIMENT DISTRIBUTION ---")
    print(f"Positive: {sentiment_counts['positive']}/15")
    print(f"Negative: {sentiment_counts['negative']}/15")
    print(f"Neutral:  {sentiment_counts['neutral']}/15")

    if big_money_news:
        print(f"\n--- BIG MONEY NEWS ({len(big_money_news)}) ---")
        for money in big_money_news:
            print(f"  {money['index']}. {money['amount']}: {money['title'][:60]}")

    if strong_keywords_news:
        print(f"\n--- STRONG KEYWORDS ({len(strong_keywords_news)}) ---")
        for kw in strong_keywords_news:
            print(f"  {kw['index']}. [{kw['type']}] '{kw['keyword']}': {kw['title']}")

    print(f"\n--- SCORE CLAUDE ---")
    print(f"Claude Score: {claude_score}/100")

    # Interpréter
    if claude_score >= 70:
        interpretation = "POSITIF - Claude voit des signaux forts positifs"
    elif claude_score >= 55:
        interpretation = "LEGEREMENT POSITIF - Tendance favorable"
    elif claude_score >= 45:
        interpretation = "NEUTRE - Pas de direction claire"
    elif claude_score >= 30:
        interpretation = "LEGEREMENT NEGATIF - Tendance defavorable"
    else:
        interpretation = "NEGATIF - Claude voit des signaux forts negatifs"

    print(f"Interpretation: {interpretation}")

    # Calculer ce que TextBlob devrait faire
    print(f"\n--- RECOMMANDATION POUR TEXTBLOB ---")

    ratio_positive = sentiment_counts['positive'] / 15
    ratio_negative = sentiment_counts['negative'] / 15

    print(f"Ratio Pos/Neg: {ratio_positive:.2f} / {ratio_negative:.2f}")

    if big_money_news:
        print(f"Gros montants: {len(big_money_news)} news")
        print(f"  => Boost modere conseille (pas trop)")

    if strong_keywords_news:
        pos_kw = sum(1 for kw in strong_keywords_news if kw['type'] == 'POSITIVE')
        neg_kw = sum(1 for kw in strong_keywords_news if kw['type'] == 'NEGATIVE')
        print(f"Keywords forts: {pos_kw} positifs, {neg_kw} negatifs")

    # Suggestion de score TextBlob
    suggested_range = None
    if claude_score >= 70:
        suggested_range = f"{claude_score-5:.0f}-{claude_score+5:.0f}"
    elif claude_score >= 55:
        suggested_range = f"{claude_score-5:.0f}-{claude_score+5:.0f}"
    else:
        suggested_range = f"{claude_score-5:.0f}-{claude_score+5:.0f}"

    print(f"\nTextBlob devrait viser: {suggested_range} (Claude: {claude_score:.0f})")

    return {
        'symbol': symbol,
        'claude_score': claude_score,
        'news_count': len(news_items),
        'sentiment_dist': sentiment_counts,
        'big_money_count': len(big_money_news),
        'strong_kw_count': len(strong_keywords_news),
        'ratio_positive': ratio_positive,
        'ratio_negative': ratio_negative
    }


async def main():
    print("="*100)
    print("ANALYSE APPROFONDIE DES DIVERGENCES")
    print("="*100)
    print("\nObjectif: Comprendre ce que Claude voit pour calibrer TextBlob")

    # Analyser les cas les plus divergents
    test_symbols = [
        ('AAPL', 'PIRE CAS V6: 84.6 vs 65.0 (diff 19.6)'),
        ('CSCO', 'GRANDE DIVERGENCE V6: 83.8 vs 70.0 (diff 13.8)'),
        ('AVGO', 'DIVERGENCE V6: 93.8 vs 85.0 (diff 8.8)'),
        ('TTE.PA', 'CAS DIFFICILE V6: 62.9 vs 55.0 (diff 7.9)'),
        ('AMZN', 'SOUS-ESTIME V6: 57.2 vs 65.0 (diff 7.8)'),
        ('ORCL', 'BON CAS V6: 62.4 vs 65.0 (diff 2.6)')
    ]

    analyzer = HistoricalNewsAnalyzer()
    scorer = UltraTextBlob()

    results = []

    for symbol, description in test_symbols:
        print(f"\n{description}")
        result = await deep_analyze_divergence(symbol, scorer, analyzer)
        if result:
            results.append(result)
        await asyncio.sleep(1)

    await scorer.close()
    await analyzer.close()

    # RESUME GLOBAL
    print("\n" + "="*100)
    print("RESUME GLOBAL - PATTERNS IDENTIFIES")
    print("="*100)

    for r in results:
        print(f"\n{r['symbol']}: Claude = {r['claude_score']:.0f}")
        print(f"  Pos/Neg ratio: {r['ratio_positive']:.2f}/{r['ratio_negative']:.2f}")
        print(f"  Big money: {r['big_money_count']}")
        print(f"  Strong keywords: {r['strong_kw_count']}")

    print("\n" + "="*100)
    print("CONCLUSIONS POUR V8:")
    print("="*100)
    print("""
1. Ne pas SUR-BOOSTER les gros montants ($XB)
   - AAPL avait $53B mais Claude = 65 (pas 84!)
   - AVGO avait $500B mais Claude = 85 (pas 93!)
   - Boost modere: 1.5-2.0 max (pas 3-5)

2. Ponderer par le RATIO positif/negatif
   - Si 8/15 positives => score ~60-70
   - Si 12/15 positives => score ~75-85
   - Pas juste compter, mais ratio!

3. Titre = 2x importance (pas 3x ou 4x)
   - Claude ne sur-pondere pas autant le titre

4. Formule finale: plus lineaire
   - Pas de boost progressif excessif
   - Rester proche de la moyenne ponderee
    """)


if __name__ == "__main__":
    asyncio.run(main())
