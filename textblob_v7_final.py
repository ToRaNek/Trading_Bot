"""
TextBlob V7 - VERSION FINALE EQUILIBREE
Basée sur V6 mais avec ajustements pour réduire l'écart moyen

Problèmes V6 identifiés:
- AAPL: 84.6 vs 65.0 (trop optimiste de 19.6 pts)
- CSCO: 83.8 vs 70.0 (trop optimiste de 13.8 pts)
- AVGO: 93.8 vs 85.0 (trop optimiste de 8.8 pts)
- AMZN: 57.2 vs 65.0 (trop pessimiste de 7.8 pts)

Solution V7:
1. Réduire le boost des montants (principal problème)
2. Réduire l'importance du titre (de 4× à 3×)
3. Ajuster la formule finale pour être moins extrême
4. Garder la détection de keywords (bon)
"""

from textblob import TextBlob
import numpy as np
import re


def score_news_v7_final(news_items: list) -> float:
    """
    VERSION 7 FINALE - Équilibrée pour matcher Claude
    Objectif: < 8 pts d'écart moyen
    """
    if not news_items:
        return 50.0

    # Keywords enrichis (GARDE V6)
    positive_keywords = {
        # Très forts
        'beats expectations': 5.0,
        'earnings beat': 5.0,
        'record profit': 5.0,
        'record earnings': 5.0,
        'surge': 4.0,
        'soars': 4.0,
        'rockets': 4.0,
        'breakthrough': 4.0,

        # Forts
        'upgrade': 3.5,
        'outperform': 3.5,
        'buy rating': 3.5,
        'strong buy': 4.0,
        'invests': 3.0,
        'investment': 2.5,
        'expansion': 2.5,
        'expands': 2.5,

        # Moyens
        'profit': 2.0,
        'growth': 1.8,
        'gains': 1.5,
        'acquisition': 2.5,
        'deal': 2.0,
        'partnership': 2.0,
        'contract': 1.8,
        'wins': 2.0,
        'beats': 2.5,
        'jumps': 2.0,

        # Faibles
        'announces': 0.8,
        'launches': 1.2,
        'approval': 1.8,
        'positive': 1.0,
        'bullish': 1.5
    }

    negative_keywords = {
        # Très forts
        'bankruptcy': 5.0,
        'bankrupt': 5.0,
        'crash': 4.5,
        'scandal': 4.0,
        'fraud': 4.5,
        'earnings miss': 5.0,

        # Forts
        'downgrade': 3.5,
        'underperform': 3.5,
        'sell rating': 3.5,
        'plunge': 3.5,
        'plunges': 3.5,
        'tumbles': 3.0,

        # Moyens
        'lawsuit': 2.5,
        'investigation': 2.5,
        'probe': 2.0,
        'recall': 2.5,
        'loss': 2.0,
        'losses': 2.0,
        'misses': 2.5,

        # Faibles
        'falls': 1.2,
        'drops': 1.2,
        'decline': 1.5,
        'bearish': 1.5,
        'concern': 1.0,
        'worried': 1.0
    }

    sentiments = []
    weights = []

    for i, article in enumerate(news_items[:50]):
        title = article.get('title', '')
        summary = article.get('summary', '')

        # TITRE = 3× (REDUIT de 4× en V6)
        title_blob = TextBlob(title)
        title_polarity = title_blob.sentiment.polarity * 3.0

        summary_blob = TextBlob(summary)
        summary_polarity = summary_blob.sentiment.polarity

        # Moyenne pondérée
        base_polarity = (title_polarity + summary_polarity) / 4.0

        text_lower = (title + ' ' + summary).lower()

        # === DETECTION MONTANTS (BOOST REDUIT - principal fix) ===
        money_boost = 0
        money_context_positive = any(word in text_lower for word in [
            'invest', 'expansion', 'deal', 'contract', 'acquisition', 'revenue', 'profit', 'earnings'
        ])

        # Billions (REDUIT de moitié vs V6)
        billions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
        if billions_match:
            amount = float(billions_match[0])
            if amount >= 10:
                money_boost = 2.5 if money_context_positive else 1.5  # V6: 5.0/3.0
            elif amount >= 5:
                money_boost = 2.0 if money_context_positive else 1.2  # V6: 4.0/2.5
            elif amount >= 1:
                money_boost = 1.5 if money_context_positive else 1.0  # V6: 3.0/2.0
            else:
                money_boost = 1.0 if money_context_positive else 0.5  # V6: 2.0/1.0

        # Millions (REDUIT)
        millions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)
        if millions_match and not billions_match:
            amount = float(millions_match[0])
            if amount >= 1000:
                money_boost = 1.5 if money_context_positive else 1.0  # V6: 3.0/2.0
            elif amount >= 500:
                money_boost = 1.2 if money_context_positive else 0.8  # V6: 2.5/1.5
            elif amount >= 100:
                money_boost = 0.8 if money_context_positive else 0.5  # V6: 1.5/1.0

        # === KEYWORDS (PONDERATION REDUITE) ===
        keyword_score = 0

        for keyword, weight in positive_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                keyword_score += weight * count * 0.3  # V6: 0.4

        for keyword, weight in negative_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                keyword_score -= weight * count * 0.3  # V6: 0.4

        # === IMPORTANCE ===
        importance = 1.0 + money_boost + abs(keyword_score) * 0.25  # V6: 0.4

        # === POLARITE FINALE ===
        final_polarity = base_polarity + (keyword_score * 0.08)  # V6: 0.12

        # === SENTIMENT PONDERE ===
        weighted_sentiment = final_polarity * importance

        # === RECENCY (REDUIT) ===
        recency_weight = 1.4 if i < 15 else 1.0  # V6: 1.6

        sentiments.append(weighted_sentiment)
        weights.append(recency_weight)

    if sentiments:
        # Moyenne pondérée
        avg_sentiment = np.average(sentiments, weights=weights)

        # === FORMULE FINALE (AJUSTEE) ===
        amplified = avg_sentiment * 3.0  # V6: 3.3

        # Conversion 0-100
        score = ((amplified + 3) / 6) * 100

        # === BOOST PROGRESSIF (REDUIT) ===
        if score > 55:
            boost = (score - 55) * 0.10  # V6: 0.15
            score += boost
        elif score < 45:
            penalty = (45 - score) * 0.10  # V6: 0.15
            score -= penalty

        score = max(0, min(100, score))
    else:
        score = 50.0

    return score


# Test rapide
if __name__ == "__main__":
    # Test avec news simulées
    test_cases = [
        {
            'name': 'AMZN (avec $3B investment)',
            'news': [
                {'title': 'Amazon invests $3 billion in new data center', 'summary': 'Major expansion'},
                {'title': 'Tech stocks show mixed results', 'summary': 'Market volatility'},
                {'title': 'Amazon Web Services grows', 'summary': 'Cloud business expands'}
            ],
            'expected': '60-65 (Claude: 65)'
        },
        {
            'name': 'AAPL (avec $53B accessory market)',
            'news': [
                {'title': 'Apple Accessories Market grows to $53 billion', 'summary': 'Strong growth'},
                {'title': 'Apple stock shows gains', 'summary': 'Positive investor sentiment'},
                {'title': 'iPhone sales remain strong', 'summary': 'Q4 results positive'}
            ],
            'expected': '65-70 (Claude: 65, V6 etait 84.6)'
        },
        {
            'name': 'AVGO (avec $500B NVDA mentions)',
            'news': [
                {'title': 'Nvidia stock surges to $500B valuation', 'summary': 'Earnings beat'},
                {'title': 'Broadcom benefits from AI boom', 'summary': 'Chip demand high'},
                {'title': 'Semiconductor sector shows strength', 'summary': 'Industry growth'}
            ],
            'expected': '75-85 (Claude: 85, V6 etait 93.8)'
        }
    ]

    print("=" * 80)
    print("TEST RAPIDE V7")
    print("=" * 80)

    for test in test_cases:
        score = score_news_v7_final(test['news'])
        print(f"\n{test['name']}")
        print(f"  Score V7: {score:.1f}/100")
        print(f"  Attendu:  {test['expected']}")
