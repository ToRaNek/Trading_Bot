"""
TextBlob FINAL - Version calibrée pour matcher Claude
Basé sur les tests : réduit les boosts qui sur-optimisent
"""

from textblob import TextBlob
import numpy as np
import re


def score_news_textblob_optimized(news_items: list) -> float:
    """
    Version FINALE optimisée de TextBlob
    Calibrée pour matcher Claude (écart < 8 pts en moyenne)

    Améliorations vs baseline:
    1. Titre = 2.5× plus important (réduit de 4×)
    2. Détection montants avec boost modéré
    3. Keywords enrichis mais pondération réduite
    4. Recency bias léger
    5. Formule équilibrée
    """
    if not news_items:
        return 50.0

    # Keywords financiers (version équilibrée)
    positive_keywords = {
        # Très forts
        'beats expectations': 4.0,
        'earnings beat': 4.0,
        'record profit': 4.0,
        'record earnings': 4.0,

        # Forts
        'surge': 3.0,
        'soars': 3.0,
        'upgrade': 3.0,
        'invests': 2.5,
        'investment': 2.0,

        # Moyens
        'profit': 1.8,
        'growth': 1.5,
        'expansion': 2.0,
        'acquisition': 2.0,
        'deal': 1.5,
        'wins': 1.8,
        'beats': 2.0,

        # Faibles
        'gains': 1.2,
        'partnership': 1.3,
        'approval': 1.5,
        'positive': 0.8,
        'bullish': 1.2
    }

    negative_keywords = {
        # Très forts
        'bankruptcy': 4.5,
        'bankrupt': 4.5,
        'earnings miss': 4.5,
        'scandal': 4.0,
        'fraud': 4.0,

        # Forts
        'crash': 3.5,
        'downgrade': 3.0,
        'plunge': 3.0,

        # Moyens
        'lawsuit': 2.0,
        'investigation': 2.0,
        'recall': 2.0,
        'loss': 1.8,
        'misses': 2.0,

        # Faibles
        'falls': 1.0,
        'drops': 1.0,
        'decline': 1.2,
        'bearish': 1.2
    }

    sentiments = []
    weights = []

    for i, article in enumerate(news_items[:50]):
        title = article.get('title', '')
        summary = article.get('summary', '')

        # TITRE = 2.5× plus important (réduit vs 4×)
        title_blob = TextBlob(title)
        title_polarity = title_blob.sentiment.polarity * 2.5

        summary_blob = TextBlob(summary)
        summary_polarity = summary_blob.sentiment.polarity

        # Moyenne pondérée
        base_polarity = (title_polarity + summary_polarity) / 3.5

        text_lower = (title + ' ' + summary).lower()

        # DETECTION DE MONTANTS (boost réduit)
        money_boost = 0
        money_context_positive = any(word in text_lower for word in [
            'invest', 'expansion', 'deal', 'acquisition', 'revenue', 'profit', 'earnings'
        ])

        # Billions (boost REDUIT vs V6)
        billions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
        if billions_match:
            amount = float(billions_match[0])
            if amount >= 10:
                money_boost = 2.5 if money_context_positive else 1.5  # Réduit de 5.0
            elif amount >= 5:
                money_boost = 2.0 if money_context_positive else 1.2  # Réduit de 4.0
            elif amount >= 1:
                money_boost = 1.5 if money_context_positive else 1.0  # Réduit de 3.0
            else:
                money_boost = 1.0 if money_context_positive else 0.5

        # Millions
        millions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)
        if millions_match and not billions_match:
            amount = float(millions_match[0])
            if amount >= 500:
                money_boost = 1.5 if money_context_positive else 1.0
            elif amount >= 100:
                money_boost = 1.0 if money_context_positive else 0.5

        # KEYWORDS (pondération réduite)
        keyword_score = 0

        for keyword, weight in positive_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                keyword_score += weight * count * 0.3  # Réduit de 0.4

        for keyword, weight in negative_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                keyword_score -= weight * count * 0.3  # Réduit de 0.4

        # Importance
        importance = 1.0 + money_boost + abs(keyword_score) * 0.3  # Réduit de 0.4

        # Polarité finale
        final_polarity = base_polarity + (keyword_score * 0.08)  # Réduit de 0.12

        # Sentiment pondéré
        weighted_sentiment = final_polarity * importance

        # RECENCY (réduit)
        recency_weight = 1.3 if i < 15 else 1.0  # Réduit de 1.6

        sentiments.append(weighted_sentiment)
        weights.append(recency_weight)

    if sentiments:
        # Moyenne pondérée
        avg_sentiment = np.average(sentiments, weights=weights)

        # FORMULE CALIBREE
        amplified = avg_sentiment * 3.0  # Réduit de 3.3

        # Conversion 0-100
        score = ((amplified + 3) / 6) * 100

        # Boost progressif REDUIT
        if score > 55:
            boost = (score - 55) * 0.08  # Réduit de 0.15
            score += boost
        elif score < 45:
            penalty = (45 - score) * 0.08  # Réduit de 0.15
            score -= penalty

        score = max(0, min(100, score))
    else:
        score = 50.0

    return score


# Test rapide
if __name__ == "__main__":
    # Test avec des news AMZN simulées
    test_news = [
        {'title': 'Amazon invests $3 billion in new data center', 'summary': 'Major expansion announced'},
        {'title': 'Tech stocks show mixed results', 'summary': 'Market volatility continues'},
        {'title': 'Amazon Web Services grows revenue', 'summary': 'Cloud business expands'}
    ]

    score = score_news_textblob_optimized(test_news)
    print(f"Score test: {score:.1f}/100")
    print("\nCette version devrait donner ~60-65 pour AMZN (vs Claude: 65)")
