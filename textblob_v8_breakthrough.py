"""
TextBlob V8 - BREAKTHROUGH VERSION
Basée sur l'analyse approfondie des divergences

DÉCOUVERTES CLÉS:
- AAPL: $53B mais Claude=50 car ratio 0.40pos/0.13neg = NEUTRE
- CSCO: Pas de $, Claude=75 grâce aux keywords "surge", "invests"
- AVGO: 3x "surge" → Claude=85 (pas les montants $500B!)
- TTE.PA: 100% neutre mais Claude=65 → contexte

NOUVELLE APPROCHE V8:
1. Base sur RATIO pos/neg (pas compte absolu)
2. Boost montants TRÈS RÉDUIT (1.0-1.5 max)
3. Keywords = signal principal
4. Titre = 2× (pas plus)
5. Formule linéaire simple
"""

from textblob import TextBlob
import numpy as np
import re


def score_news_v8_breakthrough(news_items: list) -> float:
    """
    VERSION 8 - BREAKTHROUGH
    Formule complètement repensée basée sur l'analyse des divergences
    """
    if not news_items:
        return 50.0

    # Keywords (identiques V6 mais usage différent)
    positive_keywords = {
        'beats expectations': 5.0, 'earnings beat': 5.0, 'record profit': 5.0,
        'surge': 4.0, 'soars': 4.0, 'rockets': 4.0, 'breakthrough': 4.0,
        'upgrade': 3.5, 'invests': 3.0, 'investment': 2.5,
        'expansion': 2.5, 'profit': 2.0, 'growth': 1.8,
        'acquisition': 2.5, 'deal': 2.0, 'wins': 2.0
    }

    negative_keywords = {
        'bankruptcy': 5.0, 'crash': 4.5, 'scandal': 4.0,
        'earnings miss': 5.0, 'downgrade': 3.5, 'plunge': 3.5,
        'lawsuit': 2.5, 'loss': 2.0, 'decline': 1.5
    }

    # Analyser les 15 premières (ce que Claude voit)
    top_15 = news_items[:15]

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    keyword_signals = []
    money_detected = False

    for i, article in enumerate(top_15):
        title = article.get('title', '')
        summary = article.get('summary', '')

        # Sentiment de base (TITRE = 2×)
        title_blob = TextBlob(title)
        title_pol = title_blob.sentiment.polarity * 2.0

        summary_blob = TextBlob(summary)
        summary_pol = summary_blob.sentiment.polarity

        # Moyenne
        avg_pol = (title_pol + summary_pol) / 3.0

        # Compter (seuil plus strict)
        if avg_pol > 0.15:
            positive_count += 1
        elif avg_pol < -0.15:
            negative_count += 1
        else:
            neutral_count += 1

        text_lower = (title + ' ' + summary).lower()

        # Détecter keywords (signal fort)
        kw_score = 0
        for keyword, weight in positive_keywords.items():
            if keyword in text_lower:
                kw_score += weight

        for keyword, weight in negative_keywords.items():
            if keyword in text_lower:
                kw_score -= weight

        if kw_score != 0:
            keyword_signals.append(kw_score)

        # Détecter montants (flag seulement)
        if re.search(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower):
            money_detected = True

    # === NOUVELLE FORMULE BASÉE SUR RATIO ===

    # 1. Score de base = ratio pos/neg
    total = len(top_15)
    ratio_positive = positive_count / total
    ratio_negative = negative_count / total
    ratio_neutral = neutral_count / total

    # Formule ratio (inspirée de Claude)
    base_score = 50.0  # Neutre

    # Si majoritairement positif
    if ratio_positive > ratio_negative:
        # Échelle: 0.20 pos = +5, 0.40 pos = +10, 0.60 pos = +20, 0.80 pos = +30
        pos_boost = (ratio_positive - ratio_negative) * 50  # 0-50 pts
        base_score += pos_boost

    # Si majoritairement négatif
    elif ratio_negative > ratio_positive:
        neg_penalty = (ratio_negative - ratio_positive) * 50
        base_score -= neg_penalty

    # 2. Ajustement par keywords (signal fort)
    if keyword_signals:
        avg_kw = np.mean(keyword_signals)
        # Keywords donnent +/- 10 pts max
        kw_adjustment = avg_kw * 2.0  # Weight=5.0 → +10 pts
        kw_adjustment = max(-10, min(10, kw_adjustment))
        base_score += kw_adjustment

    # 3. Montants: boost TRÈS LEGER (problème principal V6!)
    if money_detected and ratio_positive > 0.20:
        # Seulement +3-5 pts (vs +15-20 en V6!)
        base_score += 4.0

    # 4. Recency léger (top 15 déjà pris en compte)
    # Pas besoin de recency bias supplémentaire

    # Limiter à 0-100
    score = max(0, min(100, base_score))

    return score


def score_news_v8_debug(news_items: list) -> dict:
    """Version debug pour comprendre le calcul"""
    if not news_items:
        return {'score': 50.0, 'details': 'No news'}

    positive_keywords = {
        'beats expectations': 5.0, 'earnings beat': 5.0, 'record profit': 5.0,
        'surge': 4.0, 'soars': 4.0, 'rockets': 4.0, 'breakthrough': 4.0,
        'upgrade': 3.5, 'invests': 3.0, 'investment': 2.5,
        'expansion': 2.5, 'profit': 2.0, 'growth': 1.8,
        'acquisition': 2.5, 'deal': 2.0, 'wins': 2.0
    }

    negative_keywords = {
        'bankruptcy': 5.0, 'crash': 4.5, 'scandal': 4.0,
        'earnings miss': 5.0, 'downgrade': 3.5, 'plunge': 3.5,
        'lawsuit': 2.5, 'loss': 2.0, 'decline': 1.5
    }

    top_15 = news_items[:15]
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    keyword_signals = []
    money_detected = False

    for article in top_15:
        title = article.get('title', '')
        summary = article.get('summary', '')

        title_blob = TextBlob(title)
        title_pol = title_blob.sentiment.polarity * 2.0

        summary_blob = TextBlob(summary)
        summary_pol = summary_blob.sentiment.polarity

        avg_pol = (title_pol + summary_pol) / 3.0

        if avg_pol > 0.15:
            positive_count += 1
        elif avg_pol < -0.15:
            negative_count += 1
        else:
            neutral_count += 1

        text_lower = (title + ' ' + summary).lower()

        kw_score = 0
        for keyword, weight in positive_keywords.items():
            if keyword in text_lower:
                kw_score += weight

        for keyword, weight in negative_keywords.items():
            if keyword in text_lower:
                kw_score -= weight

        if kw_score != 0:
            keyword_signals.append(kw_score)

        if re.search(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower):
            money_detected = True

    total = len(top_15)
    ratio_positive = positive_count / total
    ratio_negative = negative_count / total

    base_score = 50.0

    if ratio_positive > ratio_negative:
        pos_boost = (ratio_positive - ratio_negative) * 50
        base_score += pos_boost

    elif ratio_negative > ratio_positive:
        neg_penalty = (ratio_negative - ratio_positive) * 50
        base_score -= neg_penalty

    kw_adjustment = 0
    if keyword_signals:
        avg_kw = np.mean(keyword_signals)
        kw_adjustment = avg_kw * 2.0
        kw_adjustment = max(-10, min(10, kw_adjustment))
        base_score += kw_adjustment

    money_boost = 0
    if money_detected and ratio_positive > 0.20:
        money_boost = 4.0
        base_score += money_boost

    score = max(0, min(100, base_score))

    return {
        'score': score,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'ratio_pos': ratio_positive,
        'ratio_neg': ratio_negative,
        'keyword_adj': kw_adjustment,
        'money_boost': money_boost
    }


# Test
if __name__ == "__main__":
    print("="*80)
    print("TEST V8 BREAKTHROUGH")
    print("="*80)

    # Test AAPL (devrait donner ~50-55 vs Claude 50)
    aapl_news = [
        {'title': 'Apple Accessories Market grows to $53 billion', 'summary': 'Strong growth'},
        {'title': 'Apple stock shows gains', 'summary': 'Positive sentiment'},
        {'title': 'Warren Buffett sells Apple stake', 'summary': 'Portfolio rebalancing'},
        {'title': 'Tech stocks mixed', 'summary': 'Market volatility'},
        {'title': 'iPhone sales remain steady', 'summary': 'Q4 results'},
        {'title': 'Apple services grow', 'summary': 'Revenue up'},
    ] * 3  # Simuler 18 news

    result = score_news_v8_debug(aapl_news)
    print(f"\nAAPL simulation:")
    print(f"  Score: {result['score']:.1f}/100 (Claude: 50)")
    print(f"  Pos/Neg: {result['positive_count']}/{result['negative_count']}")
    print(f"  Ratio: {result['ratio_pos']:.2f}/{result['ratio_neg']:.2f}")
    print(f"  Keyword adj: {result['keyword_adj']:.1f}")
    print(f"  Money boost: {result['money_boost']:.1f}")

    # Test CSCO (devrait donner ~70-75 vs Claude 75)
    csco_news = [
        {'title': 'Cisco invests in AI startup', 'summary': 'Major investment'},
        {'title': 'Cisco stock gains', 'summary': 'Positive momentum'},
        {'title': 'Tech sector rises', 'summary': 'Market rally'},
        {'title': 'Cisco announces partnership', 'summary': 'Strategic deal'},
    ] * 4

    result2 = score_news_v8_debug(csco_news)
    print(f"\nCSCO simulation:")
    print(f"  Score: {result2['score']:.1f}/100 (Claude: 75)")
    print(f"  Pos/Neg: {result2['positive_count']}/{result2['negative_count']}")
    print(f"  Keyword adj: {result2['keyword_adj']:.1f}")
