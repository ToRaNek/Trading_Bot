"""
TextBlob V9 - REFINED VERSION
Basée sur V8 mais avec ajustements fins

V8 Résultats:
- CSCO: 0.2 pts ✅ PARFAIT!
- AMZN: 5.0 pts ✅ BON!
- AVGO: 9.2 pts (sous-estime)
- TTE.PA: 9.7 pts (sous-estime)
- Moyenne: 10.5 pts

V9 Ajustements:
1. Boost keywords "surge" plus fort (AVGO)
2. Détection meilleure du contexte neutre→positif (TTE.PA)
3. Affiner ratio formula
"""

from textblob import TextBlob
import numpy as np
import re


def score_news_v9_refined(news_items: list) -> float:
    """
    VERSION 9 - REFINED
    Objectif: <6 pts d'écart moyen
    """
    if not news_items:
        return 50.0

    # Keywords (boost ajusté pour V9)
    positive_keywords = {
        # TRES FORTS (boost plus fort pour surge, soars)
        'beats expectations': 5.0,
        'earnings beat': 5.0,
        'record profit': 5.0,
        'surge': 5.0,  # Augmenté de 4.0 (AVGO fix)
        'soars': 5.0,  # Augmenté de 4.0
        'rockets': 4.5,

        # FORTS
        'breakthrough': 4.0,
        'upgrade': 3.5,
        'invests': 3.5,  # Augmenté de 3.0
        'investment': 3.0,  # Augmenté de 2.5

        # MOYENS
        'expansion': 2.5,
        'profit': 2.2,  # Augmenté de 2.0
        'growth': 2.0,  # Augmenté de 1.8
        'acquisition': 2.5,
        'deal': 2.2,  # Augmenté de 2.0
        'wins': 2.0
    }

    negative_keywords = {
        'bankruptcy': 5.0,
        'crash': 4.5,
        'scandal': 4.0,
        'earnings miss': 5.0,
        'downgrade': 3.5,
        'plunge': 3.5,
        'lawsuit': 2.5,
        'loss': 2.0,
        'decline': 1.5
    }

    # Analyser top 15
    top_15 = news_items[:15]

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    keyword_signals = []
    money_detected = False
    expansion_context = False  # Nouveau: détecte contexte expansion

    for article in top_15:
        title = article.get('title', '')
        summary = article.get('summary', '')

        # Sentiment
        title_blob = TextBlob(title)
        title_pol = title_blob.sentiment.polarity * 2.0

        summary_blob = TextBlob(summary)
        summary_pol = summary_blob.sentiment.polarity

        avg_pol = (title_pol + summary_pol) / 3.0

        # Seuil ajusté pour V9
        if avg_pol > 0.12:  # Réduit de 0.15 (plus sensible)
            positive_count += 1
        elif avg_pol < -0.12:  # Réduit de -0.15
            negative_count += 1
        else:
            neutral_count += 1

        text_lower = (title + ' ' + summary).lower()

        # Keywords
        kw_score = 0
        for keyword, weight in positive_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                kw_score += weight * count

        for keyword, weight in negative_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                kw_score -= weight * count

        if kw_score != 0:
            keyword_signals.append(kw_score)

        # Montants
        if re.search(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower):
            money_detected = True

        # Contexte expansion (TTE.PA fix)
        if any(word in text_lower for word in ['expansion', 'expands', 'partnership', 'deal', 'contract']):
            expansion_context = True

    # === FORMULE V9 (ajustée) ===

    total = len(top_15)
    ratio_positive = positive_count / total
    ratio_negative = negative_count / total
    ratio_neutral = neutral_count / total

    base_score = 50.0

    # Boost/penalty par ratio (formule ajustée)
    if ratio_positive > ratio_negative:
        # Augmente le multiplicateur pour donner des scores plus hauts
        pos_boost = (ratio_positive - ratio_negative) * 60  # V8: 50
        base_score += pos_boost

    elif ratio_negative > ratio_positive:
        neg_penalty = (ratio_negative - ratio_positive) * 60  # V8: 50
        base_score -= neg_penalty

    # Si 100% neutre mais contexte expansion (TTE.PA fix)
    if ratio_neutral >= 0.90 and expansion_context:
        base_score += 10  # Boost pour contexte positif même si neutre

    # Keywords (ajustement plus fort)
    if keyword_signals:
        avg_kw = np.mean(keyword_signals)
        kw_adjustment = avg_kw * 2.5  # V8: 2.0 (augmenté)
        kw_adjustment = max(-15, min(15, kw_adjustment))  # V8: -10,+10
        base_score += kw_adjustment

    # Montants (garde léger)
    if money_detected and ratio_positive > 0.15:  # V8: 0.20 (réduit seuil)
        base_score += 4.0

    score = max(0, min(100, base_score))

    return score


# Test
if __name__ == "__main__":
    print("="*80)
    print("TEST V9 REFINED")
    print("="*80)

    # Test avec news simulées basées sur l'analyse
    test_cases = [
        {
            'name': 'AVGO (3x surge)',
            'news': [
                {'title': 'Nvidia stock surges on earnings', 'summary': 'Strong results'},
                {'title': 'Tech sector surges', 'summary': 'Market rally'},
                {'title': 'Chip stocks surge', 'summary': 'AI boom'},
                {'title': 'Market shows strength', 'summary': 'Positive momentum'},
            ] * 4,
            'expected': '80-85 (Claude: 85, V8: 75.8)'
        },
        {
            'name': 'TTE.PA (100% neutre + expansion)',
            'news': [
                {'title': 'TotalEnergies expands in Africa', 'summary': 'New partnership'},
                {'title': 'Oil company signs deal', 'summary': 'Strategic contract'},
                {'title': 'Energy sector news', 'summary': 'Market update'},
            ] * 5,
            'expected': '60-65 (Claude: 65, V8: 55.3)'
        }
    ]

    for test in test_cases:
        score = score_news_v9_refined(test['news'])
        print(f"\n{test['name']}")
        print(f"  Score V9: {score:.1f}/100")
        print(f"  Attendu:  {test['expected']}")
