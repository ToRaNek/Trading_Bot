"""
TextBlob V10 - FINAL CORRECTION
Basée sur V9 mais avec corrections critiques

PROBLEMES V9:
- Ecart moyen: 10.3 pts (vs objectif 5-6)
- BNP.PA: V9=74.6, Claude=25 (diff 49.6)
- SAF.PA: V9=65.0, Claude=15 (diff 50.0)
- Pattern: V9 ne peut pas descendre assez bas (<30)

FIXES V10:
1. Retirer boost expansion_context (créait faux positifs)
2. Augmenter penalty negative (×70 vs ×60)
3. Augmenter range keywords (-20/+20 vs -15/+15)
4. Ajouter penalty si trop de news neutres sans keywords positifs
5. Détecter keywords négatifs "hidden" (faibles mais importants)
"""

from textblob import TextBlob
import numpy as np
import re


def score_news_v10_final(news_items: list) -> float:
    """
    VERSION 10 - FINAL
    Objectif: <6 pts d'écart moyen sur 40 actions
    """
    if not news_items:
        return 50.0

    # Keywords enrichis pour V10
    positive_keywords = {
        # TRES FORTS
        'beats expectations': 5.0,
        'earnings beat': 5.0,
        'record profit': 5.0,
        'surge': 5.0,
        'soars': 5.0,
        'rockets': 4.5,

        # FORTS
        'breakthrough': 4.0,
        'upgrade': 3.5,
        'invests': 3.5,
        'investment': 3.0,

        # MOYENS
        'expansion': 2.5,
        'profit': 2.2,
        'growth': 2.0,
        'acquisition': 2.5,
        'deal': 2.2,
        'wins': 2.0
    }

    negative_keywords = {
        # TRES FORTS
        'bankruptcy': 5.0,
        'crash': 4.5,
        'scandal': 4.0,
        'earnings miss': 5.0,
        'fraud': 4.5,

        # FORTS
        'downgrade': 3.5,
        'plunge': 3.5,
        'lawsuit': 2.5,
        'loss': 2.0,
        'losses': 2.0,

        # MOYENS
        'decline': 1.5,
        'falls': 1.2,
        'drops': 1.2,

        # NOUVEAUX: keywords "hidden" négatifs (V10)
        'tumbles': 2.0,
        'slump': 2.0,
        'slumps': 2.0,
        'weak': 1.5,
        'weakens': 1.5,
        'miss': 2.5,
        'misses': 2.5,
        'concern': 1.0,
        'concerns': 1.0,
        'struggle': 1.8,
        'struggles': 1.8
    }

    # Analyser top 15
    top_15 = news_items[:15]

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    keyword_signals = []
    money_detected = False
    has_strong_positive_kw = False  # V10: tracker keywords positifs forts

    for article in top_15:
        title = article.get('title', '')
        summary = article.get('summary', '')

        # Sentiment
        title_blob = TextBlob(title)
        title_pol = title_blob.sentiment.polarity * 2.0

        summary_blob = TextBlob(summary)
        summary_pol = summary_blob.sentiment.polarity

        avg_pol = (title_pol + summary_pol) / 3.0

        # Seuil (inchangé de V9)
        if avg_pol > 0.12:
            positive_count += 1
        elif avg_pol < -0.12:
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
                if weight >= 4.0:  # Keywords très forts
                    has_strong_positive_kw = True

        for keyword, weight in negative_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                kw_score -= weight * count

        if kw_score != 0:
            keyword_signals.append(kw_score)

        # Montants
        if re.search(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower):
            money_detected = True

    # === FORMULE V10 (corrigée) ===

    total = len(top_15)
    ratio_positive = positive_count / total
    ratio_negative = negative_count / total
    ratio_neutral = neutral_count / total

    base_score = 50.0

    # Boost/penalty par ratio (V10: multiplicateur augmenté pour négatif)
    if ratio_positive > ratio_negative:
        pos_boost = (ratio_positive - ratio_negative) * 60  # Inchangé
        base_score += pos_boost

    elif ratio_negative > ratio_positive:
        neg_penalty = (ratio_negative - ratio_positive) * 80  # V9: 60 → V10: 80
        base_score -= neg_penalty

    # V10: RETRAIT du boost expansion_context (source de faux positifs)
    # (lignes 144-146 de V9 supprimées)

    # Keywords (V10: range augmenté)
    if keyword_signals:
        avg_kw = np.mean(keyword_signals)
        kw_adjustment = avg_kw * 2.5
        kw_adjustment = max(-20, min(20, kw_adjustment))  # V9: -15/+15 → V10: -20/+20
        base_score += kw_adjustment

    # V10: NOUVELLE PENALTY si trop neutre sans keywords positifs forts
    if ratio_neutral >= 0.70 and not has_strong_positive_kw:
        # Si 70%+ neutres ET pas de keywords très positifs → probablement ennuyeux/négatif
        neutral_penalty = (ratio_neutral - 0.70) * 25  # V10: seuil réduit + penalty augmentée
        base_score -= neutral_penalty

    # Montants (garde léger, inchangé)
    if money_detected and ratio_positive > 0.15:
        base_score += 4.0

    score = max(0, min(100, base_score))

    return score


# Test
if __name__ == "__main__":
    print("="*80)
    print("TEST V10 FINAL")
    print("="*80)

    # Test cas problématiques V9
    test_cases = [
        {
            'name': 'BNP.PA simulé (devrait être <30)',
            'news': [
                {'title': 'Banking sector faces concerns', 'summary': 'Market uncertainty'},
                {'title': 'European banks struggle', 'summary': 'Economic headwinds'},
                {'title': 'Financial sector news', 'summary': 'Market update'},
                {'title': 'Bank stocks drop', 'summary': 'Investor concern'},
            ] * 4,
            'expected': '<30 (Claude: 25, V9: 74.6)'
        },
        {
            'name': 'SAF.PA simulé (devrait être <20)',
            'news': [
                {'title': 'Aerospace industry news', 'summary': 'Market report'},
                {'title': 'Defense sector update', 'summary': 'Industry analysis'},
                {'title': 'Manufacturing data', 'summary': 'Economic indicators'},
            ] * 5,
            'expected': '<20 (Claude: 15, V9: 65.0)'
        },
        {
            'name': 'AMZN (bon cas, devrait rester ~70)',
            'news': [
                {'title': 'Amazon expands logistics', 'summary': 'Growth strategy'},
                {'title': 'E-commerce growth continues', 'summary': 'Strong results'},
                {'title': 'Tech stocks gain', 'summary': 'Positive momentum'},
            ] * 5,
            'expected': '~70 (Claude: 70, V9: 70.8)'
        }
    ]

    for test in test_cases:
        score = score_news_v10_final(test['news'])
        print(f"\n{test['name']}")
        print(f"  Score V10: {score:.1f}/100")
        print(f"  Attendu:   {test['expected']}")
