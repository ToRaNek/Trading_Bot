"""
TextBlob V11 - INTENSITY + DIRECTION SYSTEM
Nouveau système demandé par l'utilisateur:
- Score 0-100 = INTENSITÉ du sentiment (50 = neutre)
- Direction = POSITIF ou NÉGATIF (séparé du score)

Exemples:
- 88/100 POSITIF = sentiment très fortement haussier (bullish)
- 92/100 NÉGATIF = sentiment très fortement baissier (bearish)
- 50/100 NEUTRE = pas de sentiment clair

Améliorations par rapport à V9:
1. Meilleurs mots-clés pour détecter les catastrophes (recall, emergency, directive, etc.)
2. Système séparé intensité/direction au lieu d'un seul score
3. Retourne un tuple (intensity, direction, explanation)
"""

from textblob import TextBlob
import numpy as np
import re
from typing import Tuple


def score_news_v11_intensity_direction(news_items: list) -> Tuple[float, str, str]:
    """
    VERSION 11 - INTENSITY + DIRECTION

    Returns:
        (intensity: float 0-100, direction: str, explanation: str)

    Direction values: "POSITIF", "NÉGATIF", "NEUTRE"
    """
    if not news_items:
        return 50.0, "NEUTRE", "Aucune news disponible"

    # ========== KEYWORDS AMÉLIORÉS ==========

    # TRÈS FORTEMENT POSITIFS (bullish)
    ultra_positive_keywords = {
        'beats expectations': 6.0,
        'earnings beat': 6.0,
        'record profit': 6.0,
        'record earnings': 6.0,
        'blockbuster': 5.5,
        'massive growth': 5.5,
    }

    # FORTEMENT POSITIFS
    strong_positive_keywords = {
        'surge': 5.0,
        'surges': 5.0,
        'soars': 5.0,
        'rockets': 4.5,
        'skyrockets': 5.0,
        'breakthrough': 4.5,
        'innovation': 3.5,
    }

    # POSITIFS MOYENS
    moderate_positive_keywords = {
        'upgrade': 3.5,
        'upgraded': 3.5,
        'invests': 3.5,
        'investment': 3.0,
        'expansion': 2.8,
        'expands': 2.8,
        'profit': 2.5,
        'growth': 2.3,
        'acquisition': 2.8,
        'deal': 2.5,
        'partnership': 2.5,
        'wins': 2.3,
        'approval': 3.0,
        'approved': 3.0,
        'bullish': 3.0,
        'gains': 2.0,
        'jumps': 2.5,
    }

    # TRÈS FORTEMENT NÉGATIFS (bearish) - AMÉLIORÉ pour Airbus
    ultra_negative_keywords = {
        'bankruptcy': 6.0,
        'bankrupt': 6.0,
        'crash': 5.5,
        'crashes': 5.5,
        'collapsed': 5.5,
        'emergency': 5.0,  # NOUVEAU (Airbus)
        'emergency directive': 6.0,  # NOUVEAU (Airbus FAA)
        'airworthiness directive': 5.5,  # NOUVEAU (Airbus FAA)
        'flight control': 5.0,  # NOUVEAU (Airbus safety)
        'safety concern': 5.0,  # NOUVEAU
        'scandal': 5.0,
        'fraud': 5.0,
    }

    # FORTEMENT NÉGATIFS
    strong_negative_keywords = {
        'recall': 4.5,  # AMÉLIORÉ (était 2.0 dans V9)
        'recalls': 4.5,
        'precautionary': 3.5,  # NOUVEAU (Airbus)
        'grounded': 4.5,  # NOUVEAU (aviation)
        'suspended': 4.0,
        'halted': 4.0,
        'earnings miss': 5.0,
        'misses': 4.0,
        'plunge': 4.0,
        'plunges': 4.0,
        'downgrade': 4.0,
        'downgraded': 4.0,
        'investigation': 3.5,
        'probe': 3.5,
    }

    # NÉGATIFS MOYENS
    moderate_negative_keywords = {
        'lawsuit': 3.0,
        'litigation': 3.0,
        'loss': 2.5,
        'losses': 2.5,
        'decline': 2.0,
        'declines': 2.0,
        'falls': 2.0,
        'drops': 2.0,
        'slides': 2.0,
        'tumbles': 2.5,
        'slumps': 2.5,
        'bearish': 3.0,
        'concern': 2.0,
        'concerns': 2.0,
        'worry': 2.0,
        'risk': 1.8,
        'pressure': 1.5,  # NOUVEAU (delivery pressure)
    }

    # Combine all keywords for easier access
    all_positive_keywords = {**ultra_positive_keywords, **strong_positive_keywords, **moderate_positive_keywords}
    all_negative_keywords = {**ultra_negative_keywords, **strong_negative_keywords, **moderate_negative_keywords}

    # ========== ANALYSE DES NEWS ==========

    top_15 = news_items[:15]

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    keyword_signals = []
    sentiment_sum = 0

    for article in top_15:
        title = article.get('title', '')
        summary = article.get('summary', '')

        # Sentiment TextBlob (baseline)
        title_blob = TextBlob(title)
        title_pol = title_blob.sentiment.polarity * 2.0

        summary_blob = TextBlob(summary)
        summary_pol = summary_blob.sentiment.polarity

        avg_pol = (title_pol + summary_pol) / 3.0
        sentiment_sum += avg_pol

        # Count sentiment distribution
        if avg_pol > 0.12:
            positive_count += 1
        elif avg_pol < -0.12:
            negative_count += 1
        else:
            neutral_count += 1

        # Keyword detection
        text_lower = (title + ' ' + summary).lower()

        kw_score = 0
        for keyword, weight in all_positive_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                kw_score += weight * count

        for keyword, weight in all_negative_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                kw_score -= weight * count

        if kw_score != 0:
            keyword_signals.append(kw_score)

    # ========== CALCUL INTENSITÉ + DIRECTION ==========

    total = len(top_15)
    ratio_positive = positive_count / total
    ratio_negative = negative_count / total
    ratio_neutral = neutral_count / total

    # Calculer la direction dominante
    # On utilise aussi les keywords pour déterminer la direction (pas juste les ratios)
    keyword_bias = 0
    if keyword_signals:
        keyword_bias = np.mean(keyword_signals)

    # Déterminer direction basée sur ratio ET keywords
    if ratio_positive > ratio_negative + 0.15 or (ratio_positive >= ratio_negative and keyword_bias > 2.0):
        direction = "POSITIF"
    elif ratio_negative > ratio_positive + 0.15 or (ratio_negative >= ratio_positive and keyword_bias < -2.0):
        direction = "NÉGATIF"
    else:
        direction = "NEUTRE"

    # Calculer l'intensité (0-100, 50 = neutre)
    intensity = 50.0

    # Contribution du ratio pos/neg
    sentiment_delta = abs(ratio_positive - ratio_negative)
    intensity_from_ratio = sentiment_delta * 50  # Max +50

    # Contribution des keywords (très important!)
    intensity_from_keywords = 0
    if keyword_signals:
        avg_kw = np.mean(keyword_signals)
        intensity_from_keywords = abs(avg_kw) * 4.0  # Amplifié pour détecter les cas forts
        intensity_from_keywords = min(40, intensity_from_keywords)  # Cap à +40

    # Combiner les intensités
    total_intensity = intensity_from_ratio + intensity_from_keywords

    # Appliquer l'intensité (50 = neutre, 100 = max)
    intensity = 50 + total_intensity
    intensity = max(0, min(100, intensity))

    # Si direction neutre, ramener intensité vers 50
    if direction == "NEUTRE":
        intensity = 50 + (intensity - 50) * 0.3  # Réduire l'intensité pour neutre

    # ========== EXPLICATION ==========

    explanation_parts = []
    explanation_parts.append(f"{positive_count}+ {negative_count}- {neutral_count}n")

    if keyword_signals:
        avg_kw = np.mean(keyword_signals)
        if abs(avg_kw) > 2:
            explanation_parts.append(f"Keywords: {avg_kw:+.1f}")

    if ratio_positive > 0.6:
        explanation_parts.append("Majorité positive")
    elif ratio_negative > 0.6:
        explanation_parts.append("Majorité négative")

    explanation = " | ".join(explanation_parts)

    return intensity, direction, explanation


# Test
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("="*80)
    print("TEST V11 - INTENSITY + DIRECTION SYSTEM")
    print("="*80)

    from datetime import datetime

    # Test 1: Airbus catastrophe (rappel massif)
    airbus_news = [
        {
            'title': 'Airbus calls for immediate software upgrade to A320 aircraft amid control concerns',
            'summary': 'Emergency directive. Airbus requests airlines take immediate action to upgrade software warning there could be an issue with flight controls. About 6,000 single-aisle planes need repairs.',
        },
        {
            'title': 'Airlines race to fix Airbus planes after warning solar radiation could cause pilots to lose control',
            'summary': 'Analysis revealed that intense solar radiation may corrupt data critical to the functioning of flight controls',
        },
        {
            'title': 'FAA issues Emergency Airworthiness Directive for Airbus A319 and A320/321 airplanes',
            'summary': 'The FAA is ordering operators to perform the software upgrade before the airplane flies again',
        },
        {
            'title': 'Airbus update on A320 Family precautionary fleet action',
            'summary': 'Precautionary measures implemented across approximately 6,000 aircraft following flight control issue discovery',
        },
        {
            'title': 'Airbus faces new operational adjustments for aircraft with Pratt & Whitney engines in extreme cold',
            'summary': 'The planemaker introduced restrictions for take-offs in freezing fog. Affects A320neo family',
        },
    ] * 3

    intensity, direction, explanation = score_news_v11_intensity_direction(airbus_news)

    print(f"\n📰 TEST 1: AIRBUS (rappel d'urgence 6000 avions)")
    print(f"   Résultat V11: {intensity:.0f}/100 {direction}")
    print(f"   Explication: {explanation}")
    print(f"   Attendu: 85-95/100 NÉGATIF")
    print(f"   ✅" if (80 <= intensity <= 100 and direction == "NÉGATIF") else f"   ❌")

    # Test 2: News très positives
    positive_news = [
        {
            'title': 'Apple beats expectations with record earnings',
            'summary': 'Company reports breakthrough quarter with massive growth in all segments',
        },
        {
            'title': 'Stock surges on blockbuster results',
            'summary': 'Shares soar as investors react to record profit announcement',
        },
        {
            'title': 'Analysts upgrade rating citing innovation',
            'summary': 'Multiple upgrades as company wins major deal',
        },
    ] * 5

    intensity, direction, explanation = score_news_v11_intensity_direction(positive_news)

    print(f"\n📰 TEST 2: NEWS TRÈS POSITIVES")
    print(f"   Résultat V11: {intensity:.0f}/100 {direction}")
    print(f"   Explication: {explanation}")
    print(f"   Attendu: 80-95/100 POSITIF")
    print(f"   ✅" if (75 <= intensity <= 100 and direction == "POSITIF") else f"   ❌")

    # Test 3: News neutres
    neutral_news = [
        {
            'title': 'Company announces quarterly results',
            'summary': 'Results in line with expectations',
        },
        {
            'title': 'Market update for the sector',
            'summary': 'Trading continues in normal range',
        },
    ] * 7

    intensity, direction, explanation = score_news_v11_intensity_direction(neutral_news)

    print(f"\n📰 TEST 3: NEWS NEUTRES")
    print(f"   Résultat V11: {intensity:.0f}/100 {direction}")
    print(f"   Explication: {explanation}")
    print(f"   Attendu: 45-55/100 NEUTRE")
    print(f"   ✅" if (40 <= intensity <= 60 and direction == "NEUTRE") else f"   ❌")

    print("\n" + "="*80)
