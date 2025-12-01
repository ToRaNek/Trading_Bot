"""
Test d'intégration du système V11 (Intensity + Direction)
Vérifie que le nouveau scoring fonctionne correctement dans le système complet
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import asyncio
from datetime import datetime
from textblob_v11_intensity_direction import score_news_v11_intensity_direction

# Simuler des news catastrophiques (Airbus)
catastrophic_news = [
    {
        'title': 'Emergency Airworthiness Directive issued for Airbus A320 family',
        'summary': 'FAA orders immediate software upgrade before flights resume. Flight control issue discovered affecting 6,000 aircraft.',
    },
    {
        'title': 'Airbus recalls 6,000 planes for critical safety fix',
        'summary': 'Emergency recall due to solar radiation corrupting flight control data',
    },
    {
        'title': 'Airlines ground Airbus A320 fleet pending emergency fixes',
        'summary': 'Precautionary grounding affects thousands of flights worldwide',
    },
] * 5

# Simuler des news très positives
very_positive_news = [
    {
        'title': 'Company beats expectations with record earnings',
        'summary': 'Breakthrough quarter drives massive stock surge',
    },
    {
        'title': 'Stock soars on blockbuster results',
        'summary': 'Revenue rockets 45% as innovation drives growth',
    },
    {
        'title': 'Analysts upgrade citing strong profit outlook',
        'summary': 'Multiple upgrades as company wins major deal',
    },
] * 5

# Simuler des news neutres
neutral_news = [
    {
        'title': 'Company announces quarterly results',
        'summary': 'Results meet market expectations',
    },
    {
        'title': 'Industry update for the sector',
        'summary': 'Trading continues in normal range',
    },
] * 7

print("="*80)
print("TEST D'INTÉGRATION V11 - INTENSITY + DIRECTION")
print("="*80)

# Test 1: News catastrophiques (devrait être ~90-100/100 NÉGATIF)
print("\n📰 TEST 1: NEWS CATASTROPHIQUES (Airbus)")
intensity, direction, explanation = score_news_v11_intensity_direction(catastrophic_news)
print(f"   Score: {intensity:.0f}/100 {direction}")
print(f"   Détails: {explanation}")

expected_catastro = (85 <= intensity <= 100 and direction == "NÉGATIF")
print(f"   Attendu: 85-100/100 NÉGATIF")
print(f"   {'✅ PASS' if expected_catastro else '❌ FAIL'}")

if expected_catastro:
    print(f"\n   💡 Interprétation:")
    print(f"      Le bot détecte correctement que les news sont TRÈS négatives.")
    print(f"      Avec un score composite de ~{(75 + intensity) / 2:.0f}/100, le bot:")
    if (75 + intensity) / 2 >= 65:
        print(f"      → ACHÈTERAIT quand même (erreur!)")
        print(f"      ⚠️  PROBLÈME: Le score tech (75) tire le composite vers le haut")
        print(f"      ⚠️  SOLUTION: Utiliser la DIRECTION pour bloquer les achats si NÉGATIF")
    else:
        print(f"      → NE PASSERAIT PAS le seuil de 65/100 ✅")

# Test 2: News très positives (devrait être ~85-100/100 POSITIF)
print("\n📰 TEST 2: NEWS TRÈS POSITIVES")
intensity, direction, explanation = score_news_v11_intensity_direction(very_positive_news)
print(f"   Score: {intensity:.0f}/100 {direction}")
print(f"   Détails: {explanation}")

expected_positive = (80 <= intensity <= 100 and direction == "POSITIF")
print(f"   Attendu: 80-100/100 POSITIF")
print(f"   {'✅ PASS' if expected_positive else '❌ FAIL'}")

# Test 3: News neutres (devrait être ~45-55/100 NEUTRE)
print("\n📰 TEST 3: NEWS NEUTRES")
intensity, direction, explanation = score_news_v11_intensity_direction(neutral_news)
print(f"   Score: {intensity:.0f}/100 {direction}")
print(f"   Détails: {explanation}")

expected_neutral = (40 <= intensity <= 60 and direction == "NEUTRE")
print(f"   Attendu: 40-60/100 NEUTRE")
print(f"   {'✅ PASS' if expected_neutral else '❌ FAIL'}")

# Résumé
print("\n" + "="*80)
print("RÉSUMÉ")
print("="*80)

all_pass = expected_catastro and expected_positive and expected_neutral

if all_pass:
    print("✅ TOUS LES TESTS PASSENT!")
    print("\nLe système V11 est prêt à être utilisé.")
    print("\nProchaines étapes:")
    print("1. Le bot affichera maintenant: 'Score News: XX/100 POSITIF/NÉGATIF'")
    print("2. Vous pouvez voir l'intensité ET la direction du sentiment")
    print("3. Plus besoin de deviner si 59/100 est bon ou mauvais!")
    print("\nExemples:")
    print("• 92/100 NÉGATIF = très mauvaises news (bearish) → éviter l'achat")
    print("• 88/100 POSITIF = très bonnes news (bullish) → bon signal d'achat")
    print("• 50/100 NEUTRE = pas de sentiment clair → se fier au score tech")
else:
    print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("\nVérifiez les résultats ci-dessus pour voir où le système échoue.")

print("\n" + "="*80)
