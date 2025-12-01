"""
Test du scoring avec les vraies news Airbus (AIR.PA)
Pour comprendre pourquoi le score est si bas alors que les news sont catastrophiques
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from textblob_v9_refined import score_news_v9_refined
from datetime import datetime

# Vraies news Airbus du 28-30 Nov 2025 (catastrophiques)
airbus_news = [
    {
        'title': 'Airbus calls for immediate software upgrade to A320 aircraft amid control concerns',
        'summary': 'Airbus requests airlines take immediate action to upgrade software warning there could be an issue with flight controls. About 6,000 single-aisle planes need repairs.',
        'publisher': 'NPR',
        'date': datetime(2025, 11, 28),
        'importance': 3.0,  # Très important (recall)
        'sentiment': -0.6
    },
    {
        'title': 'Airlines race to fix Airbus planes after warning solar radiation could cause pilots to lose control',
        'summary': 'Analysis of a recent event involving an A320 Family aircraft revealed that intense solar radiation may corrupt data critical to the functioning of flight controls',
        'publisher': 'CNN',
        'date': datetime(2025, 11, 28),
        'importance': 4.0,  # Critique (safety)
        'sentiment': -0.7
    },
    {
        'title': 'FAA issues Emergency Airworthiness Directive for Airbus A319 and A320/321 airplanes',
        'summary': 'The FAA is ordering operators to perform the software upgrade before the airplane flies again, and before Sunday, Nov. 30',
        'publisher': 'FAA',
        'date': datetime(2025, 11, 28),
        'importance': 4.5,  # Très critique (regulatory)
        'sentiment': -0.8
    },
    {
        'title': 'Airbus update on A320 Family precautionary fleet action',
        'summary': 'Precautionary measures implemented across approximately 6,000 aircraft following flight control issue discovery',
        'publisher': 'Airbus',
        'date': datetime(2025, 11, 28),
        'importance': 3.5,
        'sentiment': -0.5
    },
    {
        'title': 'Airbus faces new operational adjustments for aircraft with Pratt & Whitney engines in extreme cold',
        'summary': 'The planemaker introduced restrictions for take-offs in freezing fog and low visibility. Affects A320neo family including A321neo and A321LR models',
        'publisher': 'Parameter.io',
        'date': datetime(2025, 11, 29),
        'importance': 2.5,
        'sentiment': -0.4
    },
    {
        'title': 'Airbus delivery targets under pressure as November-December push begins',
        'summary': 'Airbus needs to deliver 235 jets between November and December to reach target of around 820. Requires average of 118 aircraft monthly.',
        'publisher': 'Aerospace Global News',
        'date': datetime(2025, 11, 27),
        'importance': 2.0,
        'sentiment': -0.3
    },
    {
        'title': 'American Airlines completes Airbus software fixes for impacted aircraft',
        'summary': 'Aircraft impacted by Airbus recall have received the software fixes necessary to resume flying',
        'publisher': 'CNBC',
        'date': datetime(2025, 11, 29),
        'importance': 1.5,
        'sentiment': 0.2  # Légèrement positif (problème résolu pour AA)
    }
]

print("="*80)
print("TEST SCORING AIRBUS (AIR.PA) - News du 28-30 Nov 2025")
print("="*80)
print(f"\n📰 {len(airbus_news)} news analysées:")
for i, news in enumerate(airbus_news, 1):
    print(f"\n{i}. {news['title'][:70]}...")
    print(f"   Sentiment: {news['sentiment']:+.2f} | Importance: {news['importance']:.1f}")

# Test avec le scorer actuel (V9)
score_v9 = score_news_v9_refined(airbus_news)

print("\n" + "="*80)
print("RÉSULTAT:")
print("="*80)
print(f"\n🤖 Score TextBlob V9: {score_v9:.1f}/100")
print(f"\n📊 Analyse:")
if score_v9 > 65:
    print(f"   ✅ Score > 65 → Bot ACHÈTE (BULLISH)")
elif score_v9 < 35:
    print(f"   ❌ Score < 35 → Bot VEND (BEARISH)")
else:
    print(f"   ⚠️  Score 35-65 → NEUTRE/FAIBLE signal")

print(f"\n🎯 Score attendu:")
print(f"   Ces news sont CATASTROPHIQUES:")
print(f"   - Rappel d'urgence de 6000 avions")
print(f"   - Bug critique de contrôle de vol")
print(f"   - Directive FAA d'urgence")
print(f"   - Restrictions opérationnelles supplémentaires")
print(f"   → Score devrait être 85-95/100 NÉGATIF (bearish)")

print(f"\n💡 Système proposé:")
print(f"   Au lieu de: {score_v9:.0f}/100")
print(f"   Avoir: Intensité + Direction")
print(f"   Exemple: 92/100 NÉGATIF (très fort sentiment bearish)")
print(f"           ou 88/100 POSITIF (très fort sentiment bullish)")
print(f"           ou 50/100 NEUTRE (pas de sentiment)")

print("\n" + "="*80)
