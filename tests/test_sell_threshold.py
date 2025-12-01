"""
Test spécifique du cas problématique:
SELL avec Tech 65/100 et News 75/100 POSITIF

Avant: Score = 0 (bloqué)
Après: Score devrait permettre la vente si tech est bon
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calculate_composite_score(tech_score, tech_signal, news_score, news_direction):
    """Logique mise à jour"""
    if tech_signal == 'SELL':
        if news_direction == "POSITIF":
            if news_score > 85:  # Bloque seulement >85
                composite_score = 0
                reason = "News TRÈS POSITIVES bloquent la vente"
            else:
                # News positives modérées (70-85) : léger malus
                composite_score = (tech_score * 0.70) + (news_score * 0.30)
                composite_score = max(0, composite_score - 10)
                reason = "News POSITIVES pénalisent légèrement la vente"
        else:
            composite_score = (tech_score * 0.70) + (news_score * 0.30)
            reason = "News NEUTRES"
    else:
        composite_score = tech_score
        reason = "N/A"

    return composite_score, reason

print("="*80)
print("TEST SEUIL VENTE AJUSTÉ")
print("="*80)

# Cas qui causait problème dans le backtest
test_cases = [
    ("Tech 65, News 75 POSITIF", 65, 'SELL', 75, "POSITIF"),
    ("Tech 72, News 75 POSITIF", 72, 'SELL', 75, "POSITIF"),
    ("Tech 80, News 75 POSITIF", 80, 'SELL', 75, "POSITIF"),
    ("Tech 65, News 87 POSITIF", 65, 'SELL', 87, "POSITIF"),
    ("Tech 80, News 87 POSITIF", 80, 'SELL', 87, "POSITIF"),
]

print("\n📊 RÉSULTATS:\n")

for nom, tech, signal, news, direction in test_cases:
    score, reason = calculate_composite_score(tech, signal, news, direction)
    decision = "VALIDÉ ✅" if score >= 65 else "REFUSÉ ❌"

    print(f"{nom}")
    print(f"   Score: {score:.0f}/100 → {decision}")
    print(f"   Raison: {reason}")
    print()

print("="*80)
print("ANALYSE")
print("="*80)
print("\n📈 Avant l'ajustement:")
print("   Tech 65 + News 75 POSITIF → Score 0 (BLOQUÉ)")
print("   Problème: Le bot attendait le stop loss au lieu de vendre!")
print("\n📉 Après l'ajustement:")
print("   Tech 65 + News 75 POSITIF → Score ~56/100 (REFUSÉ, mais pas à 0)")
print("   Tech 72 + News 75 POSITIF → Score ~63/100 (proche du seuil)")
print("   Tech 80 + News 75 POSITIF → Score ~69/100 (VALIDÉ ✅)")
print("\n💡 Maintenant:")
print("   • News 75 POSITIF n'est plus un blocage total")
print("   • Un bon signal technique (>72) peut quand même vendre")
print("   • Évite les stop loss inutiles sur news modérément positives")
print("   • Bloque seulement sur news VRAIMENT excellentes (>85)")

print("\n" + "="*80)
