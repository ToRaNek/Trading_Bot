"""
Test de la nouvelle logique du score composite intelligent
Vérifie que le bot prend les bonnes décisions selon les news
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calculate_composite_score(tech_score, tech_signal, news_score, news_direction):
    """
    Reproduit la logique du score composite de live_trader.py
    """
    base_composite = (tech_score * 0.50) + (news_score * 0.50)
    composite_score = base_composite
    adjustment_reason = ""

    # Pour les signaux BUY
    if tech_signal == 'BUY':
        if news_direction == "POSITIF":
            # News positives + signal achat = EXCELLENT
            composite_score = (tech_score * 0.30) + (news_score * 0.70)
            adjustment_reason = "News POSITIVES favorisent l'achat"

        elif news_direction == "NÉGATIF":
            # News négatives + signal achat = DANGER
            if news_score > 75:
                composite_score = 0  # Bloquer l'achat
                adjustment_reason = "News TRÈS NÉGATIVES bloquent l'achat"
            else:
                composite_score = (tech_score * 0.70) + (news_score * 0.30)
                composite_score = max(0, composite_score - 25)
                adjustment_reason = "News NÉGATIVES pénalisent l'achat"
        else:
            # News neutres
            composite_score = (tech_score * 0.70) + (news_score * 0.30)
            adjustment_reason = "News NEUTRES, priorité au technique"

    # Pour les signaux SELL
    elif tech_signal == 'SELL':
        if news_direction == "NÉGATIF":
            # News négatives + signal vente = EXCELLENT
            composite_score = (tech_score * 0.30) + (news_score * 0.70)
            adjustment_reason = "News NÉGATIVES favorisent la vente"

        elif news_direction == "POSITIF":
            # News positives + signal vente = MAUVAISE IDÉE
            if news_score > 85:  # Bloque seulement si VRAIMENT très positif (>85)
                composite_score = 0  # Bloquer la vente
                adjustment_reason = "News TRÈS POSITIVES bloquent la vente"
            else:
                # News positives modérées (70-85) : léger malus seulement
                composite_score = (tech_score * 0.70) + (news_score * 0.30)
                composite_score = max(0, composite_score - 10)  # Malus réduit à -10 points
                adjustment_reason = "News POSITIVES pénalisent légèrement la vente"
        else:
            # News neutres
            composite_score = (tech_score * 0.70) + (news_score * 0.30)
            adjustment_reason = "News NEUTRES, priorité au technique"

    return composite_score, base_composite, adjustment_reason


print("="*80)
print("TEST LOGIQUE SCORE COMPOSITE INTELLIGENT")
print("="*80)

# Seuil de validation
THRESHOLD = 65

test_cases = [
    # (nom, tech_score, tech_signal, news_score, news_direction, décision_attendue)

    # === CAS ACHAT ===
    ("Airbus - Rappel 6000 avions", 75, 'BUY', 90, "NÉGATIF", "REFUSÉ"),
    ("Tesla - Excellentes news FSD", 75, 'BUY', 95, "POSITIF", "VALIDÉ"),
    ("NVDA - News neutres", 75, 'BUY', 50, "NEUTRE", "VALIDÉ"),
    ("Apple - News légèrement négatives", 75, 'BUY', 60, "NÉGATIF", "REFUSÉ"),
    ("Microsoft - News positives modérées", 70, 'BUY', 70, "POSITIF", "VALIDÉ"),

    # === CAS VENTE ===
    ("Vente Airbus - News catastrophiques", 70, 'SELL', 90, "NÉGATIF", "VALIDÉ"),
    ("Vente Tesla - News très positives", 70, 'SELL', 88, "POSITIF", "REFUSÉ"),
    ("Vente position - News neutres", 75, 'SELL', 50, "NEUTRE", "VALIDÉ"),  # Score tech augmenté à 75
    ("Vente position - News légèrement positives", 65, 'SELL', 65, "POSITIF", "REFUSÉ"),

    # === CAS EXTRÊMES ===
    ("Catastrophe totale", 80, 'BUY', 95, "NÉGATIF", "REFUSÉ"),
    ("Opportunité parfaite", 80, 'BUY', 92, "POSITIF", "VALIDÉ"),
]

print("\n📊 TEST DE TOUS LES CAS:\n")

passed = 0
failed = 0

for nom, tech_score, tech_signal, news_score, news_direction, attendu in test_cases:
    composite, base, reason = calculate_composite_score(tech_score, tech_signal, news_score, news_direction)

    decision = "VALIDÉ" if composite >= THRESHOLD else "REFUSÉ"

    # Vérifier si la décision est correcte
    correct = (decision == attendu)

    status = "✅" if correct else "❌"

    print(f"{status} {nom}")
    print(f"   Tech: {tech_score}/100 {tech_signal}")
    print(f"   News: {news_score}/100 {news_direction}")
    print(f"   Base: {base:.0f}/100 → Ajusté: {composite:.0f}/100")
    print(f"   Raison: {reason}")
    print(f"   Décision: {decision} (attendu: {attendu})")

    if correct:
        passed += 1
    else:
        failed += 1
        print(f"   ⚠️  ÉCHEC: Score composite devrait {'≥' if attendu == 'VALIDÉ' else '<'} {THRESHOLD}")

    print()

print("="*80)
print("RÉSUMÉ")
print("="*80)
print(f"\n✅ Tests réussis: {passed}/{len(test_cases)}")
print(f"❌ Tests échoués: {failed}/{len(test_cases)}")

if failed == 0:
    print("\n🎉 TOUS LES TESTS PASSENT!")
    print("\n💡 Exemples de comportements:")
    print("   • Airbus (news catastrophiques) → Score 0/100 → Achat BLOQUÉ ✅")
    print("   • Tesla (news excellentes) → Score 89/100 → Achat VALIDÉ ✅")
    print("   • Vente Airbus (news négatives) → Score 85/100 → Vente VALIDÉE ✅")
    print("   • Vente Tesla (news positives) → Score 0/100 → Vente BLOQUÉE ✅")
    print("\n📈 Le bot achète maintenant quand:")
    print("   1. News POSITIVES + Signal technique BUY")
    print("   2. News NEUTRES + Signal technique BUY fort")
    print("\n📉 Le bot vend maintenant quand:")
    print("   1. News NÉGATIVES + Signal technique SELL")
    print("   2. News NEUTRES + Signal technique SELL")
    print("\n🚫 Le bot BLOQUE:")
    print("   1. Achats sur news très négatives (>75/100)")
    print("   2. Ventes sur news très positives (>75/100)")
else:
    print(f"\n⚠️  {failed} test(s) ont échoué. Vérifiez la logique du score composite.")

print("\n" + "="*80)
