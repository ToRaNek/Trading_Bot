"""Test de la nouvelle logique SELL ajustée"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calculate_sell_score(tech_score, news_score):
    """
    Nouvelle logique SELL avec news POSITIF
    """
    if news_score >= 80:
        return 0, "News TRÈS POSITIVES (≥80) bloquent la vente"
    elif news_score >= 75:
        score = (tech_score * 0.70) + (news_score * 0.30) - 5
        return max(0, score), "News POSITIVES (75-79) léger malus -5"
    else:
        score = (tech_score * 0.70) + (news_score * 0.30)
        return score, "News POSITIVES modérées (<75), priorité au technique"

print("=" * 80)
print("TEST NOUVELLE LOGIQUE SELL")
print("=" * 80)

# Seuil de validation
THRESHOLD = 65

# Cas du backtest MSFT qui posait problème
test_cases = [
    # (description, tech, news, devrait_passer)
    ("14 Nov - MSFT $509.50", 73, 65, True),   # Doit passer maintenant!
    ("14 Nov - MSFT $509.64", 73, 65, True),   # Doit passer maintenant!

    ("26 Nov - MSFT $483.24", 62, 76, False),  # Tech trop faible
    ("26 Nov - MSFT $486.79", 58, 76, False),  # Tech trop faible
    ("26 Nov - MSFT $487.25", 65, 76, True),   # Limite, devrait passer

    ("28 Nov - MSFT $490.58", 80, 75, True),   # Passait déjà, doit toujours passer

    # Cas limites
    ("Tech fort, news excellentes", 85, 85, False),  # Bloqué (news ≥80)
    ("Tech fort, news très positives", 85, 79, True), # Passe avec malus -5
    ("Tech moyen, news modérées", 70, 70, True),     # Passe sans malus
    ("Tech faible, news modérées", 60, 70, False),   # Ne passe pas
]

print("\n📊 RÉSULTATS:\n")

passed = 0
failed = 0

for description, tech, news, should_pass in test_cases:
    score, reason = calculate_sell_score(tech, news)
    will_pass = score >= THRESHOLD

    status = "✅" if will_pass == should_pass else "❌"
    result_emoji = "✅ VALIDÉ" if will_pass else "❌ REJETÉ"

    print(f"{status} {description}")
    print(f"   Tech: {tech}/100 | News: {news}/100 POSITIF")
    print(f"   Score: {score:.0f}/100 → {result_emoji}")
    print(f"   Raison: {reason}")

    if will_pass == should_pass:
        passed += 1
    else:
        failed += 1
        print(f"   ⚠️  ERREUR: Devrait {'PASSER' if should_pass else 'ÉCHOUER'}")
    print()

print("=" * 80)
print("RÉSUMÉ")
print("=" * 80)
print(f"\n✅ Tests réussis: {passed}/{len(test_cases)}")
print(f"❌ Tests échoués: {failed}/{len(test_cases)}")

if failed == 0:
    print("\n🎉 TOUS LES TESTS PASSENT!")
    print("\n💡 Changements:")
    print("   • News <75 POSITIF: Pas de malus, le technique décide")
    print("   • News 75-79 POSITIF: Malus -5 seulement")
    print("   • News ≥80 POSITIF: Blocage total (vraiment excellentes)")
    print("\n📈 Impact sur MSFT:")
    print("   • 14 Nov @ $509.50: Tech 73 + News 65 → Score 70/100 ✅")
    print("   • Au lieu de rejeter (ancien: 61/100 ❌)")
    print("   • Aurait vendu avec +1.5% profit au lieu de -3% loss!")
else:
    print(f"\n⚠️  {failed} test(s) ont échoué.")

print("\n" + "=" * 80)
