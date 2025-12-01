"""
Test du système V11 sur toutes les actions de la watchlist
Compare l'ancien système V9 vs le nouveau V11 sur des cas réels
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import asyncio
from datetime import datetime, timedelta
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from textblob_v9_refined import score_news_v9_refined
from textblob_v11_intensity_direction import score_news_v11_intensity_direction

async def test_stock(analyzer, symbol, date_str):
    """Test un stock à une date donnée"""
    target_date = datetime.strptime(date_str, '%Y-%m-%d')

    # Récupérer les news
    has_news, news_items, intensity, direction = await analyzer.get_news_for_date(symbol, target_date)

    if not has_news or len(news_items) == 0:
        return None

    # Calculer aussi avec V9 pour comparaison
    score_v9 = score_news_v9_refined(news_items)

    return {
        'symbol': symbol,
        'date': date_str,
        'news_count': len(news_items),
        'v9_score': score_v9,
        'v11_intensity': intensity,
        'v11_direction': direction,
        'news_titles': [n['title'][:80] for n in news_items[:3]]
    }

async def main():
    print("="*80)
    print("TEST SYSTÈME V11 SUR TOUTES LES ACTIONS")
    print("="*80)

    # Initialiser l'analyseur
    analyzer = HistoricalNewsAnalyzer()

    # Stocks à tester (mix US + France)
    test_cases = [
        # Actions US
        ('AAPL', '2025-11-29'),  # Apple
        ('NVDA', '2025-11-29'),  # Nvidia
        ('TSLA', '2025-11-29'),  # Tesla
        ('AMZN', '2025-11-29'),  # Amazon
        ('MSFT', '2025-11-29'),  # Microsoft

        # Actions françaises
        ('AIR.PA', '2025-11-29'),  # Airbus (on connaît déjà)
        ('MC.PA', '2025-11-29'),   # LVMH
        ('SAF.PA', '2025-11-29'),  # Safran
        ('OR.PA', '2025-11-29'),   # L'Oréal
        ('BNP.PA', '2025-11-29'),  # BNP Paribas
    ]

    print(f"\n📊 Test de {len(test_cases)} actions au 29 novembre 2025")
    print(f"⏳ Récupération des news en cours...\n")

    results = []

    for symbol, date in test_cases:
        try:
            result = await test_stock(analyzer, symbol, date)
            if result:
                results.append(result)

                # Affichage immédiat
                print(f"\n{'='*80}")
                print(f"📈 {result['symbol']} - {result['date']}")
                print(f"{'='*80}")
                print(f"📰 {result['news_count']} news trouvées:")
                for i, title in enumerate(result['news_titles'], 1):
                    print(f"   {i}. {title}...")

                print(f"\n📊 COMPARAISON DES SCORES:")
                print(f"   Ancien (V9): {result['v9_score']:.0f}/100")
                print(f"   Nouveau (V11): {result['v11_intensity']:.0f}/100 **{result['v11_direction']}**")

                # Interprétation
                delta = abs(result['v11_intensity'] - result['v9_score'])
                if delta > 15:
                    print(f"   ⚠️  ÉCART IMPORTANT: {delta:.0f} points de différence!")
                    if result['v11_direction'] == "NÉGATIF" and result['v9_score'] > 45:
                        print(f"   💡 V9 sous-estimait la négativité (score trop proche de neutre)")
                    elif result['v11_direction'] == "POSITIF" and result['v9_score'] < 55:
                        print(f"   💡 V9 sous-estimait la positivité (score trop proche de neutre)")
                else:
                    print(f"   ✅ Scores cohérents (écart: {delta:.0f} points)")

                # Impact sur trading
                tech_score = 75  # Score technique moyen simulé
                composite_v9 = (tech_score * 0.5) + (result['v9_score'] * 0.5)
                composite_v11 = (tech_score * 0.5) + (result['v11_intensity'] * 0.5)

                print(f"\n💰 IMPACT SUR LE TRADING (avec tech score {tech_score}/100):")
                achat_v9 = "ACHAT" if composite_v9 >= 65 else "PAS D'ACHAT"
                print(f"   Composite V9: {composite_v9:.0f}/100 → {achat_v9}")
                print(f"   Composite V11: {composite_v11:.0f}/100 {result['v11_direction']} → ", end='')

                if composite_v11 >= 65:
                    if result['v11_direction'] == "NÉGATIF" and result['v11_intensity'] > 80:
                        print(f"ACHAT mais NEWS TRÈS NÉGATIVES!")
                    elif result['v11_direction'] == "POSITIF":
                        print(f"ACHAT validé (news positives)")
                    else:
                        print(f"ACHAT validé")
                else:
                    print(f"PAS D'ACHAT")
            else:
                print(f"⚠️  {symbol}: Aucune news trouvée pour {date}")

        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {e}")

        # Petit délai pour éviter de surcharger les APIs
        await asyncio.sleep(0.5)

    # Résumé final
    print(f"\n{'='*80}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*80}")

    if not results:
        print("❌ Aucun résultat obtenu")
    else:
        print(f"\n✅ {len(results)} actions analysées avec succès\n")

        # Statistiques
        big_differences = [r for r in results if abs(r['v11_intensity'] - r['v9_score']) > 15]
        negative_warnings = [r for r in results if r['v11_direction'] == "NÉGATIF" and r['v11_intensity'] > 80]

        print(f"📊 Statistiques:")
        print(f"   • Écarts importants (>15 pts): {len(big_differences)}/{len(results)}")
        print(f"   • Warnings (news très négatives): {len(negative_warnings)}/{len(results)}")

        if big_differences:
            print(f"\n⚠️  Actions avec écarts importants V9 vs V11:")
            for r in big_differences:
                print(f"   • {r['symbol']}: V9={r['v9_score']:.0f} → V11={r['v11_intensity']:.0f} {r['v11_direction']}")

        if negative_warnings:
            print(f"\n🚨 Actions avec news TRÈS NÉGATIVES (attention aux achats!):")
            for r in negative_warnings:
                print(f"   • {r['symbol']}: {r['v11_intensity']:.0f}/100 NÉGATIF")

        print(f"\n💡 Le nouveau système V11 permet de:")
        print(f"   1. Voir immédiatement si les news sont POSITIVES ou NÉGATIVES")
        print(f"   2. Éviter les achats sur des actions avec news catastrophiques")
        print(f"   3. Mieux comprendre l'intensité du sentiment (pas juste un score neutre)")

    await analyzer.close()
    print(f"\n{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())
