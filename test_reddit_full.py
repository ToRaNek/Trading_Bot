#!/usr/bin/env python3
"""
Test de récupération COMPLÈTE des posts Reddit avec sauvegarde CSV
Récupère TOUS les posts disponibles (pas de limite de temps)
"""

import asyncio
import sys
sys.path.insert(0, '.')

from trading_bot_main import RedditSentimentAnalyzer
from datetime import datetime


async def test_full_reddit_scrape():
    """Récupère TOUS les posts Reddit pour NVDA et sauvegarde en CSV"""

    print("\n" + "=" * 80)
    print("🔥 RÉCUPÉRATION COMPLÈTE REDDIT - NVDA")
    print("=" * 80)

    analyzer = RedditSentimentAnalyzer()

    symbol = "NVDA"
    target_date = datetime.now()  # Maintenant

    print(f"\n📅 Date cible: {target_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Ticker: {symbol}")
    print(f"📱 Sources: r/NVDA_Stock + r/stocks ($NVDA)")
    print(f"💾 Sauvegarde: CSV activée")
    print(f"\n⚠️ ATTENTION: Va récupérer TOUS les posts via PUSHSHIFT")
    print("   - PAGINATION ILLIMITÉE jusqu'à épuisement des données")
    print("   - 500 posts par page")
    print("   - Délai de 1.5s entre chaque page (rate limiting)")
    print("   - Peut prendre 10-30 minutes selon le volume")
    print("   - Sauvegarde automatique en CSV")

    input("\nAppuyez sur ENTRÉE pour continuer...")

    print("\n" + "=" * 80)
    print("🚀 DÉMARRAGE DE LA RÉCUPÉRATION")
    print("=" * 80)

    try:
        # Récupération avec sauvegarde CSV
        score, post_count, samples = await analyzer.get_reddit_sentiment(
            symbol=symbol,
            target_date=target_date,
            lookback_hours=48,  # Paramètre ignoré (on récupère tout)
            save_csv=True  # ✅ Activer sauvegarde CSV
        )

        print("\n" + "=" * 80)
        print("✅ RÉCUPÉRATION TERMINÉE")
        print("=" * 80)

        print(f"\n📊 Résumé:")
        print(f"   Posts totaux: {post_count}")
        print(f"   Score sentiment: {score:.1f}/100")

        if post_count > 0:
            print(f"\n💾 Fichiers CSV créés:")
            print(f"   - reddit_posts_{symbol}_NVDA_Stock_*.csv")
            print(f"   - reddit_posts_{symbol}_stocks_*.csv")

            print(f"\n📋 Exemples de posts (premiers):")
            for i, sample in enumerate(samples[:5], 1):
                print(f"   {i}. {sample}")

            print(f"\n💡 Les fichiers CSV contiennent:")
            print(f"   - Date de création")
            print(f"   - Titre du post")
            print(f"   - Corps du post")
            print(f"   - Score (upvotes)")
        else:
            print(f"\n⚠️ Aucun post trouvé")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await analyzer.close()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_full_reddit_scrape())
