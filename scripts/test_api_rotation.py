"""Script de test pour vérifier le système de rotation des clés API"""

import sys
import os
import asyncio
from datetime import datetime

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_key_rotator import APIKeyRotator
from analyzers.news_analyzer import HistoricalNewsAnalyzer


def test_rotator_basic():
    """Test basique du rotateur de clés"""
    print("=" * 60)
    print("TEST 1: Fonctionnalités de base du rotateur")
    print("=" * 60)

    rotator = APIKeyRotator()
    stats = rotator.get_stats()

    print(f"\nStats initiales:")
    print(f"  Total clés: {stats['total_keys']}")
    print(f"  Clés actives: {stats['active_keys']}")
    print(f"  Clés échouées: {stats['failed_keys']}")
    print(f"  Index courant: {stats['current_index']}")

    if stats['total_keys'] > 0:
        print(f"\nClé courante: {rotator.get_current_key()[:20]}...")

        # Test de rotation
        print("\nTest de rotation:")
        for i in range(min(3, stats['total_keys'])):
            key = rotator.get_current_key()
            print(f"  Tour {i + 1}: Clé {rotator.current_index + 1} - {key[:20] if key else 'None'}...")
            rotator.rotate()

        # Test de marquage d'échec
        print("\nTest de marquage d'échec:")
        rotator.mark_current_as_failed()
        stats = rotator.get_stats()
        print(f"  Clés actives après échec: {stats['active_keys']}/{stats['total_keys']}")

    print("\n✅ Test 1 terminé\n")


async def test_news_analyzer():
    """Test du news analyzer avec rotation"""
    print("=" * 60)
    print("TEST 2: News Analyzer avec rotation des clés")
    print("=" * 60)

    analyzer = HistoricalNewsAnalyzer()

    # Test de récupération de news
    symbol = "NVDA"
    target_date = datetime(2025, 7, 15)

    print(f"\nTest de récupération des news pour {symbol} @ {target_date.strftime('%Y-%m-%d')}")

    try:
        has_news, news_items, score = await analyzer.get_news_for_date(symbol, target_date)

        print(f"\nRésultats:")
        print(f"  News trouvées: {has_news}")
        print(f"  Nombre d'articles: {len(news_items)}")
        print(f"  Score: {score:.0f}/100")

        if news_items:
            print(f"\nPremiers articles:")
            for i, article in enumerate(news_items[:3], 1):
                print(f"  {i}. {article['title'][:60]}...")
                print(f"     Source: {article['publisher']}, Importance: {article['importance']:.1f}")

        # Stats du rotateur
        stats = analyzer.newsapi_rotator.get_stats()
        print(f"\nStats du rotateur après requête:")
        print(f"  Clés actives: {stats['active_keys']}/{stats['total_keys']}")
        print(f"  Clés échouées: {stats['failed_keys']}")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await analyzer.close()

    print("\n✅ Test 2 terminé\n")


async def test_multiple_requests():
    """Test avec plusieurs requêtes pour voir la rotation en action"""
    print("=" * 60)
    print("TEST 3: Multiples requêtes pour tester la rotation")
    print("=" * 60)

    analyzer = HistoricalNewsAnalyzer()

    symbols = ["NVDA", "AAPL", "MSFT"]
    dates = [datetime(2025, 7, 15), datetime(2025, 7, 16), datetime(2025, 7, 17)]

    print(f"\nTest de {len(symbols) * len(dates)} requêtes\n")

    try:
        for date in dates:
            for symbol in symbols:
                print(f"Requête: {symbol} @ {date.strftime('%Y-%m-%d')}")
                has_news, news_items, score = await analyzer.get_news_for_date(symbol, date)
                stats = analyzer.newsapi_rotator.get_stats()
                print(f"  ✓ {len(news_items)} news, clé {analyzer.newsapi_rotator.current_index + 1}/{stats['total_keys']}, {stats['active_keys']} actives")

                # Petit délai pour ne pas spammer l'API
                await asyncio.sleep(0.5)

        print(f"\nStats finales du rotateur:")
        stats = analyzer.newsapi_rotator.get_stats()
        print(f"  Total clés: {stats['total_keys']}")
        print(f"  Clés actives: {stats['active_keys']}")
        print(f"  Clés échouées: {stats['failed_keys']}")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await analyzer.close()

    print("\n✅ Test 3 terminé\n")


async def main():
    """Fonction principale de test"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "TEST SYSTÈME DE ROTATION DES CLÉS API" + " " * 10 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Test 1: Fonctionnalités de base
    test_rotator_basic()

    # Test 2: News analyzer
    await test_news_analyzer()

    # Test 3: Multiples requêtes
    await test_multiple_requests()

    print("\n" + "=" * 60)
    print("TOUS LES TESTS TERMINÉS")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
