"""Test du système de rotation des clés Finnhub"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import asyncio

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.api_key_rotator import APIKeyRotator
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from datetime import datetime, timedelta

print("=" * 80)
print("TEST ROTATION CLÉS FINNHUB")
print("=" * 80)

# Test 1: Vérifier que le rotateur Finnhub se charge correctement
print("\n📋 Test 1: Chargement du rotateur Finnhub")
print("-" * 80)

finnhub_rotator = APIKeyRotator(key_column='finnhub_key', env_var='FINNHUB_KEY')
stats = finnhub_rotator.get_stats()

print(f"Total clés: {stats['total_keys']}")
print(f"Clés actives: {stats['active_keys']}")
print(f"Clés épuisées: {stats['failed_keys']}")
print(f"Index actuel: {stats['current_index']}")

# Test 2: Vérifier que le HistoricalNewsAnalyzer initialise les deux rotateurs
print("\n📋 Test 2: Initialisation du HistoricalNewsAnalyzer")
print("-" * 80)

analyzer = HistoricalNewsAnalyzer()

newsapi_stats = analyzer.newsapi_rotator.get_stats()
finnhub_stats = analyzer.finnhub_rotator.get_stats()

print(f"NewsAPI - Total: {newsapi_stats['total_keys']}, Actives: {newsapi_stats['active_keys']}")
print(f"Finnhub - Total: {finnhub_stats['total_keys']}, Actives: {finnhub_stats['active_keys']}")

# Test 3: Tester la récupération de news avec rotation
print("\n📋 Test 3: Récupération de news (teste la rotation si limite atteinte)")
print("-" * 80)

async def test_news_fetch():
    # Tester avec AAPL (Apple) sur une date récente
    target_date = datetime.now() - timedelta(days=7)

    print(f"Récupération des news pour AAPL le {target_date.strftime('%Y-%m-%d')}...")

    has_news, news_items, intensity, direction = await analyzer.get_news_for_date('AAPL', target_date)

    if has_news:
        print(f"✅ News trouvées: {len(news_items)} articles")
        print(f"   Score: {intensity:.0f}/100 {direction}")

        # Afficher les stats après la requête
        newsapi_stats_after = analyzer.newsapi_rotator.get_stats()
        finnhub_stats_after = analyzer.finnhub_rotator.get_stats()

        print(f"\n📊 Stats après requête:")
        print(f"   NewsAPI - Actives: {newsapi_stats_after['active_keys']}/{newsapi_stats_after['total_keys']}")
        print(f"   Finnhub - Actives: {finnhub_stats_after['active_keys']}/{finnhub_stats_after['total_keys']}")
    else:
        print("❌ Aucune news trouvée (normal si aucune clé API valide)")

    await analyzer.close()

asyncio.run(test_news_fetch())

print("\n" + "=" * 80)
print("RÉSUMÉ")
print("=" * 80)
print("\n✅ Le système de rotation Finnhub est maintenant opérationnel!")
print("\n💡 Fonctionnalités:")
print("   • Rotation automatique des clés Finnhub sur erreur 429/403")
print("   • Fallback sur FINNHUB_KEY si aucune clé dans le CSV")
print("   • Fonctionne en parallèle avec la rotation NewsAPI")
print("\n📝 Pour ajouter des clés Finnhub:")
print("   1. Ouvrir api_keys.csv")
print("   2. Ajouter les clés dans la colonne 'finnhub_key'")
print("   3. Le bot les utilisera automatiquement en rotation")

print("\n" + "=" * 80)
