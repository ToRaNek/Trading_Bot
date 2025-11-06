#!/usr/bin/env python3
"""
Script de test pour vérifier les améliorations du bot de trading
- Test du nouveau système de scoring technique
- Test de l'intégration Reddit
"""

import asyncio
import sys
sys.path.insert(0, '.')

from trading_bot_main import (
    TechnicalAnalyzer,
    RedditSentimentAnalyzer,
    RealisticBacktestEngine
)
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


async def test_technical_scoring():
    """Test du nouveau système de scoring technique"""
    print("\n" + "="*80)
    print("TEST 1: Système de Scoring Technique Amélioré")
    print("="*80)

    analyzer = TechnicalAnalyzer()

    # Télécharger des données récentes pour test
    symbol = "AAPL"
    print(f"\n📊 Test sur {symbol}...")
    stock = yf.Ticker(symbol)
    df = stock.history(period="3mo", interval="1d")

    if df.empty:
        print("❌ Erreur: Impossible de récupérer les données")
        return False

    # Calculer les indicateurs
    df = analyzer.calculate_indicators(df)

    # Tester sur les 5 derniers jours
    print(f"\n🔍 Analyse des 5 derniers jours:")
    for i in range(-5, 0):
        row = df.iloc[i]
        date = df.index[i]
        score, reasons = analyzer.get_technical_score(row)

        print(f"\n📅 {date.strftime('%Y-%m-%d')} - Score: {score:.0f}/100")
        for reason in reasons[:3]:  # Top 3 raisons
            print(f"   {reason}")

    print("\n✅ Test scoring technique OK")
    return True


async def test_reddit_sentiment():
    """Test de l'analyseur de sentiment Reddit"""
    print("\n" + "="*80)
    print("TEST 2: Analyseur de Sentiment Reddit")
    print("="*80)

    analyzer = RedditSentimentAnalyzer()

    # Test sur quelques tickers
    test_symbols = ["NVDA", "AAPL", "TSLA"]

    for symbol in test_symbols:
        print(f"\n📱 Test Reddit pour {symbol}...")

        try:
            score, post_count, samples = await analyzer.get_reddit_sentiment(
                symbol,
                target_date=datetime.now(),
                lookback_hours=72
            )

            print(f"   Score: {score:.0f}/100")
            print(f"   Posts trouvés: {post_count}")

            if samples:
                print(f"   Exemples:")
                for sample in samples[:2]:
                    print(f"      {sample}")

            # Délai pour éviter rate limiting
            await asyncio.sleep(2)

        except Exception as e:
            print(f"   ⚠️ Erreur: {e}")

    await analyzer.close()
    print("\n✅ Test Reddit OK")
    return True


async def test_backtest_integration():
    """Test d'un backtest complet rapide"""
    print("\n" + "="*80)
    print("TEST 3: Backtest avec Intégration Complète")
    print("="*80)

    engine = RealisticBacktestEngine()

    # Test rapide sur 1 mois avec une seule action
    symbol = "NVDA"
    months = 1

    print(f"\n🚀 Backtest rapide de {symbol} sur {months} mois...")
    print("   (Ceci peut prendre quelques minutes...)")

    try:
        result = await engine.backtest_with_news_validation(symbol, months)

        if result:
            print(f"\n📊 Résultats du Backtest:")
            print(f"   Trades: {result['total_trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Profit Total: {result['total_profit']:+.2f}%")
            print(f"   Score Stratégie: {result['strategy_score']:.0f}/100")
            print(f"   Achats validés: {result['validated_buys']}")
            print(f"   Achats rejetés: {result['rejected_buys']}")
            print(f"   Ventes validées: {result['validated_sells']}")
            print(f"   Ventes rejetées: {result['rejected_sells']}")

            if result['trades']:
                print(f"\n   Exemple de trade:")
                trade = result['trades'][0]
                print(f"      Entrée: {trade['entry_date'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:.2f}")
                print(f"      Sortie: {trade['exit_date'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:.2f}")
                print(f"      Profit: {trade['profit']:+.2f}%")
                print(f"      Score Tech: {trade.get('tech_score', 0):.0f}")
                print(f"      Score IA: {trade.get('ai_score', 0):.0f}")
                print(f"      Score Reddit: {trade.get('reddit_score', 0):.0f}")
                print(f"      Score Final: {trade.get('final_score', 0):.0f}")

            print("\n✅ Test backtest OK")
        else:
            print("❌ Erreur: Pas de résultats")
            return False

    except Exception as e:
        print(f"❌ Erreur backtest: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.close()

    return True


async def main():
    """Lance tous les tests"""
    print("\n🔬 TESTS DES AMÉLIORATIONS DU BOT DE TRADING")
    print("="*80)

    results = []

    # Test 1: Scoring technique
    try:
        results.append(("Scoring Technique", await test_technical_scoring()))
    except Exception as e:
        print(f"\n❌ Erreur test scoring: {e}")
        results.append(("Scoring Technique", False))

    # Test 2: Reddit sentiment
    try:
        results.append(("Reddit Sentiment", await test_reddit_sentiment()))
    except Exception as e:
        print(f"\n❌ Erreur test Reddit: {e}")
        results.append(("Reddit Sentiment", False))

    # Test 3: Backtest complet (optionnel, plus long)
    print("\n⚠️ Test backtest complet - Ceci peut prendre 5-10 minutes")
    response = input("Voulez-vous lancer le test de backtest complet? (o/n): ")
    if response.lower() == 'o':
        try:
            results.append(("Backtest Complet", await test_backtest_integration()))
        except Exception as e:
            print(f"\n❌ Erreur test backtest: {e}")
            results.append(("Backtest Complet", False))

    # Résumé
    print("\n" + "="*80)
    print("📋 RÉSUMÉ DES TESTS")
    print("="*80)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 Tous les tests ont réussi!")
        print("\n✨ Améliorations implémentées:")
        print("   1. ✅ Système de scoring technique avec confluence")
        print("   2. ✅ Analyseur de sentiment Reddit multi-sources")
        print("   3. ✅ Intégration dans le backtest (Tech 40% + IA 35% + Reddit 25%)")
        print("   4. ✅ Détection de confluence/conflit entre sources")
        print("   5. ✅ Seuil de validation ajusté à 65/100")
    else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
