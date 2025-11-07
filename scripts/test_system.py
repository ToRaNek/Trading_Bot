#!/usr/bin/env python3
"""
Script de validation du système v2.0
Vérifie que toutes les fonctionnalités sont actives
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test 1: Vérifier les imports"""
    print("\n" + "="*80)
    print("TEST 1: IMPORTS")
    print("="*80)

    try:
        from backtest import RealisticBacktestEngine
        from analyzers.ai_scorer import AIScorer
        from analyzers.news_analyzer import HistoricalNewsAnalyzer
        from analyzers.reddit_analyzer import RedditSentimentAnalyzer
        print("✅ Tous les imports fonctionnent")
        return True
    except Exception as e:
        print(f"❌ Erreur import: {e}")
        return False


def test_backtest_engine():
    """Test 2: Vérifier le backtest engine"""
    print("\n" + "="*80)
    print("TEST 2: BACKTEST ENGINE")
    print("="*80)

    try:
        from backtest import RealisticBacktestEngine

        engine = RealisticBacktestEngine(data_dir='data')

        # Vérifier stop loss / take profit
        assert engine.stop_loss_pct == -3.0, f"Stop loss incorrect: {engine.stop_loss_pct}"
        assert engine.take_profit_pct == 10.0, f"Take profit incorrect: {engine.take_profit_pct}"
        print(f"✅ Stop Loss: {engine.stop_loss_pct}%")
        print(f"✅ Take Profit: {engine.take_profit_pct}%")

        # Vérifier les analyzers
        assert hasattr(engine, 'news_analyzer'), "Pas de news_analyzer"
        assert hasattr(engine, 'reddit_analyzer'), "Pas de reddit_analyzer"
        assert hasattr(engine, 'tech_analyzer'), "Pas de tech_analyzer"
        print("✅ Analyzers présents")

        # Vérifier AI Scorer
        assert hasattr(engine.news_analyzer, 'ai_scorer'), "Pas d'ai_scorer"
        print("✅ AI Scorer actif")

        # Vérifier data_dir
        assert hasattr(engine.reddit_analyzer, 'data_dir'), "Pas de data_dir"
        assert engine.reddit_analyzer.data_dir == 'data', f"data_dir incorrect: {engine.reddit_analyzer.data_dir}"
        print(f"✅ Data dir: {engine.reddit_analyzer.data_dir}")

        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_files():
    """Test 3: Vérifier les fichiers CSV"""
    print("\n" + "="*80)
    print("TEST 3: FICHIERS CSV")
    print("="*80)

    data_dir = Path('data')
    csv_files = list(data_dir.glob('Sentiment_*.csv'))

    if csv_files:
        print(f"✅ {len(csv_files)} fichiers CSV trouvés:")
        for csv_file in sorted(csv_files):
            ticker = csv_file.stem.replace('Sentiment_', '')
            size_kb = csv_file.stat().st_size / 1024
            print(f"   - {ticker}: {size_kb:.1f} KB")
        return True
    else:
        print("⚠️  Aucun fichier CSV trouvé dans data/")
        print("   Exécutez: python scripts/scrape_all_stocks.py")
        return False


def test_config():
    """Test 4: Vérifier la configuration"""
    print("\n" + "="*80)
    print("TEST 4: CONFIGURATION")
    print("="*80)

    try:
        from config_stocks import STOCK_CONFIGS, get_all_tickers

        tickers = get_all_tickers()
        print(f"✅ {len(tickers)} actions configurées")

        # Vérifier quelques configurations
        for ticker in ['NVDA', 'AAPL', 'META']:
            if ticker in STOCK_CONFIGS:
                sources = STOCK_CONFIGS[ticker]['sources']
                print(f"✅ {ticker}: {len(sources)} source(s)")

        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_reddit_score_zero():
    """Test 5: Vérifier que le score Reddit = 0 si pas de posts"""
    print("\n" + "="*80)
    print("TEST 5: SCORE REDDIT = 0")
    print("="*80)

    try:
        import asyncio
        from analyzers.reddit_analyzer import RedditSentimentAnalyzer

        async def check_score():
            analyzer = RedditSentimentAnalyzer(data_dir='data')

            # Simuler une demande sans données (ne devrait pas trouver de posts)
            from datetime import datetime, timedelta
            old_date = datetime(2020, 1, 1)  # Date très ancienne

            score, count, samples, posts = await analyzer.get_reddit_sentiment(
                'NONEXISTENT_TICKER', old_date, lookback_hours=1
            )

            await analyzer.close()

            if count == 0 and score == 0.0:
                print(f"✅ Score Reddit = {score} avec {count} posts")
                return True
            else:
                print(f"❌ Score Reddit = {score} avec {count} posts (attendu: 0.0)")
                return False

        return asyncio.run(check_score())
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Exécuter tous les tests"""
    print("\n" + "="*80)
    print("🧪 VALIDATION SYSTÈME v2.0")
    print("="*80)

    results = []

    # Exécuter les tests
    results.append(("Imports", test_imports()))
    results.append(("Backtest Engine", test_backtest_engine()))
    results.append(("Fichiers CSV", test_csv_files()))
    results.append(("Configuration", test_config()))
    results.append(("Score Reddit = 0", test_reddit_score_zero()))

    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")

    print("\n" + "="*80)
    print(f"RÉSULTAT: {passed}/{total} tests passés")

    if passed == total:
        print("✅ SYSTÈME VALIDÉ - Toutes les fonctionnalités sont actives !")
    else:
        print("⚠️  ATTENTION - Certaines fonctionnalités ne sont pas actives")

    print("="*80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
