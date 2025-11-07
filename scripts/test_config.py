#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration des actions
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_stocks import STOCK_CONFIGS, get_stock_config, get_all_tickers


def main():
    print("\n" + "="*80)
    print("🧪 TEST CONFIGURATION DES ACTIONS")
    print("="*80)

    # Lister toutes les actions
    tickers = get_all_tickers()
    print(f"\n✅ {len(tickers)} actions configurées:")
    for ticker in tickers:
        print(f"   - {ticker}")

    # Tester get_stock_config pour chaque action
    print("\n" + "="*80)
    print("📋 DÉTAILS DES CONFIGURATIONS")
    print("="*80)

    for ticker in tickers:
        config = get_stock_config(ticker)
        sources = config.get('sources', [])

        print(f"\n{ticker}:")
        for i, source in enumerate(sources, 1):
            if source['type'] == 'subreddit':
                print(f"   {i}. Subreddit: r/{source['name']}")
            elif source['type'] == 'search':
                print(f"   {i}. Recherche: r/{source['subreddit']}/search?q={source['query']}")

    # Tester get_stock_config avec un ticker non configuré
    print("\n" + "="*80)
    print("🔍 TEST TICKER NON CONFIGURÉ")
    print("="*80)

    test_ticker = 'MSFT'
    config = get_stock_config(test_ticker)
    print(f"\n{test_ticker} (non configuré):")
    print(f"   Configuration par défaut: r/stocks/search?q=${test_ticker}")
    print(f"   Sources: {config['sources']}")

    # Vérifier les fichiers CSV existants
    print("\n" + "="*80)
    print("📁 FICHIERS CSV EXISTANTS")
    print("="*80)

    data_dir = Path(__file__).parent.parent / 'data'
    csv_files = list(data_dir.glob('Sentiment_*.csv'))

    if csv_files:
        print(f"\n✅ {len(csv_files)} fichiers trouvés:")
        for csv_file in csv_files:
            # Extraire le ticker du nom de fichier
            ticker = csv_file.stem.replace('Sentiment_', '')
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"   - {ticker}: {csv_file.name} ({size_mb:.2f} MB)")
    else:
        print("\n⚠️  Aucun fichier CSV trouvé dans data/")
        print("   Exécutez 'python scripts/scrape_all_stocks.py' pour générer les données")

    print("\n" + "="*80)
    print("✅ TEST TERMINÉ")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
