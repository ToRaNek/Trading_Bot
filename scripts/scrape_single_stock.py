#!/usr/bin/env python3
"""
Script de scraping Reddit pour UNE SEULE action
Utile pour tester rapidement ou mettre à jour une action spécifique
"""

import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importer le scraper
from scrape_all_stocks import MultiStockScraper
from config_stocks import STOCK_CONFIGS, get_stock_config


async def scrape_single(ticker: str):
    """Scrape une seule action"""

    print("\n" + "="*80)
    print(f"🔍 SCRAPING: {ticker}")
    print("="*80)

    # Récupérer la configuration
    config = get_stock_config(ticker)

    if not config.get('sources'):
        print(f"❌ Aucune source configurée pour {ticker}")
        return

    print(f"\n📋 Sources configurées:")
    for i, source in enumerate(config['sources'], 1):
        if source['type'] == 'subreddit':
            print(f"   {i}. r/{source['name']}")
        elif source['type'] == 'search':
            print(f"   {i}. r/{source['subreddit']}/search?q={source['query']}")

    print("\n⚠️  Mode: Reddit API (récent) + Pushshift (historique)")
    print(f"📁 Output: data/Sentiment_{ticker}.csv\n")

    # Initialiser le scraper
    scraper = MultiStockScraper(output_dir='data')

    try:
        # Scraper l'action
        posts = await scraper.scrape_stock(ticker, config)

        if posts:
            scraper.save_to_csv(ticker, posts)
            print(f"\n✅ Terminé ! {len(posts)} posts sauvegardés")
        else:
            print(f"\n⚠️  Aucun post trouvé pour {ticker}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()


def main():
    """Point d'entrée principal"""

    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("\n" + "="*80)
        print("📖 USAGE")
        print("="*80)
        print(f"\npython {Path(__file__).name} TICKER")
        print("\nExemple:")
        print(f"  python {Path(__file__).name} NVDA")
        print(f"  python {Path(__file__).name} AAPL")

        print("\n" + "="*80)
        print("📋 ACTIONS DISPONIBLES")
        print("="*80)
        print("\nActions configurées:")
        for ticker in STOCK_CONFIGS.keys():
            print(f"  - {ticker}")

        print("\nVous pouvez également scraper n'importe quel ticker")
        print("(il utilisera r/stocks par défaut)")
        print("="*80 + "\n")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    asyncio.run(scrape_single(ticker))


if __name__ == "__main__":
    main()
