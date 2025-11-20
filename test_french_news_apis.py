"""
Test de différentes APIs pour récupérer les news françaises (CAC 40)
Finnhub ne fonctionne pas (403), on teste les alternatives
"""

import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()


async def test_alpha_vantage(symbol: str):
    """
    Alpha Vantage - Gratuit (25 requêtes/jour)
    Supporte CAC 40 avec suffix .PAR
    """
    print(f"\n[Alpha Vantage] Test de {symbol}...")

    # Convertir MC.PA -> MC.PAR (format Alpha Vantage)
    av_symbol = symbol.replace('.PA', '.PAR')

    api_key = "demo"  # Utilise d'abord la demo key

    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': av_symbol,
        'apikey': api_key,
        'limit': 50
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    if 'feed' in data:
                        news_count = len(data['feed'])
                        print(f"   ✅ {news_count} news trouvées")

                        # Afficher les 3 premières
                        for i, article in enumerate(data['feed'][:3], 1):
                            title = article.get('title', 'N/A')
                            sentiment = article.get('overall_sentiment_label', 'N/A')
                            print(f"   {i}. [{sentiment}] {title[:70]}...")

                        return True, news_count
                    else:
                        print(f"   ❌ Pas de news dans la réponse")
                        print(f"   Réponse: {data}")
                        return False, 0
                else:
                    print(f"   ❌ Erreur {response.status}")
                    text = await response.text()
                    print(f"   {text[:200]}")
                    return False, 0

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, 0


async def test_newsapi_french(symbol: str):
    """
    NewsAPI - Gratuit (100 requêtes/jour)
    On l'utilise déjà mais testons avec meilleurs termes de recherche
    """
    print(f"\n[NewsAPI] Test de {symbol}...")

    company_names = {
        'MC.PA': 'LVMH',
        'OR.PA': 'L\'Oréal',
        'AIR.PA': 'Airbus',
        'SAN.PA': 'Sanofi',
        'TTE.PA': 'TotalEnergies',
        'BNP.PA': 'BNP Paribas',
    }

    search_term = company_names.get(symbol, symbol.replace('.PA', ''))

    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        print("   ❌ Pas de clé NewsAPI dans .env")
        return False, 0

    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f"{search_term}",
        'sortBy': 'publishedAt',
        'language': 'en',  # Tester aussi 'fr'
        'apiKey': api_key,
        'pageSize': 50
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get('status') == 'ok':
                        news_count = len(data.get('articles', []))
                        print(f"   ✅ {news_count} news trouvées")

                        # Afficher les 3 premières
                        for i, article in enumerate(data.get('articles', [])[:3], 1):
                            title = article.get('title', 'N/A')
                            source = article.get('source', {}).get('name', 'N/A')
                            print(f"   {i}. [{source}] {title[:70]}...")

                        return True, news_count
                    else:
                        print(f"   ❌ Status: {data.get('status')}")
                        return False, 0
                else:
                    print(f"   ❌ Erreur {response.status}")
                    return False, 0

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, 0


async def test_eodhd(symbol: str):
    """
    EODHD - Gratuit avec limites
    Supporte 60+ bourses dont Paris
    """
    print(f"\n[EODHD] Test de {symbol}...")

    # EODHD utilise le format MC.PA
    api_key = "demo"  # Utilise d'abord la demo key

    url = f"https://eodhd.com/api/news"
    params = {
        's': symbol,
        'api_token': api_key,
        'limit': 50,
        'fmt': 'json'
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    if isinstance(data, list):
                        news_count = len(data)
                        print(f"   ✅ {news_count} news trouvées")

                        # Afficher les 3 premières
                        for i, article in enumerate(data[:3], 1):
                            title = article.get('title', 'N/A')
                            print(f"   {i}. {title[:70]}...")

                        return True, news_count
                    else:
                        print(f"   ❌ Format de réponse inattendu")
                        print(f"   {data}")
                        return False, 0
                else:
                    print(f"   ❌ Erreur {response.status}")
                    text = await response.text()
                    print(f"   {text[:200]}")
                    return False, 0

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, 0


async def test_marketaux(symbol: str):
    """
    Marketaux - API gratuite spécialisée news financières
    """
    print(f"\n[Marketaux] Test de {symbol}...")

    company_names = {
        'MC.PA': 'LVMH',
        'AIR.PA': 'Airbus',
        'TTE.PA': 'TotalEnergies',
    }

    search_term = company_names.get(symbol, symbol.replace('.PA', ''))

    url = "https://api.marketaux.com/v1/news/all"
    params = {
        'symbols': symbol,
        'filter_entities': 'true',
        'language': 'en',
        'limit': 50,
        'api_token': 'demo'  # Demo key
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    if 'data' in data:
                        news_count = len(data['data'])
                        print(f"   ✅ {news_count} news trouvées")

                        # Afficher les 3 premières
                        for i, article in enumerate(data['data'][:3], 1):
                            title = article.get('title', 'N/A')
                            print(f"   {i}. {title[:70]}...")

                        return True, news_count
                    else:
                        print(f"   ❌ Pas de news dans la réponse")
                        print(f"   {data}")
                        return False, 0
                else:
                    print(f"   ❌ Erreur {response.status}")
                    text = await response.text()
                    print(f"   {text[:200]}")
                    return False, 0

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, 0


async def main():
    print("=" * 100)
    print("TEST DES APIs POUR NEWS FRANÇAISES (CAC 40)")
    print("=" * 100)
    print("\nObjectif: Trouver une alternative à Finnhub qui donne des 403 sur les actions .PA")
    print("\nActions testées:")

    test_symbols = ['MC.PA', 'AIR.PA', 'TTE.PA']

    results = {}

    for api_name, api_func in [
        ('Alpha Vantage', test_alpha_vantage),
        ('NewsAPI', test_newsapi_french),
        ('EODHD', test_eodhd),
        ('Marketaux', test_marketaux)
    ]:
        print(f"\n{'=' * 100}")
        print(f"API: {api_name}")
        print("=" * 100)

        api_results = []

        for symbol in test_symbols:
            success, count = await api_func(symbol)
            api_results.append({'symbol': symbol, 'success': success, 'count': count})
            await asyncio.sleep(1)  # Rate limiting

        results[api_name] = api_results

    # RECAP
    print("\n" + "=" * 100)
    print("RECAPITULATIF:")
    print("=" * 100)

    print(f"\n{'API':<20} {'MC.PA':>10} {'AIR.PA':>10} {'TTE.PA':>10} {'Total':>10}")
    print("-" * 100)

    for api_name, api_results in results.items():
        counts = [r['count'] for r in api_results]
        total = sum(counts)
        status = "✅" if total > 0 else "❌"

        print(f"{api_name:<20} {counts[0]:>10} {counts[1]:>10} {counts[2]:>10} {total:>10} {status}")

    print("\n" + "=" * 100)
    print("RECOMMANDATION:")
    print("=" * 100)

    # Trouver la meilleure API
    best_api = max(results.items(), key=lambda x: sum(r['count'] for r in x[1]))
    best_name = best_api[0]
    best_total = sum(r['count'] for r in best_api[1])

    if best_total > 0:
        print(f"\n✅ Meilleure API: {best_name} ({best_total} news au total)")
        print(f"\nUtilise {best_name} comme source secondaire pour les actions françaises (.PA)")
        print(f"Finnhub reste pour les actions US (AAPL, MSFT, etc.)")
    else:
        print("\n❌ Aucune API gratuite ne fonctionne bien pour les actions françaises")
        print("\nSolution: Continuer avec NewsAPI uniquement (déjà utilisé)")


if __name__ == "__main__":
    asyncio.run(main())
