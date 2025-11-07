# -*- coding: utf-8 -*-
"""Test simple pour recuperer les news NVDA depuis Finnhub et NewsAPI"""

import asyncio
import aiohttp
import os
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()


class SimpleNewsTest:
    """Test simple des sources de news"""

    def __init__(self):
        # Charger les cles API
        self.finnhub_key = os.getenv('FINNHUB_KEY', '')
        self.newsapi_keys = self._load_newsapi_keys()
        self.current_newsapi_index = 0

        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print(f"Finnhub Key: {'[OK] Trouvee' if self.finnhub_key else '[X] Manquante'}")
        print(f"NewsAPI Keys: {len(self.newsapi_keys)} cle(s) chargee(s)")
        if self.newsapi_keys:
            for i, key in enumerate(self.newsapi_keys, 1):
                print(f"  Cle {i}: {key[:15]}...{key[-5:]}")
        print()

    def _load_newsapi_keys(self):
        """Charge les cles NewsAPI depuis le CSV"""
        keys = []
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'api_keys.csv'
        )

        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = row.get('newsapi_key', '').strip()
                        if key and key not in ['YOUR_NEWSAPI_KEY_1', 'YOUR_NEWSAPI_KEY_2', 'YOUR_NEWSAPI_KEY_3']:
                            keys.append(key)
            except Exception as e:
                print(f"Erreur lecture CSV: {e}")

        # Fallback sur variable d'environnement
        if not keys:
            env_key = os.getenv('NEWSAPI_KEY', '')
            if env_key:
                keys.append(env_key)

        return keys

    async def test_finnhub(self, symbol="NVDA"):
        """Test de recuperation depuis Finnhub"""
        print("=" * 70)
        print("TEST FINNHUB")
        print("=" * 70)

        if not self.finnhub_key:
            print("[X] Pas de cle Finnhub configuree (FINNHUB_KEY)")
            print()
            return

        try:
            # Date d'aujourd'hui - 7 jours
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)

            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }

            print(f"Requete: {symbol} du {from_date.strftime('%Y-%m-%d')} au {to_date.strftime('%Y-%m-%d')}")
            print(f"URL: {url}")
            print()

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    print(f"Status: {response.status}")

                    if response.status == 200:
                        data = await response.json()

                        if isinstance(data, list):
                            print(f"[OK] {len(data)} articles recus\n")

                            if data:
                                print("=" * 70)
                                print("PREMIERS ARTICLES")
                                print("=" * 70)

                                for i, article in enumerate(data[:5], 1):
                                    headline = article.get('headline', 'N/A')
                                    source = article.get('source', 'N/A')
                                    timestamp = article.get('datetime', 0)
                                    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp else 'N/A'
                                    summary = article.get('summary', 'N/A')

                                    print(f"\n{i}. {headline}")
                                    print(f"   Source: {source}")
                                    print(f"   Date: {date}")
                                    print(f"   Summary: {summary[:150]}...")

                                print()
                                print(f"Total: {len(data)} articles")
                            else:
                                print("[!] Aucun article trouve")
                        else:
                            print(f"[X] Format de reponse inattendu: {type(data)}")
                            print(f"Reponse: {data}")
                    else:
                        text = await response.text()
                        print(f"[X] Erreur {response.status}")
                        print(f"Reponse: {text[:200]}")

        except Exception as e:
            print(f"[X] Exception: {e}")
            import traceback
            traceback.print_exc()

        print()

    async def test_newsapi(self, symbol="NVDA", try_all_keys=True):
        """Test de recuperation depuis NewsAPI"""
        print("=" * 70)
        print("TEST NEWSAPI")
        print("=" * 70)

        if not self.newsapi_keys:
            print("[X] Pas de cle NewsAPI configuree")
            print()
            return

        # Mapping des symboles vers les noms de compagnies
        company_names = {
            'NVDA': 'Nvidia',
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'META': 'Meta Facebook',
            'GOOG': 'Google Alphabet'
        }

        search_term = company_names.get(symbol, symbol)

        # Date d'aujourd'hui - 7 jours
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)

        keys_to_try = self.newsapi_keys if try_all_keys else [self.newsapi_keys[self.current_newsapi_index]]

        for key_index, api_key in enumerate(keys_to_try, 1):
            print(f"\n{'-' * 70}")
            print(f"Essai avec cle {key_index}/{len(self.newsapi_keys)}: {api_key[:15]}...{api_key[-5:]}")
            print(f"{'-' * 70}")

            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f"{search_term} stock OR {symbol}",
                    'from': from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'to': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'apiKey': api_key,
                    'pageSize': 100
                }

                print(f"Requete: {search_term} ({symbol})")
                print(f"Periode: {from_date.strftime('%Y-%m-%d')} au {to_date.strftime('%Y-%m-%d')}")
                print(f"URL: {url}")
                print()

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=15) as response:
                        print(f"Status: {response.status}")

                        if response.status == 200:
                            data = await response.json()

                            total_results = data.get('totalResults', 0)
                            articles = data.get('articles', [])

                            print(f"[OK] {len(articles)} articles recus (total disponible: {total_results})")

                            if articles:
                                print()
                                print("=" * 70)
                                print("PREMIERS ARTICLES")
                                print("=" * 70)

                                for i, article in enumerate(articles[:5], 1):
                                    title = article.get('title', 'N/A')
                                    source_name = article.get('source', {}).get('name', 'N/A')
                                    published_at = article.get('publishedAt', 'N/A')
                                    description = article.get('description', 'N/A')
                                    url_article = article.get('url', 'N/A')

                                    print(f"\n{i}. {title}")
                                    print(f"   Source: {source_name}")
                                    print(f"   Date: {published_at}")
                                    print(f"   Description: {description[:150] if description else 'N/A'}...")
                                    print(f"   URL: {url_article[:80]}...")

                                print()
                                print(f"[OK] Cle {key_index} fonctionne! {len(articles)} articles recuperes")
                                break  # Succes, pas besoin d'essayer les autres cles
                            else:
                                print("[!] Aucun article trouve")
                                break

                        elif response.status == 429:
                            text = await response.text()
                            print(f"[X] Limite atteinte (429) - Cette cle est epuisee")
                            print(f"Reponse: {text[:200]}")
                            if key_index < len(keys_to_try):
                                print(f"[>] Passage a la cle suivante...")
                                await asyncio.sleep(1)
                                continue
                            else:
                                print("[X] Toutes les cles ont ete essayees")
                                break

                        elif response.status == 401:
                            text = await response.text()
                            print(f"[X] Erreur d'authentification (401) - Cle invalide")
                            print(f"Reponse: {text[:200]}")
                            if key_index < len(keys_to_try):
                                print(f"[>] Passage a la cle suivante...")
                                await asyncio.sleep(1)
                                continue
                            else:
                                print("[X] Toutes les cles ont ete essayees")
                                break

                        else:
                            text = await response.text()
                            print(f"[X] Erreur {response.status}")
                            print(f"Reponse: {text[:300]}")
                            break

            except Exception as e:
                print(f"[X] Exception: {e}")
                import traceback
                traceback.print_exc()

        print()

    async def test_both(self, symbol="NVDA"):
        """Test les deux sources"""
        print("\n")
        print("+" + "=" * 68 + "+")
        print("|" + " " * 18 + f" TEST RECUPERATION NEWS - {symbol} " + " " * 18 + "|")
        print("+" + "=" * 68 + "+")
        print("\n")

        # Test Finnhub
        await self.test_finnhub(symbol)

        # Petit delai entre les tests
        await asyncio.sleep(2)

        # Test NewsAPI
        await self.test_newsapi(symbol)

        print("=" * 70)
        print("TESTS TERMINES")
        print("=" * 70)
        print()


async def main():
    """Fonction principale"""
    tester = SimpleNewsTest()

    # Tester les deux sources pour NVDA
    await tester.test_both("NVDA")


if __name__ == "__main__":
    asyncio.run(main())
