# -*- coding: utf-8 -*-
"""Script pour recuperer les news a une date precise et obtenir un score de sentiment via HuggingFace"""

import asyncio
import aiohttp
import os
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class NewsSentimentAnalyzer:
    """Analyse le sentiment des news a une date precise"""

    def __init__(self):
        self.finnhub_key = os.getenv('FINNHUB_KEY', '')
        self.newsapi_keys = self._load_newsapi_keys()
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN', '')

        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print(f"Finnhub Key: {'[OK]' if self.finnhub_key else '[X]'}")
        print(f"NewsAPI Keys: {len(self.newsapi_keys)} cle(s)")
        print(f"HuggingFace Token: {'[OK]' if self.hf_token else '[X]'}")
        print()

    def _load_newsapi_keys(self):
        """Charge les cles NewsAPI"""
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
                print(f"Erreur CSV: {e}")

        if not keys:
            env_key = os.getenv('NEWSAPI_KEY', '')
            if env_key:
                keys.append(env_key)

        return keys

    async def get_news_for_date(self, symbol, target_date):
        """Recupere les news pour une date precise"""
        print("=" * 70)
        print(f"RECUPERATION NEWS - {symbol} @ {target_date.strftime('%Y-%m-%d')}")
        print("=" * 70)

        all_articles = []

        # Fenetre: 48h avant la date cible (pour avoir le contexte)
        from_date = target_date - timedelta(hours=48)
        to_date = target_date

        # 1. Essayer Finnhub
        if self.finnhub_key:
            print("\n[1/2] Finnhub...")
            finnhub_articles = await self._get_finnhub_news(symbol, from_date, to_date)
            all_articles.extend(finnhub_articles)
            print(f"      -> {len(finnhub_articles)} articles")

        # 2. Essayer NewsAPI
        if self.newsapi_keys:
            print("[2/2] NewsAPI...")
            newsapi_articles = await self._get_newsapi_news(symbol, from_date, to_date)
            all_articles.extend(newsapi_articles)
            print(f"      -> {len(newsapi_articles)} articles")

        print(f"\nTOTAL: {len(all_articles)} articles recuperes")
        return all_articles

    async def _get_finnhub_news(self, symbol, from_date, to_date):
        """Recupere depuis Finnhub"""
        articles = []

        try:
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()

                        for article in data:
                            headline = article.get('headline', '')
                            summary = article.get('summary', '')
                            timestamp = article.get('datetime', 0)
                            pub_date = datetime.fromtimestamp(timestamp)

                            if from_date <= pub_date <= to_date:
                                articles.append({
                                    'title': headline,
                                    'content': summary,
                                    'source': article.get('source', 'Finnhub'),
                                    'date': pub_date,
                                    'url': article.get('url', '')
                                })
        except Exception as e:
            print(f"      Erreur Finnhub: {e}")

        return articles

    async def _get_newsapi_news(self, symbol, from_date, to_date):
        """Recupere depuis NewsAPI"""
        articles = []

        company_names = {
            'NVDA': 'Nvidia', 'AAPL': 'Apple', 'MSFT': 'Microsoft',
            'TSLA': 'Tesla', 'AMZN': 'Amazon', 'META': 'Meta Facebook',
            'GOOG': 'Google Alphabet'
        }

        search_term = company_names.get(symbol, symbol)

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{search_term} stock OR {symbol}",
                'from': from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'to': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.newsapi_keys[0] if self.newsapi_keys else '',
                'pageSize': 100
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()

                        for article in data.get('articles', []):
                            title = article.get('title', '')
                            description = article.get('description', '')
                            content = article.get('content', '')
                            pub_str = article.get('publishedAt', '')

                            if pub_str:
                                pub_date = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
                                pub_date = pub_date.replace(tzinfo=None)

                                if from_date <= pub_date <= to_date:
                                    articles.append({
                                        'title': title,
                                        'content': description or content,
                                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                                        'date': pub_date,
                                        'url': article.get('url', '')
                                    })
        except Exception as e:
            print(f"      Erreur NewsAPI: {e}")

        return articles

    async def analyze_sentiment_with_hf(self, symbol, articles, target_date):
        """Envoie les articles a HuggingFace pour analyse de sentiment"""
        print("\n" + "=" * 70)
        print("ANALYSE DE SENTIMENT VIA HUGGINGFACE")
        print("=" * 70)

        if not self.hf_token:
            print("[X] Token HuggingFace manquant")
            return None

        if not articles:
            print("[X] Aucun article a analyser")
            return None

        # Construire le prompt avec tous les articles
        print(f"\nPreparation de {len(articles)} articles pour l'IA...")

        articles_text = ""
        for i, article in enumerate(articles[:50], 1):  # Limiter a 50 articles max
            articles_text += f"\n{i}. [{article['source']}] {article['title']}\n"
            if article['content']:
                articles_text += f"   {article['content'][:200]}...\n"

        prompt = f"""ANALYSE DE SENTIMENT DES NEWS - {symbol}
Date d'analyse: {target_date.strftime('%Y-%m-%d')}
Nombre d'articles: {len(articles)}

ARTICLES:
{articles_text}

TACHE:
Analyse le sentiment general de ces articles de news pour {symbol}.

REPONDS EXACTEMENT DANS CE FORMAT:
SCORE: [0-100]
SENTIMENT: [POSITIF/NEGATIF/NEUTRE]
RATIO: [X% positif, Y% negatif, Z% neutre]
RAISON: [Explication courte en 2-3 phrases]

ECHELLE DE SCORE:
- 0-30: Tres negatif (mauvaises nouvelles, crash, problemes)
- 31-45: Negatif (nouvelles defavorables)
- 46-55: Neutre (nouvelles mixtes ou neutres)
- 56-70: Positif (bonnes nouvelles)
- 71-100: Tres positif (excellentes nouvelles, croissance)

Le score doit refleter l'impact probable sur le prix de l'action."""

        print(f"Taille du prompt: {len(prompt)} caracteres")
        print("\nEnvoi a HuggingFace...")

        try:
            # Nouvelle API HuggingFace
            url = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "return_full_text": False
                },
                "options": {"wait_for_model": True}
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()

                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get('generated_text', '')

                            print("\n" + "=" * 70)
                            print("REPONSE DE L'IA")
                            print("=" * 70)
                            print(generated_text)
                            print()

                            # Parser la reponse
                            score = self._extract_score(generated_text)
                            sentiment = self._extract_sentiment(generated_text)

                            return {
                                'score': score,
                                'sentiment': sentiment,
                                'full_response': generated_text,
                                'articles_count': len(articles)
                            }
                        else:
                            print(f"[X] Format de reponse inattendu: {result}")
                    elif response.status == 503:
                        text = await response.text()
                        print(f"[!] Modele en cours de chargement (503), reessai dans 20s...")
                        print(f"    Reponse: {text[:200]}")
                        await asyncio.sleep(20)
                        # Retry une fois
                        return await self._retry_hf_request(url, headers, payload)
                    else:
                        text = await response.text()
                        print(f"[X] Erreur {response.status}: {text[:200]}")

        except Exception as e:
            print(f"[X] Exception: {e}")
            import traceback
            traceback.print_exc()

        return None

    async def _retry_hf_request(self, url, headers, payload):
        """Retry HuggingFace request une fois"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get('generated_text', '')
                            print("\n" + "=" * 70)
                            print("REPONSE DE L'IA (apres retry)")
                            print("=" * 70)
                            print(generated_text)
                            print()

                            score = self._extract_score(generated_text)
                            sentiment = self._extract_sentiment(generated_text)

                            return {
                                'score': score,
                                'sentiment': sentiment,
                                'full_response': generated_text
                            }
                    else:
                        text = await response.text()
                        print(f"[X] Erreur lors du retry {response.status}: {text[:200]}")
        except Exception as e:
            print(f"[X] Exception lors du retry: {e}")

        return None

    def _extract_score(self, text):
        """Extrait le score de la reponse"""
        try:
            if 'SCORE:' in text:
                score_line = text.split('SCORE:')[1].split('\n')[0]
                score = int(''.join(filter(str.isdigit, score_line))[:3])
                return max(0, min(100, score))
        except:
            pass
        return 50  # Score par defaut

    def _extract_sentiment(self, text):
        """Extrait le sentiment de la reponse"""
        text_upper = text.upper()
        if 'POSITIF' in text_upper:
            return 'POSITIF'
        elif 'NEGATIF' in text_upper or 'NÉGATIF' in text_upper:
            return 'NEGATIF'
        else:
            return 'NEUTRE'

    async def analyze_date(self, symbol, date_str):
        """Fonction principale: analyse une date complete"""
        print("\n")
        print("+" + "=" * 68 + "+")
        print("|" + " " * 15 + f"ANALYSE NEWS SENTIMENT - {symbol}" + " " * 15 + "|")
        print("+" + "=" * 68 + "+")
        print()

        # Parser la date
        target_date = datetime.strptime(date_str, '%Y-%m-%d')

        # 1. Recuperer les news
        articles = await self.get_news_for_date(symbol, target_date)

        if not articles:
            print("\n[!] Aucun article trouve pour cette date")
            return

        # Afficher quelques exemples
        print("\n" + "-" * 70)
        print("EXEMPLES D'ARTICLES")
        print("-" * 70)
        for i, article in enumerate(articles[:3], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']} | Date: {article['date'].strftime('%Y-%m-%d %H:%M')}")
            print(f"   Content: {article['content'][:150]}...")
        print()

        # 2. Analyser avec HuggingFace
        result = await self.analyze_sentiment_with_hf(symbol, articles, target_date)

        if result:
            print("=" * 70)
            print("RESULTAT FINAL")
            print("=" * 70)
            print(f"Score: {result['score']}/100")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Articles analyses: {result['articles_count']}")
            print()


async def main():
    """Fonction principale"""
    analyzer = NewsSentimentAnalyzer()

    # Exemple: analyser NVDA a une date precise
    # Vous pouvez changer la date ici
    await analyzer.analyze_date('NVDA', '2025-07-15')

    # Pour tester plusieurs dates:
    # await analyzer.analyze_date('NVDA', '2025-06-10')
    # await analyzer.analyze_date('NVDA', '2025-06-20')


if __name__ == "__main__":
    asyncio.run(main())
