# -*- coding: utf-8 -*-
"""Script simple pour analyser le sentiment des news avec TextBlob (local, pas d'API externe)"""

import asyncio
import aiohttp
import os
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv
from textblob import TextBlob
import numpy as np

# Charger les variables d'environnement
load_dotenv()


class SimpleNewsSentiment:
    """Analyse le sentiment des news localement"""

    def __init__(self):
        self.finnhub_key = os.getenv('FINNHUB_KEY', '')
        self.newsapi_keys = self._load_newsapi_keys()

        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print(f"Finnhub Key: {'[OK]' if self.finnhub_key else '[X]'}")
        print(f"NewsAPI Keys: {len(self.newsapi_keys)} cle(s)")
        print(f"Methode: TextBlob (analyse locale)")
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

        # Fenetre: 48h avant la date cible
        from_date = target_date - timedelta(hours=48)
        to_date = target_date

        # 1. Finnhub
        if self.finnhub_key:
            print("\n[1/2] Finnhub...")
            finnhub_articles = await self._get_finnhub_news(symbol, from_date, to_date)
            all_articles.extend(finnhub_articles)
            print(f"      -> {len(finnhub_articles)} articles")

        # 2. NewsAPI
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
                                    'date': pub_date
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

        # NewsAPI gratuit: limite de 30 jours
        now = datetime.now()
        days_ago = (now - to_date).days

        if days_ago > 30:
            print(f"      [!] NewsAPI limite: date trop ancienne ({days_ago} jours)")
            print(f"      [!] NewsAPI gratuit = max 30 jours dans le passe")
            return articles

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{search_term} OR {symbol}",
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.newsapi_keys[0] if self.newsapi_keys else '',
                'pageSize': 100
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        total_results = data.get('totalResults', 0)

                        print(f"      NewsAPI: {total_results} resultats disponibles")

                        for article in data.get('articles', []):
                            title = article.get('title', '')
                            description = article.get('description', '')
                            content = article.get('content', '')
                            pub_str = article.get('publishedAt', '')

                            if pub_str and title:
                                try:
                                    pub_date = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
                                    pub_date = pub_date.replace(tzinfo=None)

                                    # Pas de filtrage supplementaire, NewsAPI a deja filtre
                                    articles.append({
                                        'title': title,
                                        'content': description or content or title,
                                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                                        'date': pub_date
                                    })
                                except Exception as e:
                                    continue
                    elif response.status == 429:
                        print(f"      [!] NewsAPI limite atteinte (429)")
                    elif response.status == 426:
                        text = await response.text()
                        print(f"      [!] NewsAPI erreur 426: {text[:200]}")
                    else:
                        text = await response.text()
                        print(f"      Erreur NewsAPI {response.status}: {text[:200]}")
        except Exception as e:
            print(f"      Erreur NewsAPI: {e}")

        return articles

    def analyze_sentiment(self, symbol, articles, target_date):
        """Analyse le sentiment avec TextBlob"""
        print("\n" + "=" * 70)
        print("ANALYSE DE SENTIMENT (TextBlob)")
        print("=" * 70)

        if not articles:
            print("[X] Aucun article a analyser")
            return None

        # Keywords pour detecter l'importance
        positive_keywords = {
            'earnings beat': 3.0, 'beats': 2.5, 'profit': 2.0, 'surge': 2.5,
            'breakthrough': 2.5, 'record': 2.0, 'growth': 1.5, 'upgrade': 2.0,
            'partnership': 1.5, 'acquisition': 2.0, 'approval': 2.0,
            'bullish': 2.0, 'gains': 1.5, 'jumps': 2.0, 'soars': 2.5
        }

        negative_keywords = {
            'earnings miss': 3.0, 'misses': 2.5, 'loss': 2.0, 'plunge': 2.5,
            'crash': 3.0, 'downgrade': 2.0, 'lawsuit': 2.0, 'investigation': 2.0,
            'recall': 2.0, 'bankruptcy': 3.0, 'bearish': 2.0, 'falls': 1.5,
            'drops': 1.5, 'slides': 1.5, 'tumbles': 2.0, 'slumps': 2.0
        }

        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        print(f"\nAnalyse de {len(articles)} articles...\n")

        for i, article in enumerate(articles, 1):
            text = f"{article['title']} {article['content']}"
            if len(text) < 10:
                continue

            # Analyse TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 (negatif) a +1 (positif)

            # Detecter les keywords importants
            text_lower = text.lower()
            importance = 1.0

            for keyword, weight in positive_keywords.items():
                if keyword in text_lower:
                    importance += weight

            for keyword, weight in negative_keywords.items():
                if keyword in text_lower:
                    importance += weight

            # Ponderer le sentiment par l'importance
            weighted_sentiment = polarity * importance
            sentiments.append(weighted_sentiment)

            # Compter les sentiments
            if polarity > 0.1:
                positive_count += 1
            elif polarity < -0.1:
                negative_count += 1
            else:
                neutral_count += 1

            # Afficher quelques exemples
            if i <= 5:
                sentiment_str = "POSITIF" if polarity > 0.1 else "NEGATIF" if polarity < -0.1 else "NEUTRE"
                print(f"{i}. [{sentiment_str} {polarity:+.2f}] {article['title'][:60]}...")

        # Calculer le score final
        if sentiments:
            avg_sentiment = np.mean(sentiments)

            # Convertir en score 0-100
            # avg_sentiment est entre -5 et +5 environ
            score = ((avg_sentiment + 3) / 6) * 100
            score = max(0, min(100, score))

            # Determiner le sentiment global
            if score > 60:
                overall_sentiment = "POSITIF"
            elif score < 40:
                overall_sentiment = "NEGATIF"
            else:
                overall_sentiment = "NEUTRE"

            # Calculer les ratios
            total = positive_count + negative_count + neutral_count
            pos_ratio = (positive_count / total * 100) if total > 0 else 0
            neg_ratio = (negative_count / total * 100) if total > 0 else 0
            neu_ratio = (neutral_count / total * 100) if total > 0 else 0

            print("\n" + "=" * 70)
            print("RESULTAT FINAL")
            print("=" * 70)
            print(f"Score: {score:.0f}/100")
            print(f"Sentiment: {overall_sentiment}")
            print(f"Ratio: {pos_ratio:.0f}% positif, {neg_ratio:.0f}% negatif, {neu_ratio:.0f}% neutre")
            print(f"Articles analyses: {len(articles)}")
            print(f"Sentiment moyen brut: {avg_sentiment:.2f}")
            print()

            return {
                'score': int(score),
                'sentiment': overall_sentiment,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'positive_ratio': pos_ratio,
                'negative_ratio': neg_ratio,
                'neutral_ratio': neu_ratio,
                'articles_count': len(articles),
                'raw_sentiment': avg_sentiment
            }

        return None

    async def analyze_date(self, symbol, date_str):
        """Fonction principale"""
        print("\n")
        print("+" + "=" * 68 + "+")
        print("|" + " " * 12 + f"ANALYSE SENTIMENT NEWS - {symbol} - {date_str}" + " " * 12 + "|")
        print("+" + "=" * 68 + "+")
        print()

        target_date = datetime.strptime(date_str, '%Y-%m-%d')

        # Recuperer les news
        articles = await self.get_news_for_date(symbol, target_date)

        if not articles:
            print("\n[!] Aucun article trouve")
            return

        # Analyser le sentiment
        result = self.analyze_sentiment(symbol, articles, target_date)

        return result


async def main():
    """Fonction principale"""
    analyzer = SimpleNewsSentiment()

    # Tester plusieurs dates RECENTES (NewsAPI gratuit = 30 jours max)
    # Pour dates historiques, seul Finnhub fonctionnera

    print("=" * 70)
    print("IMPORTANT: NewsAPI gratuit limite a 30 jours dans le passe")
    print("Pour dates anciennes, seul Finnhub retournera des resultats")
    print("=" * 70)
    print()

    # Exemple avec date recente (NewsAPI + Finnhub)
    await analyzer.analyze_date('NVDA', '2025-11-05')
    print("\n" + "=" * 70 + "\n")
    await asyncio.sleep(2)

    # Exemple avec date ancienne (Finnhub seulement)
    await analyzer.analyze_date('NVDA', '2025-07-15')
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
