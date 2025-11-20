"""
Test de différentes améliorations pour TextBlob
Pour réduire l'écart avec Claude
"""

import asyncio
import os
import sys
from datetime import datetime
import aiohttp
from dotenv import load_dotenv
from textblob import TextBlob
import numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzers.news_analyzer import HistoricalNewsAnalyzer

load_dotenv()


class ImprovedTextBlobScorer:
    """Différentes versions améliorées de TextBlob"""

    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.session = None

    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    def version_1_basic(self, news_items: list) -> float:
        """Version actuelle (baseline)"""
        if not news_items:
            return 50.0

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
        for article in news_items[:50]:
            title = article.get('title', '')
            summary = article.get('summary', '')
            text = f"{title} {summary}"

            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            text_lower = text.lower()
            importance = 1.0

            for keyword, weight in positive_keywords.items():
                if keyword in text_lower:
                    importance += weight

            for keyword, weight in negative_keywords.items():
                if keyword in text_lower:
                    importance += weight

            weighted_sentiment = polarity * importance
            sentiments.append(weighted_sentiment)

        if sentiments:
            avg_sentiment = np.mean(sentiments)
            amplified_sentiment = avg_sentiment * 3.0
            score = ((amplified_sentiment + 3) / 6) * 100
            score = max(0, min(100, score))
        else:
            score = 50.0

        return score

    def version_2_weighted_title(self, news_items: list) -> float:
        """
        AMELIORATION 1: Titre compte 3x plus que description
        Les titres sont plus importants pour le sentiment
        """
        if not news_items:
            return 50.0

        positive_keywords = {
            'earnings beat': 3.0, 'beats': 2.5, 'profit': 2.0, 'surge': 2.5,
            'breakthrough': 2.5, 'record': 2.0, 'growth': 1.5, 'upgrade': 2.0,
            'partnership': 1.5, 'acquisition': 2.0, 'approval': 2.0,
            'bullish': 2.0, 'gains': 1.5, 'jumps': 2.0, 'soars': 2.5,
            'invests': 2.5, 'investment': 2.0, 'expansion': 2.0
        }

        negative_keywords = {
            'earnings miss': 3.0, 'misses': 2.5, 'loss': 2.0, 'plunge': 2.5,
            'crash': 3.0, 'downgrade': 2.0, 'lawsuit': 2.0, 'investigation': 2.0,
            'recall': 2.0, 'bankruptcy': 3.0, 'bearish': 2.0, 'falls': 1.5,
            'drops': 1.5, 'slides': 1.5, 'tumbles': 2.0, 'slumps': 2.0
        }

        sentiments = []
        for article in news_items[:50]:
            title = article.get('title', '')
            summary = article.get('summary', '')

            # Analyser titre séparément (3x plus important)
            title_blob = TextBlob(title)
            title_polarity = title_blob.sentiment.polarity * 3.0

            # Analyser description
            summary_blob = TextBlob(summary)
            summary_polarity = summary_blob.sentiment.polarity

            # Moyenne pondérée
            polarity = (title_polarity + summary_polarity) / 4.0

            text_lower = (title + ' ' + summary).lower()
            importance = 1.0

            for keyword, weight in positive_keywords.items():
                if keyword in text_lower:
                    importance += weight

            for keyword, weight in negative_keywords.items():
                if keyword in text_lower:
                    importance += weight

            weighted_sentiment = polarity * importance
            sentiments.append(weighted_sentiment)

        if sentiments:
            avg_sentiment = np.mean(sentiments)
            amplified_sentiment = avg_sentiment * 3.0
            score = ((amplified_sentiment + 3) / 6) * 100
            score = max(0, min(100, score))
        else:
            score = 50.0

        return score

    def version_3_money_detection(self, news_items: list) -> float:
        """
        AMELIORATION 2: Détection de montants importants
        "$3 billion investment" = très positif
        """
        if not news_items:
            return 50.0

        positive_keywords = {
            'earnings beat': 3.0, 'beats': 2.5, 'profit': 2.0, 'surge': 2.5,
            'breakthrough': 2.5, 'record': 2.0, 'growth': 1.5, 'upgrade': 2.0,
            'partnership': 1.5, 'acquisition': 2.0, 'approval': 2.0,
            'bullish': 2.0, 'gains': 1.5, 'jumps': 2.0, 'soars': 2.5,
            'invests': 3.0, 'investment': 2.5, 'expansion': 2.0, 'expands': 2.0
        }

        negative_keywords = {
            'earnings miss': 3.0, 'misses': 2.5, 'loss': 2.0, 'plunge': 2.5,
            'crash': 3.0, 'downgrade': 2.0, 'lawsuit': 2.0, 'investigation': 2.0,
            'recall': 2.0, 'bankruptcy': 3.0, 'bearish': 2.0, 'falls': 1.5,
            'drops': 1.5, 'slides': 1.5, 'tumbles': 2.0, 'slumps': 2.0
        }

        sentiments = []
        for article in news_items[:50]:
            title = article.get('title', '')
            summary = article.get('summary', '')

            # Analyser titre séparément (3x plus important)
            title_blob = TextBlob(title)
            title_polarity = title_blob.sentiment.polarity * 3.0

            summary_blob = TextBlob(summary)
            summary_polarity = summary_blob.sentiment.polarity

            polarity = (title_polarity + summary_polarity) / 4.0

            text_lower = (title + ' ' + summary).lower()

            # DETECTION DE MONTANTS
            money_boost = 0

            # Billions (très important)
            billions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
            if billions_match:
                amount = float(billions_match[0])
                if amount >= 1:
                    money_boost = 2.5  # $1B+ = très important
                    # Si contexte positif (investment, expansion), booster davantage
                    if any(word in text_lower for word in ['invest', 'expansion', 'deal', 'contract']):
                        money_boost = 3.5

            # Millions (important si > 100M)
            millions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)
            if millions_match and not billions_match:
                amount = float(millions_match[0])
                if amount >= 500:
                    money_boost = 2.0
                elif amount >= 100:
                    money_boost = 1.0

            importance = 1.0 + money_boost

            # Keywords
            for keyword, weight in positive_keywords.items():
                if keyword in text_lower:
                    importance += weight * 0.5

            for keyword, weight in negative_keywords.items():
                if keyword in text_lower:
                    importance += weight * 0.5

            weighted_sentiment = polarity * importance
            sentiments.append(weighted_sentiment)

        if sentiments:
            avg_sentiment = np.mean(sentiments)
            amplified_sentiment = avg_sentiment * 3.0
            score = ((amplified_sentiment + 3) / 6) * 100
            score = max(0, min(100, score))
        else:
            score = 50.0

        return score

    def version_4_recency_bias(self, news_items: list) -> float:
        """
        AMELIORATION 3: Pondération par récence
        Les 10 premières news comptent plus
        """
        if not news_items:
            return 50.0

        positive_keywords = {
            'earnings beat': 3.0, 'beats': 2.5, 'profit': 2.0, 'surge': 2.5,
            'breakthrough': 2.5, 'record': 2.0, 'growth': 1.5, 'upgrade': 2.0,
            'partnership': 1.5, 'acquisition': 2.0, 'approval': 2.0,
            'bullish': 2.0, 'gains': 1.5, 'jumps': 2.0, 'soars': 2.5,
            'invests': 3.0, 'investment': 2.5, 'expansion': 2.0, 'expands': 2.0
        }

        negative_keywords = {
            'earnings miss': 3.0, 'misses': 2.5, 'loss': 2.0, 'plunge': 2.5,
            'crash': 3.0, 'downgrade': 2.0, 'lawsuit': 2.0, 'investigation': 2.0,
            'recall': 2.0, 'bankruptcy': 3.0, 'bearish': 2.0, 'falls': 1.5,
            'drops': 1.5, 'slides': 1.5, 'tumbles': 2.0, 'slumps': 2.0
        }

        sentiments = []
        weights = []

        for i, article in enumerate(news_items[:50]):
            title = article.get('title', '')
            summary = article.get('summary', '')

            # Analyser titre séparément (3x plus important)
            title_blob = TextBlob(title)
            title_polarity = title_blob.sentiment.polarity * 3.0

            summary_blob = TextBlob(summary)
            summary_polarity = summary_blob.sentiment.polarity

            polarity = (title_polarity + summary_polarity) / 4.0

            text_lower = (title + ' ' + summary).lower()

            # DETECTION DE MONTANTS
            money_boost = 0
            billions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
            if billions_match:
                amount = float(billions_match[0])
                if amount >= 1:
                    money_boost = 2.5
                    if any(word in text_lower for word in ['invest', 'expansion', 'deal', 'contract']):
                        money_boost = 3.5

            millions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)
            if millions_match and not billions_match:
                amount = float(millions_match[0])
                if amount >= 500:
                    money_boost = 2.0
                elif amount >= 100:
                    money_boost = 1.0

            importance = 1.0 + money_boost

            for keyword, weight in positive_keywords.items():
                if keyword in text_lower:
                    importance += weight * 0.5

            for keyword, weight in negative_keywords.items():
                if keyword in text_lower:
                    importance += weight * 0.5

            weighted_sentiment = polarity * importance

            # RECENCY BIAS: Les 10 premières news comptent 50% plus
            recency_weight = 1.5 if i < 10 else 1.0

            sentiments.append(weighted_sentiment)
            weights.append(recency_weight)

        if sentiments:
            # Moyenne pondérée par récence
            avg_sentiment = np.average(sentiments, weights=weights)
            amplified_sentiment = avg_sentiment * 3.0
            score = ((amplified_sentiment + 3) / 6) * 100
            score = max(0, min(100, score))
        else:
            score = 50.0

        return score

    def version_5_ultimate(self, news_items: list) -> float:
        """
        AMELIORATION ULTIME: Combine tout
        + Ajustement final pour se rapprocher de Claude
        """
        if not news_items:
            return 50.0

        # Keywords enrichis
        positive_keywords = {
            # Très forts
            'earnings beat': 4.0, 'beats expectations': 4.0, 'record profit': 4.0,
            'breakthrough': 3.5, 'surge': 3.0, 'soars': 3.0, 'rockets': 3.0,

            # Forts
            'upgrade': 3.0, 'invests': 3.0, 'investment': 2.5, 'expansion': 2.5,
            'profit': 2.5, 'growth': 2.0, 'acquisition': 2.5, 'deal': 2.0,

            # Moyens
            'beats': 2.0, 'gains': 1.5, 'jumps': 2.0, 'partnership': 1.5,
            'approval': 2.0, 'bullish': 2.0, 'contract': 1.5, 'expands': 2.0,
            'announces': 1.0
        }

        negative_keywords = {
            # Très forts
            'bankruptcy': 4.0, 'bankrupt': 4.0, 'crash': 3.5, 'scandal': 3.5,
            'earnings miss': 4.0, 'fraud': 3.5,

            # Forts
            'downgrade': 3.0, 'plunge': 3.0, 'lawsuit': 2.5, 'investigation': 2.5,
            'recall': 2.5, 'loss': 2.5,

            # Moyens
            'misses': 2.0, 'falls': 1.5, 'drops': 1.5, 'bearish': 2.0,
            'decline': 2.0, 'slumps': 2.0, 'tumbles': 2.0
        }

        sentiments = []
        weights = []

        for i, article in enumerate(news_items[:50]):
            title = article.get('title', '')
            summary = article.get('summary', '')

            # Titre = 3x plus important
            title_blob = TextBlob(title)
            title_polarity = title_blob.sentiment.polarity * 3.0

            summary_blob = TextBlob(summary)
            summary_polarity = summary_blob.sentiment.polarity

            polarity = (title_polarity + summary_polarity) / 4.0

            text_lower = (title + ' ' + summary).lower()

            # DETECTION DE MONTANTS (boost important)
            money_boost = 0
            billions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
            if billions_match:
                amount = float(billions_match[0])
                if amount >= 1:
                    money_boost = 3.0  # Augmenté
                    if any(word in text_lower for word in ['invest', 'expansion', 'deal', 'contract', 'acquisition']):
                        money_boost = 4.0  # Encore plus si contexte positif

            millions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)
            if millions_match and not billions_match:
                amount = float(millions_match[0])
                if amount >= 500:
                    money_boost = 2.5
                elif amount >= 100:
                    money_boost = 1.5

            importance = 1.0 + money_boost

            # Keywords (pondération réduite pour équilibrer)
            keyword_boost = 0
            for keyword, weight in positive_keywords.items():
                if keyword in text_lower:
                    keyword_boost += weight * 0.4

            for keyword, weight in negative_keywords.items():
                if keyword in text_lower:
                    keyword_boost -= weight * 0.4

            importance += abs(keyword_boost) * 0.5
            final_polarity = polarity + keyword_boost * 0.1

            weighted_sentiment = final_polarity * importance

            # Récence
            recency_weight = 1.5 if i < 10 else 1.0

            sentiments.append(weighted_sentiment)
            weights.append(recency_weight)

        if sentiments:
            avg_sentiment = np.average(sentiments, weights=weights)

            # FORMULE AJUSTEE pour se rapprocher de Claude
            # Claude tend à donner des scores légèrement plus élevés
            amplified_sentiment = avg_sentiment * 3.2  # Augmenté de 3.0 à 3.2
            score = ((amplified_sentiment + 3) / 6) * 100

            # Ajustement final: +2 points si positif, -2 si négatif
            if score > 52:
                score += 2
            elif score < 48:
                score -= 2

            score = max(0, min(100, score))
        else:
            score = 50.0

        return score

    async def score_with_claude(self, symbol: str, news_items: list) -> float:
        """Score Claude (référence)"""
        if not news_items or not self.api_key:
            return 50.0

        try:
            top_news = news_items[:15]
            news_summary = []
            for i, news in enumerate(top_news, 1):
                title = news.get('title', '')
                description = news.get('description', '')
                published = news.get('published_at', 'Date inconnue')
                news_summary.append(f"{i}. [{published}] {title}\n   {description[:200]}")

            news_text = "\n\n".join(news_summary)
            prompt = f"""Tu es un analyste financier expert. Analyse les actualites recentes sur {symbol} et determine leur sentiment global.

ACTUALITES ({len(top_news)} news les plus recentes sur {len(news_items)} au total):
{news_text}

Ta mission: Donne un score entre 0-100 pour le sentiment global.

Reponds UNIQUEMENT au format JSON: {{"score": <0-100>}}"""

            session = await self.get_session()
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            }

            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('content', [{}])[0].get('text', '')

                    import json
                    json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        score = float(data.get('score', 50))
                        return max(0, min(100, score))

        except Exception:
            pass

        return 50.0

    async def close(self):
        if self.session:
            await self.session.close()


async def main():
    print("=" * 100)
    print("TEST DES AMELIORATIONS TEXTBLOB")
    print("=" * 100)

    # Test sur les actions avec les plus grandes divergences
    test_symbols = ['TTE.PA', 'CSCO', 'AVGO', 'NFLX', 'AMZN', 'AAPL']

    news_analyzer = HistoricalNewsAnalyzer()
    scorer = ImprovedTextBlobScorer()
    target_date = datetime.now()

    results = []

    for symbol in test_symbols:
        print(f"\nTest de {symbol}...", end=" ")

        has_news, news_items, _ = await news_analyzer.get_news_for_date(symbol, target_date)

        if not has_news or not news_items:
            print("Pas de news - SKIP")
            continue

        print(f"{len(news_items)} news")

        # Tester toutes les versions
        v1 = scorer.version_1_basic(news_items)
        v2 = scorer.version_2_weighted_title(news_items)
        v3 = scorer.version_3_money_detection(news_items)
        v4 = scorer.version_4_recency_bias(news_items)
        v5 = scorer.version_5_ultimate(news_items)
        claude = await scorer.score_with_claude(symbol, news_items)

        results.append({
            'symbol': symbol,
            'v1_basic': v1,
            'v2_title': v2,
            'v3_money': v3,
            'v4_recency': v4,
            'v5_ultimate': v5,
            'claude': claude,
            'diff_v1': abs(claude - v1),
            'diff_v5': abs(claude - v5)
        })

        await asyncio.sleep(0.5)

    await scorer.close()
    await news_analyzer.close()

    # RECAP
    print("\n" + "=" * 100)
    print("RESULTATS:")
    print("=" * 100)
    print(f"{'Symbol':<10} {'V1':>6} {'V2':>6} {'V3':>6} {'V4':>6} {'V5':>6} {'Claude':>7} {'Diff V1':>9} {'Diff V5':>9}")
    print("-" * 100)

    for r in results:
        print(f"{r['symbol']:<10} "
              f"{r['v1_basic']:>6.1f} {r['v2_title']:>6.1f} {r['v3_money']:>6.1f} "
              f"{r['v4_recency']:>6.1f} {r['v5_ultimate']:>6.1f} {r['claude']:>7.1f} "
              f"{r['diff_v1']:>9.1f} {r['diff_v5']:>9.1f}")

    # Stats
    avg_diff_v1 = np.mean([r['diff_v1'] for r in results])
    avg_diff_v5 = np.mean([r['diff_v5'] for r in results])
    improvement = avg_diff_v1 - avg_diff_v5

    print("\n" + "=" * 100)
    print(f"AMELIORATION:")
    print("=" * 100)
    print(f"Diff moyenne V1 (Basic):    {avg_diff_v1:.1f} points")
    print(f"Diff moyenne V5 (Ultimate): {avg_diff_v5:.1f} points")
    print(f"Amelioration:               {improvement:.1f} points ({improvement/avg_diff_v1*100:.1f}%)")

    if improvement > 2:
        print("\n=> Version 5 AMELIORE significativement TextBlob!")
    elif improvement > 0:
        print("\n=> Version 5 ameliore legerement TextBlob")
    else:
        print("\n=> Pas d'amelioration significative")


if __name__ == "__main__":
    asyncio.run(main())
