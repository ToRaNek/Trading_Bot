"""
Analyse approfondie des écarts TextBlob vs Claude
Pour créer la meilleure version possible
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


class UltraTextBlob:
    """Version ULTRA optimisée de TextBlob"""

    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.session = None

    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    def analyze_news_details(self, news_items: list) -> dict:
        """Analyse détaillée pour comprendre les divergences"""
        big_money_news = []
        very_positive_keywords = []
        very_negative_keywords = []

        for i, article in enumerate(news_items[:20]):
            title = article.get('title', '')
            summary = article.get('summary', '')
            text = f"{title} {summary}".lower()

            # Détecter gros montants
            billions = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text)
            if billions:
                big_money_news.append({
                    'index': i,
                    'title': title[:60],
                    'amount': f"${billions[0]}B"
                })

            # Détecter keywords très positifs
            strong_positive = ['beats expectations', 'record profit', 'surge', 'soars', 'breakthrough']
            for kw in strong_positive:
                if kw in text:
                    very_positive_keywords.append({
                        'index': i,
                        'keyword': kw,
                        'title': title[:60]
                    })

            # Détecter keywords très négatifs
            strong_negative = ['bankruptcy', 'crash', 'scandal', 'plunge']
            for kw in strong_negative:
                if kw in text:
                    very_negative_keywords.append({
                        'index': i,
                        'keyword': kw,
                        'title': title[:60]
                    })

        return {
            'big_money': big_money_news,
            'strong_positive': very_positive_keywords,
            'strong_negative': very_negative_keywords
        }

    def score_textblob_ultra_v6(self, news_items: list) -> float:
        """
        VERSION 6 - ULTRA OPTIMISEE
        Basée sur l'analyse des divergences les plus importantes
        """
        if not news_items:
            return 50.0

        # Keywords ultra-enrichis basés sur l'analyse manuelle
        positive_keywords = {
            # TRES FORTS (donnent un gros boost)
            'beats expectations': 5.0,
            'earnings beat': 5.0,
            'record profit': 5.0,
            'record earnings': 5.0,
            'surge': 4.0,
            'soars': 4.0,
            'rockets': 4.0,
            'breakthrough': 4.0,

            # FORTS
            'upgrade': 3.5,
            'outperform': 3.5,
            'buy rating': 3.5,
            'strong buy': 4.0,
            'invests': 3.0,
            'investment': 2.5,
            'expansion': 2.5,
            'expands': 2.5,

            # MOYENS
            'profit': 2.0,
            'growth': 1.8,
            'gains': 1.5,
            'acquisition': 2.5,
            'deal': 2.0,
            'partnership': 2.0,
            'contract': 1.8,
            'wins': 2.0,
            'beats': 2.5,
            'jumps': 2.0,

            # FAIBLES MAIS POSITIFS
            'announces': 0.8,
            'launches': 1.2,
            'approval': 1.8,
            'positive': 1.0,
            'bullish': 1.5
        }

        negative_keywords = {
            # TRES FORTS
            'bankruptcy': 5.0,
            'bankrupt': 5.0,
            'crash': 4.5,
            'scandal': 4.0,
            'fraud': 4.5,
            'earnings miss': 5.0,

            # FORTS
            'downgrade': 3.5,
            'underperform': 3.5,
            'sell rating': 3.5,
            'plunge': 3.5,
            'plunges': 3.5,
            'tumbles': 3.0,

            # MOYENS
            'lawsuit': 2.5,
            'investigation': 2.5,
            'probe': 2.0,
            'recall': 2.5,
            'loss': 2.0,
            'losses': 2.0,
            'misses': 2.5,

            # FAIBLES
            'falls': 1.2,
            'drops': 1.2,
            'decline': 1.5,
            'bearish': 1.5,
            'concern': 1.0,
            'worried': 1.0
        }

        sentiments = []
        weights = []

        for i, article in enumerate(news_items[:50]):
            title = article.get('title', '')
            summary = article.get('summary', '')

            # TITRE = 4x plus important (augmenté de 3x)
            title_blob = TextBlob(title)
            title_polarity = title_blob.sentiment.polarity * 4.0

            summary_blob = TextBlob(summary)
            summary_polarity = summary_blob.sentiment.polarity

            # Moyenne pondérée (titre domine)
            base_polarity = (title_polarity + summary_polarity) / 5.0

            text_lower = (title + ' ' + summary).lower()

            # === DETECTION DE MONTANTS (CRUCIAL) ===
            money_boost = 0
            money_context_positive = any(word in text_lower for word in [
                'invest', 'expansion', 'deal', 'contract', 'acquisition', 'revenue', 'profit', 'earnings'
            ])

            # Billions
            billions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
            if billions_match:
                amount = float(billions_match[0])
                if amount >= 10:
                    money_boost = 5.0 if money_context_positive else 3.0
                elif amount >= 5:
                    money_boost = 4.0 if money_context_positive else 2.5
                elif amount >= 1:
                    money_boost = 3.0 if money_context_positive else 2.0
                else:
                    money_boost = 2.0 if money_context_positive else 1.0

            # Millions (si pas de billions)
            millions_match = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)
            if millions_match and not billions_match:
                amount = float(millions_match[0])
                if amount >= 1000:
                    money_boost = 3.0 if money_context_positive else 2.0
                elif amount >= 500:
                    money_boost = 2.5 if money_context_positive else 1.5
                elif amount >= 100:
                    money_boost = 1.5 if money_context_positive else 1.0

            # === DETECTION DE KEYWORDS ===
            keyword_score = 0

            for keyword, weight in positive_keywords.items():
                count = text_lower.count(keyword)
                if count > 0:
                    keyword_score += weight * count

            for keyword, weight in negative_keywords.items():
                count = text_lower.count(keyword)
                if count > 0:
                    keyword_score -= weight * count

            # === COMBINAISON FINALE ===
            # Importance basée sur montants et keywords
            importance = 1.0 + money_boost + abs(keyword_score) * 0.4

            # Polarité finale (combine TextBlob + keywords)
            final_polarity = base_polarity + (keyword_score * 0.12)

            # Sentiment final pondéré
            weighted_sentiment = final_polarity * importance

            # === RECENCY BIAS ===
            # Les 15 premières news = 60% plus importantes (augmenté)
            recency_weight = 1.6 if i < 15 else 1.0

            sentiments.append(weighted_sentiment)
            weights.append(recency_weight)

        if sentiments:
            # Moyenne pondérée
            avg_sentiment = np.average(sentiments, weights=weights)

            # === FORMULE OPTIMISEE ===
            # Amplification ajustée
            amplified = avg_sentiment * 3.3  # Augmenté de 3.2

            # Conversion en 0-100
            score = ((amplified + 3) / 6) * 100

            # === AJUSTEMENT FINAL PROGRESSIF ===
            # Au lieu d'un boost fixe, boost progressif
            if score > 55:
                boost = (score - 55) * 0.15  # Plus c'est positif, plus on booste
                score += boost
            elif score < 45:
                penalty = (45 - score) * 0.15  # Plus c'est négatif, plus on pénalise
                score -= penalty

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
            prompt = f"""Score these {symbol} news from 0-100.

{news_text}

Reply with ONLY this exact format: {{"score": 65}}

No explanations. Just the JSON."""

            session = await self.get_session()
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 150,
                "temperature": 0.5,
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

                    # Debug
                    # print(f"[DEBUG] Response: {json.dumps(result, indent=2)[:200]}")

                    content = result.get('content', [{}])[0].get('text', '')

                    import json
                    json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)
                        score = float(data.get('score', 50))
                        score = max(0, min(100, score))
                        return score
                    else:
                        print(f"[Claude Warning] No JSON found in: {content[:100]}")
                        return 50.0
                else:
                    error_text = await response.text()
                    print(f"[Claude Error] HTTP {response.status}: {error_text[:200]}")
                    return 50.0

        except Exception as e:
            print(f"[Claude Exception] {e}")
            import traceback
            traceback.print_exc()
            return 50.0

    async def close(self):
        if self.session:
            await self.session.close()


async def main():
    print("=" * 100)
    print("TEST FINAL - TEXTBLOB ULTRA V6 vs CLAUDE")
    print("=" * 100)

    # Tester sur les cas les plus difficiles
    test_symbols = [
        ('TTE.PA', 'Plus grande divergence (22.4 pts)'),
        ('CSCO', 'Grande divergence (20.0 pts)'),
        ('AVGO', 'Grande divergence (19.9 pts)'),
        ('AMZN', 'Divergence moyenne (4.9 pts)'),
        ('AAPL', 'Petite divergence (2.5 pts)'),
        ('ORCL', 'Tres petite divergence (1.0 pt)')
    ]

    news_analyzer = HistoricalNewsAnalyzer()
    scorer = UltraTextBlob()
    target_date = datetime.now()

    results = []

    for symbol, description in test_symbols:
        print(f"\n[{symbol}] {description}")

        has_news, news_items, _ = await news_analyzer.get_news_for_date(symbol, target_date)

        if not has_news or not news_items:
            print("   Pas de news - SKIP")
            continue

        print(f"   {len(news_items)} news recuperees")

        # Analyser les détails
        details = scorer.analyze_news_details(news_items)

        if details['big_money']:
            print(f"   $ Gros montants detectes: {len(details['big_money'])}")
            for money in details['big_money'][:2]:
                print(f"     - {money['amount']}: {money['title']}")

        if details['strong_positive']:
            print(f"   + Keywords tres positifs: {len(details['strong_positive'])}")

        if details['strong_negative']:
            print(f"   - Keywords tres negatifs: {len(details['strong_negative'])}")

        # Scorer
        textblob_v6 = scorer.score_textblob_ultra_v6(news_items)
        claude = await scorer.score_with_claude(symbol, news_items)

        diff = abs(claude - textblob_v6)

        print(f"   TextBlob V6: {textblob_v6:.1f}")
        print(f"   Claude:      {claude:.1f}")
        print(f"   Diff:        {diff:.1f} pts")

        results.append({
            'symbol': symbol,
            'description': description,
            'textblob_v6': textblob_v6,
            'claude': claude,
            'diff': diff
        })

        await asyncio.sleep(0.5)

    await scorer.close()
    await news_analyzer.close()

    # RECAP FINAL
    print("\n" + "=" * 100)
    print("RESULTATS FINAUX:")
    print("=" * 100)

    print(f"\n{'Symbol':<10} {'TextBlob V6':>12} {'Claude':>8} {'Diff':>8} {'Description'}")
    print("-" * 100)

    for r in results:
        print(f"{r['symbol']:<10} {r['textblob_v6']:>12.1f} {r['claude']:>8.1f} {r['diff']:>8.1f} {r['description']}")

    # Stats
    avg_diff = np.mean([r['diff'] for r in results])
    max_diff = max([r['diff'] for r in results])
    min_diff = min([r['diff'] for r in results])

    print("\n" + "=" * 100)
    print("STATISTIQUES:")
    print("=" * 100)
    print(f"Difference moyenne: {avg_diff:.1f} pts")
    print(f"Difference min:     {min_diff:.1f} pts")
    print(f"Difference max:     {max_diff:.1f} pts")

    # Comparaison avec V1 (baseline)
    print("\n" + "=" * 100)
    print("AMELIORATION vs BASELINE:")
    print("=" * 100)
    print("Baseline V1 (de test_all_stocks_comparison.py):")
    print("  - Difference moyenne: 8.7 pts")
    print(f"\nTextBlob Ultra V6:")
    print(f"  - Difference moyenne: {avg_diff:.1f} pts")

    improvement = 8.7 - avg_diff
    if improvement > 0:
        print(f"\nAMELIORATION: {improvement:.1f} pts ({improvement/8.7*100:.1f}%)")
    else:
        print(f"\nPAS D'AMELIORATION ({improvement:.1f} pts)")

    if avg_diff < 5:
        print("\n=> EXCELLENT! TextBlob V6 est quasi identique a Claude!")
    elif avg_diff < 8:
        print("\n=> TRES BON! TextBlob V6 est tres proche de Claude")
    else:
        print("\n=> CORRECT mais encore de la marge d'amelioration")


if __name__ == "__main__":
    asyncio.run(main())
