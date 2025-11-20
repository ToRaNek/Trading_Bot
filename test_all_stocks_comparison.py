"""
Test de scoring sur TOUTES les actions de la watchlist
Compare TextBlob Basic vs Claude pour voir la différence réelle
"""

import asyncio
import os
import sys
from datetime import datetime
import aiohttp
from dotenv import load_dotenv
from textblob import TextBlob
import numpy as np

# Importer les analyseurs existants
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from config import WATCHLIST

load_dotenv()


class QuickScorer:
    """Scorer rapide pour comparer TextBlob vs Claude"""

    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.session = None

    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    def score_with_textblob(self, news_items: list) -> dict:
        """Score TextBlob Basic (méthode actuelle)"""
        if not news_items:
            return {'score': 50.0, 'sentiment': 'neutral'}

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

        if score >= 60:
            sentiment = 'positive'
        elif score <= 40:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {'score': score, 'sentiment': sentiment}

    async def score_with_claude(self, symbol: str, news_items: list) -> dict:
        """Score Claude AI"""
        if not news_items or not self.api_key:
            return {'score': 50.0, 'sentiment': 'neutral'}

        try:
            top_news = news_items[:15]

            news_summary = []
            for i, news in enumerate(top_news, 1):
                title = news.get('title', '')
                description = news.get('description', '')
                published = news.get('published_at', 'Date inconnue')

                news_summary.append(
                    f"{i}. [{published}] {title}\n"
                    f"   {description[:200]}"
                )

            news_text = "\n\n".join(news_summary)

            prompt = f"""Tu es un analyste financier expert. Analyse les actualites recentes sur {symbol} et determine leur sentiment global.

ACTUALITES ({len(top_news)} news les plus recentes sur {len(news_items)} au total):
{news_text}

Ta mission:
1. Evalue le sentiment global de ces actualites
2. Donne un score entre 0-100:
   - 0-20: Tres negatif (scandales majeurs, faillite, pertes massives)
   - 20-40: Negatif (problemes importants, baisse, mauvais resultats)
   - 40-60: Neutre (pas d'impact clair ou mixte)
   - 60-80: Positif (bonnes nouvelles, croissance, succes)
   - 80-100: Tres positif (innovations majeures, resultats exceptionnels)

Reponds UNIQUEMENT au format JSON suivant (PAS de markdown, juste le JSON):
{{
    "score": <nombre entre 0-100>,
    "sentiment": "<positive/negative/neutral>"
}}"""

            session = await self.get_session()
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 150,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
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
                    import re

                    json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)

                        score = float(data.get('score', 50))
                        score = max(0, min(100, score))

                        return {
                            'score': score,
                            'sentiment': data.get('sentiment', 'neutral')
                        }

        except Exception as e:
            pass

        return {'score': 50.0, 'sentiment': 'neutral'}

    async def close(self):
        if self.session:
            await self.session.close()


async def main():
    print("=" * 100)
    print("TEST DE SCORING SUR TOUTES LES ACTIONS DE LA WATCHLIST")
    print("=" * 100)
    print(f"\nNombre d'actions a tester: {len(WATCHLIST)}")
    print("Methodes: TextBlob Basic (GRATUIT) vs Claude AI ($0.003/requete)")
    print("\n" + "=" * 100)

    # Initialiser
    news_analyzer = HistoricalNewsAnalyzer()
    scorer = QuickScorer()
    target_date = datetime.now()

    results = []
    total_cost = 0

    for i, symbol in enumerate(WATCHLIST, 1):
        print(f"\n[{i}/{len(WATCHLIST)}] Analyse de {symbol}...", end=" ")

        try:
            # Récupérer les news
            has_news, news_items, _ = await news_analyzer.get_news_for_date(symbol, target_date)

            if not has_news or not news_items:
                print(f"Aucune news - SKIP")
                continue

            print(f"{len(news_items)} news", end=" | ")

            # Score TextBlob
            textblob_result = scorer.score_with_textblob(news_items)
            tb_score = textblob_result['score']
            tb_sentiment = textblob_result['sentiment']

            print(f"TextBlob: {tb_score:.1f}", end=" | ")

            # Score Claude
            claude_result = await scorer.score_with_claude(symbol, news_items)
            cl_score = claude_result['score']
            cl_sentiment = claude_result['sentiment']

            print(f"Claude: {cl_score:.1f}", end=" | ")

            # Calculer la différence
            diff = abs(cl_score - tb_score)
            print(f"Diff: {diff:.1f}")

            # Stocker
            results.append({
                'symbol': symbol,
                'news_count': len(news_items),
                'textblob_score': tb_score,
                'textblob_sentiment': tb_sentiment,
                'claude_score': cl_score,
                'claude_sentiment': cl_sentiment,
                'diff': diff
            })

            total_cost += 0.003  # Coût par requête Claude

            # Pause pour ne pas surcharger l'API
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"ERREUR: {e}")
            continue

    # Fermer
    await scorer.close()
    await news_analyzer.close()

    # RECAP
    print("\n" + "=" * 100)
    print("RECAPITULATIF COMPLET:")
    print("=" * 100)

    if not results:
        print("\nAucun resultat disponible!")
        return

    # Trier par différence
    results_sorted = sorted(results, key=lambda x: x['diff'], reverse=True)

    # Statistiques
    differences = [r['diff'] for r in results]
    avg_diff = np.mean(differences)
    max_diff = max(differences)
    min_diff = min(differences)
    std_diff = np.std(differences)

    # Accords de sentiment
    same_sentiment = sum(1 for r in results if r['textblob_sentiment'] == r['claude_sentiment'])
    total = len(results)
    agreement_pct = (same_sentiment / total) * 100

    print(f"\nACTIONS TESTEES: {total}/{len(WATCHLIST)}")
    print(f"COUT TOTAL: ${total_cost:.2f}")
    print("\n" + "-" * 100)
    print("STATISTIQUES:")
    print("-" * 100)
    print(f"Difference moyenne:        {avg_diff:.1f} points")
    print(f"Difference min:            {min_diff:.1f} points")
    print(f"Difference max:            {max_diff:.1f} points")
    print(f"Ecart-type:                {std_diff:.1f}")
    print(f"Accord sur le sentiment:   {same_sentiment}/{total} ({agreement_pct:.1f}%)")

    # Répartition par différence
    very_close = sum(1 for d in differences if d < 5)
    close = sum(1 for d in differences if 5 <= d < 10)
    moderate = sum(1 for d in differences if 10 <= d < 20)
    far = sum(1 for d in differences if d >= 20)

    print("\n" + "-" * 100)
    print("REPARTITION DES DIFFERENCES:")
    print("-" * 100)
    print(f"Tres proche (< 5 points):   {very_close:>3} actions ({very_close/total*100:.1f}%)")
    print(f"Proche (5-10 points):       {close:>3} actions ({close/total*100:.1f}%)")
    print(f"Modere (10-20 points):      {moderate:>3} actions ({moderate/total*100:.1f}%)")
    print(f"Eloigne (>= 20 points):     {far:>3} actions ({far/total*100:.1f}%)")

    # Top 10 des plus grandes différences
    print("\n" + "-" * 100)
    print("TOP 10 DES PLUS GRANDES DIFFERENCES:")
    print("-" * 100)
    print(f"{'Symbole':<10} {'News':>5} {'TextBlob':>10} {'Claude':>10} {'Diff':>8}")
    print("-" * 100)

    for r in results_sorted[:10]:
        print(f"{r['symbol']:<10} {r['news_count']:>5} "
              f"{r['textblob_score']:>10.1f} {r['claude_score']:>10.1f} "
              f"{r['diff']:>8.1f}")

    # Top 10 des plus petites différences
    print("\n" + "-" * 100)
    print("TOP 10 DES PLUS PETITES DIFFERENCES (Meilleure coherence):")
    print("-" * 100)
    print(f"{'Symbole':<10} {'News':>5} {'TextBlob':>10} {'Claude':>10} {'Diff':>8}")
    print("-" * 100)

    results_sorted_asc = sorted(results, key=lambda x: x['diff'])
    for r in results_sorted_asc[:10]:
        print(f"{r['symbol']:<10} {r['news_count']:>5} "
              f"{r['textblob_score']:>10.1f} {r['claude_score']:>10.1f} "
              f"{r['diff']:>8.1f}")

    # RECOMMANDATION FINALE
    print("\n" + "=" * 100)
    print("RECOMMANDATION FINALE:")
    print("=" * 100)

    if avg_diff < 5:
        print("\nTextBlob Basic est QUASI IDENTIQUE a Claude!")
        print(f"Difference moyenne de seulement {avg_diff:.1f} points")
        print("=> UTILISE TEXTBLOB BASIC (GRATUIT)")
        print(f"   Tu economises ${total_cost:.2f} par analyse complete")
    elif avg_diff < 10:
        print("\nTextBlob Basic est COHERENT avec Claude")
        print(f"Difference moyenne de {avg_diff:.1f} points")
        print(f"Accord sur le sentiment: {agreement_pct:.1f}%")
        print("=> UTILISE TEXTBLOB BASIC pour economiser")
        print(f"   ou Claude si tu veux plus de precision (cout: ${total_cost:.2f}/analyse)")
    else:
        print("\nClaude donne des resultats SIGNIFICATIVEMENT DIFFERENTS")
        print(f"Difference moyenne de {avg_diff:.1f} points")
        print(f"Accord sur le sentiment: {agreement_pct:.1f}%")
        print("=> UTILISE CLAUDE pour une meilleure precision")
        print(f"   Cout par analyse complete: ${total_cost:.2f}")
        print(f"   Cout mensuel (1x/jour): ~${total_cost * 30:.2f}")
        print(f"   Cout mensuel (toutes les 30min): ~${total_cost * 48 * 30:.2f}")

    print("\n" + "=" * 100)
    print("FIN DU TEST")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
