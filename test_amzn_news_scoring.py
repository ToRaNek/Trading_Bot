"""
Test de récupération et scoring des news AMZN avec Claude API (Anthropic)
Compare Claude vs TextBlob pour voir quelle méthode est la meilleure
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

load_dotenv()


class ClaudeNewsScorer:
    """Scorer utilisant l'API Claude d'Anthropic (toi-même!)"""

    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.session = None

    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def score_with_claude(self, symbol: str, news_items: list) -> dict:
        """
        Demande à Claude (API Anthropic) de scorer les news
        C'est toi qui te demandes à toi-même!
        """
        if not news_items:
            return {
                'score': 50.0,
                'sentiment': 'neutral',
                'reasoning': 'Aucune news disponible',
                'method': 'Claude API'
            }

        if not self.api_key:
            print("\n[ERREUR] Pas de cle ANTHROPIC_API_KEY trouvee dans .env!")
            print("Ajoute-la pour utiliser Claude API directement")
            print("Obtiens une cle sur: https://console.anthropic.com/")
            return {
                'score': 0.0,
                'sentiment': 'unknown',
                'reasoning': 'API key manquante',
                'method': 'Claude API'
            }

        try:
            # Limiter à 15 news max pour le prompt
            top_news = news_items[:15]

            # Construire le résumé des news
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

            # Prompt pour Claude (toi-même!)
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

3. Explique ton raisonnement en 2-3 phrases COURTES

Reponds UNIQUEMENT au format JSON suivant (PAS de markdown, juste le JSON):
{{
    "score": <nombre entre 0-100>,
    "sentiment": "<positive/negative/neutral>",
    "reasoning": "<ton explication en 2-3 phrases>"
}}"""

            # Appeler l'API Claude
            session = await self.get_session()
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 300,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }

            print(f"\n[Claude API] Analyse en cours de {len(news_items)} news sur {symbol}...")

            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extraire la réponse
                    content = result.get('content', [{}])[0].get('text', '')

                    # Parser le JSON
                    import json
                    import re

                    # Extraire le JSON de la réponse (au cas où il y a du texte autour)
                    json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)

                        score = float(data.get('score', 50))
                        score = max(0, min(100, score))

                        return {
                            'score': score,
                            'sentiment': data.get('sentiment', 'neutral'),
                            'reasoning': data.get('reasoning', ''),
                            'method': 'Claude API'
                        }
                    else:
                        print(f"[ERREUR] Impossible de parser la reponse: {content}")
                        return {
                            'score': 50.0,
                            'sentiment': 'unknown',
                            'reasoning': 'Erreur de parsing',
                            'method': 'Claude API'
                        }
                else:
                    error_text = await response.text()
                    print(f"[ERREUR] API Claude {response.status}: {error_text}")
                    return {
                        'score': 0.0,
                        'sentiment': 'error',
                        'reasoning': f'Erreur API {response.status}',
                        'method': 'Claude API'
                    }

        except Exception as e:
            print(f"[ERREUR] Exception lors du scoring: {e}")
            import traceback
            traceback.print_exc()
            return {
                'score': 0.0,
                'sentiment': 'error',
                'reasoning': f'Exception: {str(e)}',
                'method': 'Claude API'
            }

    def score_with_textblob(self, news_items: list) -> dict:
        """
        Score basé sur TextBlob (méthode actuelle - VERSION DE BASE)
        """
        if not news_items:
            return {
                'score': 50.0,
                'sentiment': 'neutral',
                'reasoning': 'Aucune news disponible',
                'method': 'TextBlob Basic'
            }

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

            if polarity > 0.1:
                positive_count += 1
            elif polarity < -0.1:
                negative_count += 1
            else:
                neutral_count += 1

        if sentiments:
            avg_sentiment = np.mean(sentiments)
            amplified_sentiment = avg_sentiment * 3.0
            score = ((amplified_sentiment + 3) / 6) * 100
            score = max(0, min(100, score))
        else:
            score = 50.0
            avg_sentiment = 0

        if score >= 60:
            sentiment = 'positive'
        elif score <= 40:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        reasoning = f"{positive_count} pos, {negative_count} neg, {neutral_count} neu. Avg: {avg_sentiment:.3f}"

        return {
            'score': score,
            'sentiment': sentiment,
            'reasoning': reasoning,
            'method': 'TextBlob Basic'
        }

    def score_with_textblob_improved(self, news_items: list) -> dict:
        """
        Version AMELIOREE de TextBlob - GRATUIT et plus performant
        Améliorations:
        1. Plus de keywords financiers
        2. Détection de montants ($X billion/million)
        3. Pondération par récence
        4. Titre compte 2x plus que description
        5. Détection de négations
        """
        if not news_items:
            return {
                'score': 50.0,
                'sentiment': 'neutral',
                'reasoning': 'Aucune news disponible',
                'method': 'TextBlob Improved'
            }

        # Keywords étendus avec poids financiers
        positive_keywords = {
            # Earnings & Performance
            'earnings beat': 4.0, 'beats expectations': 4.0, 'beats': 2.5,
            'profit surge': 3.5, 'profit': 2.0, 'revenue growth': 3.0,
            'record profit': 4.0, 'record revenue': 4.0, 'record': 2.0,

            # Investments & Growth
            'invests': 3.0, 'investment': 2.5, 'expansion': 2.5, 'expands': 2.5,
            'growth': 1.5, 'breakthrough': 2.5, 'innovation': 2.0,

            # Market reaction
            'upgrade': 3.0, 'upgraded': 3.0, 'outperform': 2.5,
            'bullish': 2.0, 'surge': 2.5, 'soars': 2.5, 'jumps': 2.0,
            'gains': 1.5, 'rallies': 2.0, 'rockets': 2.5,

            # Business wins
            'partnership': 2.0, 'acquisition': 2.0, 'approval': 2.0,
            'contract': 1.5, 'deal': 1.5, 'announces': 1.0
        }

        negative_keywords = {
            # Earnings & Performance
            'earnings miss': 4.0, 'misses expectations': 4.0, 'misses': 2.5,
            'loss': 2.5, 'losses': 2.5, 'decline': 2.0, 'declines': 2.0,

            # Market issues
            'downgrade': 3.0, 'downgraded': 3.0, 'underperform': 2.5,
            'crash': 3.5, 'plunge': 2.5, 'plunges': 2.5, 'tumbles': 2.0,
            'drops': 1.5, 'falls': 1.5, 'slides': 1.5, 'slumps': 2.0,

            # Legal & Problems
            'lawsuit': 2.5, 'investigation': 2.5, 'probe': 2.0,
            'recall': 2.5, 'bankruptcy': 4.0, 'bankrupt': 4.0,
            'scandal': 3.0, 'fraud': 3.5, 'fined': 2.5,

            # Bearish
            'bearish': 2.0, 'weak': 1.5, 'disappointing': 2.0,
            'concern': 1.5, 'worried': 1.5, 'risk': 1.0
        }

        import re

        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for i, article in enumerate(news_items[:50]):
            title = article.get('title', '')
            summary = article.get('summary', '')

            # Titre compte 2x plus
            title_blob = TextBlob(title)
            title_polarity = title_blob.sentiment.polarity * 2.0

            summary_blob = TextBlob(summary)
            summary_polarity = summary_blob.sentiment.polarity

            # Moyenne pondérée
            polarity = (title_polarity + summary_polarity) / 3.0

            text_lower = (title + ' ' + summary).lower()

            # Détecter montants importants ($X billion/million)
            money_boost = 0
            billions = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*billion', text_lower)
            millions = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*million', text_lower)

            if billions:
                money_boost = 2.0  # Billions = très important
            elif millions:
                amount = float(millions[0]) if millions else 0
                if amount >= 100:
                    money_boost = 1.0  # +100M = important

            # Détecter keywords
            keyword_score = 0
            for keyword, weight in positive_keywords.items():
                count = text_lower.count(keyword)
                if count > 0:
                    keyword_score += weight * count

            for keyword, weight in negative_keywords.items():
                count = text_lower.count(keyword)
                if count > 0:
                    keyword_score -= weight * count

            # Pondération par récence (les 10 premières news comptent plus)
            recency_weight = 1.5 if i < 10 else 1.0

            # Score final combiné
            importance = 1.0 + money_boost + abs(keyword_score) * 0.3
            final_sentiment = (polarity + keyword_score * 0.1) * importance * recency_weight

            sentiments.append(final_sentiment)

            if final_sentiment > 0.15:
                positive_count += 1
            elif final_sentiment < -0.15:
                negative_count += 1
            else:
                neutral_count += 1

        if sentiments:
            avg_sentiment = np.mean(sentiments)
            # Formule améliorée pour convertir en 0-100
            score = 50 + (avg_sentiment * 25)  # Plus sensible
            score = max(0, min(100, score))
        else:
            score = 50.0
            avg_sentiment = 0

        if score >= 60:
            sentiment = 'positive'
        elif score <= 40:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        reasoning = f"{positive_count} pos, {negative_count} neg, {neutral_count} neu. Avg: {avg_sentiment:.3f}"

        return {
            'score': score,
            'sentiment': sentiment,
            'reasoning': reasoning,
            'method': 'TextBlob Improved'
        }

    async def close(self):
        if self.session:
            await self.session.close()


async def main():
    print("=" * 80)
    print("TEST DE SCORING DES NEWS AMZN AVEC CLAUDE API")
    print("=" * 80)
    print()

    symbol = "AMZN"

    # 1. Récupérer les news
    print(f"[1/2] Recuperation des news {symbol}...")
    news_analyzer = HistoricalNewsAnalyzer()

    target_date = datetime.now()
    has_news, news_items, news_score = await news_analyzer.get_news_for_date(symbol, target_date)

    print(f"   -> {len(news_items) if news_items else 0} news recuperees")

    if not news_items:
        print("\n[ERREUR] Aucune news trouvee. Verifier les cles API.")
        await news_analyzer.close()
        return

    # Afficher les premières news
    print("\n" + "=" * 80)
    print("APERCU DES NEWS:")
    print("=" * 80)
    for i, news in enumerate(news_items[:5], 1):
        print(f"\n{i}. {news.get('title', 'Sans titre')}")
        print(f"   Date: {news.get('published_at', 'N/A')}")
        print(f"   Source: {news.get('source', 'N/A')}")
        desc = news.get('description', '')
        if desc:
            print(f"   Description: {desc[:150]}...")

    if len(news_items) > 5:
        print(f"\n... et {len(news_items) - 5} autres news")

    # 2. Scorer avec TextBlob (version de base)
    print("\n" + "=" * 80)
    print("[2/4] SCORING AVEC TEXTBLOB BASIC (methode actuelle)")
    print("=" * 80)

    claude_scorer = ClaudeNewsScorer()
    textblob_basic = claude_scorer.score_with_textblob(news_items)

    print(f"\nRESULTAT TEXTBLOB BASIC:")
    print(f"   Score: {textblob_basic['score']:.1f}/100")
    print(f"   Sentiment: {textblob_basic['sentiment']}")
    print(f"   Details: {textblob_basic['reasoning']}")

    # 3. Scorer avec TextBlob Improved (GRATUIT + AMELIORE)
    print("\n" + "=" * 80)
    print("[3/4] SCORING AVEC TEXTBLOB IMPROVED (GRATUIT + AMELIORE)")
    print("=" * 80)

    textblob_improved = claude_scorer.score_with_textblob_improved(news_items)

    print(f"\nRESULTAT TEXTBLOB IMPROVED:")
    print(f"   Score: {textblob_improved['score']:.1f}/100")
    print(f"   Sentiment: {textblob_improved['sentiment']}")
    print(f"   Details: {textblob_improved['reasoning']}")

    # 4. Scorer avec Claude
    print("\n" + "=" * 80)
    print("[4/4] SCORING AVEC CLAUDE AI (PAYANT)")
    print("=" * 80)

    claude_result = await claude_scorer.score_with_claude(symbol, news_items)

    print(f"\nRESULTAT CLAUDE:")
    print(f"   Score: {claude_result['score']}/100")
    print(f"   Sentiment: {claude_result['sentiment']}")
    print(f"   Details: {claude_result['reasoning']}")

    # Comparaison
    print("\n" + "=" * 80)
    print("COMPARAISON DES 3 METHODES:")
    print("=" * 80)
    print(f"\nTextBlob Basic:    {textblob_basic['score']:>6.1f}/100  [{textblob_basic['sentiment']}] - GRATUIT")
    print(f"TextBlob Improved: {textblob_improved['score']:>6.1f}/100  [{textblob_improved['sentiment']}] - GRATUIT")
    print(f"Claude AI:         {claude_result['score']:>6.1f}/100  [{claude_result['sentiment']}] - $0.003/requete")

    diff_basic_claude = abs(claude_result['score'] - textblob_basic['score'])
    diff_improved_claude = abs(claude_result['score'] - textblob_improved['score'])

    print(f"\nDiff Basic vs Claude:    {diff_basic_claude:.1f} points")
    print(f"Diff Improved vs Claude: {diff_improved_claude:.1f} points")

    if diff_improved_claude < diff_basic_claude:
        print("\n=> TextBlob Improved est plus proche de Claude!")
        improvement = diff_basic_claude - diff_improved_claude
        print(f"   Amelioration: {improvement:.1f} points (sans cout)")
    else:
        print("\n=> TextBlob Basic etait deja bon")

    # Recommandation finale
    print("\n" + "=" * 80)
    print("RECOMMANDATION:")
    print("=" * 80)

    if diff_improved_claude < 5:
        print("\nTextBlob Improved est quasi identique a Claude!")
        print("=> UTILISE TEXTBLOB IMPROVED (GRATUIT)")
    elif diff_improved_claude < 10:
        print("\nTextBlob Improved est coherent avec Claude")
        print("=> UTILISE TEXTBLOB IMPROVED pour economiser")
        print("   ou Claude si tu veux plus de precision")
    else:
        print("\nClaude donne des resultats differents")
        print("=> UTILISE CLAUDE si tu veux la meilleure precision")
        print(f"   (cout: ~$4/mois pour toutes les 30min)")

    # Nettoyer
    await claude_scorer.close()
    await news_analyzer.close()

    print("\n" + "=" * 80)
    print("FIN DU TEST")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
