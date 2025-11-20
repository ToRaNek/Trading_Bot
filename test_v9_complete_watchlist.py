"""
Test COMPLET de TextBlob V9 sur les 40 actions de la watchlist
"""

import asyncio
import sys
sys.path.insert(0, '.')

from datetime import datetime
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from textblob_v9_refined import score_news_v9_refined
from optimize_textblob_final import UltraTextBlob
from config import WATCHLIST
import numpy as np


async def test_complete_watchlist():
    print("="*100)
    print("TEST COMPLET V9 - TOUTE LA WATCHLIST (40 ACTIONS)")
    print("="*100)
    print(f"\nNombre d'actions: {len(WATCHLIST)}")
    print("Objectif: Ecart moyen < 6 pts")
    print()

    analyzer = HistoricalNewsAnalyzer()
    scorer = UltraTextBlob()

    results = []
    skipped = []

    print(f"{'#':<4} {'Symbol':<10} {'V9':>7} {'Claude':>7} {'Diff':>7} {'Status'}")
    print("-"*100)

    for i, symbol in enumerate(WATCHLIST, 1):
        try:
            has_news, news_items, _ = await analyzer.get_news_for_date(symbol, datetime.now())

            if not has_news or not news_items:
                skipped.append(symbol)
                print(f"{i:<4} {symbol:<10} {'---':>7} {'---':>7} {'---':>7} Pas de news")
                continue

            v9_score = score_news_v9_refined(news_items)
            claude_score = await scorer.score_with_claude(symbol, news_items)
            diff = abs(claude_score - v9_score)

            # Status
            if diff <= 3:
                status = "PARFAIT"
            elif diff <= 6:
                status = "EXCELLENT"
            elif diff <= 10:
                status = "BON"
            else:
                status = "A ameliorer"

            print(f"{i:<4} {symbol:<10} {v9_score:>7.1f} {claude_score:>7.1f} {diff:>7.1f} {status}")

            results.append({
                'symbol': symbol,
                'v9': v9_score,
                'claude': claude_score,
                'diff': diff,
                'news_count': len(news_items)
            })

            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"{i:<4} {symbol:<10} {'ERR':>7} {'ERR':>7} {'ERR':>7} Erreur: {e}")
            skipped.append(symbol)

    await scorer.close()
    await analyzer.close()

    # STATISTIQUES COMPLETES
    print("\n" + "="*100)
    print("STATISTIQUES COMPLETES")
    print("="*100)

    if not results:
        print("\nAucun resultat disponible!")
        return

    differences = [r['diff'] for r in results]

    avg_diff = np.mean(differences)
    median_diff = np.median(differences)
    std_diff = np.std(differences)
    min_diff = min(differences)
    max_diff = max(differences)

    print(f"\nActions testees:      {len(results)}/{len(WATCHLIST)}")
    print(f"Actions sans news:    {len(skipped)}")
    print()
    print(f"Ecart MOYEN:          {avg_diff:.1f} pts")
    print(f"Ecart MEDIAN:         {median_diff:.1f} pts")
    print(f"Ecart MIN:            {min_diff:.1f} pts")
    print(f"Ecart MAX:            {max_diff:.1f} pts")
    print(f"Ecart-type:           {std_diff:.1f}")

    # Répartition
    perfect = sum(1 for d in differences if d <= 3)
    excellent = sum(1 for d in differences if 3 < d <= 6)
    good = sum(1 for d in differences if 6 < d <= 10)
    to_improve = sum(1 for d in differences if d > 10)

    total = len(results)

    print()
    print("-"*100)
    print("REPARTITION:")
    print("-"*100)
    print(f"Parfait (<=3 pts):        {perfect:>3} actions ({perfect/total*100:>5.1f}%)")
    print(f"Excellent (3-6 pts):      {excellent:>3} actions ({excellent/total*100:>5.1f}%)")
    print(f"Bon (6-10 pts):           {good:>3} actions ({good/total*100:>5.1f}%)")
    print(f"A ameliorer (>10 pts):    {to_improve:>3} actions ({to_improve/total*100:>5.1f}%)")

    under_6 = perfect + excellent
    print()
    print(f"TOTAL <= 6 pts:           {under_6:>3} actions ({under_6/total*100:>5.1f}%)")

    # Top 10 meilleures
    print()
    print("-"*100)
    print("TOP 10 MEILLEURES PREDICTIONS:")
    print("-"*100)
    print(f"{'Symbol':<10} {'V9':>7} {'Claude':>7} {'Diff':>7}")
    print("-"*100)

    sorted_results = sorted(results, key=lambda x: x['diff'])
    for r in sorted_results[:10]:
        print(f"{r['symbol']:<10} {r['v9']:>7.1f} {r['claude']:>7.1f} {r['diff']:>7.1f}")

    # Top 10 pires
    print()
    print("-"*100)
    print("TOP 10 PLUS GRANDES DIVERGENCES:")
    print("-"*100)
    print(f"{'Symbol':<10} {'V9':>7} {'Claude':>7} {'Diff':>7}")
    print("-"*100)

    for r in sorted_results[-10:]:
        print(f"{r['symbol']:<10} {r['v9']:>7.1f} {r['claude']:>7.1f} {r['diff']:>7.1f}")

    # COMPARAISON BASELINE
    print()
    print("="*100)
    print("COMPARAISON vs BASELINE:")
    print("="*100)
    print(f"Baseline V1:    8.7 pts moyenne")
    print(f"V9 Refined:     {avg_diff:.1f} pts moyenne")
    print()

    improvement = 8.7 - avg_diff
    if improvement > 0:
        print(f"AMELIORATION:   {improvement:.1f} pts ({improvement/8.7*100:.1f}%)")
    else:
        print(f"Pas d'amelioration")

    # VERDICT FINAL
    print()
    print("="*100)
    print("VERDICT FINAL:")
    print("="*100)

    if avg_diff <= 5:
        print(f"\n*** EXCEPTIONNEL! Ecart moyen de {avg_diff:.1f} pts ***")
        print("V9 est quasi identique a Claude!")
    elif avg_diff <= 6:
        print(f"\n*** OBJECTIF ATTEINT! Ecart moyen de {avg_diff:.1f} pts ***")
        print("V9 rivalise avec Claude!")
    elif avg_diff <= 8:
        print(f"\nTres bon! Ecart moyen de {avg_diff:.1f} pts")
        print("V9 est bien calibre")
    else:
        print(f"\nBon mais peut mieux faire: {avg_diff:.1f} pts")

    print(f"\n{under_6}/{total} actions ({under_6/total*100:.0f}%) ont un ecart <= 6 pts")


if __name__ == "__main__":
    asyncio.run(test_complete_watchlist())
