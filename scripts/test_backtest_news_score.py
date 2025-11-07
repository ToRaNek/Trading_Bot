# -*- coding: utf-8 -*-
"""Test rapide pour verifier que le score de news s'affiche correctement dans le backtest"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from backtest.backtest_engine import RealisticBacktestEngine

load_dotenv()


async def test_backtest():
    """Test rapide du backtest avec les nouveaux scores"""

    print("=" * 70)
    print("TEST DU BACKTEST AVEC SCORES DE NEWS")
    print("=" * 70)
    print()

    # Créer le backtest engine
    engine = RealisticBacktestEngine()

    # Lancer le backtest sur NVDA (3 mois pour avoir assez de données)
    results = await engine.backtest_with_news_validation('NVDA', months=3)

    print("\n" + "=" * 70)
    print("TEST TERMINE")
    print("=" * 70)

    if results:
        print(f"\nTrades executes: {results.get('total_trades', 0)}")
        print(f"Win rate: {results.get('win_rate', 0):.1f}%")
        print(f"Profit total: {results.get('total_profit', 0):.2f}%")
        print(f"Profit max: {results.get('max_profit', 0):.2f}%")
        print(f"Max loss: {results.get('max_loss', 0):.2f}%")
        print(f"Duree moyenne: {results.get('avg_hold_days', 0):.0f}j")
    else:
        print("\nAucun resultat - donnees insuffisantes")

    await engine.close()


if __name__ == "__main__":
    asyncio.run(test_backtest())
