"""
Moteur de backtest réaliste - EN TRANSITION

Pour l'instant, importe depuis trading_bot_main.py
TODO: Extraire complètement la classe RealisticBacktestEngine ici
"""

# Import temporaire depuis l'ancien fichier
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trading_bot_main import RealisticBacktestEngine

# Ré-exporter pour que les imports fonctionnent
__all__ = ['RealisticBacktestEngine']

# TODO: Remplacer par:
# from analyzers import TechnicalAnalyzer, HistoricalNewsAnalyzer, RedditSentimentAnalyzer
# import yfinance as yf
# etc...
