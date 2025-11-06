"""
Bot Discord de trading - EN TRANSITION

Pour l'instant, importe depuis trading_bot_main.py
TODO: Extraire complètement la classe TradingBot + commandes Discord ici
"""

# Import temporaire depuis l'ancien fichier
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trading_bot_main import TradingBot, bot

# Ré-exporter pour que les imports fonctionnent
__all__ = ['TradingBot', 'bot']

# TODO: Remplacer par:
# import discord
# from discord.ext import commands
# from backtest import RealisticBacktestEngine
# from config import WATCHLIST
# etc...
