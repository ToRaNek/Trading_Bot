"""
Bot Discord de trading - ARCHITECTURE MODULAIRE
Utilise les modules séparés (analyzers/, backtest/)
"""

import discord
from discord.ext import commands
import logging
import time
import glob

from backtest import RealisticBacktestEngine
from config import WATCHLIST

logger = logging.getLogger('TradingBot')


class TradingBot(commands.Bot):
    """Bot Discord pour le trading avec backtest réaliste"""

    def __init__(self, reddit_csv_file: str = None, data_dir: str = 'data'):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True

        super().__init__(command_prefix='!', intents=intents, help_command=None)

        # Chercher automatiquement un fichier CSV Reddit si non spécifié
        if reddit_csv_file is None:
            csv_files = glob.glob('pushshift_*_ALL_*.csv')
            if csv_files:
                reddit_csv_file = csv_files[0]
                logger.info(f"[Init] Fichier CSV Reddit trouvé: {reddit_csv_file}")
            else:
                logger.warning("[Init] Aucun fichier CSV Reddit trouvé - Les requêtes API seront utilisées")

        # Utiliser le backtest engine modulaire
        self.backtest_engine = RealisticBacktestEngine(
            reddit_csv_file=reddit_csv_file,
            data_dir=data_dir
        )

    async def on_ready(self):
        logger.info(f'{self.user} connecté!')
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="backtests avec IA 🤖"
            )
        )


# Créer l'instance du bot
bot = TradingBot()


@bot.command(name='backtest')
async def backtest(ctx, months: int = 6):
    """
    Backtest réaliste avec validation IA + Sentiment Reddit - ANALYSE QUOTIDIENNE
    Analyse chaque jour de trading avec les actualités + sentiment Reddit
    Exemple: !backtest 6 (analyse ~120 jours de trading)
    """
    if months < 1 or months > 24:
        await ctx.send("❌ Période invalide. Utilisez entre 1 et 24 mois.")
        return

    embed = discord.Embed(
        title="⏳ Backtest Réaliste en cours...",
        description=f"Analyse QUOTIDIENNE sur {months} mois (~{months*20} jours)\n"
                   f"✅ Validation IA (News)\n"
                   f"✅ Sentiment Reddit\n"
                   f"✅ Analyse technique améliorée\n"
                   f"⚠️ Cela peut prendre 15-40 minutes...",
        color=0xffff00
    )
    embed.add_field(
        name="📋 Watchlist",
        value=f"{len(WATCHLIST)} actions",
        inline=True
    )
    embed.add_field(
        name="🤖 Sources",
        value="News + Reddit + Tech",
        inline=True
    )
    message = await ctx.send(embed=embed)

    start_time = time.time()

    try:
        results = await bot.backtest_engine.backtest_watchlist(WATCHLIST, months)
        elapsed = time.time() - start_time

        if not results:
            embed = discord.Embed(
                title="❌ Aucun résultat",
                description="Impossible de récupérer les données",
                color=0xff0000
            )
            await message.edit(embed=embed)
            return

        embed = discord.Embed(
            title=f"📊 Backtest Réaliste Terminé - {months} mois",
            description=f"{len(results)} actions analysées en {elapsed/60:.1f} minutes",
            color=0x00ff00
        )

        total_trades = sum(r['total_trades'] for r in results)
        total_validated = sum(r['validated_buys'] + r['validated_sells'] for r in results)
        total_rejected = sum(r['rejected_buys'] + r['rejected_sells'] for r in results)

        embed.add_field(name="⏱️ Temps", value=f"{elapsed/60:.1f}min", inline=True)
        embed.add_field(name="✅ Actions", value=f"{len(results)}", inline=True)
        embed.add_field(name="💼 Trades", value=f"{total_trades}", inline=True)
        embed.add_field(name="🤖 Validés", value=f"{total_validated}", inline=True)
        embed.add_field(name="❌ Rejetés", value=f"{total_rejected}", inline=True)
        embed.add_field(name="📊 Taux validation", value=f"{total_validated/(total_validated+total_rejected)*100:.0f}%" if (total_validated+total_rejected) > 0 else "N/A", inline=True)

        # Top 5 résultats
        for i, r in enumerate(results[:5], 1):
            perf = f"**{r['symbol']}** - {r['period']}\n"
            perf += f"💰 Profit: **{r['total_profit']:+.2f}%**\n"
            perf += f"📈 Win Rate: {r['win_rate']:.0f}%\n"
            perf += f"💼 Trades: {r['total_trades']} ({r['profitable_trades']} gagnants)\n"
            perf += f"⏱️ Durée moy: {r['avg_hold_hours']:.1f}h\n"
            perf += f"🤖 Validés: {r['validated_buys']}B / {r['validated_sells']}S\n"
            perf += f"❌ Rejetés: {r['rejected_buys']}B / {r['rejected_sells']}S\n"
            perf += f"📊 vs Hold: {r['strategy_vs_hold']:+.2f}%\n"
            perf += f"⭐ Score: **{r['strategy_score']:.0f}/100**"

            embed.add_field(name=f"#{i}", value=perf, inline=True)

            if i % 2 == 0:
                embed.add_field(name="\u200b", value="\u200b", inline=False)

        embed.set_footer(text="🤖 Chaque décision validée par IA (News) + Sentiment Reddit + Technique")

        await message.edit(embed=embed)

    except Exception as e:
        logger.error(f"Erreur backtest: {e}")
        import traceback
        traceback.print_exc()
        await message.edit(content=f"❌ Erreur: {str(e)}")


@bot.command(name='detail')
async def detail(ctx, symbol: str, months: int = 6):
    """
    Backtest détaillé d'une action avec tous les trades
    Exemple: !detail AAPL 6
    """
    symbol = symbol.upper()

    if months < 1 or months > 24:
        await ctx.send("❌ Période invalide. Utilisez entre 1 et 24 mois.")
        return

    embed = discord.Embed(
        title=f"⏳ Analyse détaillée {symbol}...",
        description=f"Backtest sur {months} mois avec validation IA",
        color=0xffff00
    )
    message = await ctx.send(embed=embed)

    try:
        result = await bot.backtest_engine.backtest_with_news_validation(symbol, months)

        if not result:
            embed = discord.Embed(
                title=f"❌ Erreur - {symbol}",
                description="Impossible de récupérer les données",
                color=0xff0000
            )
            await message.edit(embed=embed)
            return

        # Embed principal
        embed = discord.Embed(
            title=f"📊 Backtest Détaillé - {symbol}",
            description=f"Période: {result['period']} | Score: **{result['strategy_score']:.0f}/100**",
            color=0x00ff00 if result['total_profit'] > 0 else 0xff0000
        )

        # Stats générales
        embed.add_field(name="💰 Profit Total", value=f"**{result['total_profit']:+.2f}%**", inline=True)
        embed.add_field(name="📈 Win Rate", value=f"{result['win_rate']:.0f}%", inline=True)
        embed.add_field(name="💼 Trades", value=f"{result['total_trades']}", inline=True)

        embed.add_field(name="📊 Profit Moyen", value=f"{result['avg_profit']:+.2f}%", inline=True)
        embed.add_field(name="🎯 Max Profit", value=f"{result['max_profit']:+.2f}%", inline=True)
        embed.add_field(name="⚠️ Max Loss", value=f"{result['max_loss']:+.2f}%", inline=True)

        embed.add_field(name="⏱️ Durée Moy", value=f"{result['avg_hold_hours']:.1f}h", inline=True)
        embed.add_field(name="📅 Points décision", value=f"{result['decision_points']}", inline=True)
        embed.add_field(name="🏦 Buy & Hold", value=f"{result['buy_hold_return']:+.2f}%", inline=True)

        # Validation IA
        validation_text = f"🤖 **Validation IA:**\n"
        validation_text += f"✅ Achats validés: {result['validated_buys']}\n"
        validation_text += f"❌ Achats rejetés: {result['rejected_buys']}\n"
        validation_text += f"✅ Ventes validées: {result['validated_sells']}\n"
        validation_text += f"❌ Ventes rejetées: {result['rejected_sells']}"
        embed.add_field(name="🤖 Décisions IA", value=validation_text, inline=False)

        await message.edit(embed=embed)

        # Envoyer les trades détaillés si disponibles
        if result['trades']:
            trades_embed = discord.Embed(
                title=f"💼 Détail des Trades - {symbol}",
                color=0x00ffff
            )

            for i, trade in enumerate(result['trades'][:10], 1):  # Max 10 trades
                profit_emoji = "🟢" if trade['profit'] > 0 else "🔴"
                trade_text = f"{profit_emoji} **{trade['profit']:+.2f}%**\n"
                trade_text += f"📅 Entrée: {trade['entry_date'].strftime('%Y-%m-%d %H:%M')}\n"
                trade_text += f"💰 Prix: ${trade['entry_price']:.2f}\n"
                trade_text += f"📅 Sortie: {trade['exit_date'].strftime('%Y-%m-%d %H:%M')}\n"
                trade_text += f"💰 Prix: ${trade['exit_price']:.2f}\n"
                trade_text += f"⏱️ Durée: {trade['hold_hours']:.1f}h"

                trades_embed.add_field(
                    name=f"Trade #{i}",
                    value=trade_text,
                    inline=True
                )

                if i % 2 == 0:
                    trades_embed.add_field(name="\u200b", value="\u200b", inline=False)

            if len(result['trades']) > 10:
                trades_embed.set_footer(text=f"... et {len(result['trades'])-10} autres trades")

            await ctx.send(embed=trades_embed)

    except Exception as e:
        logger.error(f"Erreur detail: {e}")
        import traceback
        traceback.print_exc()
        await message.edit(content=f"❌ Erreur: {str(e)}")


@bot.command(name='aide')
async def aide(ctx):
    """Affiche l'aide"""
    embed = discord.Embed(
        title="📚 Guide des Commandes",
        description="Bot de Trading avec Backtest Réaliste, Validation IA et Sentiment Reddit",
        color=0x00ffff
    )

    embed.add_field(
        name="⏱️ **!backtest [mois]**",
        value="Backtest quotidien avec validation multi-sources\n"
              "Analyse CHAQUE JOUR de trading (~20 jours/mois)\n"
              "Le bot prend des décisions quotidiennes\n"
              "Score composite : Tech + IA/News + Reddit\n"
              "Exemple: `!backtest 6` (analyse ~120 jours)",
        inline=False
    )

    embed.add_field(
        name="📊 **!detail [SYMBOL] [mois]**",
        value="Backtest détaillé d'une action avec tous les trades\n"
              "Affiche les scores Tech, IA et Reddit pour chaque trade\n"
              "Exemple: `!detail AAPL 6`",
        inline=False
    )

    embed.add_field(
        name="🤖 **Comment ça marche?**",
        value="1️⃣ Analyse technique AMÉLIORÉE (système de confluence)\n"
              "   • RSI, MACD, SMA, Bollinger, Volume (score 0-100)\n"
              "2️⃣ Le bot décide: BUY, SELL ou HOLD\n"
              "3️⃣ Si BUY/SELL: récupération News + Reddit\n"
              "4️⃣ Score composite pondéré:\n"
              "   • Technique: 40%\n"
              "   • IA/News: 35%\n"
              "   • Reddit: 25%\n"
              "5️⃣ Si score final > 65, le trade est exécuté ✅\n"
              "6️⃣ Sinon, le trade est rejeté ❌",
        inline=False
    )

    embed.add_field(
        name="📱 **Sources Reddit**",
        value="Subreddits dédiés (r/NVDA_Stock, r/AAPL, etc.)\n"
              "Recherche r/stocks pour tous les tickers\n"
              "Analyse sentiment basée sur posts et upvotes\n"
              "Détection de confluence/conflit avec les news",
        inline=False
    )

    embed.add_field(
        name="💡 **Avantages**",
        value="✅ Système technique amélioré avec confluence\n"
              "✅ Simulation temps réel (analyse quotidienne)\n"
              "✅ Actualités historiques pour chaque jour\n"
              "✅ Sentiment Reddit en temps réel\n"
              "✅ Score composite multi-sources\n"
              "✅ Évite les faux signaux techniques\n"
              "✅ Compare avec Buy & Hold\n"
              "✅ Cache intelligent pour optimiser les API",
        inline=False
    )

    embed.set_footer(text="🔥 Backtest ultra-réaliste : Tech + IA + Reddit")

    await ctx.send(embed=embed)


# Exporter pour que main.py puisse l'utiliser
__all__ = ['bot', 'TradingBot']
