"""
Bot Discord de trading - ARCHITECTURE MODULAIRE
Utilise les modules séparés (analyzers/, backtest/)
"""

import discord
from discord.ext import commands
import logging
import time
import glob
import asyncio
from datetime import datetime

from backtest import RealisticBacktestEngine
from config import WATCHLIST
from trading import LiveTrader
from trading.participants import ParticipantsManager
from utils import StockInfo

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

        # Live trader pour le mode dry-run
        self.live_trader = None
        self.live_trader_task = None

        # Gestionnaire de participants pour le trading manuel
        self.participants_manager = ParticipantsManager()

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
        description="Bot de Trading avec Backtest Réaliste, Validation IA et Trading en Temps Réel\n"
                   "**NOUVEAU**: Trading avec respect des horaires de marché !",
        color=0x00ffff
    )

    # SECTION 1: Trading en temps réel
    embed.add_field(
        name="🚀 **TRADING EN TEMPS RÉEL**",
        value="━━━━━━━━━━━━━━━━━━━━━━━━━",
        inline=False
    )

    embed.add_field(
        name="⚡ **!start**",
        value="Démarre le bot en mode temps réel\n"
              "• Analyses automatiques pendant les horaires de marché\n"
              "• **PING automatique** des participants sur chaque signal\n"
              "• Vous exécutez les trades **MANUELLEMENT**\n"
              "• Le bot garde trace des positions\n"
              "• Tourne en continu jusqu'à `!stop`\n"
              "Exemple: `!start`",
        inline=False
    )

    embed.add_field(
        name="⏹️ **!stop**",
        value="Arrête le bot en mode dry-run\n"
              "Affiche les statistiques finales",
        inline=False
    )

    embed.add_field(
        name="📊 **!status**",
        value="Affiche le statut du bot en temps réel\n"
              "Performance, positions, statistiques",
        inline=False
    )

    embed.add_field(
        name="👥 **!participer**",
        value="S'enregistre comme participant\n"
              "• Tu seras pingé sur chaque signal de trading\n"
              "• Donne accès à la commande `!cash`\n"
              "• Une seule fois par utilisateur\n"
              "Exemple: `!participer`",
        inline=False
    )

    embed.add_field(
        name="💰 **!cash [montant]**",
        value="Gère ton cash disponible (participants uniquement)\n"
              "• Sans argument: affiche ton cash actuel\n"
              "• Avec montant: définit ton cash disponible\n"
              "• Permet au bot de te suggérer des montants\n"
              "Exemple: `!cash 5000` (tu as 5000€ disponibles)",
        inline=False
    )

    # SECTION 2: Backtests
    embed.add_field(
        name="📈 **BACKTESTS HISTORIQUES**",
        value="━━━━━━━━━━━━━━━━━━━━━━━━━",
        inline=False
    )

    embed.add_field(
        name="⏱️ **!backtest [mois]**",
        value="Backtest quotidien avec validation multi-sources\n"
              "Analyse CHAQUE JOUR de trading (~20 jours/mois)\n"
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
        value="1️⃣ **Vérification horaires de marché**\n"
              "   • US: 15:30-21:45 (heure FR)\n"
              "   • France: 09:00-17:15 (heure FR)\n"
              "   • Pas de trading hors horaires !\n"
              "2️⃣ Analyse technique AMÉLIORÉE (système de confluence)\n"
              "   • RSI, MACD, SMA, Bollinger, Volume (score 0-100)\n"
              "3️⃣ Le bot décide: BUY, SELL ou HOLD\n"
              "4️⃣ Si BUY/SELL: récupération News + Reddit\n"
              "5️⃣ Score composite pondéré:\n"
              "   • Technique: 50%\n"
              "   • IA/News: 50%\n"
              "6️⃣ Si score final > 65, le trade est exécuté ✅\n"
              "7️⃣ Sinon, le trade est rejeté ❌",
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
        value="✅ Respect des horaires de marché (US/FR)\n"
              "✅ Noms complets des actions (pas que les tickers)\n"
              "✅ Système technique amélioré avec confluence\n"
              "✅ Simulation temps réel (analyse quotidienne)\n"
              "✅ Actualités historiques pour chaque jour\n"
              "✅ Sentiment Reddit en temps réel\n"
              "✅ Score composite multi-sources\n"
              "✅ Évite les faux signaux techniques\n"
              "✅ Compare avec Buy & Hold\n"
              "✅ Cache intelligent pour optimiser les API",
        inline=False
    )

    embed.add_field(
        name="🕐 **Horaires de Trading**",
        value="**Marchés US** (NVDA, AAPL, etc.)\n"
              "• Ouverture: 15:30 (heure FR)\n"
              "• Fermeture: 22:00 (heure FR)\n"
              "• Dernière analyse: 21:45\n\n"
              "**Marchés France** (MC.PA, OR.PA, etc.)\n"
              "• Ouverture: 09:00 (heure FR)\n"
              "• Fermeture: 17:30 (heure FR)\n"
              "• Dernière analyse: 17:15\n\n"
              "⚠️ Pas de trading le week-end !",
        inline=False
    )

    embed.set_footer(text="🔥 Trading Bot avec IA : Backtest + Trading Temps Réel avec horaires de marché")

    await ctx.send(embed=embed)


@bot.command(name='start')
async def start(ctx):
    """
    Démarre le bot en mode temps réel (signaux manuels)
    Le bot analyse les actions et envoie des signaux aux participants
    Exemple: !start
    """
    if bot.live_trader and bot.live_trader.is_running:
        await ctx.send("❌ Le bot est déjà en cours d'exécution. Utilisez `!stop` pour l'arrêter d'abord.")
        return

    # Compter les participants
    num_participants = len(bot.participants_manager.participants)

    embed = discord.Embed(
        title="🚀 Démarrage du Bot en Temps Réel",
        description=f"Le bot va analyser le marché et envoyer des signaux de trading",
        color=0x00ff00
    )
    embed.add_field(name="👥 Participants", value=f"{num_participants}", inline=True)
    embed.add_field(name="📊 Watchlist", value=f"{len(WATCHLIST)} actions", inline=True)
    embed.add_field(name="⏰ Analyses", value="Toutes les heures", inline=True)
    embed.add_field(name="🕐 Horaires", value="US: 15:30-21:45\nFR: 09:00-17:15", inline=True)
    embed.add_field(name="🤖 Sources", value="Tech + News + Reddit", inline=True)
    embed.add_field(name="🎯 Seuil", value="65/100", inline=True)
    embed.add_field(
        name="ℹ️ Fonctionnement",
        value="• Le bot analyse chaque action pendant les horaires de marché\n"
              "• Quand un signal BUY/SELL est validé, **tous les participants sont pingés**\n"
              "• Vous exécutez les trades **manuellement** sur votre plateforme\n"
              "• Le bot garde une trace des positions pour les prochains signaux\n"
              "• Vous pouvez redémarrer le bot sans perdre les positions",
        inline=False
    )

    if num_participants == 0:
        embed.add_field(
            name="⚠️ Attention",
            value="Aucun participant enregistré ! Utilisez `!participer` pour vous inscrire.",
            inline=False
        )

    await ctx.send(embed=embed)

    # Créer et démarrer le live trader avec restauration de l'état
    bot.live_trader = LiveTrader(
        initial_cash=1000.0,
        watchlist=WATCHLIST,
        discord_channel=ctx.channel,
        portfolio_file='portfolio_temps_reel.json'  # Fichier de persistance
    )

    # Connecter le participants manager au trader
    bot.live_trader.participants_manager = bot.participants_manager

    # Lancer le trader dans une tâche asynchrone (en continu, pas de durée)
    bot.live_trader_task = bot.loop.create_task(bot.live_trader.start(duration_days=None))

    logger.info(f"[Discord] Bot démarré en temps réel par {ctx.author}")


@bot.command(name='stop')
async def stop_trading(ctx):
    """
    Arrête le bot en mode dry-run
    Exemple: !stop
    """
    if not bot.live_trader or not bot.live_trader.is_running:
        await ctx.send("❌ Le bot n'est pas en cours d'exécution.")
        return

    embed = discord.Embed(
        title="⏹️ Arrêt du Bot",
        description="Arrêt en cours...",
        color=0xff0000
    )
    await ctx.send(embed=embed)

    # Arrêter le trader
    if bot.live_trader_task:
        bot.live_trader_task.cancel()
        try:
            await bot.live_trader_task
        except asyncio.CancelledError:
            pass

    await bot.live_trader.stop()

    logger.info(f"[Discord] Bot arrêté par {ctx.author}")


@bot.command(name='status')
async def status(ctx):
    """
    Affiche le statut du bot en dry-run
    Exemple: !status
    """
    if not bot.live_trader:
        embed = discord.Embed(
            title="📊 Statut du Bot",
            description="Le bot n'a jamais été démarré. Utilisez `!start` pour le lancer.",
            color=0x808080
        )
        await ctx.send(embed=embed)
        return

    # Calculer les prix actuels
    current_prices = {}
    for symbol in bot.live_trader.portfolio.positions.keys():
        try:
            import yfinance as yf
            stock = yf.Ticker(symbol)
            current_prices[symbol] = stock.history(period='1d', interval='1m')['Close'].iloc[-1]
        except:
            pass

    performance = bot.live_trader.portfolio.get_performance(current_prices)

    # Créer l'embed
    status_text = "🟢 **EN COURS**" if bot.live_trader.is_running else "🔴 **ARRÊTÉ**"
    color = 0x00ff00 if bot.live_trader.is_running else 0xff0000

    embed = discord.Embed(
        title="📊 Statut du Bot - Dry-Run",
        description=status_text,
        color=color,
        timestamp=datetime.now()
    )

    # Performance
    profit_emoji = "📈" if performance['total_return'] > 0 else "📉"
    embed.add_field(
        name="💰 Performance",
        value=f"{profit_emoji} **{performance['total_return_pct']:+.2f}%**\n"
              f"Capital: ${performance['total_value']:.2f}\n"
              f"Initial: ${performance['initial_cash']:.2f}",
        inline=True
    )

    # Statistiques de trading
    embed.add_field(
        name="📊 Statistiques",
        value=f"Trades: {performance['total_trades']}\n"
              f"Win Rate: {performance['win_rate']:.1f}%\n"
              f"Jours: {performance['days_running']}",
        inline=True
    )

    # Positions ouvertes
    positions_text = ""
    if bot.live_trader.portfolio.positions:
        for symbol, position in bot.live_trader.portfolio.positions.items():
            price = current_prices.get(symbol, 0)
            if price > 0:
                profit_pct = ((price - position['avg_price']) / position['avg_price']) * 100
                positions_text += f"**{symbol}**: {position['shares']} @ ${position['avg_price']:.2f} ({profit_pct:+.2f}%)\n"
            else:
                positions_text += f"**{symbol}**: {position['shares']} @ ${position['avg_price']:.2f}\n"
    else:
        positions_text = "Aucune position ouverte"

    embed.add_field(
        name="📋 Positions",
        value=positions_text[:1024],  # Limiter à 1024 caractères
        inline=False
    )

    # Statistiques d'analyse
    if bot.live_trader.is_running:
        embed.add_field(
            name="🤖 Activité",
            value=f"Analyses: {bot.live_trader.analysis_count}\n"
                  f"Signaux BUY: {bot.live_trader.buy_signals}\n"
                  f"Signaux SELL: {bot.live_trader.sell_signals}",
            inline=True
        )

        embed.add_field(
            name="✅ Décisions IA",
            value=f"Validés: {bot.live_trader.validated_trades}\n"
                  f"Rejetés: {bot.live_trader.rejected_trades}\n"
                  f"Taux: {bot.live_trader.validated_trades/(bot.live_trader.validated_trades+bot.live_trader.rejected_trades)*100:.0f}%" if (bot.live_trader.validated_trades+bot.live_trader.rejected_trades) > 0 else "N/A",
            inline=True
        )

    await ctx.send(embed=embed)


@bot.command(name='cash')
async def update_cash(ctx, amount: float = None):
    """
    Met à jour ton cash disponible (réservé aux participants)
    Exemple: !cash 5000 (définit que tu as 5000€ disponibles)
    """
    user_id = ctx.author.id
    username = ctx.author.name

    # Vérifier que l'utilisateur est un participant
    if user_id not in bot.participants_manager.participants:
        embed = discord.Embed(
            title="❌ Non Participant",
            description="Tu dois d'abord t'enregistrer avec `!participer`",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return

    if amount is None:
        # Afficher le cash actuel de l'utilisateur
        participant = bot.participants_manager.participants[user_id]
        embed = discord.Embed(
            title="💰 Ton Cash",
            description=f"Voici ton cash disponible",
            color=0x00ff00
        )
        embed.add_field(name="Cash disponible", value=f"${participant['cash']:.2f}", inline=True)
        embed.add_field(name="Positions", value=f"{len(participant['positions'])}", inline=True)
        embed.add_field(name="Profit total", value=f"${participant['total_profit']:+.2f}", inline=True)

        await ctx.send(embed=embed)
        return

    if amount < 0:
        await ctx.send("❌ Le montant doit être positif.")
        return

    # Mettre à jour le cash de l'utilisateur
    bot.participants_manager.update_cash(user_id, amount)

    embed = discord.Embed(
        title="💰 Cash Mis à Jour",
        description=f"Ton cash a été défini à **${amount:.2f}**",
        color=0x00ff00,
        timestamp=datetime.now()
    )
    embed.add_field(name="Participant", value=ctx.author.mention, inline=True)
    embed.add_field(name="Nouveau cash", value=f"${amount:.2f}", inline=True)

    await ctx.send(embed=embed)
    logger.info(f"[Discord] Cash mis à jour pour {username}: ${amount:.2f}")


@bot.command(name='participer')
async def participer(ctx):
    """
    S'enregistre comme participant pour recevoir les signaux de trading
    Exemple: !participer
    """
    user_id = ctx.author.id
    username = ctx.author.name

    # Vérifier si l'utilisateur est déjà enregistré
    if user_id in bot.participants_manager.participants:
        embed = discord.Embed(
            title="✅ Déjà Participant",
            description=f"Tu es déjà enregistré comme participant !",
            color=0x00ff00,
            timestamp=datetime.now()
        )

        # Afficher les infos du participant
        participant = bot.participants_manager.participants[user_id]
        embed.add_field(name="Nom", value=username, inline=True)
        embed.add_field(name="Cash", value=f"${participant['cash']:.2f}", inline=True)
        embed.add_field(name="Positions", value=f"{len(participant['positions'])}", inline=True)

        await ctx.send(embed=embed)
        return

    # Enregistrer le nouveau participant
    bot.participants_manager.add_participant(user_id, username, initial_cash=0.0)

    embed = discord.Embed(
        title="🎉 Participant Enregistré",
        description=f"Bienvenue {username} ! Tu recevras maintenant tous les signaux de trading.",
        color=0x00ff00,
        timestamp=datetime.now()
    )

    embed.add_field(
        name="📝 Prochaines Étapes",
        value="1️⃣ Utilise `!cash <montant>` pour définir ton cash disponible\n"
              "2️⃣ Attends les signaux du bot (tu seras pingé)\n"
              "3️⃣ Execute les trades manuellement sur ta plateforme\n"
              "4️⃣ Utilise `!status` pour voir les positions du bot",
        inline=False
    )

    embed.add_field(
        name="💡 Info",
        value=f"Cash actuel: $0.00\n"
              f"Tu peux le mettre à jour avec `!cash <montant>`",
        inline=False
    )

    await ctx.send(embed=embed)
    logger.info(f"[Discord] Nouveau participant enregistré: {username} (ID: {user_id})")


# Exporter pour que main.py puisse l'utiliser
__all__ = ['bot', 'TradingBot']
