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


@bot.command(name='reel_backtest')
async def backtest(ctx, symbol: str = None, months: int = 6):
    """
    Backtest réaliste avec validation IA (News only, Reddit désactivé)
    Analyse chaque jour de trading avec les actualités

    Exemples:
    !reel_backtest MSFT 6 (backtest MSFT sur 6 mois)
    !reel_backtest AAPL (backtest AAPL sur 6 mois par défaut)
    !reel_backtest 3 (backtest toute la watchlist sur 3 mois)
    """
    # Si le premier paramètre est un nombre, c'est months sans symbole
    if symbol and symbol.isdigit():
        months = int(symbol)
        symbol = None

    if months < 1 or months > 24:
        await ctx.send("❌ Période invalide. Utilisez entre 1 et 24 mois.")
        return

    # Déterminer les symboles à tester
    if symbol:
        symbols = [symbol.upper()]
        watchlist_str = f"1 action ({symbol.upper()})"
    else:
        symbols = WATCHLIST
        watchlist_str = f"{len(WATCHLIST)} actions"

    embed = discord.Embed(
        title="⏳ Backtest Réaliste en cours...",
        description=f"Analyse sur {months} mois (~{months*20} jours)\n"
                   f"✅ Validation IA (News)\n"
                   f"✅ Analyse technique V11\n"
                   f"✅ Score composite intelligent\n"
                   f"⚠️ Cela peut prendre quelques minutes...",
        color=0xffff00
    )
    embed.add_field(
        name="📋 Actions",
        value=watchlist_str,
        inline=True
    )
    embed.add_field(
        name="🤖 Sources",
        value="News + Tech",
        inline=True
    )
    message = await ctx.send(embed=embed)

    start_time = time.time()

    try:
        results = await bot.backtest_engine.backtest_watchlist(symbols, months)
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


@bot.command(name='reel_detail')
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


@bot.command(name='reel_aide')
async def aide(ctx):
    """Affiche l'aide"""
    embed = discord.Embed(
        title="📚 Commandes du Bot",
        description="Bot de Trading en Temps Réel avec IA",
        color=0x00ffff
    )

    embed.add_field(
        name="🚀 **Commandes Principales**",
        value="`!reel_participer` - S'inscrire pour recevoir les signaux\n"
              "`!reel_cash 5000` - Définir ton cash disponible\n"
              "`!reel_start` - Démarrer le bot (admin)\n"
              "`!reel_stop` - Arrêter le bot (admin)\n"
              "`!reel_status` - Voir les positions actuelles",
        inline=False
    )

    embed.add_field(
        name="📊 **Backtests**",
        value="`!reel_backtest 6` - Backtest sur 6 mois\n"
              "`!reel_detail NVDA 6` - Détails d'une action",
        inline=False
    )

    embed.add_field(
        name="⚡ **Comment ça marche ?**",
        value="1️⃣ Inscris-toi avec `!reel_participer`\n"
              "2️⃣ Définis ton cash avec `!reel_cash 5000`\n"
              "3️⃣ Reçois les signaux dans ton channel privé\n"
              "4️⃣ Exécute les trades manuellement\n"
              "5️⃣ Le bot track tes positions",
        inline=False
    )

    embed.add_field(
        name="🕐 **Horaires**",
        value="**US**: 15:30-21:45 | **FR**: 09:00-17:15 (heure FR)\n"
              "Pas de trading le week-end",
        inline=False
    )

    embed.add_field(
        name="🤖 **Validation IA**",
        value="Score = 50% Technique + 50% News\n"
              "Signal envoyé si score ≥ 65/100",
        inline=False
    )

    await ctx.send(embed=embed)


@bot.command(name='reel_start')
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
            value="Aucun participant enregistré ! Utilisez `!reel_participer` pour vous inscrire.",
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


@bot.command(name='reel_stop')
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


@bot.command(name='reel_status')
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


@bot.command(name='reel_cash')
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
            description="Tu dois d'abord t'enregistrer avec `!reel_participer`",
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


@bot.command(name='reel_participer')
async def participer(ctx):
    """
    S'enregistre comme participant pour recevoir les signaux de trading
    Exemple: !reel_participer
    """
    user_id = ctx.author.id
    username = ctx.author.name
    user = ctx.author

    # Vérifier si l'utilisateur est déjà enregistré
    if user_id in bot.participants_manager.participants:
        # Envoyer en DM
        embed = discord.Embed(
            title="✅ Déjà Participant",
            description=f"Tu es déjà enregistré !",
            color=0x00ff00,
            timestamp=datetime.now()
        )

        participant = bot.participants_manager.participants[user_id]
        embed.add_field(name="Cash", value=f"${participant['cash']:.2f}", inline=True)
        embed.add_field(name="Positions", value=f"{len(participant['positions'])}", inline=True)

        # Récupérer le channel s'il existe
        channel_id = participant.get('private_channel_id')
        if channel_id:
            channel = bot.get_channel(channel_id)
            if channel:
                embed.add_field(name="Channel", value=f"<#{channel_id}>", inline=True)

        try:
            await user.send(embed=embed)
            await ctx.message.add_reaction('✅')
        except:
            await ctx.send(f"{user.mention} Je ne peux pas t'envoyer de DM. Active les messages privés.", delete_after=10)
        return

    # Créer un channel privé pour le participant
    guild = ctx.guild
    category = discord.utils.get(guild.categories, name="📊 Trading Signaux")

    # Créer la catégorie si elle n'existe pas
    if not category:
        category = await guild.create_category("📊 Trading Signaux")

    # Créer le channel privé
    overwrites = {
        guild.default_role: discord.PermissionOverwrite(read_messages=False),
        user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
        guild.me: discord.PermissionOverwrite(read_messages=True, send_messages=True)
    }

    channel = await guild.create_text_channel(
        name=f"signals-{username.lower()}",
        category=category,
        overwrites=overwrites,
        topic=f"Signaux de trading privés pour {username}"
    )

    # Enregistrer le participant avec son channel
    bot.participants_manager.add_participant(user_id, username, initial_cash=0.0)
    bot.participants_manager.participants[user_id]['private_channel_id'] = channel.id
    bot.participants_manager.save_state()

    # Message dans le channel privé
    channel_embed = discord.Embed(
        title="🎉 Bienvenue sur ton Channel Privé !",
        description=f"Salut {username} ! C'est ici que tu recevras tous les signaux de trading.",
        color=0x00ff00,
        timestamp=datetime.now()
    )

    channel_embed.add_field(
        name="📝 Prochaines Étapes",
        value="1️⃣ Utilise `!reel_cash <montant>` pour définir ton cash\n"
              "2️⃣ Attends les signaux (tu seras pingé ici)\n"
              "3️⃣ Exécute les trades manuellement\n"
              "4️⃣ Utilise `!reel_status` pour voir les positions",
        inline=False
    )

    channel_embed.add_field(
        name="💰 Cash Actuel",
        value="$0.00 - Utilise `!reel_cash <montant>` pour le définir",
        inline=False
    )

    await channel.send(f"{user.mention}", embed=channel_embed)

    # Envoyer en DM aussi
    dm_embed = discord.Embed(
        title="✅ Inscription Réussie !",
        description=f"Tu es maintenant participant. Ton channel privé : <#{channel.id}>",
        color=0x00ff00
    )

    try:
        await user.send(embed=dm_embed)
    except:
        pass

    # Réaction dans le channel public
    await ctx.message.add_reaction('✅')

    logger.info(f"[Discord] Participant enregistré: {username} (ID: {user_id}, Channel: {channel.id})")


# Exporter pour que main.py puisse l'utiliser
__all__ = ['bot', 'TradingBot']
