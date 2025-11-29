"""
Système de trading en temps réel (dry-run) avec portefeuille simulé
Analyse les actions chaque heure et prend des décisions autonomes
"""

import asyncio
import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import discord

from .portfolio import Portfolio
from analyzers import TechnicalAnalyzer, HistoricalNewsAnalyzer, RedditSentimentAnalyzer
from config import WATCHLIST, VALIDATION_THRESHOLD
from utils import MarketHours, StockInfo
import os

logger = logging.getLogger('TradingBot')


class LiveTrader:
    """
    Trading bot qui fonctionne en temps réel (simulé)
    - Analyse toutes les heures
    - Récupère les news et posts Reddit du jour
    - Prend des décisions d'achat/vente
    - Gère un portefeuille simulé
    """

    def __init__(self, initial_cash: float = 1000.0, watchlist: List[str] = None,
                 portfolio_file: str = 'portfolio_live.json', discord_channel=None):
        """
        Initialise le trader en temps réel

        Args:
            initial_cash: Capital initial
            watchlist: Liste des actions à trader (par défaut: WATCHLIST)
            portfolio_file: Fichier de sauvegarde du portefeuille
            discord_channel: Canal Discord pour les notifications
        """
        self.portfolio = Portfolio(initial_cash, save_file=portfolio_file)
        self.watchlist = watchlist or WATCHLIST
        self.discord_channel = discord_channel
        self.is_running = False
        self.participants_manager = None  # Sera défini par le bot Discord

        # Initialiser les analyseurs
        self.tech_analyzer = TechnicalAnalyzer()
        self.news_analyzer = HistoricalNewsAnalyzer()

        # RedditSentimentAnalyzer avec OAuth pour contourner le blocage IP Azure
        reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_analyzer = RedditSentimentAnalyzer(
            reddit_client_id=reddit_client_id,
            reddit_client_secret=reddit_client_secret
        )

        # Configuration du trading
        self.validation_threshold = VALIDATION_THRESHOLD  # Score minimum pour trader
        self.max_position_size = 0.2  # 20% du portfolio max par position
        self.stop_loss_pct = -4.0  # Stop loss à -4%
        self.take_profit_pct = 16.0  # Take profit à +16%

        # Statistiques de session
        self.analysis_count = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.validated_trades = 0
        self.rejected_trades = 0

    async def send_discord_notification(self, embed: discord.Embed):
        """Envoie une notification Discord si un canal est configuré"""
        if self.discord_channel:
            try:
                await self.discord_channel.send(embed=embed)
            except Exception as e:
                logger.error(f"[LiveTrader] Erreur envoi Discord: {e}")

    async def send_heartbeat(self):
        """Envoie un heartbeat Discord toutes les 15 minutes (sauf à :30)"""
        if not self.discord_channel:
            return

        now = datetime.now()
        current_prices = {}

        # Récupérer les prix actuels des positions
        for symbol in self.portfolio.positions.keys():
            try:
                stock = yf.Ticker(symbol)
                current_prices[symbol] = stock.history(period='1d', interval='1m')['Close'].iloc[-1]
            except:
                pass

        performance = self.portfolio.get_performance(current_prices)

        embed = discord.Embed(
            title="💓 Heartbeat - Bot actif",
            description=f"Statut au {now.strftime('%H:%M')}",
            color=0x00aaff,
            timestamp=now
        )
        embed.add_field(name="Cash", value=f"${self.portfolio.cash:.2f}", inline=True)
        embed.add_field(name="Valeur totale", value=f"${performance['total_value']:.2f}", inline=True)
        embed.add_field(name="Performance", value=f"{performance['total_return_pct']:+.2f}%", inline=True)
        embed.add_field(name="Positions", value=f"{len(self.portfolio.positions)}", inline=True)
        embed.add_field(name="Trades", value=f"{performance['total_trades']}", inline=True)
        embed.add_field(name="Win Rate", value=f"{performance['win_rate']:.1f}%" if performance['total_trades'] > 0 else "N/A", inline=True)

        await self.send_discord_notification(embed)

    async def send_daily_cash_reminder(self):
        """Envoie un rappel quotidien à 23h pour mettre à jour le cash"""
        if not self.discord_channel or not self.participants_manager:
            return

        participants_needing_update = self.participants_manager.get_participants_needing_cash_update()

        if not participants_needing_update:
            logger.info("[LiveTrader] Tous les participants ont un cash à jour ✅")
            return

        # Créer le message de rappel
        embed = discord.Embed(
            title="💰 Rappel Quotidien - Mise à jour du Cash",
            description=f"⏰ **Il est {datetime.now().strftime('%H')}h !** Mettez à jour votre cash disponible avec `!reel_cash <montant>`",
            color=0xffa500,
            timestamp=datetime.now()
        )

        # Lister les participants qui doivent update
        participants_text = ""
        mentions = []
        for user_id, username, reason in participants_needing_update[:10]:  # Max 10
            participants_text += f"• **{username}**: {reason}\n"
            mentions.append(f"<@{user_id}>")

        if len(participants_needing_update) > 10:
            participants_text += f"\n... et {len(participants_needing_update) - 10} autres participants"

        embed.add_field(
            name=f"👥 {len(participants_needing_update)} participant(s) à jour à faire",
            value=participants_text,
            inline=False
        )

        embed.add_field(
            name="📝 Instructions",
            value="Utilisez `!reel_cash <montant>` pour mettre à jour votre cash disponible\n"
                  "Exemple: `!reel_cash 5000`\n\n"
                  "⚠️ **Important**: Si vous ne mettez pas à jour dans les 24h:\n"
                  "• Rappel à nouveau demain à 23h\n"
                  "• Encore un rappel à 7h le lendemain\n"
                  "• Après ça, on continue avec votre ancien cash",
            inline=False
        )

        # Ping les participants
        mentions_text = " ".join(mentions) + "\n\n"

        await self.discord_channel.send(content=mentions_text, embed=embed)
        logger.info(f"[LiveTrader] Rappel cash envoyé à {len(participants_needing_update)} participants")

    async def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """
        Analyse une action en temps réel

        Returns:
            Dict avec les scores et la décision, ou None si erreur
        """
        try:
            # Vérifier si le marché est ouvert et si on peut trader
            can_trade, reason = MarketHours.can_trade_now(symbol)

            if not can_trade:
                logger.info(f"[LiveTrader] {symbol} ({StockInfo.get_full_name(symbol)}): {reason}")
                return None

            logger.info(f"\n[LiveTrader] 📊 Analyse de {symbol} ({StockInfo.get_full_name(symbol)})")

            # 1. Récupérer les données de prix récentes (14 jours pour avoir assez de points horaires)
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)

            df = stock.history(start=start_date, end=end_date, interval='1h')

            if df.empty or len(df) < 30:
                logger.warning(f"[LiveTrader] {symbol}: Données insuffisantes (seulement {len(df)} points)")
                return None

            # Normaliser l'index
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # 2. Analyse technique
            df = self.tech_analyzer.calculate_indicators(df)
            latest_row = df.iloc[-1]
            current_price = latest_row['Close']

            # Obtenir le signal technique
            tech_signal, tech_score, tech_details = self.tech_analyzer.get_technical_score(latest_row)
            logger.info(f"[LiveTrader] {symbol}: Tech Score={tech_score:.0f}/100 Signal={tech_signal}")

            # Si le signal technique n'est pas BUY ou SELL, ne pas continuer
            if tech_signal not in ['BUY', 'SELL']:
                logger.info(f"[LiveTrader] {symbol}: Signal HOLD → pas d'action")
                return {
                    'symbol': symbol,
                    'signal': 'HOLD',
                    'tech_score': tech_score,
                    'price': current_price
                }

            # 3 & 4. Récupérer news et Reddit en parallèle pour gagner du temps
            now = datetime.now()

            # Donner la main à la boucle d'événements avant les requêtes réseau
            await asyncio.sleep(0)

            # Lancer les deux requêtes en parallèle
            news_task = self.news_analyzer.get_news_for_date(symbol, now)
            reddit_task = self.reddit_analyzer.get_reddit_sentiment(symbol=symbol, target_date=now)

            # Attendre les résultats en parallèle
            (has_news, news_items, news_score), (reddit_score, reddit_count, reddit_samples, reddit_posts) = await asyncio.gather(
                news_task, reddit_task
            )

            logger.info(f"[LiveTrader] {symbol}: News Score={news_score:.0f}/100 ({len(news_items)} news)")
            logger.info(f"[LiveTrader] {symbol}: Reddit posts: {reddit_count}")

            # 5. Calculer le score composite (Tech: 50%, News: 50% - Reddit removed due to API blocking)
            composite_score = (tech_score * 0.50) + (news_score * 0.50)

            logger.info(f"[LiveTrader] {symbol}: Score Composite={composite_score:.0f}/100 (seuil: {self.validation_threshold})")

            # 6. Décision finale
            decision = {
                'symbol': symbol,
                'signal': tech_signal,
                'price': current_price,
                'tech_score': tech_score,
                'news_score': news_score,
                'composite_score': composite_score,
                'validated': composite_score >= self.validation_threshold,
                'news_count': len(news_items),
                'timestamp': now
            }

            return decision

        except Exception as e:
            logger.error(f"[LiveTrader] Erreur analyse {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def execute_trade(self, decision: Dict) -> bool:
        """
        Exécute un trade basé sur la décision

        Returns:
            True si le trade a été exécuté, False sinon
        """
        symbol = decision['symbol']
        signal = decision['signal']
        price = decision['price']
        validated = decision['validated']

        if not validated:
            logger.info(f"[LiveTrader] {symbol}: Trade rejeté (score trop bas)")
            self.rejected_trades += 1
            return False

        # Calculer la taille de la position
        current_prices = {symbol: price}
        portfolio_value = self.portfolio.get_total_value(current_prices)

        if signal == 'BUY':
            # Vérifier si on a déjà une position
            if self.portfolio.has_position(symbol):
                logger.info(f"[LiveTrader] {symbol}: Position déjà ouverte → pas d'achat")
                return False

            # Calculer le nombre d'actions à acheter (fractional shares supportées)
            # IMPORTANT: Utiliser le cash disponible, pas la valeur totale du portfolio
            max_investment = min(self.portfolio.cash, portfolio_value * self.max_position_size)

            # Calculer les shares en tant que float (permet les fractions)
            shares = max_investment / price

            if shares < 0.001:  # Minimum 0.001 actions
                logger.warning(f"[LiveTrader] {symbol}: Solde insuffisant pour acheter (cash: ${self.portfolio.cash:.2f}, prix: ${price:.2f}, max_inv: ${max_investment:.2f})")
                return False

            # Acheter
            success = self.portfolio.buy(symbol, price, shares, decision['timestamp'])

            if success:
                self.validated_trades += 1
                logger.info(f"[LiveTrader] ✅ ACHAT VALIDÉ: {shares} {symbol} @ ${price:.2f}")

                # Notification Discord avec ping des participants
                stock_name = StockInfo.get_full_name(symbol)
                embed = discord.Embed(
                    title=f"🟢 SIGNAL ACHAT: {stock_name}",
                    description=f"✅ **Signal validé par l'IA** | Ticker: {symbol}\n\n"
                               f"⚠️ **Exécutez ce trade MANUELLEMENT sur votre plateforme**",
                    color=0x00ff00,
                    timestamp=decision['timestamp']
                )
                embed.add_field(name="📌 Action", value=f"**ACHETER {stock_name}**", inline=False)
                embed.add_field(name="💰 Prix actuel", value=f"${price:.2f}", inline=True)
                embed.add_field(name="📊 Quantité suggérée (bot)", value=f"{shares:.4f}", inline=True)
                embed.add_field(name="💵 Coût total (bot)", value=f"${price*shares:.2f}", inline=True)
                embed.add_field(name="🔍 Score Technique", value=f"{decision['tech_score']:.0f}/100 (50%)", inline=True)
                embed.add_field(name="📰 Score News", value=f"{decision['news_score']:.0f}/100 (50%)", inline=True)
                embed.add_field(name="⭐ Score Final", value=f"**{decision['composite_score']:.0f}/100**", inline=True)
                embed.add_field(
                    name="📝 Instructions",
                    value=f"1️⃣ Ouvrez votre plateforme de trading\n"
                          f"2️⃣ Cherchez **{stock_name}** (ticker: {symbol})\n"
                          f"3️⃣ Achetez selon votre cash disponible\n"
                          f"4️⃣ Le bot garde trace de la position",
                    inline=False
                )

                # Ping uniquement les participants avec du cash > 0
                participants_ping = ""
                participants_without_cash = []
                if self.participants_manager:
                    participants_with_cash = self.participants_manager.get_participants_with_cash()

                    if participants_with_cash:
                        mentions = [f"<@{user_id}>" for user_id in participants_with_cash]
                        participants_ping = " ".join(mentions) + "\n\n"

                        # Marquer qu'ils ont maintenant cette position
                        for user_id in participants_with_cash:
                            self.participants_manager.add_position_to_participant(user_id, symbol)

                    # Identifier les participants sans cash
                    for user_id, participant in self.participants_manager.participants.items():
                        if participant['cash'] <= 0:
                            participants_without_cash.append(participant['username'])

                # Ajouter une note si certains participants n'ont pas de cash
                if participants_without_cash:
                    embed.add_field(
                        name="⚠️ Participants non pingés",
                        value=f"Cash = 0: {', '.join(participants_without_cash[:5])}" +
                              (f" et {len(participants_without_cash)-5} autres" if len(participants_without_cash) > 5 else ""),
                        inline=False
                    )

                await self.discord_channel.send(content=participants_ping, embed=embed)

                return True

        elif signal == 'SELL':
            # Vérifier si on a une position à vendre
            if not self.portfolio.has_position(symbol):
                logger.info(f"[LiveTrader] {symbol}: Pas de position à vendre")
                return False

            # Vendre toute la position
            position = self.portfolio.get_position(symbol)
            success = self.portfolio.sell(symbol, price, timestamp=decision['timestamp'])

            if success:
                self.validated_trades += 1
                logger.info(f"[LiveTrader] ✅ VENTE VALIDÉE: {symbol} @ ${price:.2f}")

                # Calculer le profit pour la notification
                last_trade = self.portfolio.trades_history[-1]

                # Notification Discord avec ping des participants
                stock_name = StockInfo.get_full_name(symbol)
                profit_emoji = "📈" if last_trade['profit'] > 0 else "📉"
                embed = discord.Embed(
                    title=f"🔴 SIGNAL VENTE: {stock_name}",
                    description=f"✅ **Signal validé par l'IA** | Ticker: {symbol}\n\n"
                               f"⚠️ **Exécutez ce trade MANUELLEMENT sur votre plateforme**",
                    color=0xff0000 if last_trade['profit'] < 0 else 0x00ff00,
                    timestamp=decision['timestamp']
                )
                embed.add_field(name="📌 Action", value=f"**VENDRE {stock_name}**", inline=False)
                embed.add_field(name="💰 Prix actuel", value=f"${price:.2f}", inline=True)
                embed.add_field(name="📊 Quantité (bot)", value=f"{last_trade['shares']:.4f}", inline=True)
                embed.add_field(name="💵 Valeur totale", value=f"${last_trade['proceeds']:.2f}", inline=True)
                embed.add_field(name=f"{profit_emoji} Profit (bot)", value=f"**${last_trade['profit']:.2f}** ({last_trade['profit_pct']:+.2f}%)", inline=False)
                embed.add_field(name="🔍 Score Technique", value=f"{decision['tech_score']:.0f}/100 (50%)", inline=True)
                embed.add_field(name="📰 Score News", value=f"{decision['news_score']:.0f}/100 (50%)", inline=True)
                embed.add_field(name="⭐ Score Final", value=f"**{decision['composite_score']:.0f}/100**", inline=True)
                embed.add_field(
                    name="📝 Instructions",
                    value=f"1️⃣ Ouvrez votre plateforme de trading\n"
                          f"2️⃣ Cherchez **{stock_name}** (ticker: {symbol})\n"
                          f"3️⃣ Vendez votre position complète\n"
                          f"4️⃣ Le bot ferme également sa position",
                    inline=False
                )

                # Ping uniquement les participants qui ont cette position
                participants_ping = ""
                participants_without_position = []
                if self.participants_manager:
                    participants_with_position = self.participants_manager.get_participants_with_position(symbol)

                    if participants_with_position:
                        mentions = [f"<@{user_id}>" for user_id in participants_with_position]
                        participants_ping = " ".join(mentions) + "\n\n"

                        # Retirer la position de ces participants
                        for user_id in participants_with_position:
                            self.participants_manager.remove_position_from_participant(user_id, symbol)

                    # Identifier les participants sans cette position
                    for user_id, participant in self.participants_manager.participants.items():
                        if symbol not in participant.get('positions', {}):
                            participants_without_position.append(participant['username'])

                # Ajouter une note si certains participants n'ont pas la position
                if participants_without_position:
                    embed.add_field(
                        name="ℹ️ Participants non pingés",
                        value=f"Pas de position sur {stock_name}: {', '.join(participants_without_position[:5])}" +
                              (f" et {len(participants_without_position)-5} autres" if len(participants_without_position) > 5 else ""),
                        inline=False
                    )

                await self.discord_channel.send(content=participants_ping, embed=embed)

                return True

        return False

    async def check_stop_loss_take_profit(self):
        """
        Vérifie les stop loss et take profit pour toutes les positions ouvertes
        """
        if not self.portfolio.positions:
            return

        logger.info(f"\n[LiveTrader] 🔍 Vérification SL/TP pour {len(self.portfolio.positions)} positions")

        for symbol in list(self.portfolio.positions.keys()):
            try:
                position = self.portfolio.positions[symbol]
                avg_price = position['avg_price']
                shares = position['shares']

                # Récupérer le prix actuel
                stock = yf.Ticker(symbol)
                current_price = stock.history(period='1d', interval='1m')['Close'].iloc[-1]

                # Calculer le profit/perte actuel
                profit_pct = ((current_price - avg_price) / avg_price) * 100

                logger.info(f"[LiveTrader] {symbol}: ${current_price:.2f} (Entrée: ${avg_price:.2f}, P/L: {profit_pct:+.2f}%)")

                # Vérifier stop loss
                if profit_pct <= self.stop_loss_pct:
                    logger.warning(f"[LiveTrader] {symbol}: ⚠️ STOP LOSS atteint ({profit_pct:+.2f}% <= {self.stop_loss_pct}%)")
                    self.portfolio.sell(symbol, current_price, timestamp=datetime.now())

                    # Notification Discord
                    stock_name = StockInfo.get_full_name(symbol)
                    embed = discord.Embed(
                        title=f"⛔ STOP LOSS: {stock_name}",
                        description=f"Position fermée automatiquement | Ticker: {symbol}",
                        color=0xff0000,
                        timestamp=datetime.now()
                    )
                    embed.add_field(name="Prix d'entrée", value=f"${avg_price:.2f}", inline=True)
                    embed.add_field(name="Prix de sortie", value=f"${current_price:.2f}", inline=True)
                    embed.add_field(name="Perte", value=f"**{profit_pct:+.2f}%**", inline=True)

                    await self.send_discord_notification(embed)

                # Vérifier take profit
                elif profit_pct >= self.take_profit_pct:
                    logger.info(f"[LiveTrader] {symbol}: ✅ TAKE PROFIT atteint ({profit_pct:+.2f}% >= {self.take_profit_pct}%)")
                    self.portfolio.sell(symbol, current_price, timestamp=datetime.now())

                    # Notification Discord
                    stock_name = StockInfo.get_full_name(symbol)
                    embed = discord.Embed(
                        title=f"💰 TAKE PROFIT: {stock_name}",
                        description=f"Position fermée automatiquement | Ticker: {symbol}",
                        color=0x00ff00,
                        timestamp=datetime.now()
                    )
                    embed.add_field(name="Prix d'entrée", value=f"${avg_price:.2f}", inline=True)
                    embed.add_field(name="Prix de sortie", value=f"${current_price:.2f}", inline=True)
                    embed.add_field(name="Profit", value=f"**{profit_pct:+.2f}%**", inline=True)

                    await self.send_discord_notification(embed)

            except Exception as e:
                logger.error(f"[LiveTrader] Erreur vérification SL/TP pour {symbol}: {e}")

    async def hourly_analysis(self):
        """Analyse complète de la watchlist (appelée chaque heure)"""
        logger.info("\n" + "="*80)
        logger.info(f"[LiveTrader] 🚀 ANALYSE HORAIRE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("="*80)

        self.analysis_count += 1

        # 1. Vérifier les stop loss / take profit
        await self.check_stop_loss_take_profit()

        # Donner la main à la boucle d'événements pour éviter le blocage du heartbeat
        await asyncio.sleep(0)

        # 2. Analyser chaque action de la watchlist
        decisions = []
        for symbol in self.watchlist:
            decision = await self.analyze_stock(symbol)
            if decision:
                decisions.append(decision)

            # Pause augmentée pour éviter le rate limiting Reddit (403)
            # ET pour laisser le heartbeat Discord respirer
            await asyncio.sleep(0.5)

        # 3. Exécuter les trades validés
        logger.info(f"\n[LiveTrader] 📝 Décisions prises: {len(decisions)}")

        # Donner la main à la boucle d'événements
        await asyncio.sleep(0)

        for decision in decisions:
            if decision['signal'] in ['BUY', 'SELL']:
                if decision['signal'] == 'BUY':
                    self.buy_signals += 1
                else:
                    self.sell_signals += 1

                await self.execute_trade(decision)
                await asyncio.sleep(0.5)

        # 4. Afficher le résumé
        current_prices = {}
        for symbol in self.portfolio.positions.keys():
            try:
                stock = yf.Ticker(symbol)
                current_prices[symbol] = stock.history(period='1d', interval='1m')['Close'].iloc[-1]
            except:
                pass

        performance = self.portfolio.get_performance(current_prices)

        logger.info(f"\n[LiveTrader] 📊 RÉSUMÉ DE SESSION:")
        logger.info(f"   • Analyses effectuées: {self.analysis_count}")
        logger.info(f"   • Signaux BUY: {self.buy_signals}")
        logger.info(f"   • Signaux SELL: {self.sell_signals}")
        logger.info(f"   • Trades validés: {self.validated_trades}")
        logger.info(f"   • Trades rejetés: {self.rejected_trades}")
        logger.info(f"   • Valeur portfolio: ${performance['total_value']:.2f}")
        logger.info(f"   • Performance: {performance['total_return_pct']:+.2f}%")
        logger.info(f"   • Positions ouvertes: {len(self.portfolio.positions)}")

        logger.info("\n" + "="*80 + "\n")

    async def start(self, duration_days: int = None):
        """
        Démarre le trading en temps réel

        Args:
            duration_days: Durée du dry-run en jours (None = en continu jusqu'à !stop)
        """
        self.is_running = True

        # Forcer la recréation de la session Reddit avec les nouveaux headers
        await self.reddit_analyzer.reset_session()

        mode_text = f"{duration_days} jours" if duration_days else "en continu (jusqu'à !stop)"

        logger.info(f"\n🚀 [LiveTrader] DÉMARRAGE DU BOT EN TEMPS RÉEL")
        logger.info(f"   • Capital bot: ${self.portfolio.initial_cash:.2f}")
        logger.info(f"   • Positions sauvegardées: {len(self.portfolio.positions)}")
        logger.info(f"   • Mode: {mode_text}")
        logger.info(f"   • Watchlist: {len(self.watchlist)} actions")
        logger.info(f"   • Heartbeat Discord: Toutes les 15 min (sauf :30)")
        logger.info(f"   • Analyses marché: À :30 de chaque heure")
        logger.info(f"   • Horaires: US 15:30-21:45 / FR 09:00-17:15")
        logger.info(f"   • Seuil validation: {self.validation_threshold}/100")
        logger.info(f"   • Stop Loss: {self.stop_loss_pct}%")
        logger.info(f"   • Take Profit: {self.take_profit_pct}%")

        # Notification Discord de démarrage
        embed = discord.Embed(
            title="🚀 BOT DÉMARRÉ",
            description=f"Trading en temps réel {mode_text}\n\n"
                       f"✅ Positions restaurées: {len(self.portfolio.positions)}",
            color=0x00ff00,
            timestamp=datetime.now()
        )
        embed.add_field(name="Capital (bot)", value=f"${self.portfolio.initial_cash:.2f}", inline=True)
        embed.add_field(name="Actions", value=f"{len(self.watchlist)}", inline=True)
        embed.add_field(name="Seuil", value=f"{self.validation_threshold}/100", inline=True)
        embed.add_field(name="Heartbeat", value="Toutes les 15min", inline=True)
        embed.add_field(name="Analyses", value="À :30 de chaque heure", inline=True)
        embed.add_field(name="Mode", value=mode_text, inline=True)

        await self.send_discord_notification(embed)

        end_date = datetime.now() + timedelta(days=duration_days) if duration_days else None

        try:
            while self.is_running and (end_date is None or datetime.now() < end_date):
                now = datetime.now()
                current_minute = now.minute
                current_hour = now.hour

                # Rappel cash quotidien à 23:00 et 7:00
                if (current_hour == 23 or current_hour == 7) and current_minute == 0:
                    logger.info(f"\n💰 {now.strftime('%H:%M')} - Rappel cash")
                    await self.send_daily_cash_reminder()
                    await asyncio.sleep(60)

                # Analyses de marché : uniquement à :30
                elif current_minute == 30:
                    logger.info(f"\n⏰ {now.strftime('%H:%M')} - Déclenchement de l'analyse de marché")
                    await self.hourly_analysis()
                    # Attendre 60 secondes pour ne pas retriggerer à :30
                    await asyncio.sleep(60)

                # Heartbeat Discord : à :00, :15, :45 (PAS à :30)
                elif current_minute in [0, 15, 45] and current_hour != 23:  # Pas à 23h (déjà utilisé)
                    logger.info(f"\n💓 {now.strftime('%H:%M')} - Heartbeat Discord")
                    await self.send_heartbeat()
                    # Attendre 60 secondes pour ne pas retriggerer
                    await asyncio.sleep(60)

                else:
                    # Vérifier toutes les 30 secondes si on est à la bonne minute
                    await asyncio.sleep(30)

        except asyncio.CancelledError:
            logger.info("[LiveTrader] ⏹️ Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"[LiveTrader] ❌ Erreur fatale: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()

    async def stop(self):
        """Arrête le bot et affiche les statistiques finales"""
        self.is_running = False

        logger.info("\n" + "="*80)
        logger.info(f"[LiveTrader] ⏹️ ARRÊT DU BOT")
        logger.info("="*80)

        # Calculer les statistiques finales
        current_prices = {}
        for symbol in self.portfolio.positions.keys():
            try:
                stock = yf.Ticker(symbol)
                current_prices[symbol] = stock.history(period='1d', interval='1m')['Close'].iloc[-1]
            except:
                pass

        performance = self.portfolio.get_performance(current_prices)

        logger.info(f"\n📊 STATISTIQUES FINALES:")
        logger.info(f"   • Durée: {performance['days_running']} jours")
        logger.info(f"   • Capital initial: ${performance['initial_cash']:.2f}")
        logger.info(f"   • Capital final: ${performance['total_value']:.2f}")
        logger.info(f"   • Performance: {performance['total_return_pct']:+.2f}%")
        logger.info(f"   • Trades totaux: {performance['total_trades']}")
        logger.info(f"   • Win Rate: {performance['win_rate']:.1f}%")
        logger.info(f"   • Profit moyen: ${performance['avg_profit']:.2f}")
        logger.info(f"   • Perte moyenne: ${performance['avg_loss']:.2f}")

        # Notification Discord finale
        embed = discord.Embed(
            title="⏹️ BOT ARRÊTÉ",
            description=f"Statistiques finales du dry-run",
            color=0x00ff00 if performance['total_return'] > 0 else 0xff0000,
            timestamp=datetime.now()
        )
        embed.add_field(name="Durée", value=f"{performance['days_running']} jours", inline=True)
        embed.add_field(name="Performance", value=f"**{performance['total_return_pct']:+.2f}%**", inline=True)
        embed.add_field(name="Capital Final", value=f"${performance['total_value']:.2f}", inline=True)
        embed.add_field(name="Trades", value=f"{performance['total_trades']}", inline=True)
        embed.add_field(name="Win Rate", value=f"{performance['win_rate']:.1f}%", inline=True)
        embed.add_field(name="Profit/Perte", value=f"${performance['total_return']:.2f}", inline=True)

        await self.send_discord_notification(embed)

        logger.info("="*80 + "\n")

        # Fermer les sessions
        await self.news_analyzer.ai_scorer.close()
        if self.reddit_analyzer.session:
            await self.reddit_analyzer.session.close()
