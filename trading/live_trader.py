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

        # Initialiser les analyseurs
        self.tech_analyzer = TechnicalAnalyzer()
        self.news_analyzer = HistoricalNewsAnalyzer()
        self.reddit_analyzer = RedditSentimentAnalyzer()

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

    async def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """
        Analyse une action en temps réel

        Returns:
            Dict avec les scores et la décision, ou None si erreur
        """
        try:
            logger.info(f"\n[LiveTrader] 📊 Analyse de {symbol}")

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

            # 3. Récupérer les news des dernières 48h
            now = datetime.now()
            has_news, news_items, news_score = await self.news_analyzer.get_news_for_date(symbol, now)
            logger.info(f"[LiveTrader] {symbol}: News Score={news_score:.0f}/100 ({len(news_items)} news)")

            # 4. Récupérer le sentiment Reddit du jour
            reddit_score, reddit_count, reddit_samples, reddit_posts = await self.reddit_analyzer.get_reddit_sentiment(
                symbol=symbol,
                target_date=now
            )
            logger.info(f"[LiveTrader] {symbol}: Reddit Score={reddit_score:.0f}/100 ({reddit_count} posts)")

            # 5. Calculer le score composite (même pondération que le backtest)
            # Tech: 40%, News: 35%, Reddit: 25%
            composite_score = (tech_score * 0.40) + (news_score * 0.35) + (reddit_score * 0.25)

            logger.info(f"[LiveTrader] {symbol}: Score Composite={composite_score:.0f}/100 (seuil: {self.validation_threshold})")

            # 6. Décision finale
            decision = {
                'symbol': symbol,
                'signal': tech_signal,
                'price': current_price,
                'tech_score': tech_score,
                'news_score': news_score,
                'reddit_score': reddit_score,
                'composite_score': composite_score,
                'validated': composite_score >= self.validation_threshold,
                'news_count': len(news_items),
                'reddit_count': reddit_count,
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

            # Calculer le nombre d'actions à acheter
            max_investment = portfolio_value * self.max_position_size
            shares = int(max_investment / price)

            if shares < 1:
                logger.warning(f"[LiveTrader] {symbol}: Solde insuffisant pour acheter")
                return False

            # Acheter
            success = self.portfolio.buy(symbol, price, shares, decision['timestamp'])

            if success:
                self.validated_trades += 1
                logger.info(f"[LiveTrader] ✅ ACHAT VALIDÉ: {shares} {symbol} @ ${price:.2f}")

                # Notification Discord
                embed = discord.Embed(
                    title=f"🟢 ACHAT: {symbol}",
                    description=f"Trade validé par l'IA",
                    color=0x00ff00,
                    timestamp=decision['timestamp']
                )
                embed.add_field(name="Prix", value=f"${price:.2f}", inline=True)
                embed.add_field(name="Quantité", value=f"{shares}", inline=True)
                embed.add_field(name="Coût", value=f"${price*shares:.2f}", inline=True)
                embed.add_field(name="Score Tech", value=f"{decision['tech_score']:.0f}/100", inline=True)
                embed.add_field(name="Score News", value=f"{decision['news_score']:.0f}/100", inline=True)
                embed.add_field(name="Score Reddit", value=f"{decision['reddit_score']:.0f}/100", inline=True)
                embed.add_field(name="Score Final", value=f"**{decision['composite_score']:.0f}/100**", inline=False)

                await self.send_discord_notification(embed)

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

                # Notification Discord
                embed = discord.Embed(
                    title=f"🔴 VENTE: {symbol}",
                    description=f"Trade validé par l'IA",
                    color=0xff0000 if last_trade['profit'] < 0 else 0x00ff00,
                    timestamp=decision['timestamp']
                )
                embed.add_field(name="Prix", value=f"${price:.2f}", inline=True)
                embed.add_field(name="Quantité", value=f"{last_trade['shares']}", inline=True)
                embed.add_field(name="Gain", value=f"${last_trade['proceeds']:.2f}", inline=True)
                embed.add_field(name="Profit", value=f"${last_trade['profit']:.2f} ({last_trade['profit_pct']:+.2f}%)", inline=False)
                embed.add_field(name="Score Tech", value=f"{decision['tech_score']:.0f}/100", inline=True)
                embed.add_field(name="Score News", value=f"{decision['news_score']:.0f}/100", inline=True)
                embed.add_field(name="Score Reddit", value=f"{decision['reddit_score']:.0f}/100", inline=True)
                embed.add_field(name="Score Final", value=f"**{decision['composite_score']:.0f}/100**", inline=False)

                await self.send_discord_notification(embed)

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
                    embed = discord.Embed(
                        title=f"⛔ STOP LOSS: {symbol}",
                        description=f"Position fermée automatiquement",
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
                    embed = discord.Embed(
                        title=f"💰 TAKE PROFIT: {symbol}",
                        description=f"Position fermée automatiquement",
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

        # 2. Analyser chaque action de la watchlist
        decisions = []
        for symbol in self.watchlist:
            decision = await self.analyze_stock(symbol)
            if decision:
                decisions.append(decision)

            # Pause entre chaque analyse pour ne pas surcharger les APIs
            await asyncio.sleep(2)

        # 3. Exécuter les trades validés
        logger.info(f"\n[LiveTrader] 📝 Décisions prises: {len(decisions)}")

        for decision in decisions:
            if decision['signal'] in ['BUY', 'SELL']:
                if decision['signal'] == 'BUY':
                    self.buy_signals += 1
                else:
                    self.sell_signals += 1

                await self.execute_trade(decision)
                await asyncio.sleep(1)

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

    async def start(self, duration_days: int = 90):
        """
        Démarre le trading en temps réel

        Args:
            duration_days: Durée du dry-run en jours (par défaut: 90 jours = 3 mois)
        """
        self.is_running = True
        logger.info(f"\n🚀 [LiveTrader] DÉMARRAGE DU BOT EN DRY-RUN")
        logger.info(f"   • Capital initial: ${self.portfolio.initial_cash:.2f}")
        logger.info(f"   • Durée: {duration_days} jours")
        logger.info(f"   • Watchlist: {len(self.watchlist)} actions")
        logger.info(f"   • Analyses: Toutes les heures")
        logger.info(f"   • Seuil validation: {self.validation_threshold}/100")
        logger.info(f"   • Stop Loss: {self.stop_loss_pct}%")
        logger.info(f"   • Take Profit: {self.take_profit_pct}%")

        # Notification Discord de démarrage
        embed = discord.Embed(
            title="🚀 BOT DÉMARRÉ",
            description=f"Trading en dry-run pendant {duration_days} jours",
            color=0x00ff00,
            timestamp=datetime.now()
        )
        embed.add_field(name="Capital", value=f"${self.portfolio.initial_cash:.2f}", inline=True)
        embed.add_field(name="Actions", value=f"{len(self.watchlist)}", inline=True)
        embed.add_field(name="Seuil", value=f"{self.validation_threshold}/100", inline=True)

        await self.send_discord_notification(embed)

        end_date = datetime.now() + timedelta(days=duration_days)

        try:
            while self.is_running and datetime.now() < end_date:
                # Effectuer l'analyse horaire
                await self.hourly_analysis()

                # Attendre 1 heure
                logger.info(f"[LiveTrader] ⏸️ Pause d'1 heure... (prochain cycle: {(datetime.now() + timedelta(hours=1)).strftime('%H:%M')})")
                await asyncio.sleep(3600)  # 3600 secondes = 1 heure

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
