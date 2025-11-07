"""
Moteur de backtest réaliste v2.0 avec AI Scoring Multi-Niveau
"""

import asyncio
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from time import sleep
import time

# Import des analyzers DEPUIS le module analyzers/
from analyzers.technical_analyzer import TechnicalAnalyzer
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from analyzers.reddit_analyzer import RedditSentimentAnalyzer

logger = logging.getLogger('TradingBot')


class RealisticBacktestEngine:
    """
    Backtest réaliste v2.0 avec:
    - Stop Loss / Take Profit automatiques
    - AI Scoring multi-niveau (Reddit + News séparés)
    - 5 derniers prix pour contexte
    - Prix cibles (buy_price / sell_price)
    - Réduction auto si données manquantes
    """

    def __init__(self, reddit_csv_file: str = None, data_dir: str = 'data'):
        # Utiliser les classes depuis analyzers/ (version modulaire)
        self.news_analyzer = HistoricalNewsAnalyzer()
        self.reddit_analyzer = RedditSentimentAnalyzer(
            csv_file=reddit_csv_file,
            data_dir=data_dir
        )
        self.tech_analyzer = TechnicalAnalyzer()

        # Configuration stop loss / take profit
        self.stop_loss_pct = -3.0  # -3%
        self.take_profit_pct = 10.0  # +10%

    async def backtest_with_news_validation(self, symbol: str, months: int = 6) -> Optional[Dict]:
        """
        Backtest sur X mois avec validation IA multi-niveau
        """
        start_time = time.time()
        logger.info(f"[>>] Backtest réaliste {symbol} - {months} mois")

        try:
            # Récupérer les données historiques
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months*30 + 60)  # +60j pour indicateurs

            df = stock.history(start=start_date, end=end_date, interval='1d')

            if df.empty or len(df) < 100:
                logger.warning(f"   [X] Données insuffisantes pour {symbol}")
                return None

            # Normaliser l'index en timezone-naive
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            logger.info(f"   [✓] {len(df)} jours de données")

            # Calculer les indicateurs
            df = self.tech_analyzer.calculate_indicators(df)

            # Analyser CHAQUE JOUR après le warm-up (60 jours pour les indicateurs)
            warm_up_days = 60
            decision_points = list(range(warm_up_days, len(df)))

            if len(decision_points) < 10:
                logger.warning(f"   [X] Pas assez de points de décision")
                return None

            logger.info(f"   [✓] {len(decision_points)} jours d'analyse (backtest quotidien)")

            # Simuler les trades
            trades = []
            position = 0  # 0 = pas de position, 1 = long
            entry_price = 0
            entry_date = None
            entry_idx = 0

            validated_buys = 0
            validated_sells = 0
            rejected_buys = 0
            rejected_sells = 0

            for idx in decision_points:
                row = df.iloc[idx]
                current_date = df.index[idx]
                current_price = row['Close']

                # PRIORITÉ 1: Vérifier stop loss / take profit
                if position == 1:
                    profit_pct = (current_price - entry_price) / entry_price * 100

                    # Stop loss prioritaire (-3%)
                    if profit_pct <= self.stop_loss_pct:
                        position = 0
                        exit_price = current_price
                        hold_days = (current_date - entry_date).days

                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit_pct,
                            'hold_days': hold_days,
                            'final_score': 0,
                            'tech_confidence': 0,
                            'reddit_score': 0,
                            'news_count': 0,
                            'reddit_posts': 0,
                            'exit_reason': 'STOP_LOSS'
                        })

                        logger.info(f"   🛑 STOP LOSS @ ${exit_price:.2f} | "
                                  f"Perte: {profit_pct:.2f}% | Durée: {hold_days}j")
                        continue

                    # Take profit prioritaire (+10%)
                    elif profit_pct >= self.take_profit_pct:
                        position = 0
                        exit_price = current_price
                        hold_days = (current_date - entry_date).days

                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit_pct,
                            'hold_days': hold_days,
                            'final_score': 100,
                            'tech_confidence': 100,
                            'reddit_score': 0,
                            'news_count': 0,
                            'reddit_posts': 0,
                            'exit_reason': 'TAKE_PROFIT'
                        })

                        logger.info(f"   🎯 TAKE PROFIT @ ${exit_price:.2f} | "
                                  f"Gain: {profit_pct:.2f}% | Durée: {hold_days}j")
                        continue

                # Analyse technique
                tech_decision, tech_confidence, tech_reasons = self.tech_analyzer.get_technical_score(row)
                bot_decision = tech_decision

                # Récupérer les actualités
                has_news, news_data, news_score = await self.news_analyzer.get_news_for_date(
                    symbol, current_date
                )

                sleep(0.3)

                # Récupérer le sentiment Reddit
                reddit_score = 0  # Défaut = 0 si pas de posts
                reddit_post_count = 0
                reddit_samples = []
                reddit_posts_details = []

                if bot_decision in ["BUY", "SELL"]:
                    reddit_score, reddit_post_count, reddit_samples, reddit_posts_details = await self.reddit_analyzer.get_reddit_sentiment(
                        symbol, current_date, lookback_hours=48
                    )
                    await asyncio.sleep(0.3)

                # Validation IA
                final_score = tech_confidence
                ai_reason = "Pas de validation IA"

                if bot_decision in ["BUY", "SELL"]:
                    # Préparer les 5 derniers prix
                    last_5_prices = []
                    if idx >= 5:
                        last_5_prices = df['Close'].iloc[idx-4:idx+1].tolist()

                    # Prix cibles
                    buy_price = None
                    sell_price = None
                    if bot_decision == "BUY":
                        buy_price = current_price
                    elif bot_decision == "SELL" and position == 1:
                        sell_price = current_price

                    # Appeler HF avec TOUS les nouveaux paramètres
                    final_score, ai_reason, reddit_score_ai, news_score_ai = await self.news_analyzer.ask_ai_decision(
                        symbol, bot_decision, news_data, current_price, tech_confidence,
                        reddit_posts=reddit_posts_details,
                        target_date=current_date,
                        last_5_prices=last_5_prices,
                        buy_price=buy_price,
                        sell_price=sell_price
                    )
                else:
                    reddit_score_ai = 0.0
                    news_score_ai = 0.0

                # Logs - Afficher les scores AI si disponibles, sinon 0
                if bot_decision != "HOLD" or idx % 20 == 0:
                    logger.info(f"   [{current_date.strftime('%Y-%m-%d')}] Decision: {bot_decision} | "
                              f"Tech: {tech_confidence:.0f}/100 | "
                              f"Reddit: {reddit_score_ai:.0f}/100 ({reddit_post_count}p) | "
                              f"News: {news_score_ai:.0f}/100 | "
                              f"FINAL: {final_score:.0f}/100")

                # Exécuter le trade si score > 65
                if bot_decision == "BUY" and position == 0:
                    if final_score > 65:
                        position = 1
                        entry_price = current_price
                        entry_date = current_date
                        entry_idx = idx
                        validated_buys += 1
                        logger.info(f"   ✅ BUY validé (Score: {final_score:.0f}) @ ${current_price:.2f}")
                    else:
                        rejected_buys += 1
                        logger.info(f"   ❌ BUY rejeté (Score: {final_score:.0f})")

                elif bot_decision == "SELL" and position == 1:
                    if final_score > 65:
                        position = 0
                        exit_price = current_price
                        profit = (exit_price - entry_price) / entry_price * 100
                        hold_days = (current_date - entry_date).days

                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit,
                            'hold_days': hold_days,
                            'final_score': final_score,
                            'tech_confidence': tech_confidence,
                            'reddit_score': reddit_score,
                            'news_count': len(news_data),
                            'reddit_posts': reddit_post_count,
                            'exit_reason': 'AI_VALIDATED_SELL'
                        })

                        validated_sells += 1
                        logger.info(f"   ✅ SELL validé (Score: {final_score:.0f}) @ ${exit_price:.2f} | "
                                  f"Profit: {profit:+.2f}% | Durée: {hold_days}j")
                    else:
                        rejected_sells += 1
                        logger.info(f"   ❌ SELL rejeté (Score: {final_score:.0f})")

                # Délai API
                if bot_decision in ["BUY", "SELL"] and has_news:
                    await asyncio.sleep(0.2)

            # Clôturer position si encore ouverte
            if position == 1:
                final_price = df['Close'].iloc[-1]
                profit = (final_price - entry_price) / entry_price * 100
                hold_days = (df.index[-1] - entry_date).days

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[-1],
                    'entry_price': entry_price,
                    'exit_price': final_price,
                    'profit': profit,
                    'hold_days': hold_days,
                    'final_score': 0,
                    'tech_confidence': 0,
                    'reddit_score': 0,
                    'news_count': 0,
                    'reddit_posts': 0,
                    'exit_reason': 'END_OF_BACKTEST'
                })

                logger.info(f"   💼 Position clôturée: {profit:+.2f}%")

            # Calculer les métriques
            if not trades:
                logger.warning(f"   [X] Aucun trade exécuté")
                return None

            profits = [t['profit'] for t in trades]
            total_profit = sum(profits)
            avg_profit = np.mean(profits)
            win_rate = len([p for p in profits if p > 0]) / len(profits) * 100

            duration = time.time() - start_time

            logger.info(f"\n{'='*80}")
            logger.info(f"📊 RÉSULTATS BACKTEST {symbol}")
            logger.info(f"{'='*80}")
            logger.info(f"   Trades: {len(trades)}")
            logger.info(f"   BUY validés: {validated_buys} | rejetés: {rejected_buys}")
            logger.info(f"   SELL validés: {validated_sells} | rejetés: {rejected_sells}")
            logger.info(f"   Profit total: {total_profit:+.2f}%")
            logger.info(f"   Profit moyen: {avg_profit:+.2f}%")
            logger.info(f"   Win rate: {win_rate:.1f}%")
            logger.info(f"   Durée: {duration:.1f}s")
            logger.info(f"{'='*80}\n")

            return {
                'symbol': symbol,
                'trades': trades,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'win_rate': win_rate,
                'validated_buys': validated_buys,
                'rejected_buys': rejected_buys,
                'validated_sells': validated_sells,
                'rejected_sells': rejected_sells
            }

        except Exception as e:
            logger.error(f"❌ Erreur backtest {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def close(self):
        """Fermer les sessions"""
        await self.news_analyzer.close()
        await self.reddit_analyzer.close()


__all__ = ['RealisticBacktestEngine']
