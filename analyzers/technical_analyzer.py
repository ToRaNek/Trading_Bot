"""Analyseur technique basé sur RSI, MACD, SMA, Bollinger Bands, Volume"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class TechnicalAnalyzer:
    """Analyse technique avec décision BUY/SELL/HOLD et score de confiance"""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcul des indicateurs techniques"""
        data_size = len(df)

        # SMA
        df['sma_20'] = df['Close'].rolling(window=min(20, data_size//2)).mean()
        if data_size > 50:
            df['sma_50'] = df['Close'].rolling(window=50).mean()
        if data_size > 200:
            df['sma_200'] = df['Close'].rolling(window=200).mean()

        # EMA et MACD
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_window = min(20, data_size//2)
        bb_sma = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)

        # Volume
        df['volume_sma'] = df['Volume'].rolling(window=min(20, data_size//2)).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma'].replace(0, 1)

        return df

    def get_technical_score(self, row: pd.Series) -> Tuple[str, float, List[str]]:
        """
        Analyse technique qui détermine BUY/SELL/HOLD avec un score de confiance (0-100)

        Returns:
            Tuple[decision, confidence_score, reasons]
            - decision: "BUY", "SELL", ou "HOLD"
            - confidence_score: 0-100 (confiance dans la décision)
            - reasons: Liste des raisons
        """
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        reasons = []
        signal_strengths = []  # Pour calculer la confiance

        # 1. RSI (Relative Strength Index)
        # Règle: RSI < 30 = oversold (BUY), RSI > 70 = overbought (SELL)
        if pd.notna(row.get('rsi')):
            rsi = row['rsi']
            if rsi < 25:
                buy_signals += 1
                signal_strengths.append(90)  # Signal très fort
                reasons.append(f"🔥🔥 RSI TRÈS survendu ({rsi:.1f}) → STRONG BUY")
            elif rsi < 30:
                buy_signals += 1
                signal_strengths.append(75)
                reasons.append(f"🔥 RSI survendu ({rsi:.1f}) → BUY")
            elif rsi < 40:
                buy_signals += 1
                signal_strengths.append(55)
                reasons.append(f"✅ RSI bas ({rsi:.1f}) → BUY")
            elif rsi > 75:
                sell_signals += 1
                signal_strengths.append(90)
                reasons.append(f"❄️❄️ RSI TRÈS suracheté ({rsi:.1f}) → STRONG SELL")
            elif rsi > 70:
                sell_signals += 1
                signal_strengths.append(75)
                reasons.append(f"❄️ RSI suracheté ({rsi:.1f}) → SELL")
            elif rsi > 60:
                sell_signals += 1
                signal_strengths.append(55)
                reasons.append(f"⚠️ RSI élevé ({rsi:.1f}) → SELL")
            else:
                hold_signals += 1
                signal_strengths.append(30)
                reasons.append(f"➡️ RSI neutre ({rsi:.1f}) → HOLD")

        # 2. MACD (Moving Average Convergence Divergence)
        # Règle: MACD > Signal = BUY, MACD < Signal = SELL
        if pd.notna(row.get('macd')) and pd.notna(row.get('macd_signal')):
            macd_diff = row['macd'] - row['macd_signal']
            macd_pct = (macd_diff / abs(row['macd_signal'])) * 100 if row['macd_signal'] != 0 else 0

            if macd_diff > 0:
                # MACD au-dessus de la signal line = BUY
                if macd_pct > 5:
                    buy_signals += 1
                    signal_strengths.append(85)
                    reasons.append(f"🚀🚀 MACD très bullish (+{macd_pct:.1f}%) → STRONG BUY")
                elif macd_pct > 2:
                    buy_signals += 1
                    signal_strengths.append(70)
                    reasons.append(f"🚀 MACD bullish (+{macd_pct:.1f}%) → BUY")
                else:
                    buy_signals += 1
                    signal_strengths.append(50)
                    reasons.append(f"✅ MACD positif (+{macd_pct:.1f}%) → BUY")
            else:
                # MACD en-dessous de la signal line = SELL
                if macd_pct < -5:
                    sell_signals += 1
                    signal_strengths.append(85)
                    reasons.append(f"📉📉 MACD très bearish ({macd_pct:.1f}%) → STRONG SELL")
                elif macd_pct < -2:
                    sell_signals += 1
                    signal_strengths.append(70)
                    reasons.append(f"📉 MACD bearish ({macd_pct:.1f}%) → SELL")
                else:
                    sell_signals += 1
                    signal_strengths.append(50)
                    reasons.append(f"⚠️ MACD négatif ({macd_pct:.1f}%) → SELL")

        # 3. SMA (Simple Moving Average) - Tendance
        # Règle: Prix > SMA20 > SMA50 = uptrend (BUY), inverse = downtrend (SELL)
        if pd.notna(row.get('sma_20')) and pd.notna(row.get('sma_50')):
            price = row['Close']
            sma_20 = row['sma_20']
            sma_50 = row['sma_50']

            # Golden Cross / Death Cross
            sma_cross = (sma_20 - sma_50) / sma_50 * 100
            price_vs_sma20 = (price - sma_20) / sma_20 * 100

            # Uptrend: Prix > SMA20 > SMA50
            if price > sma_20 and sma_20 > sma_50:
                if sma_cross > 3:
                    buy_signals += 1
                    signal_strengths.append(85)
                    reasons.append(f"📈📈 Forte tendance haussière (Golden Cross +{sma_cross:.1f}%) → STRONG BUY")
                elif sma_cross > 1:
                    buy_signals += 1
                    signal_strengths.append(70)
                    reasons.append(f"📈 Tendance haussière (+{sma_cross:.1f}%) → BUY")
                else:
                    buy_signals += 1
                    signal_strengths.append(55)
                    reasons.append(f"✅ Uptrend confirmé → BUY")
            # Downtrend: Prix < SMA20 < SMA50
            elif price < sma_20 and sma_20 < sma_50:
                if sma_cross < -3:
                    sell_signals += 1
                    signal_strengths.append(85)
                    reasons.append(f"📉📉 Forte tendance baissière (Death Cross {sma_cross:.1f}%) → STRONG SELL")
                elif sma_cross < -1:
                    sell_signals += 1
                    signal_strengths.append(70)
                    reasons.append(f"📉 Tendance baissière ({sma_cross:.1f}%) → SELL")
                else:
                    sell_signals += 1
                    signal_strengths.append(55)
                    reasons.append(f"⚠️ Downtrend confirmé → SELL")
            # Pas de tendance claire
            else:
                hold_signals += 1
                signal_strengths.append(30)
                reasons.append(f"➡️ Tendance mixte → HOLD")

        # 4. Bollinger Bands
        # Règle: Prix près BB inférieure = BUY (oversold), près BB supérieure = SELL (overbought)
        if pd.notna(row.get('bb_lower')) and pd.notna(row.get('bb_upper')):
            bb_range = row['bb_upper'] - row['bb_lower']
            if bb_range > 0:
                price = row['Close']
                bb_position = (price - row['bb_lower']) / bb_range  # 0 = bas, 1 = haut

                if bb_position < 0.1:
                    buy_signals += 1
                    signal_strengths.append(85)
                    reasons.append(f"🎯🎯 Prix TRÈS proche BB inf ({bb_position*100:.0f}%) → STRONG BUY")
                elif bb_position < 0.25:
                    buy_signals += 1
                    signal_strengths.append(70)
                    reasons.append(f"🎯 Prix proche BB inf ({bb_position*100:.0f}%) → BUY")
                elif bb_position < 0.4:
                    buy_signals += 1
                    signal_strengths.append(50)
                    reasons.append(f"✅ Prix bas dans BB ({bb_position*100:.0f}%) → BUY")
                elif bb_position > 0.9:
                    sell_signals += 1
                    signal_strengths.append(85)
                    reasons.append(f"⚠️⚠️ Prix TRÈS proche BB sup ({bb_position*100:.0f}%) → STRONG SELL")
                elif bb_position > 0.75:
                    sell_signals += 1
                    signal_strengths.append(70)
                    reasons.append(f"⚠️ Prix proche BB sup ({bb_position*100:.0f}%) → SELL")
                elif bb_position > 0.6:
                    sell_signals += 1
                    signal_strengths.append(50)
                    reasons.append(f"🔸 Prix haut dans BB ({bb_position*100:.0f}%) → SELL")
                else:
                    hold_signals += 1
                    signal_strengths.append(30)
                    reasons.append(f"➡️ Prix milieu BB ({bb_position*100:.0f}%) → HOLD")

        # 5. Volume (bonus de confirmation, pas un signal en soi)
        volume_bonus = 0
        if pd.notna(row.get('volume_ratio')):
            vol_ratio = row['volume_ratio']
            if vol_ratio > 2.5:
                volume_bonus = 15
                reasons.append(f"📊📊 Volume TRÈS élevé ({vol_ratio:.1f}x) - Confirmation forte")
            elif vol_ratio > 1.5:
                volume_bonus = 10
                reasons.append(f"📊 Volume élevé ({vol_ratio:.1f}x) - Bonne confirmation")
            elif vol_ratio > 1.0:
                volume_bonus = 5
                reasons.append(f"✅ Volume normal ({vol_ratio:.1f}x)")
            else:
                volume_bonus = -5
                reasons.append(f"📉 Volume faible ({vol_ratio:.1f}x) - Signal moins fiable")

        # DÉCISION FINALE PAR VOTE MAJORITAIRE
        total_signals = buy_signals + sell_signals + hold_signals

        if total_signals == 0:
            # Aucun signal valide
            return "HOLD", 0, ["⚠️ Aucun indicateur valide"]

        # Déterminer la décision par majorité
        if buy_signals > sell_signals and buy_signals > hold_signals:
            decision = "BUY"
        elif sell_signals > buy_signals and sell_signals > hold_signals:
            decision = "SELL"
        else:
            decision = "HOLD"

        # CALCUL DU SCORE DE CONFIANCE (0-100)
        if decision == "BUY":
            # Calculer confiance basée sur les signaux BUY uniquement
            relevant_strengths = []
            signal_idx = 0
            if buy_signals > 0:
                # Prendre les signaux correspondants aux BUY
                for reason in reasons:
                    if "→ BUY" in reason or "→ STRONG BUY" in reason:
                        if signal_idx < len(signal_strengths):
                            relevant_strengths.append(signal_strengths[signal_idx])
                        signal_idx += 1
                    elif "→ SELL" in reason or "→ HOLD" in reason:
                        signal_idx += 1

            if relevant_strengths:
                base_confidence = np.mean(relevant_strengths)
            else:
                base_confidence = 40

            # Bonus de confluence
            confluence_bonus = min(20, buy_signals * 5)
            confidence = min(100, base_confidence + confluence_bonus + volume_bonus)

            reasons.insert(0, f"🟢 DÉCISION: BUY (Confiance: {confidence:.0f}/100)")
            reasons.insert(1, f"📊 Signaux: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")

        elif decision == "SELL":
            # Même logique pour SELL
            relevant_strengths = []
            signal_idx = 0
            if sell_signals > 0:
                for reason in reasons:
                    if "→ SELL" in reason or "→ STRONG SELL" in reason:
                        if signal_idx < len(signal_strengths):
                            relevant_strengths.append(signal_strengths[signal_idx])
                        signal_idx += 1
                    elif "→ BUY" in reason or "→ HOLD" in reason:
                        signal_idx += 1

            if relevant_strengths:
                base_confidence = np.mean(relevant_strengths)
            else:
                base_confidence = 40

            confluence_bonus = min(20, sell_signals * 5)
            confidence = min(100, base_confidence + confluence_bonus + volume_bonus)

            reasons.insert(0, f"🔴 DÉCISION: SELL (Confiance: {confidence:.0f}/100)")
            reasons.insert(1, f"📊 Signaux: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")

        else:  # HOLD
            confidence = 30 + volume_bonus  # Confiance faible pour HOLD
            confidence = max(0, min(100, confidence))

            reasons.insert(0, f"🟡 DÉCISION: HOLD (Confiance: {confidence:.0f}/100)")
            reasons.insert(1, f"📊 Signaux: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")

        confidence = max(0, min(100, confidence))
        return decision, confidence, reasons
