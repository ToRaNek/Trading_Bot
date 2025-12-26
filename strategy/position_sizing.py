"""
Position Sizing
Base sur MASTER_TRADING_SKILL PARTIE XI - Section 88

Calcul de la taille optimale des positions selon:
- Risk per trade (2% max)
- Kelly Criterion (optionnel)
- Volatility-based sizing
"""
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from config.settings import (
    RISK_PER_TRADE, MAX_POSITION_PERCENT, INITIAL_CAPITAL
)

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calcule la taille optimale des positions

    Methodes:
    1. Fixed Fractional (% du capital)
    2. Fixed Risk (% de risque par trade)
    3. Volatility-based (ajuste selon ATR)
    4. Kelly Criterion (optimal mathematique)
    """

    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.capital = capital

    def update_capital(self, new_capital: float):
        """Met a jour le capital"""
        self.capital = new_capital

    # =========================================================================
    # METHODE PRINCIPALE: FIXED RISK
    # =========================================================================

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = None,
        capital: float = None
    ) -> Dict:
        """
        Calcule la taille de position basee sur le risque fixe

        Formule: shares = (capital * risk%) / risk_per_share

        MASTER_TRADING_SKILL:
        "position_size = (capital * risk_percent) / (entry - stop_loss)"

        Args:
            entry_price: Prix d'entree prevu
            stop_loss: Niveau de stop loss
            risk_percent: % du capital a risquer (defaut: RISK_PER_TRADE)
            capital: Capital a utiliser (defaut: self.capital)

        Returns:
            Dict avec shares, position_value, risk_amount, etc.
        """
        if risk_percent is None:
            risk_percent = RISK_PER_TRADE
        if capital is None:
            capital = self.capital

        # Calculer le risque par action
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            logger.warning("Invalid stop loss: must be different from entry")
            return self._empty_result()

        # Montant a risquer
        risk_amount = capital * risk_percent

        # Nombre d'actions
        shares = risk_amount / risk_per_share

        # Valeur de la position
        position_value = shares * entry_price

        # Verifier max position
        max_position_value = capital * MAX_POSITION_PERCENT
        if position_value > max_position_value:
            shares = max_position_value / entry_price
            position_value = max_position_value
            risk_amount = shares * risk_per_share

        # Arrondir vers le bas
        shares = int(shares)
        if shares <= 0:
            return self._empty_result()

        position_value = shares * entry_price
        actual_risk = shares * risk_per_share
        actual_risk_percent = actual_risk / capital

        return {
            'shares': shares,
            'position_value': position_value,
            'risk_amount': actual_risk,
            'risk_percent': actual_risk_percent,
            'risk_per_share': risk_per_share,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'capital_used_percent': position_value / capital
        }

    # =========================================================================
    # VOLATILITY-BASED SIZING
    # =========================================================================

    def calculate_volatility_adjusted_size(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        risk_percent: float = None,
        capital: float = None
    ) -> Dict:
        """
        Taille de position ajustee selon la volatilite (ATR)

        Stop loss = entry - (ATR * multiplier)
        Plus volatile = position plus petite

        Args:
            entry_price: Prix d'entree
            atr: Average True Range
            atr_multiplier: Multiplicateur ATR pour SL
            risk_percent: % de risque
            capital: Capital

        Returns:
            Dict avec sizing info
        """
        if risk_percent is None:
            risk_percent = RISK_PER_TRADE
        if capital is None:
            capital = self.capital

        # Stop loss base sur ATR
        stop_distance = atr * atr_multiplier
        stop_loss = entry_price - stop_distance

        return self.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_percent=risk_percent,
            capital=capital
        )

    # =========================================================================
    # KELLY CRITERION
    # =========================================================================

    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.5
    ) -> float:
        """
        Calcule la taille optimale selon Kelly Criterion

        Kelly % = W - [(1-W) / R]
        W = win rate
        R = ratio gain moyen / perte moyenne

        Note: On utilise generalement une fraction de Kelly (ex: 0.5)
        pour reduire la variance.

        Args:
            win_rate: Taux de reussite (0-1)
            avg_win: Gain moyen
            avg_loss: Perte moyenne (valeur absolue)
            kelly_fraction: Fraction de Kelly a utiliser (0.5 = Half Kelly)

        Returns:
            % du capital a risquer
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return RISK_PER_TRADE  # Retourner valeur par defaut

        # Ratio win/loss
        r = avg_win / avg_loss

        # Kelly formula
        kelly = win_rate - ((1 - win_rate) / r)

        # Limiter Kelly
        kelly = max(0, min(kelly, 0.25))  # Max 25%

        # Appliquer fraction
        adjusted_kelly = kelly * kelly_fraction

        # Ne jamais depasser le risk per trade max
        return min(adjusted_kelly, RISK_PER_TRADE * 2)

    # =========================================================================
    # ANTI-MARTINGALE
    # =========================================================================

    def calculate_anti_martingale_size(
        self,
        base_risk: float,
        consecutive_wins: int,
        max_multiplier: float = 2.0
    ) -> float:
        """
        Anti-Martingale: augmente la taille apres les gains

        Contrairement a Martingale (augmenter apres pertes),
        on augmente progressivement apres les gains.

        Args:
            base_risk: Risque de base (%)
            consecutive_wins: Nombre de gains consecutifs
            max_multiplier: Multiplicateur maximum

        Returns:
            % de risque ajuste
        """
        if consecutive_wins <= 0:
            return base_risk

        # Augmenter de 25% par gain consecutif, jusqu'au max
        multiplier = min(1 + (consecutive_wins * 0.25), max_multiplier)

        return base_risk * multiplier

    # =========================================================================
    # PYRAMIDING
    # =========================================================================

    def calculate_pyramid_size(
        self,
        initial_shares: int,
        pyramid_level: int,
        reduction_factor: float = 0.5
    ) -> int:
        """
        Calcule la taille pour pyramider (ajouter a une position gagnante)

        Chaque niveau de pyramide = taille precedente * reduction_factor

        Args:
            initial_shares: Taille initiale
            pyramid_level: Niveau de pyramide (1, 2, 3...)
            reduction_factor: Facteur de reduction (0.5 = moitie)

        Returns:
            Nombre d'actions pour ce niveau
        """
        if pyramid_level <= 0:
            return initial_shares

        shares = initial_shares * (reduction_factor ** pyramid_level)
        return max(1, int(shares))

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _empty_result(self) -> Dict:
        """Retourne un resultat vide"""
        return {
            'shares': 0,
            'position_value': 0,
            'risk_amount': 0,
            'risk_percent': 0,
            'risk_per_share': 0,
            'entry_price': 0,
            'stop_loss': 0,
            'capital_used_percent': 0
        }

    def validate_position(
        self,
        shares: int,
        entry_price: float,
        capital: float = None
    ) -> Tuple[bool, str]:
        """
        Valide qu'une position respecte les limites

        Returns:
            (is_valid, reason)
        """
        if capital is None:
            capital = self.capital

        if shares <= 0:
            return False, "Nombre d'actions invalide"

        position_value = shares * entry_price
        position_percent = position_value / capital

        if position_percent > MAX_POSITION_PERCENT:
            return False, f"Position trop grande ({position_percent:.1%} > {MAX_POSITION_PERCENT:.1%})"

        if position_value > capital:
            return False, "Position depasse le capital disponible"

        return True, "OK"

    def get_max_shares(self, price: float, capital: float = None) -> int:
        """
        Retourne le nombre maximum d'actions pour un prix donne
        """
        if capital is None:
            capital = self.capital

        max_value = capital * MAX_POSITION_PERCENT
        return int(max_value / price)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def calculate_risk_reward(
    entry: float,
    stop_loss: float,
    take_profit: float,
    side: str = 'long'
) -> float:
    """
    Calcule le ratio Risk/Reward

    R:R = Reward / Risk
    """
    if side == 'long':
        risk = entry - stop_loss
        reward = take_profit - entry
    else:  # short
        risk = stop_loss - entry
        reward = entry - take_profit

    if risk <= 0:
        return 0

    return reward / risk


def calculate_breakeven_price(
    entry: float,
    shares: int,
    commission: float = 0
) -> float:
    """
    Calcule le prix de break-even (incluant commissions)
    """
    total_cost = (entry * shares) + (commission * 2)  # Aller-retour
    return total_cost / shares


def calculate_partial_exit_size(
    total_shares: int,
    exit_percent: float = 0.25
) -> Tuple[int, int]:
    """
    Calcule la taille pour une sortie partielle

    Returns:
        (shares_to_exit, shares_remaining)
    """
    shares_to_exit = int(total_shares * exit_percent)
    shares_remaining = total_shares - shares_to_exit

    return shares_to_exit, shares_remaining


# =============================================================================
# SINGLETON
# =============================================================================
_position_sizer = None

def get_position_sizer() -> PositionSizer:
    """Retourne l'instance singleton"""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = PositionSizer()
    return _position_sizer
