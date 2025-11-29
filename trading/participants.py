"""
Gestion des participants et de leurs montants pour le trading manuel
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger('TradingBot')


class ParticipantsManager:
    """Gère les participants et leurs montants disponibles"""

    def __init__(self, save_file: str = 'participants.json'):
        """
        Initialise le gestionnaire de participants

        Args:
            save_file: Fichier de sauvegarde des participants
        """
        self.save_file = save_file
        self.participants = {}  # {user_id: {'username': str, 'cash': float, 'positions': {}, 'last_cash_update': timestamp}}
        self.current_cash_pool = 0.0  # Cash disponible pour la prochaine analyse
        self.load_state()

    def add_participant(self, user_id: int, username: str, initial_cash: float = 0.0):
        """
        Ajoute un participant

        Args:
            user_id: ID Discord de l'utilisateur
            username: Nom d'utilisateur Discord
            initial_cash: Cash initial (optionnel)
        """
        if user_id not in self.participants:
            from datetime import datetime
            self.participants[user_id] = {
                'username': username,
                'cash': initial_cash,
                'positions': {},  # {symbol: {'shares': float, 'avg_price': float}}
                'total_invested': 0.0,
                'total_profit': 0.0,
                'last_cash_update': None,  # Aucune mise à jour encore
                'needs_cash_update': True  # Doit définir son cash
            }
            logger.info(f"[Participants] ✅ Participant ajouté: {username} (cash: ${initial_cash})")
            self.save_state()
            return True
        else:
            logger.warning(f"[Participants] ⚠️ Participant déjà existant: {username}")
            return False

    def update_cash(self, user_id: int, amount: float):
        """
        Met à jour le cash d'un participant

        Args:
            user_id: ID Discord de l'utilisateur
            amount: Nouveau montant de cash
        """
        if user_id in self.participants:
            from datetime import datetime
            old_cash = self.participants[user_id]['cash']
            self.participants[user_id]['cash'] = amount
            self.participants[user_id]['last_cash_update'] = datetime.now().isoformat()
            self.participants[user_id]['needs_cash_update'] = False
            logger.info(f"[Participants] 💰 Cash mis à jour pour {self.participants[user_id]['username']}: ${old_cash:.2f} -> ${amount:.2f}")
            self.save_state()
            return True
        else:
            logger.warning(f"[Participants] ❌ Participant non trouvé: {user_id}")
            return False

    def set_cash_pool(self, amount: float):
        """
        Définit le cash disponible pour la prochaine analyse

        Args:
            amount: Montant total disponible
        """
        self.current_cash_pool = amount
        logger.info(f"[Participants] 💰 Cash pool défini: ${amount:.2f}")
        self.save_state()

    def get_participant_allocations(self, symbol: str, suggested_amount: float) -> List[Dict]:
        """
        Calcule les allocations pour chaque participant basé sur leur cash disponible

        Args:
            symbol: Ticker de l'action
            suggested_amount: Montant suggéré pour cette action

        Returns:
            Liste des allocations par participant
        """
        allocations = []
        total_cash = sum(p['cash'] for p in self.participants.values())

        if total_cash == 0:
            logger.warning("[Participants] ⚠️ Pas de cash disponible")
            return []

        for user_id, participant in self.participants.items():
            if participant['cash'] > 0:
                # Calculer la part proportionnelle
                proportion = participant['cash'] / total_cash
                allocated_amount = suggested_amount * proportion

                allocations.append({
                    'user_id': user_id,
                    'username': participant['username'],
                    'cash_available': participant['cash'],
                    'proportion': proportion,
                    'allocated_amount': allocated_amount,
                    'symbol': symbol
                })

        return allocations

    def record_manual_buy(self, user_id: int, symbol: str, shares: float, price: float):
        """
        Enregistre un achat manuel effectué par un participant

        Args:
            user_id: ID Discord de l'utilisateur
            symbol: Ticker de l'action
            shares: Nombre d'actions
            price: Prix d'achat
        """
        if user_id not in self.participants:
            logger.warning(f"[Participants] ❌ Participant non trouvé: {user_id}")
            return False

        participant = self.participants[user_id]
        cost = shares * price

        if cost > participant['cash']:
            logger.warning(f"[Participants] ❌ Solde insuffisant pour {participant['username']}")
            return False

        # Débiter le cash
        participant['cash'] -= cost

        # Ajouter/mettre à jour la position
        if symbol in participant['positions']:
            old_shares = participant['positions'][symbol]['shares']
            old_avg_price = participant['positions'][symbol]['avg_price']
            new_shares = old_shares + shares
            new_avg_price = ((old_shares * old_avg_price) + (shares * price)) / new_shares

            participant['positions'][symbol] = {
                'shares': new_shares,
                'avg_price': new_avg_price
            }
        else:
            participant['positions'][symbol] = {
                'shares': shares,
                'avg_price': price
            }

        participant['total_invested'] += cost

        logger.info(f"[Participants] ✅ Achat enregistré pour {participant['username']}: {shares} {symbol} @ ${price:.2f}")
        self.save_state()
        return True

    def record_manual_sell(self, user_id: int, symbol: str, shares: float, price: float):
        """
        Enregistre une vente manuelle effectuée par un participant

        Args:
            user_id: ID Discord de l'utilisateur
            symbol: Ticker de l'action
            shares: Nombre d'actions
            price: Prix de vente
        """
        if user_id not in self.participants:
            logger.warning(f"[Participants] ❌ Participant non trouvé: {user_id}")
            return False

        participant = self.participants[user_id]

        if symbol not in participant['positions']:
            logger.warning(f"[Participants] ❌ Pas de position sur {symbol} pour {participant['username']}")
            return False

        position = participant['positions'][symbol]

        if shares > position['shares']:
            logger.warning(f"[Participants] ❌ Pas assez d'actions {symbol} pour {participant['username']}")
            return False

        # Calculer le profit
        proceeds = price * shares
        cost_basis = position['avg_price'] * shares
        profit = proceeds - cost_basis

        # Créditer le cash
        participant['cash'] += proceeds

        # Mettre à jour la position
        position['shares'] -= shares
        if position['shares'] <= 0:
            del participant['positions'][symbol]

        participant['total_profit'] += profit

        logger.info(f"[Participants] ✅ Vente enregistrée pour {participant['username']}: {shares} {symbol} @ ${price:.2f} (Profit: ${profit:.2f})")
        self.save_state()
        return True

    def get_participant_summary(self, user_id: int, current_prices: Dict[str, float] = None) -> Optional[Dict]:
        """
        Retourne un résumé pour un participant

        Args:
            user_id: ID Discord de l'utilisateur
            current_prices: Prix actuels des actions

        Returns:
            Dict avec les stats du participant
        """
        if user_id not in self.participants:
            return None

        participant = self.participants[user_id]
        current_prices = current_prices or {}

        # Calculer la valeur des positions
        positions_value = 0.0
        for symbol, position in participant['positions'].items():
            price = current_prices.get(symbol, position['avg_price'])
            positions_value += position['shares'] * price

        total_value = participant['cash'] + positions_value

        return {
            'username': participant['username'],
            'cash': participant['cash'],
            'positions_value': positions_value,
            'total_value': total_value,
            'total_invested': participant['total_invested'],
            'total_profit': participant['total_profit'],
            'positions': participant['positions']
        }

    def get_all_participants_summary(self, current_prices: Dict[str, float] = None) -> List[Dict]:
        """
        Retourne un résumé pour tous les participants

        Args:
            current_prices: Prix actuels des actions

        Returns:
            Liste des résumés de tous les participants
        """
        summaries = []
        for user_id in self.participants.keys():
            summary = self.get_participant_summary(user_id, current_prices)
            if summary:
                summary['user_id'] = user_id
                summaries.append(summary)

        return summaries

    def save_state(self):
        """Sauvegarde l'état des participants dans un fichier JSON"""
        state = {
            'participants': self.participants,
            'current_cash_pool': self.current_cash_pool
        }

        try:
            with open(self.save_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"[Participants] ✅ État sauvegardé dans {self.save_file}")
        except Exception as e:
            logger.error(f"[Participants] ❌ Erreur lors de la sauvegarde: {e}")

    def load_state(self):
        """Charge l'état des participants depuis un fichier JSON"""
        if not Path(self.save_file).exists():
            logger.info(f"[Participants] Nouveau fichier de participants créé")
            return

        try:
            with open(self.save_file, 'r') as f:
                state = json.load(f)

            # Convertir les clés user_id de string à int
            self.participants = {int(k): v for k, v in state.get('participants', {}).items()}
            self.current_cash_pool = state.get('current_cash_pool', 0.0)

            logger.info(f"[Participants] ✅ État chargé: {len(self.participants)} participants")
        except Exception as e:
            logger.error(f"[Participants] ❌ Erreur lors du chargement: {e}")

    def reset(self):
        """Réinitialise tous les participants"""
        self.participants = {}
        self.current_cash_pool = 0.0
        self.save_state()
        logger.info("[Participants] 🔄 Participants réinitialisés")

    def has_position(self, user_id: int, symbol: str) -> bool:
        """
        Vérifie si un participant a une position sur une action

        Args:
            user_id: ID Discord de l'utilisateur
            symbol: Ticker de l'action

        Returns:
            True si le participant a la position, False sinon
        """
        if user_id not in self.participants:
            return False

        return symbol in self.participants[user_id]['positions']

    def add_position_to_participant(self, user_id: int, symbol: str):
        """
        Marque qu'un participant a acheté une action

        Args:
            user_id: ID Discord de l'utilisateur
            symbol: Ticker de l'action
        """
        if user_id in self.participants:
            # On ne track que si le participant a la position ou non, pas les montants
            # (car c'est manuel)
            if symbol not in self.participants[user_id]['positions']:
                self.participants[user_id]['positions'][symbol] = True
                logger.info(f"[Participants] Position ajoutée: {self.participants[user_id]['username']} -> {symbol}")
                self.save_state()

    def remove_position_from_participant(self, user_id: int, symbol: str):
        """
        Marque qu'un participant a vendu une action

        Args:
            user_id: ID Discord de l'utilisateur
            symbol: Ticker de l'action
        """
        if user_id in self.participants:
            if symbol in self.participants[user_id]['positions']:
                del self.participants[user_id]['positions'][symbol]
                logger.info(f"[Participants] Position retirée: {self.participants[user_id]['username']} -> {symbol}")
                self.save_state()

    def get_participants_with_position(self, symbol: str) -> list:
        """
        Retourne la liste des participants qui ont une position sur cette action

        Args:
            symbol: Ticker de l'action

        Returns:
            Liste des user_id qui ont la position
        """
        participants_with_position = []
        for user_id, participant in self.participants.items():
            if symbol in participant['positions']:
                participants_with_position.append(user_id)

        return participants_with_position

    def get_participants_with_cash(self) -> list:
        """
        Retourne la liste des participants qui ont du cash disponible (> 0)

        Returns:
            Liste des user_id qui ont du cash
        """
        participants_with_cash = []
        for user_id, participant in self.participants.items():
            if participant['cash'] > 0:
                participants_with_cash.append(user_id)

        return participants_with_cash

    def get_participants_needing_cash_update(self) -> list:
        """
        Retourne la liste des participants qui doivent mettre à jour leur cash

        Returns:
            Liste des (user_id, username, reason)
        """
        from datetime import datetime, timedelta

        participants_needing_update = []
        now = datetime.now()

        for user_id, participant in self.participants.items():
            # Cas 1: Jamais mis à jour le cash
            if participant.get('needs_cash_update', True):
                participants_needing_update.append((
                    user_id,
                    participant['username'],
                    "Aucune mise à jour depuis l'inscription"
                ))
                continue

            # Cas 2: Dernière mise à jour il y a plus de 24h
            last_update = participant.get('last_cash_update')
            if last_update:
                last_update_dt = datetime.fromisoformat(last_update)
                if now - last_update_dt > timedelta(hours=24):
                    hours_ago = (now - last_update_dt).total_seconds() / 3600
                    participants_needing_update.append((
                        user_id,
                        participant['username'],
                        f"Dernière mise à jour il y a {hours_ago:.0f}h"
                    ))

        return participants_needing_update

    def mark_participant_for_cash_update(self, user_id: int):
        """
        Marque qu'un participant doit mettre à jour son cash

        Args:
            user_id: ID Discord de l'utilisateur
        """
        if user_id in self.participants:
            self.participants[user_id]['needs_cash_update'] = True
            self.save_state()
