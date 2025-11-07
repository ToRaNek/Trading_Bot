"""Système de rotation des clés API pour éviter les limitations"""

import csv
import os
import logging
from typing import List, Optional

logger = logging.getLogger('TradingBot')


class APIKeyRotator:
    """Gère la rotation des clés API depuis un fichier CSV"""

    def __init__(self, csv_path: str = None):
        """
        Initialise le rotateur de clés API

        Args:
            csv_path: Chemin vers le fichier CSV contenant les clés
                     Format: une colonne 'newsapi_key' avec une clé par ligne
        """
        if csv_path is None:
            # Par défaut, chercher api_keys.csv dans le dossier Trading_Bot
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, 'api_keys.csv')

        self.csv_path = csv_path
        self.keys: List[str] = []
        self.current_index = 0
        self.failed_keys = set()  # Ensemble des clés qui ont échoué

        self._load_keys()

    def _load_keys(self):
        """Charge les clés depuis le fichier CSV"""
        try:
            if not os.path.exists(self.csv_path):
                logger.warning(f"[APIKeyRotator] Fichier CSV non trouvé: {self.csv_path}")
                logger.info(f"[APIKeyRotator] Utilisation de la clé d'environnement NEWSAPI_KEY comme fallback")
                env_key = os.getenv('NEWSAPI_KEY', '')
                if env_key:
                    self.keys = [env_key]
                return

            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row.get('newsapi_key', '').strip()
                    if key and key not in ['YOUR_NEWSAPI_KEY_1', 'YOUR_NEWSAPI_KEY_2', 'YOUR_NEWSAPI_KEY_3']:
                        self.keys.append(key)

            if not self.keys:
                logger.warning(f"[APIKeyRotator] Aucune clé valide trouvée dans {self.csv_path}")
                logger.info(f"[APIKeyRotator] Utilisation de la clé d'environnement NEWSAPI_KEY comme fallback")
                env_key = os.getenv('NEWSAPI_KEY', '')
                if env_key:
                    self.keys = [env_key]
            else:
                logger.info(f"[APIKeyRotator] {len(self.keys)} clés API chargées depuis {self.csv_path}")

        except Exception as e:
            logger.error(f"[APIKeyRotator] Erreur lors du chargement des clés: {e}")
            # Fallback sur la variable d'environnement
            env_key = os.getenv('NEWSAPI_KEY', '')
            if env_key:
                self.keys = [env_key]

    def get_current_key(self) -> Optional[str]:
        """
        Retourne la clé API courante

        Returns:
            La clé API courante ou None si aucune clé disponible
        """
        if not self.keys:
            return None

        # Si toutes les clés ont échoué, réinitialiser
        if len(self.failed_keys) >= len(self.keys):
            logger.warning("[APIKeyRotator] Toutes les clés ont échoué, réinitialisation")
            self.failed_keys.clear()

        # Trouver la prochaine clé non échouée
        attempts = 0
        while attempts < len(self.keys):
            current_key = self.keys[self.current_index]
            if current_key not in self.failed_keys:
                return current_key

            # Passer à la clé suivante
            self.current_index = (self.current_index + 1) % len(self.keys)
            attempts += 1

        # Aucune clé disponible
        return None

    def rotate(self):
        """Passe à la clé suivante dans la rotation"""
        if not self.keys:
            return

        old_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.keys)
        logger.info(f"[APIKeyRotator] Rotation: clé {old_index + 1} -> clé {self.current_index + 1}")

    def mark_current_as_failed(self):
        """Marque la clé courante comme ayant échoué (limite atteinte)"""
        if not self.keys:
            return

        current_key = self.keys[self.current_index]
        self.failed_keys.add(current_key)
        logger.warning(f"[APIKeyRotator] Clé {self.current_index + 1} marquée comme épuisée")

        # Passer automatiquement à la suivante
        self.rotate()

    def mark_current_as_success(self):
        """Marque la clé courante comme fonctionnelle (retire du set des échecs si présente)"""
        if not self.keys:
            return

        current_key = self.keys[self.current_index]
        if current_key in self.failed_keys:
            self.failed_keys.discard(current_key)
            logger.info(f"[APIKeyRotator] Clé {self.current_index + 1} fonctionne à nouveau")

    def get_stats(self) -> dict:
        """
        Retourne des statistiques sur l'état des clés

        Returns:
            Dict avec total_keys, active_keys, failed_keys, current_index
        """
        return {
            'total_keys': len(self.keys),
            'active_keys': len(self.keys) - len(self.failed_keys),
            'failed_keys': len(self.failed_keys),
            'current_index': self.current_index
        }
