"""
Informations complètes sur les actions (noms complets, marchés, etc.)
"""

from typing import Dict, Optional


class StockInfo:
    """Informations sur les actions"""

    # Dictionnaire des actions avec leurs noms complets
    STOCK_NAMES = {
        # Actions US
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc. (Google)',
        'AMZN': 'Amazon.com Inc.',
        'NVDA': 'NVIDIA Corporation',
        'META': 'Meta Platforms Inc. (Facebook)',
        'TSLA': 'Tesla Inc.',
        'BRK-B': 'Berkshire Hathaway Inc.',
        'BRK.B': 'Berkshire Hathaway Inc.',
        'JPM': 'JPMorgan Chase & Co.',
        'V': 'Visa Inc.',
        'JNJ': 'Johnson & Johnson',
        'WMT': 'Walmart Inc.',
        'PG': 'Procter & Gamble Co.',
        'MA': 'Mastercard Inc.',
        'DIS': 'The Walt Disney Company',
        'NFLX': 'Netflix Inc.',
        'ADBE': 'Adobe Inc.',
        'CRM': 'Salesforce Inc.',
        'AMD': 'Advanced Micro Devices Inc.',
        'ORCL': 'Oracle Corporation',
        'INTC': 'Intel Corporation',
        'CSCO': 'Cisco Systems Inc.',
        'PEP': 'PepsiCo Inc.',
        'COST': 'Costco Wholesale Corporation',
        'AVGO': 'Broadcom Inc.',

        # Actions Françaises (CAC 40)
        'MC.PA': 'LVMH Moët Hennessy Louis Vuitton',
        'OR.PA': "L'Oréal S.A.",
        'AIR.PA': 'Airbus SE',
        'SAN.PA': 'Sanofi S.A.',
        'TTE.PA': 'TotalEnergies SE',
        'BNP.PA': 'BNP Paribas S.A.',
        'SAF.PA': 'Safran S.A.',
        'ACA.PA': 'Crédit Agricole S.A.',
        'SU.PA': 'Schneider Electric SE',
        'DG.PA': 'Vinci S.A.',
        'BN.PA': 'Danone S.A.',
        'RMS.PA': 'Hermès International S.A.',
        'DSY.PA': 'Dassault Systèmes SE',
        'CA.PA': 'Carrefour S.A.',
        'EN.PA': 'Bouygues S.A.',
    }

    # Secteurs d'activité
    STOCK_SECTORS = {
        # US
        'AAPL': 'Technology - Consumer Electronics',
        'MSFT': 'Technology - Software',
        'GOOGL': 'Technology - Internet Services',
        'AMZN': 'Consumer Cyclical - E-commerce',
        'NVDA': 'Technology - Semiconductors',
        'META': 'Technology - Social Media',
        'TSLA': 'Automotive - Electric Vehicles',
        'BRK-B': 'Financial - Conglomerate',
        'BRK.B': 'Financial - Conglomerate',
        'JPM': 'Financial - Banking',
        'V': 'Financial - Payment Processing',
        'JNJ': 'Healthcare - Pharmaceuticals',
        'WMT': 'Consumer Defensive - Retail',
        'PG': 'Consumer Defensive - Personal Products',
        'MA': 'Financial - Payment Processing',
        'DIS': 'Communication - Entertainment',
        'NFLX': 'Communication - Streaming',
        'ADBE': 'Technology - Software',
        'CRM': 'Technology - Cloud/CRM',
        'AMD': 'Technology - Semiconductors',
        'ORCL': 'Technology - Software/Cloud',
        'INTC': 'Technology - Semiconductors',
        'CSCO': 'Technology - Networking',
        'PEP': 'Consumer Defensive - Beverages',
        'COST': 'Consumer Defensive - Retail',
        'AVGO': 'Technology - Semiconductors',

        # France
        'MC.PA': 'Luxe - Mode & Accessoires',
        'OR.PA': 'Beauté - Cosmétiques',
        'AIR.PA': 'Industrie - Aéronautique',
        'SAN.PA': 'Santé - Pharmacie',
        'TTE.PA': 'Énergie - Pétrole & Gaz',
        'BNP.PA': 'Finance - Banque',
        'SAF.PA': 'Industrie - Aéronautique & Défense',
        'ACA.PA': 'Finance - Banque',
        'SU.PA': 'Industrie - Équipements Électriques',
        'DG.PA': 'Industrie - Construction & Concessions',
        'BN.PA': 'Alimentation - Agroalimentaire',
        'RMS.PA': 'Luxe - Mode & Accessoires',
        'DSY.PA': 'Technologie - Logiciels',
        'CA.PA': 'Commerce - Distribution',
        'EN.PA': 'Télécoms & Construction',
    }

    @classmethod
    def get_full_name(cls, symbol: str) -> str:
        """
        Retourne le nom complet de l'action

        Args:
            symbol: Ticker de l'action

        Returns:
            Nom complet (ou le ticker si non trouvé)
        """
        return cls.STOCK_NAMES.get(symbol, symbol)

    @classmethod
    def get_sector(cls, symbol: str) -> str:
        """
        Retourne le secteur de l'action

        Args:
            symbol: Ticker de l'action

        Returns:
            Secteur (ou "Non défini" si non trouvé)
        """
        return cls.STOCK_SECTORS.get(symbol, "Non défini")

    @classmethod
    def get_display_name(cls, symbol: str, include_sector: bool = False) -> str:
        """
        Retourne un nom formaté pour l'affichage

        Args:
            symbol: Ticker de l'action
            include_sector: Inclure le secteur dans le nom

        Returns:
            Nom formaté pour affichage
        """
        full_name = cls.get_full_name(symbol)

        if include_sector:
            sector = cls.get_sector(symbol)
            return f"{full_name} ({symbol}) - {sector}"
        else:
            return f"{full_name} ({symbol})"

    @classmethod
    def get_market(cls, symbol: str) -> str:
        """
        Retourne le marché de l'action

        Args:
            symbol: Ticker de l'action

        Returns:
            "US" ou "France"
        """
        return "France" if symbol.endswith('.PA') else "US"

    @classmethod
    def format_for_search(cls, symbol: str) -> str:
        """
        Retourne le nom formaté pour la recherche manuelle

        Args:
            symbol: Ticker de l'action

        Returns:
            Nom complet sans ticker (pour chercher manuellement)
        """
        return cls.get_full_name(symbol)

    @classmethod
    def get_stock_info(cls, symbol: str) -> Dict:
        """
        Retourne toutes les informations sur une action

        Args:
            symbol: Ticker de l'action

        Returns:
            Dict avec toutes les infos
        """
        return {
            'symbol': symbol,
            'full_name': cls.get_full_name(symbol),
            'sector': cls.get_sector(symbol),
            'market': cls.get_market(symbol),
            'display_name': cls.get_display_name(symbol),
            'search_name': cls.format_for_search(symbol)
        }
