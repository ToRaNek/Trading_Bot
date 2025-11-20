"""Configuration du bot de trading"""

# Watchlist des actions à analyser
WATCHLIST = [
    # Actions US
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'DIS', 'NFLX', 'ADBE',
    'CRM', 'AMD', 'ORCL', 'INTC', 'CSCO', 'PEP', 'COST', 'AVGO',

    # Actions Françaises (CAC 40)
    'MC.PA',      # LVMH - Luxe
    'OR.PA',      # L'Oréal - Cosmétiques
    'AIR.PA',     # Airbus - Aéronautique
    'SAN.PA',     # Sanofi - Pharmacie
    'TTE.PA',     # TotalEnergies - Énergie
    'BNP.PA',     # BNP Paribas - Banque
    'SAF.PA',     # Safran - Aéronautique/Défense
    'ACA.PA',     # Crédit Agricole - Banque
    'SU.PA',      # Schneider Electric - Industrie
    'DG.PA',      # Vinci - Construction/Concessions
    'BN.PA',      # Danone - Agroalimentaire
    'RMS.PA',     # Hermès - Luxe
    'DSY.PA',     # Dassault Systèmes - Software
    'CA.PA',      # Carrefour - Distribution
    'EN.PA'       # Bouygues - Télécoms/Construction
]

# Seuil de validation pour exécuter un trade
VALIDATION_THRESHOLD = 65  # Score minimum pour exécuter (0-100)

# Configuration logging
LOG_FILE = 'trading_bot.log'
LOG_LEVEL = 'INFO'

# Configuration Reddit
# Sur Azure/VPS: mettre à False car les IP de datacenters sont bloquées par Reddit
# Sur PC local: mettre à True pour utiliser l'API Reddit en temps réel
ENABLE_REDDIT_API = False  # Désactiver sur Azure VPS (IP bloquée)
