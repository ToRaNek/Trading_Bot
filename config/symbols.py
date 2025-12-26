"""
Liste des symboles a trader
Actions US (compatibles Alpaca) et Europeennes (Yahoo Finance)
"""

# =============================================================================
# US STOCKS - S&P 500 TOP
# =============================================================================
US_STOCKS = {
    "tech": [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet
        "AMZN",   # Amazon
        "META",   # Meta
        "NVDA",   # NVIDIA
        "TSLA",   # Tesla
        "AMD",    # AMD
        "INTC",   # Intel
        "CRM",    # Salesforce
    ],
    "finance": [
        "JPM",    # JP Morgan
        "BAC",    # Bank of America
        "WFC",    # Wells Fargo
        "GS",     # Goldman Sachs
        "MS",     # Morgan Stanley
        "V",      # Visa
        "MA",     # Mastercard
    ],
    "healthcare": [
        "JNJ",    # Johnson & Johnson
        "UNH",    # UnitedHealth
        "PFE",    # Pfizer
        "MRK",    # Merck
        "ABBV",   # AbbVie
    ],
    "consumer": [
        "WMT",    # Walmart
        "PG",     # Procter & Gamble
        "KO",     # Coca-Cola
        "PEP",    # PepsiCo
        "COST",   # Costco
        "NKE",    # Nike
        "MCD",    # McDonald's
    ],
    "industrial": [
        "CAT",    # Caterpillar
        "BA",     # Boeing
        "HON",    # Honeywell
        "UPS",    # UPS
        "RTX",    # Raytheon
    ],
    "energy": [
        "XOM",    # Exxon Mobil
        "CVX",    # Chevron
        "COP",    # ConocoPhillips
    ],
}

# =============================================================================
# EUROPEAN STOCKS - CAC 40 (Yahoo Finance suffix .PA)
# =============================================================================
EU_STOCKS = {
    "cac40": [
        "AI.PA",      # Air Liquide
        "AIR.PA",     # Airbus
        "ALO.PA",     # Alstom
        "MT.PA",      # ArcelorMittal
        "CS.PA",      # AXA
        "BNP.PA",     # BNP Paribas
        "EN.PA",      # Bouygues
        "CAP.PA",     # Capgemini
        "CA.PA",      # Carrefour
        "SGO.PA",     # Saint-Gobain
        "SAN.PA",     # Sanofi
        "SU.PA",      # Schneider Electric
        "GLE.PA",     # Societe Generale
        "STLA.PA",    # Stellantis
        "STM.PA",     # STMicroelectronics
        "TEP.PA",     # Teleperformance
        "HO.PA",      # Thales
        "TTE.PA",     # TotalEnergies
        "URW.PA",     # Unibail
        "VIE.PA",     # Veolia
        "DG.PA",      # Vinci
        "VIV.PA",     # Vivendi
        "WLN.PA",     # Worldline
        "MC.PA",      # LVMH
        "OR.PA",      # L'Oreal
        "RI.PA",      # Pernod Ricard
        "KER.PA",     # Kering
        "HMC.PA",     # Hermes
    ],
}

# =============================================================================
# ETFs - Pour diversification
# =============================================================================
ETFS = {
    "us_index": [
        "SPY",    # S&P 500
        "QQQ",    # Nasdaq 100
        "DIA",    # Dow Jones
        "IWM",    # Russell 2000
    ],
    "sector": [
        "XLK",    # Tech
        "XLF",    # Finance
        "XLV",    # Healthcare
        "XLE",    # Energy
        "XLI",    # Industrials
    ],
    "international": [
        "EFA",    # EAFE (Europe, Australasia, Far East)
        "EEM",    # Emerging Markets
        "FXI",    # China
    ],
}

# =============================================================================
# INDICES (pour analyse, pas trading direct)
# =============================================================================
INDICES = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "Nasdaq",
    "^FCHI": "CAC 40",
    "^GDAXI": "DAX",
    "^VIX": "VIX (Volatility)",
}

# =============================================================================
# WATCHLIST PAR DEFAUT
# =============================================================================
def get_default_watchlist():
    """Retourne la watchlist par defaut (US Tech + Finance)"""
    return US_STOCKS["tech"][:5] + US_STOCKS["finance"][:3]


def get_all_us_stocks():
    """Retourne toutes les actions US"""
    all_stocks = []
    for sector_stocks in US_STOCKS.values():
        all_stocks.extend(sector_stocks)
    return all_stocks


def get_all_eu_stocks():
    """Retourne toutes les actions EU"""
    all_stocks = []
    for market_stocks in EU_STOCKS.values():
        all_stocks.extend(market_stocks)
    return all_stocks


def get_sector(symbol):
    """Retourne le secteur d'un symbole"""
    for sector, stocks in US_STOCKS.items():
        if symbol in stocks:
            return sector
    for market, stocks in EU_STOCKS.items():
        if symbol in stocks:
            return market
    return "unknown"


def get_all_symbols():
    """Retourne tous les symboles disponibles"""
    symbols = []
    symbols.extend(get_all_us_stocks())
    symbols.extend(get_all_eu_stocks())
    for etf_list in ETFS.values():
        symbols.extend(etf_list)
    return symbols


# =============================================================================
# SYMBOLES ACTIFS (a modifier selon preferences)
# =============================================================================
ACTIVE_SYMBOLS = get_default_watchlist()

# =============================================================================
# WATCHLIST - Liste principale pour le scanner
# =============================================================================
WATCHLIST = get_all_us_stocks() + get_all_eu_stocks()
