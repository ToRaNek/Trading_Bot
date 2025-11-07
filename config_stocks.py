"""
Configuration centralisée des actions et leurs sources Reddit

Pour ajouter une nouvelle action :
1. Si l'action a un subreddit dédié (ex: r/NVDA_Stock) :
   - Ajouter {'type': 'subreddit', 'name': 'NOM_SUBREDDIT'}

2. Si l'action n'a pas de subreddit dédié :
   - Ajouter {'type': 'search', 'subreddit': 'stocks', 'query': '$TICKER'}

3. Pour rechercher dans plusieurs sources, ajouter plusieurs entrées dans 'sources'
"""

# Configuration des actions avec leurs sources Reddit
STOCK_CONFIGS = {
    'NVDA': {
        'sources': [
            {'type': 'subreddit', 'name': 'NVDA_Stock'},
            {'type': 'search', 'subreddit': 'stocks', 'query': '$NVDA'}
        ]
    },
    'AAPL': {
        'sources': [
            {'type': 'subreddit', 'name': 'AAPL'},
            {'type': 'search', 'subreddit': 'stocks', 'query': '$AAPL'}
        ]
    },
    'GOOG': {
        'sources': [
            {'type': 'subreddit', 'name': 'GOOG_Stock'},
            {'type': 'search', 'subreddit': 'stocks', 'query': '$GOOG'}
        ]
    },
    'AMZN': {
        'sources': [
            {'type': 'subreddit', 'name': 'amzn'},
            {'type': 'search', 'subreddit': 'stocks', 'query': '$AMZN'}
        ]
    },
    'META': {
        'sources': [
            {'type': 'search', 'subreddit': 'stocks', 'query': '$meta'}
        ]
    },
    'TSLA': {
        'sources': [
            {'type': 'subreddit', 'name': 'TSLA'},
            {'type': 'search', 'subreddit': 'stocks', 'query': '$TSLA'}
        ]
    },
    'BRK.B': {
        'sources': [
            {'type': 'subreddit', 'name': 'BerkshireHathaway'},
            {'type': 'search', 'subreddit': 'stocks', 'query': '$BRK'}
        ]
    },
    'JPM': {
        'sources': [
            {'type': 'subreddit', 'name': 'JPMorganChase'},
            {'type': 'search', 'subreddit': 'stocks', 'query': '$JPM'}
        ]
    },
    'V': {
        'sources': [
            {'type': 'search', 'subreddit': 'stocks', 'query': '$visa'}
        ]
    },
    'JNJ': {
        'sources': [
            {'type': 'search', 'subreddit': 'ValueInvesting', 'query': 'JNJ'},
            {'type': 'search', 'subreddit': 'stocks', 'query': '$JNJ'}
        ]
    },
    'WMT': {
        'sources': [
            {'type': 'search', 'subreddit': 'stocks', 'query': 'wmt'}
        ]
    }
}


def get_stock_config(ticker: str) -> dict:
    """Récupère la configuration d'une action"""
    return STOCK_CONFIGS.get(ticker, {
        'sources': [
            {'type': 'search', 'subreddit': 'stocks', 'query': f'${ticker}'}
        ]
    })


def get_all_tickers() -> list:
    """Retourne la liste de tous les tickers configurés"""
    return list(STOCK_CONFIGS.keys())


# Exemple d'utilisation :
# from config_stocks import STOCK_CONFIGS, get_stock_config
# config = get_stock_config('NVDA')
# sources = config['sources']
