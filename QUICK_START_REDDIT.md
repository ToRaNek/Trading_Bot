# Guide Rapide - Scraping Reddit

## 🚀 Démarrage Rapide

### 1. Activer l'environnement virtuel

```bash
source .venv/bin/activate
```

### 2. Tester la configuration

```bash
python scripts/test_config.py
```

### 3. Scraper une seule action (test rapide)

```bash
python scripts/scrape_single_stock.py NVDA
```

### 4. Scraper toutes les actions

```bash
python scripts/scrape_all_stocks.py
```

Les données seront sauvegardées dans `data/Sentiment_[TICKER].csv`

## 📋 Actions Configurées

- **NVDA** - Nvidia
- **AAPL** - Apple
- **GOOG** - Google
- **AMZN** - Amazon
- **META** - Meta
- **TSLA** - Tesla
- **BRK.B** - Berkshire Hathaway
- **JPM** - JPMorgan Chase
- **V** - Visa
- **JNJ** - Johnson & Johnson
- **WMT** - Walmart

## 🔧 Ajouter une Nouvelle Action

1. Éditer `config_stocks.py`
2. Ajouter la configuration :

```python
'TICKER': {
    'sources': [
        {'type': 'subreddit', 'name': 'NOM_SUBREDDIT'},  # Si dédié
        {'type': 'search', 'subreddit': 'stocks', 'query': '$TICKER'}  # Sinon
    ]
}
```

3. Scraper :

```bash
python scripts/scrape_single_stock.py TICKER
```

## 📁 Structure des Fichiers

```
data/
└── Sentiment_[TICKER].csv    # Format : created, source, title, body, upvotes, downvotes...

scripts/
├── scrape_all_stocks.py      # Scrape toutes les actions
├── scrape_single_stock.py    # Scrape une seule action
└── test_config.py             # Teste la configuration

config_stocks.py               # Configuration centralisée
```

## 💡 Astuces

- Les CSV sont utilisés **automatiquement** par le backtest
- Les doublons sont **automatiquement** supprimés
- Le système combine **Reddit API** (récent) + **Pushshift** (historique)

## 📖 Documentation Complète

Voir `README_REDDIT_SCRAPING.md` pour plus de détails.
