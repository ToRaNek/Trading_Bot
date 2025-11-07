# Guide de Scraping Reddit pour le Trading Bot

## 📁 Structure des fichiers

```
Trading_Bot/
├── data/                          # Dossier de données
│   ├── Sentiment_NVDA.csv        # Données sentiment pour NVDA
│   ├── Sentiment_AAPL.csv        # Données sentiment pour AAPL
│   └── Sentiment_[TICKER].csv    # Etc...
│
├── scripts/
│   ├── scrape_all_stocks.py      # Script principal pour scraper toutes les actions
│   └── scrape_pushshift_csv.py   # Ancien script (uniquement NVDA)
│
├── analyzers/
│   └── reddit_analyzer.py         # Analyseur de sentiment (lit automatiquement les CSV)
│
└── config_stocks.py               # Configuration centralisée des actions
```

## 🚀 Comment scraper les données Reddit

### 1. Scraper toutes les actions configurées

```bash
cd scripts/
python scrape_all_stocks.py
```

Ce script va :
- Scraper toutes les actions définies dans `config_stocks.py`
- Récupérer les posts via Reddit API (posts récents) et Pushshift (historique)
- Sauvegarder chaque action dans `data/Sentiment_[TICKER].csv`

### 2. Ajouter une nouvelle action

Éditez le fichier `config_stocks.py` :

```python
STOCK_CONFIGS = {
    'MSFT': {  # Nouveau ticker
        'sources': [
            # Option 1 : Subreddit dédié
            {'type': 'subreddit', 'name': 'microsoft'},

            # Option 2 : Recherche dans r/stocks
            {'type': 'search', 'subreddit': 'stocks', 'query': '$MSFT'}
        ]
    }
}
```

**Règles** :
- Si l'action a un subreddit dédié (ex: r/NVDA_Stock), utilisez `type: 'subreddit'`
- Sinon, utilisez `type: 'search'` avec r/stocks et le ticker
- Vous pouvez combiner plusieurs sources (subreddit dédié + recherche)

### 3. Format des fichiers CSV

Chaque fichier `data/Sentiment_[TICKER].csv` contient :

```csv
created,source,title,body,upvotes,downvotes,author,url,id
2025-11-06 15:30:00,r/NVDA_Stock,"NVDA earnings beat!","Great quarter...",145,12,user123,https://...,abc123
```

Colonnes :
- **created** : Date/heure du post
- **source** : Subreddit source (ex: r/NVDA_Stock)
- **title** : Titre du post
- **body** : Contenu du post
- **upvotes** : Nombre d'upvotes
- **downvotes** : Nombre de downvotes
- **author** : Auteur du post
- **url** : URL du post
- **id** : ID unique du post

## 📊 Utilisation dans le backtest

Le `RedditSentimentAnalyzer` charge automatiquement le bon fichier CSV selon l'action :

```python
from analyzers.reddit_analyzer import RedditSentimentAnalyzer

# Initialiser l'analyseur (va chercher dans data/)
analyzer = RedditSentimentAnalyzer(data_dir='data')

# Analyser le sentiment pour NVDA
sentiment, post_count, samples, posts = await analyzer.get_reddit_sentiment(
    symbol='NVDA',
    target_date=datetime.now(),
    lookback_hours=48
)

print(f"Sentiment NVDA: {sentiment}/100 ({post_count} posts)")
```

**Le système charge automatiquement** `data/Sentiment_NVDA.csv` !

## 🎯 Actions configurées actuellement

| Ticker | Subreddits / Sources |
|--------|---------------------|
| NVDA   | r/NVDA_Stock + r/stocks ($NVDA) |
| AAPL   | r/AAPL + r/stocks ($AAPL) |
| GOOG   | r/GOOG_Stock + r/stocks ($GOOG) |
| AMZN   | r/amzn + r/stocks ($AMZN) |
| META   | r/stocks ($meta) |
| TSLA   | r/TSLA + r/stocks ($TSLA) |
| BRK.B  | r/BerkshireHathaway + r/stocks ($BRK) |
| JPM    | r/JPMorganChase + r/stocks ($JPM) |
| V      | r/stocks ($visa) |
| JNJ    | r/ValueInvesting (JNJ) + r/stocks ($JNJ) |
| WMT    | r/stocks (wmt) |

## ⚙️ Configuration avancée

### Scraper une seule action

Modifiez `scrape_all_stocks.py` pour limiter le scraping :

```python
# Scraper uniquement NVDA et AAPL
STOCK_CONFIGS = {k: v for k, v in STOCK_CONFIGS.items() if k in ['NVDA', 'AAPL']}
```

### Modifier les paramètres de scraping

Dans `scrape_all_stocks.py`, vous pouvez ajuster :
- `limit` : Nombre max de posts par source (défaut: 1000)
- `size` : Taille des pages Pushshift (défaut: 100)
- Délais entre requêtes pour éviter rate limiting

## 🔧 Dépannage

### Erreur "Fichier Sentiment_XXX.csv introuvable"

1. Vérifiez que vous avez bien scrapé l'action avec `scrape_all_stocks.py`
2. Vérifiez que le fichier existe dans `data/`

### Rate limiting Reddit

Si vous êtes bloqué par Reddit :
- Augmentez les délais entre requêtes (`await asyncio.sleep(...)`)
- Utilisez un VPN ou changez d'IP
- Attendez quelques heures

### Pushshift ne répond pas

Pushshift/PullPush peut être instable :
- Le script a des retries automatiques (3 tentatives)
- Si ça échoue, les données Reddit API seront quand même sauvegardées
- Réessayez plus tard pour l'historique Pushshift

## 📝 Notes importantes

1. **Reddit API** : Limité aux ~1000 posts les plus récents par source
2. **Pushshift** : Accès à l'historique complet mais peut être lent/instable
3. **Déduplication** : Les doublons entre sources sont automatiquement supprimés
4. **Rate limiting** : Respectez les limites pour éviter d'être bloqué

## 🎓 Exemples d'utilisation

### Exemple 1 : Scraper toutes les actions

```bash
python scripts/scrape_all_stocks.py
```

### Exemple 2 : Analyser le sentiment dans le backtest

```python
from backtest.backtest_engine import RealisticBacktestEngine

# Le backtest va automatiquement charger data/Sentiment_[SYMBOL].csv
engine = RealisticBacktestEngine()
results = await engine.backtest_with_news_validation('NVDA', months=6)
```

### Exemple 3 : Ajouter MSFT

1. Éditez `config_stocks.py` :
```python
'MSFT': {
    'sources': [
        {'type': 'subreddit', 'name': 'microsoft'},
        {'type': 'search', 'subreddit': 'stocks', 'query': '$MSFT'}
    ]
}
```

2. Scrapez :
```bash
python scripts/scrape_all_stocks.py
```

3. Le fichier `data/Sentiment_MSFT.csv` sera créé automatiquement

## 📞 Support

Pour toute question ou problème, consultez :
- `GUIDE_TEST.md` : Guide de test complet
- `README_STRUCTURE.md` : Structure du projet
- `RECAP_FINAL.md` : Récapitulatif du système
