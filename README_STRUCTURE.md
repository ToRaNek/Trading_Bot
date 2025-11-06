# Structure Modulaire du Bot de Trading

## 📁 Organisation des fichiers

```
Trading_Bot/
├── analyzers/              # Modules d'analyse ✅
│   ├── __init__.py         ✅
│   ├── technical_analyzer.py    ✅ Analyse technique (RSI, MACD, SMA, BB)
│   ├── news_analyzer.py         ✅ Analyse news + validation IA HuggingFace
│   └── reddit_analyzer.py       ✅ Analyse sentiment Reddit (CSV + APIs)
│
├── backtest/               # Module de backtest ⚠️
│   ├── __init__.py         ✅
│   └── backtest_engine.py       ⚠️ Moteur (en transition, importe de trading_bot_main.py)
│
├── bot/                    # Bot Discord ⚠️
│   ├── __init__.py         ✅
│   └── discord_bot.py           ⚠️ Commandes (en transition, importe de trading_bot_main.py)
│
├── config.py               ✅ Configuration (watchlist, seuils)
├── main.py                 ✅ Point d'entrée principal avec imports modulaires
├── test_modules.py         ✅ Tests des modules créés
├── README_STRUCTURE.md     ✅ Cette doc
└── trading_bot_main.py     ✅ Fichier original (backup fonctionnel)
```

**Légende:**
- ✅ Complètement extrait et fonctionnel
- ⚠️ En transition (importe encore de trading_bot_main.py)
- ❌ À créer

## 🎯 Modules créés

### ✅ `analyzers/technical_analyzer.py`
**Classe:** `TechnicalAnalyzer`

**Méthodes:**
- `calculate_indicators(df)` - Calcule RSI, MACD, SMA, Bollinger Bands, Volume
- `get_technical_score(row)` - Retourne (decision, confidence, reasons)
  - Decision: "BUY" / "SELL" / "HOLD"
  - Confidence: 0-100
  - Reasons: Liste des signaux

**Exemple:**
```python
from analyzers import TechnicalAnalyzer

analyzer = TechnicalAnalyzer()
df = analyzer.calculate_indicators(df)
decision, confidence, reasons = analyzer.get_technical_score(df.iloc[-1])

# Output: ("BUY", 72, ["🟢 DÉCISION: BUY (Confiance: 72/100)", ...])
```

### ✅ `analyzers/news_analyzer.py`
**Classe:** `HistoricalNewsAnalyzer`

**Méthodes:**
- `get_news_for_date(symbol, date)` - Récupère news historiques (cache)
- `ask_ai_decision(symbol, decision, news, price, tech_confidence, reddit_posts)` - Validation IA avec HuggingFace
  - Prompt enrichi avec: news complètes, Reddit posts (upvotes/downvotes), décision tech
  - Retourne: (final_score, reason)

**Exemple:**
```python
from analyzers import HistoricalNewsAnalyzer

analyzer = HistoricalNewsAnalyzer()
has_news, news_data, news_score = await analyzer.get_news_for_date("NVDA", datetime.now())
final_score, reason = await analyzer.ask_ai_decision("NVDA", "BUY", news_data, 500.0, 72, reddit_posts)
```

### ✅ `config.py`
Configuration centrale:
- `WATCHLIST` - Liste des actions à analyser
- `VALIDATION_THRESHOLD = 65` - Score minimum pour exécuter un trade
- `LOG_FILE`, `LOG_LEVEL`

## 🚀 Utilisation

### Pour l'instant (transition):
```bash
python main.py
```
→ Utilise encore `trading_bot_main.py` en arrière-plan

### Après migration complète:
Les modules seront importés depuis les dossiers séparés:
```python
from analyzers import TechnicalAnalyzer, HistoricalNewsAnalyzer, RedditSentimentAnalyzer
from backtest import RealisticBacktestEngine
from bot import TradingBot
from config import WATCHLIST, VALIDATION_THRESHOLD
```

### ✅ `analyzers/reddit_analyzer.py`
**Classe:** `RedditSentimentAnalyzer`

**Méthodes:**
- `load_csv_data()` - Charge les posts Reddit depuis CSV (backtest sans requêtes API)
- `get_posts_from_csv(symbol, date)` - Filtre posts par date
- `get_reddit_sentiment(symbol, date)` - Retourne (score, count, samples, all_posts_details)
  - Utilise CSV si disponible, sinon fait des requêtes API
  - Supporte Reddit API (< 7j) et Pushshift (> 7j)
- `_get_subreddit_posts()`, `_search_reddit_comments()` - API Reddit
- `_get_pushshift_posts()`, `_search_pushshift()` - API Pushshift avec pagination
- `save_posts_to_csv()` - Sauvegarde posts (avec upvotes/downvotes)

**Exemple:**
```python
from analyzers import RedditSentimentAnalyzer

analyzer = RedditSentimentAnalyzer(csv_file="pushshift_NVDA_ALL.csv")
score, count, samples, posts = await analyzer.get_reddit_sentiment("NVDA", datetime.now())
# posts contient: title, body, upvotes, downvotes, score, created, author, source
```

## 📝 Statut des modules

✅ **Complètement extraits et fonctionnels:**
- `analyzers/technical_analyzer.py`
- `analyzers/news_analyzer.py`
- `analyzers/reddit_analyzer.py`
- `config.py`
- `main.py` (imports modulaires)

⚠️ **En transition (importent de trading_bot_main.py):**
- `backtest/backtest_engine.py`
- `bot/discord_bot.py`

💡 Ces modules fonctionnent mais utilisent encore trading_bot_main.py en arrière-plan. Extraction complète possible mais pas urgente.

## ✨ Avantages de la structure modulaire

1. **Séparation des responsabilités** - Chaque module a un rôle clair
2. **Réutilisabilité** - Les analyseurs peuvent être utilisés indépendamment
3. **Tests unitaires** - Plus facile de tester chaque composant
4. **Maintenance** - Modifications isolées sans impacter le reste
5. **Lisibilité** - Code organisé et navigable

## 🔧 Prochaines étapes

1. Tester que `main.py` fonctionne correctement
2. Créer les modules manquants (reddit, backtest, bot)
3. Migrer les imports dans `main.py`
4. Supprimer `trading_bot_main.py` (ou le garder en backup)

## 📊 Flux de données

```
main.py
  ↓
TradingBot (Discord)
  ↓
RealisticBacktestEngine
  ↓
├─→ TechnicalAnalyzer → BUY/SELL/HOLD + Confidence
├─→ HistoricalNewsAnalyzer → News + Score IA (HuggingFace)
└─→ RedditSentimentAnalyzer → Sentiment + Posts détaillés
  ↓
Décision finale (score > 65 → Execute)
```
