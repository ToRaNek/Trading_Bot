# Système de Scoring IA - Documentation

## 🎯 Aperçu

Le système de trading utilise maintenant une approche **multi-niveau** pour valider les décisions :

1. **Analyse Technique** → Décision initiale (BUY/SELL/HOLD)
2. **Stop Loss / Take Profit** → Sorties automatiques prioritaires
3. **AI Scoring** → Validation intelligente par Hugging Face
4. **Décision Finale** → Exécution du trade ou rejet

## 🔄 Flux de Décision

```
┌──────────────────────────┐
│  Analyse Technique       │
│  (RSI, MACD, Bollinger)  │
└────────────┬─────────────┘
             │
             ▼
    ┌────────────────┐
    │  BUY/SELL?     │──── HOLD ──> Aucune action
    └────────┬───────┘
             │
             ▼ (En position)
    ┌────────────────────┐
    │ PRIORITÉ 1:        │
    │ Stop Loss (-3%)    │──── Activé ──> Vente automatique
    │ Take Profit (+10%) │
    └────────┬───────────┘
             │
             ▼ (Pas activé)
    ┌────────────────────┐
    │ AI Scorer:         │
    │ 1. Score Reddit    │
    │ 2. Score News      │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Hugging Face:      │
    │ Décision Finale    │
    │ (0-100)            │
    └────────┬───────────┘
             │
             ▼
    Score > 65 ? ──> OUI ──> Exécution trade
             │
             └──> NON ──> Rejet
```

## 📊 Système de Scoring Multi-Niveau

### Niveau 1 : Stop Loss / Take Profit (PRIORITAIRE)

**Configuration actuelle :**
- **Stop Loss** : -3% (protection contre les pertes)
- **Take Profit** : +10% (sécurisation des gains)

**Fonctionnement :**
- Vérifié à **chaque jour** si en position
- **Prioritaire** sur toute autre décision
- **Pas de requête IA** si activé (économie de tokens)
- Sortie **immédiate** et automatique

```python
# Configuration dans RealisticBacktestEngine
self.stop_loss_pct = -3.0   # -3%
self.take_profit_pct = 10.0  # +10%
```

### Niveau 2 : AI Scorer - Pré-analyse

**Nouveau système** : Avant la décision finale, Hugging Face analyse :

#### 2.1 Score Reddit (0-100)

**Entrées :**
- Titres des posts
- Contenu des posts
- Upvotes / Downvotes
- Volume de discussions

**Analyse :**
```python
from analyzers.ai_scorer import AIScorer

scorer = AIScorer(hf_token)
reddit_score = await scorer.score_reddit_posts(symbol, posts, target_date)
```

**Règles :**
- **0 posts** → Score = 0 (personne n'en parle)
- **Peu d'upvotes** → Score réduit
- **Beaucoup de discussions** → Score augmenté
- **Sentiment positif** → Score élevé (70-100)
- **Sentiment négatif** → Score faible (0-30)

#### 2.2 Score News (0-100)

**Entrées :**
- Titres des actualités
- Descriptions
- Sources (importance)
- Volume de news

**Analyse :**
```python
news_score = await scorer.score_news(symbol, news_items, target_date)
```

**Règles :**
- **0 news** → Score = 0 (pas d'actualité)
- **News positives** (croissance, innovation) → Score élevé
- **News négatives** (scandales, pertes) → Score faible
- **Beaucoup de news** → Événement important

### Niveau 3 : Hugging Face - Décision Finale

**Entrées consolidées :**
1. **Score Technique** : RSI, MACD, Bollinger (0-100)
2. **Score Reddit** : Pré-calculé par AI Scorer (0-100)
3. **Score News** : Pré-calculé par AI Scorer (0-100)
4. **5 derniers prix** : Contexte de tendance
5. **Prix cible** : buy_price ou sell_price
6. **Type de décision** : BUY ou SELL

**Prompt optimisé :**
```
TRADING DECISION VALIDATION - OPTIMIZED SYSTEM

🎯 STOCK: NVDA
💰 Current Price: $145.50
🎯 Target Buy Price: $145.50

📈 LAST 5 PRICES (trend context):
   1. $142.30
   2. $143.80
   3. $144.20
   4. $145.00
   5. $145.50
   Trend: 📈 UPTREND (+2.25%)

🤖 TECHNICAL DECISION: BUY
📊 Technical Confidence: 75/100

📊 PRE-CALCULATED SCORES (by AI Scorer):
💬 Reddit Community Score: 82/100 (45 posts analyzed)
📰 News Sentiment Score: 68/100 (8 news analyzed)

TASK: Provide the FINAL TRADING SCORE (0-100)
```

**Réductions automatiques :**
- **Pas de Reddit** (score = 0) → Score final × 0.7 (-30%)
- **Pas de News** (score = 0) → Score final × 0.7 (-30%)
- **Aucun des deux** → Score final = 0 (rejet automatique)

**Échelle de décision :**
- **0-30** : Mauvaise décision, données contradictoires
- **31-50** : Décision faible, signaux mixtes
- **51-70** : Bonne décision, modérément supportée
- **71-100** : Excellente décision, fortement supportée

**Seuil d'exécution :** Score > 65

## 🔧 Configuration

### Variables d'environnement

```bash
# .env
HUGGINGFACE_TOKEN=your_token_here
NEWSAPI_KEY=your_newsapi_key
FINNHUB_KEY=your_finnhub_key
```

### Paramètres du backtest

```python
from trading_bot_main import RealisticBacktestEngine

# Initialisation avec data_dir pour charger les CSV par action
engine = RealisticBacktestEngine(
    reddit_csv_file=None,  # Pas de CSV global
    data_dir='data'         # Dossier contenant Sentiment_[TICKER].csv
)

# Modifier stop loss / take profit
engine.stop_loss_pct = -5.0   # -5% au lieu de -3%
engine.take_profit_pct = 15.0  # +15% au lieu de +10%

# Lancer le backtest
results = await engine.backtest_with_news_validation('NVDA', months=6)
```

## 📈 Exemple de Décision

### Scénario : Signal BUY sur NVDA

**Données :**
- Prix actuel : $145.50
- Analyse technique : BUY (Confidence: 75/100)
- Reddit : 45 posts (Score AI: 82/100)
- News : 8 actualités (Score AI: 68/100)
- Tendance : +2.25% sur 5 derniers jours

**Flux de décision :**

1. **Analyse Technique** → BUY suggéré
2. **Stop Loss/Take Profit** → Pas en position (skip)
3. **AI Scorer** :
   - Reddit : 82/100 ✅ (communauté très positive)
   - News : 68/100 ✅ (actualités positives)
4. **Hugging Face** :
   - Combine : Tech (75) + Reddit (82) + News (68) + Trend (+2.25%)
   - Score final : **88/100** ✅
5. **Décision** : 88 > 65 → **BUY EXÉCUTÉ**

### Scénario : Signal SELL sans données

**Données :**
- Prix actuel : $148.20
- Analyse technique : SELL (Confidence: 70/100)
- Reddit : 0 posts (Score: 0/100) ⚠️
- News : 0 news (Score: 0/100) ⚠️
- Tendance : -1.5% sur 5 derniers jours

**Flux de décision :**

1. **Analyse Technique** → SELL suggéré
2. **Stop Loss/Take Profit** → Pas activé
3. **AI Scorer** :
   - Reddit : 0/100 ❌ (personne n'en parle)
   - News : 0/100 ❌ (aucune actualité)
4. **Hugging Face** :
   - Aucune donnée disponible
   - Score final : **0/100** ❌
5. **Décision** : 0 < 65 → **SELL REJETÉ**

## 🎓 Bonnes Pratiques

### 1. Scraper les données avant le backtest

```bash
# Scraper toutes les actions
python scripts/scrape_all_stocks.py

# Les données seront dans data/Sentiment_[TICKER].csv
# Le backtest les chargera automatiquement
```

### 2. Vérifier la configuration

```bash
python scripts/test_config.py
```

### 3. Tester avec une seule action

```python
# Test rapide sur 3 mois
results = await engine.backtest_with_news_validation('NVDA', months=3)
```

### 4. Surveiller les logs

```python
import logging
logging.basicConfig(level=logging.INFO)

# Vous verrez :
# [AI Scorer] Reddit NVDA: Score 82/100 (45 posts)
# [AI Scorer] News NVDA: Score 68/100 (8 news)
# [AI Decision] NVDA: FINAL SCORE 88/100
```

## 📊 Résultats et Statistiques

Le backtest génère maintenant des statistiques détaillées :

```python
{
    'symbol': 'NVDA',
    'trades': [
        {
            'entry_date': '2024-05-01',
            'exit_date': '2024-05-15',
            'entry_price': 145.50,
            'exit_price': 160.05,
            'profit': 10.0,  # +10%
            'hold_days': 14,
            'final_score': 100,
            'exit_reason': 'TAKE_PROFIT'  # Nouveau !
        },
        {
            'exit_reason': 'STOP_LOSS'     # -3%
        },
        {
            'exit_reason': 'AI_VALIDATED_SELL'  # Score > 65
        }
    ]
}
```

**Exit reasons :**
- `TAKE_PROFIT` : +10% atteint (prioritaire)
- `STOP_LOSS` : -3% atteint (prioritaire)
- `AI_VALIDATED_SELL` : Vente validée par l'IA

## 🔍 Dépannage

### Erreur "Token HuggingFace manquant"

```bash
# Vérifier .env
cat .env | grep HUGGINGFACE

# Ajouter si manquant
echo "HUGGINGFACE_TOKEN=your_token" >> .env
```

### Score toujours à 0

**Causes possibles :**
1. Pas de données Reddit → Scraper l'action
2. Pas de news → Normal si période calme
3. Les deux → Score final = 0 (normal)

**Solution :**
```bash
# Scraper les données Reddit
python scripts/scrape_single_stock.py NVDA
```

### Trop de rejets

**Causes :**
- Seuil trop élevé (> 65)
- Données insuffisantes

**Solution :**
```python
# Baisser le seuil temporairement
if final_score > 60:  # Au lieu de 65
    # Exécuter trade
```

## 📚 Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `analyzers/ai_scorer.py` | Scoring Reddit/News par HF |
| `analyzers/news_analyzer.py` | Récupération news + décision finale |
| `analyzers/reddit_analyzer.py` | Récupération Reddit (charge CSV) |
| `trading_bot_main.py` | Backtest engine avec stop loss/take profit |
| `data/Sentiment_[TICKER].csv` | Données Reddit par action |

## 🚀 Prochaines Étapes

1. ✅ Scraper les données de toutes les actions
2. ✅ Vérifier la configuration avec `test_config.py`
3. ✅ Tester le backtest sur une action
4. 📊 Analyser les résultats
5. 🎯 Ajuster les seuils si nécessaire

---

**Version :** 2.0 - AI Scoring Multi-Niveau
**Date :** 2025-11-07
