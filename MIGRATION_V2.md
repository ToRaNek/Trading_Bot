# 🔄 Guide de Migration v2.0

## ⚠️ Problème Identifié

Le projet a **deux versions** de certaines classes :

1. **Version modulaire** (recommandée) :
   - `analyzers/reddit_analyzer.py`
   - `analyzers/news_analyzer.py`
   - `analyzers/ai_scorer.py`

2. **Version monolithique** (ancienne) :
   - `trading_bot_main.py` (contient TOUT le code en un seul fichier)

## ✅ Ce qui a été corrigé

### 1. Erreur `data_dir`
**Problème :**
```
TypeError: RedditSentimentAnalyzer.__init__() got an unexpected keyword argument 'data_dir'
```

**Solution :**
J'ai ajouté le paramètre `data_dir` dans **LES DEUX** versions :
- ✅ `analyzers/reddit_analyzer.py`
- ✅ `trading_bot_main.py`

### 2. Paramètres manquants dans `ask_ai_decision()`

Les nouvelles fonctionnalités sont dans `analyzers/news_analyzer.py` mais **PAS** dans la version de `trading_bot_main.py`.

## 🚀 Actions à faire

### Option 1 : Utiliser les modules (RECOMMANDÉ)

**Avantages :**
- Code organisé et maintenable
- Nouvelles fonctionnalités (AI Scorer, Stop Loss/Take Profit)
- Séparation des responsabilités

**Fichiers à utiliser :**
```python
from analyzers.reddit_analyzer import RedditSentimentAnalyzer
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from analyzers.ai_scorer import AIScorer
from analyzers.technical_analyzer import TechnicalAnalyzer
```

**Modifier `backtest/backtest_engine.py` :**
```python
# AVANT
from trading_bot_main import RealisticBacktestEngine

# APRÈS - Extraire la classe dans backtest_engine.py
from analyzers import TechnicalAnalyzer, HistoricalNewsAnalyzer, RedditSentimentAnalyzer
```

### Option 2 : Synchroniser trading_bot_main.py

Si vous voulez continuer à utiliser `trading_bot_main.py`, vous devez **synchroniser** toutes les modifications :

**Classes à mettre à jour dans `trading_bot_main.py` :**

1. ✅ `RedditSentimentAnalyzer.__init__()` - FAIT
2. ❌ `HistoricalNewsAnalyzer.__init__()` - Ajouter `self.ai_scorer`
3. ❌ `HistoricalNewsAnalyzer.ask_ai_decision()` - Ajouter nouveaux paramètres
4. ❌ `RealisticBacktestEngine` - Ajouter stop_loss/take_profit logic

## 📋 Checklist de Migration

### Pour utiliser la version modulaire :

- [ ] Extraire `RealisticBacktestEngine` de `trading_bot_main.py` vers `backtest/backtest_engine.py`
- [ ] Modifier les imports dans `backtest/backtest_engine.py`
- [ ] Supprimer les duplications de classes dans `trading_bot_main.py`
- [ ] Tester avec `python main.py`

### Pour synchroniser trading_bot_main.py :

- [x] Ajouter `data_dir` à `RedditSentimentAnalyzer`
- [ ] Ajouter `ai_scorer` à `HistoricalNewsAnalyzer`
- [ ] Mettre à jour `ask_ai_decision()` avec nouveaux paramètres
- [ ] Ajouter stop_loss/take_profit dans la boucle de backtest
- [ ] Ajouter les 5 derniers prix
- [ ] Tester avec `python main.py`

## 🔧 Code de Migration

### Extraire RealisticBacktestEngine (Recommandé)

**Créer `backtest/backtest_engine_v2.py` :**

```python
"""Moteur de backtest v2 avec AI Scoring Multi-Niveau"""

import asyncio
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from time import sleep

from analyzers.technical_analyzer import TechnicalAnalyzer
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from analyzers.reddit_analyzer import RedditSentimentAnalyzer

logger = logging.getLogger('TradingBot')


class RealisticBacktestEngine:
    """
    Backtest réaliste v2.0 avec:
    - Stop Loss / Take Profit automatiques
    - AI Scoring multi-niveau
    - Scores Reddit et News pré-calculés
    - 5 derniers prix pour contexte
    """

    def __init__(self, reddit_csv_file: str = None, data_dir: str = 'data'):
        self.news_analyzer = HistoricalNewsAnalyzer()
        self.reddit_analyzer = RedditSentimentAnalyzer(
            csv_file=reddit_csv_file,
            data_dir=data_dir
        )
        self.tech_analyzer = TechnicalAnalyzer()

        # Configuration stop loss / take profit
        self.stop_loss_pct = -3.0
        self.take_profit_pct = 10.0

    async def backtest_with_news_validation(self, symbol: str, months: int = 6):
        """Backtest avec validation IA multi-niveau"""
        # ... (copier le code depuis trading_bot_main.py lignes 1345-1550)
```

**Puis modifier `backtest/__init__.py` :**

```python
from .backtest_engine_v2 import RealisticBacktestEngine

__all__ = ['RealisticBacktestEngine']
```

## 🐛 Problèmes Connus

### 1. Import Circulaire

**Problème :**
```
backtest/backtest_engine.py → from trading_bot_main import RealisticBacktestEngine
trading_bot_main.py → bot = TradingBot() → RealisticBacktestEngine()
```

**Solution :**
Extraire `RealisticBacktestEngine` dans son propre fichier.

### 2. Deux versions de classes

**Classes dupliquées :**
- `RedditSentimentAnalyzer` (dans `analyzers/` ET `trading_bot_main.py`)
- `HistoricalNewsAnalyzer` (dans `analyzers/` ET `trading_bot_main.py`)

**Solution :**
Supprimer les classes de `trading_bot_main.py` et importer depuis `analyzers/`.

## 📊 État Actuel

| Composant | Module | trading_bot_main.py | Synchro |
|-----------|--------|---------------------|---------|
| `RedditSentimentAnalyzer.__init__()` | ✅ data_dir | ✅ data_dir | ✅ OUI |
| `RedditSentimentAnalyzer.load_csv_data()` | ✅ par symbole | ❌ ancien | ❌ NON |
| `HistoricalNewsAnalyzer.__init__()` | ✅ ai_scorer | ❌ pas ai_scorer | ❌ NON |
| `HistoricalNewsAnalyzer.ask_ai_decision()` | ✅ nouveaux params | ❌ anciens params | ❌ NON |
| `AIScorer` | ✅ existe | ❌ n'existe pas | N/A |
| `RealisticBacktestEngine` stop/take | ✅ existe | ✅ existe | ✅ OUI |

## 🎯 Recommandation

**Utiliser l'architecture modulaire** et extraire complètement `RealisticBacktestEngine` dans son propre fichier.

**Avantages :**
- Code maintenable ✅
- Pas de duplication ✅
- Toutes les nouvelles fonctionnalités ✅
- Tests plus faciles ✅

**Inconvénient :**
- Nécessite un peu de refactoring (~30 min)

## 📝 Script de Migration Automatique

```bash
# 1. Créer le nouveau fichier backtest_engine_v2.py
cp backtest/backtest_engine.py backtest/backtest_engine_v2.py

# 2. Remplacer l'import
sed -i 's/from trading_bot_main import RealisticBacktestEngine/from analyzers import TechnicalAnalyzer, HistoricalNewsAnalyzer, RedditSentimentAnalyzer/' backtest/backtest_engine_v2.py

# 3. Copier la classe depuis trading_bot_main.py
# (extraction manuelle recommandée)

# 4. Mettre à jour __init__.py
echo "from .backtest_engine_v2 import RealisticBacktestEngine" > backtest/__init__.py
echo "__all__ = ['RealisticBacktestEngine']" >> backtest/__init__.py

# 5. Tester
python3 -c "from backtest import RealisticBacktestEngine; print('✅ Import OK')"
```

## ✅ Vérification Post-Migration

```python
# Test rapide
python3 -c "
from analyzers.ai_scorer import AIScorer
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from analyzers.reddit_analyzer import RedditSentimentAnalyzer
from backtest import RealisticBacktestEngine

print('✅ Tous les imports fonctionnent')
"
```

## 🆘 En Cas de Problème

**Si le bot ne démarre plus :**

1. Vérifier les imports :
```bash
python3 -c "from backtest import RealisticBacktestEngine"
```

2. Vérifier que `data_dir` est bien ajouté partout :
```bash
grep -n "def __init__.*data_dir" analyzers/reddit_analyzer.py trading_bot_main.py
```

3. Revenir à la version précédente :
```bash
git checkout analyzers/reddit_analyzer.py trading_bot_main.py
```

## 📞 Support

Consultez :
- `README_AI_SCORING.md` - Guide complet du système
- `CHANGEMENTS_V2.md` - Liste des changements
- `RESUME_CHANGEMENTS.txt` - Résumé rapide

---

**Version :** v2.0
**Date :** 2025-11-07
**Statut :** ✅ Fonctionnel (avec duplications)
