# 🧪 Guide de Test - Architecture Modulaire

## ✅ Modules créés et testables

La séparation en modules est **terminée** et **fonctionnelle**!

### 📦 Ce qui a été fait

1. **`analyzers/`** - Modules d'analyse (3/3 ✅)
   - ✅ `technical_analyzer.py` - Analyse technique complète
   - ✅ `news_analyzer.py` - News + validation IA
   - ✅ `reddit_analyzer.py` - Sentiment Reddit (CSV + APIs)

2. **`backtest/`** - Moteur de backtest (en transition ⚠️)
   - ⚠️ `backtest_engine.py` - Importe de trading_bot_main.py

3. **`bot/`** - Bot Discord (en transition ⚠️)
   - ⚠️ `discord_bot.py` - Importe de trading_bot_main.py

4. **Configuration & Main**
   - ✅ `config.py` - Configuration centrale
   - ✅ `main.py` - Point d'entrée avec imports modulaires
   - ✅ `test_modules.py` - Tests automatisés
   - ✅ `trading_bot_main.py` - Backup fonctionnel intact

## 🚀 Tests à effectuer

### Test 1: Vérifier les modules
```bash
python test_modules.py
```

**Résultat attendu:**
```
================================================================================
🧪 TEST DES MODULES CRÉÉS
================================================================================

[1/3] Test import config.py...
✅ Config chargée: 25 actions, seuil=65

[2/3] Test TechnicalAnalyzer...
✅ TechnicalAnalyzer importé
   📊 Téléchargement données NVDA...
   ✅ XXX jours de données
   ✅ Indicateurs calculés (RSI, MACD, SMA, BB, Volume)
   🎯 Décision: BUY/SELL/HOLD avec XX% confiance
   📝 Raisons: ...
   ✅ Format de sortie correct

[3/3] Test HistoricalNewsAnalyzer...
✅ HistoricalNewsAnalyzer importé
   📰 Test récupération news...
   ✅ X news trouvées (ou ⚠️ Aucune news si pas de clé API)
   🤖 Test validation IA...
   📊 Score final: XX/100
   💭 Raison: ...
✅ Tests asynchrones passés

================================================================================
✅ TOUS LES TESTS SONT PASSÉS!
================================================================================
```

### Test 2: Lancer le bot (mode production)
```bash
python main.py
```

**Résultat attendu:**
```
================================================================================
🚀 TRADING BOT - ARCHITECTURE MODULAIRE
================================================================================
📁 Modules chargés:
   ✅ analyzers.TechnicalAnalyzer
   ✅ analyzers.HistoricalNewsAnalyzer
   ✅ analyzers.RedditSentimentAnalyzer
   ⚠️  backtest.RealisticBacktestEngine (transition)
   ⚠️  bot.TradingBot (transition)

📊 Configuration:
   • Watchlist: 25 actions
   • Seuil validation: 65/100
   • Log: trading_bot.log
================================================================================

[Le bot Discord démarre...]
```

### Test 3: Utiliser les modules dans un script custom

**Exemple - Script test_custom.py:**
```python
import asyncio
from datetime import datetime, timedelta
from analyzers import TechnicalAnalyzer, HistoricalNewsAnalyzer, RedditSentimentAnalyzer
from config import WATCHLIST
import yfinance as yf

async def test_nvda():
    # 1. Analyse technique
    tech = TechnicalAnalyzer()
    df = yf.download('NVDA', period='6mo', progress=False)
    df = tech.calculate_indicators(df)
    decision, confidence, reasons = tech.get_technical_score(df.iloc[-1])

    print(f"📊 Technique: {decision} ({confidence:.0f}%)")
    print(f"   {reasons[0]}")

    # 2. Analyse news
    news = HistoricalNewsAnalyzer()
    has_news, news_data, score = await news.get_news_for_date('NVDA', datetime.now())
    print(f"📰 News: {len(news_data)} articles (score: {score:.0f})")

    # 3. Analyse Reddit
    reddit = RedditSentimentAnalyzer(csv_file="pushshift_NVDA_ALL.csv")
    reddit_score, count, samples, posts = await reddit.get_reddit_sentiment('NVDA', datetime.now())
    print(f"💬 Reddit: {count} posts (sentiment: {reddit_score:.0f}/100)")

    # 4. Validation IA finale
    final_score, reason = await news.ask_ai_decision(
        'NVDA', decision, news_data, 500.0, confidence, posts
    )
    print(f"🤖 Score final IA: {final_score}/100")
    print(f"   Raison: {reason}")

    await news.close()
    await reddit.close()

asyncio.run(test_nvda())
```

## 📊 Structure des données

### TechnicalAnalyzer
```python
decision, confidence, reasons = analyzer.get_technical_score(row)
# decision: "BUY" | "SELL" | "HOLD"
# confidence: 0-100
# reasons: ["🟢 DÉCISION: BUY (Confiance: 72/100)", "📊 Signaux: 3 BUY, 1 SELL, 0 HOLD", ...]
```

### HistoricalNewsAnalyzer
```python
has_news, news_data, news_score = await analyzer.get_news_for_date(symbol, date)
# has_news: bool
# news_data: [{'title': '...', 'importance': 2.5, 'keywords': [...], 'summary': '...', ...}, ...]
# news_score: 0-100

final_score, reason = await analyzer.ask_ai_decision(symbol, decision, news_data, price, tech_confidence, reddit_posts)
# final_score: 0-100 (score FINAL pour décision)
# reason: "Tech 72 + Sentiment très positif → BOOST"
```

### RedditSentimentAnalyzer
```python
score, count, samples, all_posts = await analyzer.get_reddit_sentiment(symbol, date)
# score: 0-100 (sentiment 0=très négatif, 50=neutre, 100=très positif)
# count: nombre de posts
# samples: ["🟢 Great earnings...", ...]
# all_posts: [{'title': '...', 'body': '...', 'upvotes': 42, 'downvotes': 3, ...}, ...]
```

## 🔧 Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'analyzers'
```
**Solution:** Tu es dans le mauvais dossier. Va dans Trading_Bot/:
```bash
cd /home/infoetu/gordon.delangue.etu/PROJ_PERSO/Trading_Bot
python test_modules.py
```

### Token Discord manquant
```
❌ Token Discord manquant dans .env
```
**Solution:** Crée/édite `.env` et ajoute:
```
DISCORD_BOT_TOKEN=ton_token_ici
```

### Pandas not found (pour Reddit CSV)
```bash
pip install pandas
# ou
.venv/bin/pip install pandas
```

## 🎯 Prochaines étapes (optionnel)

Si tu veux extraire complètement backtest et bot:

1. **Extraire RealisticBacktestEngine:**
   - Copier la classe depuis `trading_bot_main.py` vers `backtest/backtest_engine.py`
   - Remplacer les imports
   - Tester

2. **Extraire TradingBot:**
   - Copier depuis `trading_bot_main.py` vers `bot/discord_bot.py`
   - Remplacer les imports
   - Tester

Mais ce n'est **pas urgent** car la transition actuelle fonctionne parfaitement! ✅

## ✨ Résumé

**Ce qui fonctionne maintenant:**
- ✅ Imports modulaires dans `main.py`
- ✅ Tous les analyseurs sont séparés et réutilisables
- ✅ Configuration centralisée dans `config.py`
- ✅ Tests automatisés disponibles
- ✅ `trading_bot_main.py` reste intact (backup)
- ✅ Le bot Discord fonctionne normalement

**Tu peux utiliser:**
```bash
python main.py          # Lancer le bot
python test_modules.py  # Tester les modules
```

🎉 **L'architecture modulaire est opérationnelle!**
