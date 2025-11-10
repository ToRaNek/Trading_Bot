# 🚀 Trading Bot - Mode Dry-Run (Temps Réel)

## Vue d'ensemble

Le bot de trading peut maintenant fonctionner en mode **dry-run** (simulation en temps réel) pendant 3 mois avec un portefeuille simulé de **$1000**.

## Fonctionnalités

### ✅ Ce que fait le bot automatiquement

- **Analyse horaire** : Toutes les heures, le bot analyse chaque action de la watchlist
- **Récupération des données** :
  - News des dernières 48h (Finnhub + NewsAPI)
  - Posts Reddit du jour (API Reddit en temps réel)
  - Données de prix en temps réel (Yahoo Finance)
- **Analyse technique** : RSI, MACD, SMA, Bollinger Bands, Volume
- **Score composite IA** :
  - Technique : 40%
  - News : 35%
  - Reddit : 25%
- **Décisions de trading** :
  - Achat si score ≥ 65/100 et signal BUY
  - Vente si score ≥ 65/100 et signal SELL
- **Gestion du risque** :
  - Stop Loss automatique : -4%
  - Take Profit automatique : +16%
  - Taille de position max : 20% du portfolio
- **Notifications Discord** : Pour chaque trade, stop loss, take profit

## Commandes Discord

### `!start [jours]`

Démarre le bot en mode dry-run pour un nombre de jours spécifié (par défaut : 90 jours = 3 mois).

```
!start 90
```

**Paramètres :**
- `jours` : Durée du dry-run (1-365 jours)

**Ce qui se passe :**
1. Le bot crée un portefeuille simulé avec $1000
2. Il commence à analyser les actions toutes les heures
3. Il envoie des notifications Discord pour chaque action importante
4. L'état du portefeuille est sauvegardé dans `portfolio_live.json`

### `!stop`

Arrête le bot et affiche les statistiques finales.

```
!stop
```

### `!status`

Affiche l'état actuel du bot en temps réel.

```
!status
```

**Informations affichées :**
- Performance globale (% de profit/perte)
- Capital actuel vs initial
- Nombre de trades
- Win rate
- Positions ouvertes avec profit/perte en temps réel
- Statistiques d'activité (analyses, signaux, validations IA)

## Architecture

### Structure des fichiers

```
trading/
├── __init__.py           # Exports du module
├── portfolio.py          # Gestion du portefeuille simulé
└── live_trader.py        # Logique de trading en temps réel
```

### Système de portefeuille (`Portfolio`)

Le système de portefeuille gère :
- **Cash** : Argent liquide disponible
- **Positions** : Actions détenues (symbole, quantité, prix moyen)
- **Historique** : Tous les trades effectués
- **Sauvegarde** : État persistant dans un fichier JSON

**Méthodes principales :**
- `buy(symbol, price, shares, timestamp)` : Acheter des actions
- `sell(symbol, price, shares, timestamp)` : Vendre des actions
- `get_total_value(current_prices)` : Valeur totale du portfolio
- `get_performance(current_prices)` : Statistiques de performance

### Trader en temps réel (`LiveTrader`)

Le trader en temps réel gère :
- **Boucle horaire** : Analyse complète toutes les heures
- **Analyse de stock** : Technique + News + Reddit
- **Exécution de trades** : Validation et exécution des ordres
- **Stop Loss / Take Profit** : Vérification et fermeture automatique
- **Notifications Discord** : Alertes pour tous les événements importants

**Méthodes principales :**
- `analyze_stock(symbol)` : Analyse complète d'une action
- `execute_trade(decision)` : Exécute un trade si validé
- `check_stop_loss_take_profit()` : Vérifie les seuils SL/TP
- `hourly_analysis()` : Cycle d'analyse complet
- `start(duration_days)` : Démarre le bot
- `stop()` : Arrête le bot

## Flux de trading

### 1. Analyse horaire

```
Pour chaque action dans la watchlist:
  1. Récupérer les données de prix (7 derniers jours)
  2. Calculer les indicateurs techniques
  3. Obtenir le signal technique (BUY/SELL/HOLD)
  4. Si BUY ou SELL:
     a. Récupérer les news des 48 dernières heures
     b. Récupérer le sentiment Reddit du jour
     c. Calculer le score composite
     d. Si score ≥ seuil: exécuter le trade
```

### 2. Vérification SL/TP

```
Pour chaque position ouverte:
  1. Récupérer le prix actuel
  2. Calculer le profit/perte actuel
  3. Si perte ≤ -4%: STOP LOSS → vendre
  4. Si profit ≥ +16%: TAKE PROFIT → vendre
```

### 3. Exécution de trade

```
Si signal BUY:
  1. Vérifier si on a déjà une position
  2. Calculer la taille de position (max 20% du portfolio)
  3. Acheter si solde suffisant
  4. Envoyer notification Discord

Si signal SELL:
  1. Vérifier si on a une position à vendre
  2. Vendre toute la position
  3. Calculer le profit/perte
  4. Envoyer notification Discord
```

## Notifications Discord

Le bot envoie des notifications pour :

### 🟢 Achat
- Prix d'achat
- Quantité
- Coût total
- Scores (Tech, News, Reddit, Composite)

### 🔴 Vente
- Prix de vente
- Quantité
- Gain total
- Profit/Perte (% et $)
- Scores (Tech, News, Reddit, Composite)

### ⛔ Stop Loss
- Prix d'entrée et de sortie
- Perte en %

### 💰 Take Profit
- Prix d'entrée et de sortie
- Profit en %

### 🚀 Démarrage / ⏹️ Arrêt
- Statistiques de session
- Performance finale

## Exemple d'utilisation

```discord
Utilisateur: !start 90

Bot: 🚀 Démarrage du Bot en Dry-Run
     Le bot va trader automatiquement pendant 90 jours
     💰 Capital initial: $1000
     📊 Watchlist: 25 actions
     ⏰ Fréquence: Toutes les heures
     ...

[1 heure plus tard]

Bot: 🟢 ACHAT: NVDA
     Trade validé par l'IA
     Prix: $500.00
     Quantité: 3
     Score Final: 78/100

[Quelques heures plus tard]

Bot: 💰 TAKE PROFIT: NVDA
     Position fermée automatiquement
     Prix d'entrée: $500.00
     Prix de sortie: $580.00
     Profit: +16.00%

[Plus tard]

Utilisateur: !status

Bot: 📊 Statut du Bot - Dry-Run
     🟢 EN COURS
     💰 Performance: +12.50%
     Capital: $1125.00
     Trades: 8
     Win Rate: 75.0%
     ...

[Après 90 jours ou sur demande]

Utilisateur: !stop

Bot: ⏹️ ARRÊT DU BOT
     Statistiques finales du dry-run
     Durée: 90 jours
     Performance: +35.00%
     Capital Final: $1350.00
     Trades: 42
     Win Rate: 71.4%
```

## Configuration

Les paramètres suivants peuvent être ajustés dans `trading/live_trader.py` :

```python
self.validation_threshold = 65        # Score minimum pour trader (0-100)
self.max_position_size = 0.2          # Taille max par position (20%)
self.stop_loss_pct = -4.0             # Stop loss (-4%)
self.take_profit_pct = 16.0           # Take profit (+16%)
```

## Fichiers générés

- `portfolio_live.json` : État du portefeuille (sauvegarde automatique)
- `trading_bot.log` : Logs détaillés de toutes les opérations

## Dépendances

Les modules utilisés :
- `yfinance` : Données de prix en temps réel
- `analyzers.TechnicalAnalyzer` : Analyse technique
- `analyzers.HistoricalNewsAnalyzer` : Récupération et analyse des news
- `analyzers.RedditSentimentAnalyzer` : Analyse du sentiment Reddit
- `discord.py` : Intégration Discord

## Notes importantes

1. **Mode simulé** : Aucun argent réel n'est utilisé. C'est un portefeuille 100% simulé.
2. **Limite d'API** : Les APIs (NewsAPI, Finnhub, Reddit) ont des limites de requêtes. Le bot gère la rotation automatique des clés.
3. **Performances** : Les performances passées ne garantissent pas les performances futures.
4. **Données en temps réel** : Le bot utilise les données les plus récentes disponibles, mais il peut y avoir un léger délai.
5. **Sauvegarde** : L'état du portefeuille est sauvegardé automatiquement après chaque trade.

## Troubleshooting

### Le bot ne démarre pas
- Vérifiez que toutes les variables d'environnement sont définies dans `.env`
- Vérifiez les logs dans `trading_bot.log`

### Pas de trades exécutés
- Le seuil de validation (65/100) est peut-être trop élevé
- Les scores techniques/news/reddit peuvent être tous faibles
- Vérifiez les logs pour voir les scores de chaque analyse

### Erreurs d'API
- Vérifiez que les clés API sont valides
- Certaines APIs ont des limites de requêtes horaires/quotidiennes
- Le bot utilise un système de rotation automatique des clés

## Support

Pour toute question ou problème, consultez les logs dans `trading_bot.log` ou contactez le développeur.

---

**Bon trading ! 🚀📈**
