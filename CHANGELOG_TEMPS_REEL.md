# Changelog - Branche Temps-Reel

## Nouvelles Fonctionnalités

### 1. 🕐 Respect des Horaires de Marché

Le bot ne trade plus que pendant les heures d'ouverture des marchés :

#### Marchés US (NVDA, AAPL, TSLA, etc.)
- **Ouverture** : 15:30 (heure française)
- **Fermeture** : 22:00 (heure française)
- **Dernière analyse** : 21:45 (pour avoir le temps de trader avant la fermeture)

#### Marchés France (MC.PA, OR.PA, AIR.PA, etc.)
- **Ouverture** : 09:00 (heure française)
- **Fermeture** : 17:30 (heure française)
- **Dernière analyse** : 17:15 (pour avoir le temps de trader avant la fermeture)

#### Week-end
- ❌ Pas de trading le samedi et dimanche (marchés fermés)

**Nouveau module** : `utils/market_hours.py`
- Fonction `MarketHours.can_trade_now(symbol)` pour vérifier si on peut trader
- Fonction `MarketHours.is_market_open(symbol)` pour vérifier si le marché est ouvert
- Gestion automatique du timezone Paris
- Distinction automatique entre marchés US et France basée sur le ticker (.PA pour France)

---

### 2. 💰 Commande `/cash` - Gestion du Cash Disponible

Nouvelle commande Discord pour gérer le cash disponible pour le trading :

#### Utilisation
```
!cash              # Affiche le cash actuel
!cash 5000         # Définit 5000€ comme cash disponible
```

#### Fonctionnalités
- Affiche le cash pool global
- Liste tous les participants avec leur cash individuel
- Met à jour le montant disponible pour les prochains trades
- Sauvegarde automatique dans `participants.json`

---

### 3. 👥 Commande `/participer` - Notifications aux Participants

Nouvelle commande Discord pour notifier tous les participants avec les détails des positions :

#### Utilisation
```
!participer
```

#### Fonctionnalités
- ✅ **Ping automatique** de tous les participants enregistrés
- 📊 **Positions actuelles** avec :
  - Nom complet de l'action (ex: "NVIDIA Corporation" au lieu de "NVDA")
  - Quantité d'actions
  - Prix moyen d'achat
  - Prix actuel
  - Valeur totale de la position
  - Profit/Perte en pourcentage
- 💰 **Cash disponible** pour chaque participant
- 📈 **Valeur totale** du portefeuille de chaque participant
- 💸 **Profit total** réalisé par chaque participant

**Nouveau module** : `trading/participants.py`
- Gestion des participants et de leurs montants
- Calcul des allocations proportionnelles
- Enregistrement des trades manuels (buy/sell)
- Sauvegarde automatique dans `participants.json`

---

### 4. 📝 Noms Complets des Actions

Toutes les notifications Discord affichent maintenant les **noms complets** des actions au lieu des symboles :

#### Avant
```
🟢 ACHAT: NVDA
```

#### Maintenant
```
🟢 ACHAT: NVIDIA Corporation
Ticker: NVDA
```

#### Exemples de Noms Complets

**Actions US :**
- NVDA → NVIDIA Corporation
- AAPL → Apple Inc.
- TSLA → Tesla Inc.
- META → Meta Platforms Inc. (Facebook)

**Actions Françaises :**
- MC.PA → LVMH Moët Hennessy Louis Vuitton
- OR.PA → L'Oréal S.A.
- AIR.PA → Airbus SE
- SAN.PA → Sanofi S.A.

**Nouveau module** : `utils/stock_info.py`
- Dictionnaire complet des noms d'actions
- Secteurs d'activité
- Fonctions pour récupérer le nom complet, le secteur, le marché

---

## Fichiers Modifiés

### Nouveaux Fichiers
1. `utils/market_hours.py` - Gestion des horaires de marché
2. `utils/stock_info.py` - Informations complètes sur les actions
3. `trading/participants.py` - Gestion des participants
4. `participants.json` - Sauvegarde des participants (créé automatiquement)

### Fichiers Modifiés
1. `bot/discord_bot.py` - Ajout des commandes `/cash` et `/participer`
2. `trading/live_trader.py` - Vérification horaires + noms complets dans notifications
3. `utils/__init__.py` - Export des nouveaux modules
4. `bot/discord_bot.py` (aide) - Documentation des nouvelles commandes

---

## Architecture des Nouveaux Modules

### utils/market_hours.py

```python
from utils import MarketHours

# Vérifier si on peut trader maintenant
can_trade, reason = MarketHours.can_trade_now('NVDA')
# Résultat : (False, "Marché US fermé (fermeture à 22:00)")

# Vérifier si le marché est ouvert
is_open, reason = MarketHours.is_market_open('MC.PA')
# Résultat : (True, "Marché français ouvert")
```

### utils/stock_info.py

```python
from utils import StockInfo

# Nom complet
name = StockInfo.get_full_name('NVDA')
# Résultat : "NVIDIA Corporation"

# Secteur
sector = StockInfo.get_sector('NVDA')
# Résultat : "Technology - Semiconductors"

# Marché
market = StockInfo.get_market('MC.PA')
# Résultat : "France"

# Toutes les infos
info = StockInfo.get_stock_info('NVDA')
# Résultat : {
#   'symbol': 'NVDA',
#   'full_name': 'NVIDIA Corporation',
#   'sector': 'Technology - Semiconductors',
#   'market': 'US',
#   'display_name': 'NVIDIA Corporation (NVDA)',
#   'search_name': 'NVIDIA Corporation'
# }
```

### trading/participants.py

```python
from trading.participants import ParticipantsManager

# Créer le gestionnaire
manager = ParticipantsManager()

# Ajouter un participant
manager.add_participant(user_id=123456, username="JohnDoe", initial_cash=5000.0)

# Mettre à jour le cash
manager.update_cash(user_id=123456, amount=7000.0)

# Définir le cash pool global
manager.set_cash_pool(10000.0)

# Récupérer les allocations
allocations = manager.get_participant_allocations(symbol='NVDA', suggested_amount=3000.0)
```

---

## Commandes Discord Mises à Jour

### Commandes de Trading en Temps Réel

| Commande | Description |
|----------|-------------|
| `!start [jours]` | Démarre le bot en mode trading simulé |
| `!stop` | Arrête le bot |
| `!status` | Affiche le statut et les performances |
| `!cash [montant]` | **NOUVEAU** - Gère le cash disponible |
| `!participer` | **NOUVEAU** - Ping participants + positions |

### Commandes de Backtest

| Commande | Description |
|----------|-------------|
| `!backtest [mois]` | Backtest sur N mois |
| `!detail [SYMBOL] [mois]` | Backtest détaillé d'une action |

### Commande d'Aide

| Commande | Description |
|----------|-------------|
| `!aide` | Affiche l'aide complète (mise à jour avec nouvelles fonctionnalités) |

---

## Comportement du Bot

### Avant (Simul-Temps-Reel)
- ❌ Tradait 24/7 sans vérifier les horaires
- ❌ Affichait uniquement les tickers (NVDA, AAPL, etc.)
- ❌ Pas de gestion de cash multi-utilisateurs
- ❌ Pas de notifications aux participants

### Maintenant (Temps-Reel)
- ✅ Respect strict des horaires de marché (US/France)
- ✅ Affichage des noms complets dans toutes les notifications
- ✅ Gestion du cash avec commande `/cash`
- ✅ Notifications aux participants avec `/participer`
- ✅ Détails complets de chaque position (P/L, valeur, prix)
- ✅ Trading automatique uniquement pendant les heures d'ouverture
- ✅ Log clair quand une action est skippée (marché fermé)

---

## Migration depuis Simul-Temps-Reel

Pour migrer depuis la branche `Simul-Temps-Reel` :

1. **Checkout la nouvelle branche**
   ```bash
   git checkout Temps-Reel
   ```

2. **Installer les dépendances** (si nouvelles)
   ```bash
   pip install pytz
   ```

3. **Tester les imports**
   ```bash
   python -c "from utils import MarketHours, StockInfo; print('OK')"
   python -c "from trading.participants import ParticipantsManager; print('OK')"
   ```

4. **Démarrer le bot**
   ```bash
   python main.py
   ```

---

## Fichiers de Configuration

### participants.json (créé automatiquement)
```json
{
  "participants": {
    "123456789": {
      "username": "User1",
      "cash": 5000.0,
      "positions": {},
      "total_invested": 0.0,
      "total_profit": 0.0
    }
  },
  "current_cash_pool": 10000.0
}
```

---

## Notes Importantes

1. **Horaires en heure française** : Tous les horaires sont en heure de Paris (UTC+1 en hiver, UTC+2 en été)
2. **Pas de trading le week-end** : Le bot skip automatiquement samedi/dimanche
3. **Marge de sécurité** : Les dernières analyses sont 15 minutes avant la fermeture pour avoir le temps de trader
4. **Noms complets partout** : Discord notifications, logs, commandes - tout affiche maintenant les noms complets
5. **Gestion participants** : Le fichier `participants.json` est sauvegardé automatiquement à chaque modification

---

## Tests Effectués

- ✅ Import des nouveaux modules (MarketHours, StockInfo, ParticipantsManager)
- ✅ Vérification horaires de marché (détecte correctement week-end)
- ✅ Récupération noms complets (NVDA → "NVIDIA Corporation")
- ✅ Import du bot Discord sans erreurs
- ✅ Toutes les fonctionnalités existantes préservées

---

## Prochaines Étapes (Suggestions)

1. ⏰ **Commande `/horaires`** pour afficher les horaires d'ouverture/fermeture en temps réel
2. 👤 **Commande `/participant add`** pour ajouter des participants directement depuis Discord
3. 📊 **Graphiques** des performances de chaque participant
4. 🔔 **Notifications personnalisées** pour chaque participant sur leurs trades
5. 🌍 **Support d'autres marchés** (Asie, etc.)
