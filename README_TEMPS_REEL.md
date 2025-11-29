# 🚀 Trading Bot - Mode Temps Réel

## 🎯 Concept

Ce bot analyse le marché en temps réel et **envoie des signaux de trading** que les participants exécutent **manuellement** sur leur plateforme. Le bot ne peut pas trader automatiquement car ça nécessite une API payante, donc il sert de **système de signaux collaboratif**.

---

## 📋 Fonctionnement

### 1️⃣ **S'enregistrer comme participant**
```
!participer
```
- Tu rejoins le groupe de traders
- Tu recevras un **ping Discord** sur chaque signal
- Tu obtiens accès à `!cash` pour gérer ton capital

### 2️⃣ **Définir ton cash disponible**
```
!cash 5000
```
- Le bot saura que tu as 5000€ disponibles
- Il pourra te suggérer des montants adaptés
- Ton cash est sauvegardé automatiquement

### 3️⃣ **Démarrer le bot**
```
!start
```
- Le bot démarre et tourne **en continu**
- Il analyse les marchés pendant les horaires d'ouverture
- Il envoie des signaux quand les conditions sont bonnes

### 4️⃣ **Recevoir les signaux**

Quand le bot détecte une opportunité, **tous les participants sont pingés** avec un message comme :

```
@Participant1 @Participant2 @Participant3

🟢 SIGNAL ACHAT: NVIDIA Corporation

✅ Signal validé par l'IA | Ticker: NVDA
⚠️ Exécutez ce trade MANUELLEMENT sur votre plateforme

📌 Action: ACHETER NVIDIA Corporation
💰 Prix actuel: $875.42
📊 Quantité suggérée (bot): 1.1423
💵 Coût total (bot): $1000.00

🔍 Score Technique: 78/100 (50%)
📰 Score News: 72/100 (50%)
⭐ Score Final: 75/100

📝 Instructions:
1️⃣ Ouvrez votre plateforme de trading
2️⃣ Cherchez NVIDIA Corporation (ticker: NVDA)
3️⃣ Achetez selon votre cash disponible
4️⃣ Le bot garde trace de la position
```

### 5️⃣ **Exécuter manuellement**

- Tu vas sur **ta plateforme de trading** (Degiro, Trade Republic, etc.)
- Tu cherches l'action (NVDA dans l'exemple)
- Tu achètes avec ton propre argent
- Le bot garde une trace de **sa** position virtuelle pour les prochains signaux

---

## 🕐 Horaires de Trading

### Marchés US (NVDA, AAPL, TSLA, META, etc.)
- **Ouverture** : 15:30 (heure française)
- **Fermeture** : 22:00 (heure française)
- **Dernière analyse** : 21:45

### Marchés France (MC.PA, OR.PA, AIR.PA, etc.)
- **Ouverture** : 09:00 (heure française)
- **Fermeture** : 17:30 (heure française)
- **Dernière analyse** : 17:15

### Week-end
- ❌ **Pas de trading** samedi et dimanche
- Le bot ne fait rien pendant le week-end

---

## 💾 Persistance et Reprise

### Le bot peut s'éteindre sans problème

Si le bot s'éteint (crash, redémarrage serveur, etc.), **rien n'est perdu** :

1. **Positions sauvegardées** dans `portfolio_temps_reel.json`
2. **Participants sauvegardés** dans `participants.json`
3. Au redémarrage (`!start`), tout est restauré automatiquement

### Fichiers de sauvegarde

- `portfolio_temps_reel.json` → Positions du bot
- `participants.json` → Liste des participants + leur cash

Ces fichiers sont mis à jour automatiquement à chaque changement.

---

## 🤖 Signaux Envoyés

Le bot ping les participants pour :

### 🟢 Signal ACHAT
- Quand le score composite > 65/100
- Pendant les horaires de marché
- Avec le nom complet de l'action + instructions

### 🔴 Signal VENTE
- Quand le bot décide de vendre
- Ou quand Stop Loss / Take Profit atteint
- Avec profit/perte calculé

---

## 📊 Commandes Disponibles

### Trading
| Commande | Description |
|----------|-------------|
| `!participer` | S'enregistrer comme participant |
| `!cash [montant]` | Gérer son cash (participants uniquement) |
| `!start` | Démarrer le bot (en continu) |
| `!stop` | Arrêter le bot |
| `!status` | Voir les positions actuelles du bot |

### Backtest
| Commande | Description |
|----------|-------------|
| `!backtest [mois]` | Backtest historique |
| `!detail [SYMBOL] [mois]` | Backtest détaillé d'une action |

### Aide
| Commande | Description |
|----------|-------------|
| `!aide` | Guide complet |

---

## 🔍 Analyse Multi-Sources

Chaque signal est validé par **3 sources** :

### 1. Analyse Technique (50%)
- RSI, MACD, SMA, Bollinger, Volume
- Système de confluence
- Score 0-100

### 2. News IA (50%)
- Actualités du jour analysées par IA
- Sentiment positif/négatif
- Score 0-100

### 3. Horaires de Marché
- Vérifie que le marché est ouvert
- Skip automatiquement si fermé

**Score final = (Technique × 50%) + (News × 50%)**

Si score ≥ 65/100 → Signal envoyé ✅

---

## 📝 Exemple de Session

### Étape 1 : Configuration initiale
```discord
User1: !participer
Bot: 🎉 Participant Enregistré - Bienvenue User1 !

User1: !cash 5000
Bot: 💰 Cash Mis à Jour - Ton cash a été défini à $5000.00

User2: !participer
Bot: 🎉 Participant Enregistré - Bienvenue User2 !

User2: !cash 3000
Bot: 💰 Cash Mis à Jour - Ton cash a été défini à $3000.00
```

### Étape 2 : Démarrage
```discord
Admin: !start
Bot: 🚀 Démarrage du Bot en Temps Réel
     👥 Participants: 2
     📊 Watchlist: 40 actions
     ⏰ Analyses: Toutes les heures
```

### Étape 3 : Réception des signaux
```discord
Bot: @User1 @User2

     🟢 SIGNAL ACHAT: Apple Inc.

     ✅ Signal validé par l'IA | Ticker: AAPL
     ⚠️ Exécutez ce trade MANUELLEMENT

     💰 Prix actuel: $178.50
     ⭐ Score Final: 72/100

     📝 Instructions:
     1️⃣ Ouvrez votre plateforme
     2️⃣ Cherchez Apple Inc. (AAPL)
     3️⃣ Achetez selon votre cash
```

### Étape 4 : Exécution manuelle
- User1 achète 28 actions (28 × $178.50 = $4998)
- User2 achète 16 actions (16 × $178.50 = $2856)
- Le bot garde trace de SA position virtuelle

### Étape 5 : Signal de vente
```discord
Bot: @User1 @User2

     🔴 SIGNAL VENTE: Apple Inc.

     ✅ Signal validé par l'IA | Ticker: AAPL
     ⚠️ Exécutez ce trade MANUELLEMENT

     💰 Prix actuel: $185.20
     📈 Profit (bot): +$37.66 (+3.76%)

     📝 Instructions:
     1️⃣ Ouvrez votre plateforme
     2️⃣ Vendez votre position complète sur AAPL
```

---

## ⚡ Avantages

### Pour les participants
- ✅ **Gratuit** - Pas d'API payante nécessaire
- ✅ **Collaboratif** - Tout le monde reçoit les mêmes signaux
- ✅ **Flexible** - Tu choisis combien investir
- ✅ **Éducatif** - Tu apprends en voyant les analyses
- ✅ **Transparent** - Tous les scores sont affichés

### Pour le système
- ✅ **Persistant** - Résiste aux crashes
- ✅ **Intelligent** - Respect des horaires de marché
- ✅ **Multi-sources** - Tech + News + Horaires
- ✅ **Sauvegardé** - Tout est dans des fichiers JSON

---

## ⚠️ Important

### Ce que le bot FAIT
- ✅ Analyse les marchés en temps réel
- ✅ Envoie des signaux validés par IA
- ✅ Garde trace de ses positions virtuelles
- ✅ Ping tous les participants
- ✅ Respecte les horaires de marché

### Ce que le bot NE FAIT PAS
- ❌ N'exécute PAS les trades automatiquement
- ❌ N'a PAS accès à votre plateforme
- ❌ Ne gère PAS votre argent réel
- ❌ Ne garantit PAS les profits

**Tu es responsable de tes propres trades et de ton propre argent.**

---

## 🐛 Résolution de Problèmes

### Le bot ne répond pas
```
1. Vérifier qu'il est démarré (!status)
2. Redémarrer avec !stop puis !start
```

### Pas de signaux reçus
```
1. Vérifier que tu es participant (!participer)
2. Vérifier les horaires de marché (US: 15:30-22:00, FR: 09:00-17:30)
3. Le bot analyse à :30 de chaque heure
```

### Positions perdues après crash
```
1. Vérifier que portfolio_temps_reel.json existe
2. Le bot restaure automatiquement au !start
3. Si problème, vérifier les logs
```

---

## 📂 Structure des Fichiers

```
Trading_Bot/
├── portfolio_temps_reel.json    # Positions du bot
├── participants.json             # Participants + cash
├── trading_bot.log              # Logs détaillés
├── bot/
│   └── discord_bot.py           # Commandes Discord
├── trading/
│   ├── live_trader.py           # Logique de trading
│   ├── portfolio.py             # Gestion portfolio
│   └── participants.py          # Gestion participants
└── utils/
    ├── market_hours.py          # Horaires de marché
    └── stock_info.py            # Noms complets
```

---

## 🚀 Prochaines Améliorations Possibles

1. 📊 Commande `!stats` pour voir les performances de chaque participant
2. 📈 Graphiques des profits/pertes
3. 🔔 Notifications push personnalisées
4. 📱 Intégration Telegram en plus de Discord
5. 🌍 Support d'autres marchés (Asie, etc.)

---

## 📞 Support

Pour toute question :
- Utilisez `!aide` dans Discord
- Consultez les logs dans `trading_bot.log`
- Vérifiez les fichiers de sauvegarde `.json`
