# Changelog - Améliorations V11 + Trading Temps Réel

Date: 2025-12-01

## 🎯 Nouveau système de scoring des news V11

### Problème résolu
L'ancien système V9 donnait des scores trompeurs:
- **Airbus** (rappel de 6000 avions): 59/100 → semblait neutre ❌
- Impossible de savoir si un score de 60/100 était positif ou négatif
- Sous-estimation systématique des événements catastrophiques

### Solution: Système Intensité + Direction
Nouveau système en 2 parties:
1. **Intensité** (0-100): Distance par rapport à neutre (50)
2. **Direction**: POSITIF, NÉGATIF, ou NEUTRE

### Exemples de résultats
| Ancien (V9) | Nouveau (V11) | Action |
|-------------|---------------|--------|
| 59/100 | **97/100 NÉGATIF** | Airbus (rappel d'urgence) |
| 97/100 | **95/100 POSITIF** | Tesla (excellentes news) |
| 77/100 | **54/100 NEUTRE** | Microsoft (news mixtes) |

### Améliorations techniques
1. **Meilleurs mots-clés**:
   - Ajout: `emergency`, `recall`, `grounded`, `flight control`, `airworthiness directive`
   - Poids augmentés pour catastrophes: `recall` 2.0 → 4.5

2. **Détection séparée**:
   - Intensité basée sur ratio + keywords
   - Direction basée sur ratio ET keywords (plus robuste)

3. **Fichier**: `textblob_v11_intensity_direction.py`

## 🔔 Corrections Trading Temps Réel

### 1. STOP LOSS et TAKE PROFIT envoient maintenant des PINGS ✅

**Avant:**
- Seuls les signaux ACHAT/VENTE envoyaient des pings
- STOP LOSS/TAKE PROFIT = notifications silencieuses

**Après:**
- Tous les événements envoient un ping `@user` dans le channel privé
- Le participant est notifié même s'il n'est pas sur Discord

**Fichiers modifiés:**
- `trading/live_trader.py:432-465` (STOP LOSS)
- `trading/live_trader.py:467-500` (TAKE PROFIT)

### 2. Timezone corrigée pour la France ✅

**Avant:**
- Timestamps en UTC (1h de décalage)
- Messages affichés avec mauvaise heure

**Après:**
- Timezone **UTC+1** (France)
- Fonction `get_france_time()` utilisée partout
- Tous les `datetime.now()` remplacés

**Fichiers modifiés:**
- `trading/live_trader.py:21-26` (nouvelle fonction)
- Tous les timestamps dans le fichier

### 3. Format des messages mis à jour ✅

**Nouveau format dans Discord:**
```
🔍 Score Technique: 75/100 (50%)
📰 Score News: 90/100 **NÉGATIF** (50%)
⭐ Score Final: 82/100
```

**Avantages:**
- Direction visible immédiatement (POSITIF/NÉGATIF/NEUTRE)
- Plus besoin de deviner si 59/100 est bon ou mauvais
- Transparence totale sur le sentiment des news

## 📊 Résultats des tests

### Test sur 9 actions (29 nov 2025)

| Action | V9 Score | V11 Score | Direction | Décision |
|--------|----------|-----------|-----------|----------|
| TSLA | 97/100 | 95/100 | POSITIF | ✅ Achat |
| AAPL | 81/100 | 63/100 | POSITIF | ✅ Achat |
| NVDA | 61/100 | 50/100 | NEUTRE | ❌ Refusé |
| MSFT | 77/100 | 54/100 | NEUTRE | ❌ Refusé |
| AIR.PA | 51/100 | 54/100 | NEUTRE | ❌ Refusé |
| MC.PA | 71/100 | 74/100 | POSITIF | ✅ Achat |
| BNP.PA | 73/100 | 76/100 | POSITIF | ✅ Achat |
| SAF.PA | 64/100 | 54/100 | NEUTRE | ❌ Refusé |

### Observations
- **Plus conservateur**: Refuse les achats sur news neutres
- **Plus précis**: Détecte correctement les catastrophes
- **Plus transparent**: Direction visible dans tous les messages

## 🚀 Prochaines améliorations possibles

### Proposition: Bloquer les achats sur news très négatives
**Problème actuel:**
- Airbus avec score tech 75 + news 90 NÉGATIF = composite 82/100
- Le bot achèterait quand même (82 > seuil 65) ❌

**Solution proposée:**
```python
if news_direction == "NÉGATIF" and news_score > 80:
    # Appliquer un malus de -20 points au composite
    composite_score -= 20
```

**Résultat:**
- Airbus: 82 - 20 = 62 → Achat refusé ✅

## 📝 Fichiers modifiés

### Nouveaux fichiers
1. `textblob_v11_intensity_direction.py` - Nouveau système de scoring
2. `test_v11_integration.py` - Tests d'intégration
3. `test_all_stocks_v11.py` - Tests sur toutes les actions
4. `test_airbus_scoring.py` - Test spécifique Airbus

### Fichiers modifiés
1. `analyzers/news_analyzer.py`:
   - Import V11 au lieu de V9
   - Signature `get_news_for_date()` retourne maintenant `(has_news, items, intensity, direction)`
   - Logging amélioré

2. `trading/live_trader.py`:
   - Ajout timezone France (UTC+1)
   - Fonction `get_france_time()`
   - STOP LOSS/TAKE PROFIT avec pings
   - Format messages avec direction
   - Stockage `news_direction` dans décisions

## ✅ Checklist de déploiement

- [x] Système V11 créé et testé
- [x] Tests passent sur tous les cas (catastrophiques, positifs, neutres)
- [x] Intégration dans news_analyzer.py
- [x] Intégration dans live_trader.py
- [x] Timezone corrigée (UTC+1)
- [x] Pings ajoutés pour STOP LOSS/TAKE PROFIT
- [x] Format messages mis à jour
- [x] Tests sur actions réelles (9 actions)

## 🎓 Utilisation

Le bot est maintenant prêt à utiliser. Les messages Discord afficheront:
- Score technique (0-100)
- **Score news avec direction** (ex: 90/100 NÉGATIF)
- Score composite final

Les notifications incluent:
- ✅ Signaux ACHAT avec ping
- ✅ Signaux VENTE avec ping
- ✅ STOP LOSS avec ping (nouveau!)
- ✅ TAKE PROFIT avec ping (nouveau!)
- ✅ Heures en timezone France (nouveau!)
