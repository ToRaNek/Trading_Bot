# Système de Rotation des Clés API NewsAPI

## Vue d'ensemble

Ce système permet de gérer automatiquement la rotation entre plusieurs clés API NewsAPI pour éviter les limitations de taux (rate limits). Lorsqu'une clé atteint sa limite, le système passe automatiquement à la clé suivante.

## Configuration

### 1. Fichier CSV des clés API

Créez ou modifiez le fichier `api_keys.csv` à la racine du dossier Trading_Bot:

```csv
newsapi_key
votre_cle_1_ici
votre_cle_2_ici
votre_cle_3_ici
```

**Important:**
- Le fichier doit avoir une colonne nommée exactement `newsapi_key`
- Une clé par ligne
- Les clés d'exemple (YOUR_NEWSAPI_KEY_X) sont automatiquement ignorées

### 2. Emplacement du fichier

Par défaut, le système cherche le fichier à:
```
Trading_Bot/api_keys.csv
```

Vous pouvez spécifier un chemin personnalisé:
```python
analyzer = HistoricalNewsAnalyzer(api_keys_csv_path='/chemin/vers/vos/cles.csv')
```

### 3. Fallback automatique

Si le fichier CSV n'existe pas ou ne contient aucune clé valide, le système utilise automatiquement la variable d'environnement `NEWSAPI_KEY` comme fallback.

## Fonctionnalités

### Rotation automatique

Le système gère automatiquement:
- ✅ Rotation entre les clés disponibles
- ✅ Détection des erreurs 429 (rate limit exceeded)
- ✅ Passage automatique à la clé suivante en cas de limite atteinte
- ✅ Marquage des clés épuisées pour éviter de les réutiliser
- ✅ Réinitialisation automatique quand toutes les clés sont épuisées

### Statistiques

Vous pouvez obtenir des statistiques sur l'état des clés:

```python
stats = analyzer.newsapi_rotator.get_stats()
print(f"Clés actives: {stats['active_keys']}/{stats['total_keys']}")
```

Retourne:
- `total_keys`: Nombre total de clés chargées
- `active_keys`: Nombre de clés encore utilisables
- `failed_keys`: Nombre de clés ayant atteint leur limite
- `current_index`: Index de la clé actuellement utilisée

## Utilisation

### Dans votre code

Le système est déjà intégré dans `HistoricalNewsAnalyzer`:

```python
from analyzers.news_analyzer import HistoricalNewsAnalyzer
from datetime import datetime

# Initialisation (utilise api_keys.csv par défaut)
analyzer = HistoricalNewsAnalyzer()

# Ou avec un chemin personnalisé
analyzer = HistoricalNewsAnalyzer(api_keys_csv_path='mon_fichier.csv')

# Utilisation normale
has_news, news_items, score = await analyzer.get_news_for_date('NVDA', datetime.now())
```

### Script de test

Un script de test complet est disponible:

```bash
python scripts/test_api_rotation.py
```

Ce script teste:
1. Les fonctionnalités de base du rotateur
2. L'intégration avec le news analyzer
3. Le comportement avec plusieurs requêtes

## Logs

Le système génère des logs détaillés:

```
[NewsAPI] Rotateur initialisé: 3/3 clés actives
[News] NVDA @ 2025-07-15: 45 news NewsAPI (clé 1)
[News] AAPL: NewsAPI clé 2 limite atteinte (429)
[APIKeyRotator] Clé 2 marquée comme épuisée
[APIKeyRotator] Rotation: clé 2 -> clé 3
[News] AAPL: Tentative 2/3 avec clé 3
```

## Avantages

1. **Augmentation de la capacité**: Multipliez votre limite de requêtes par le nombre de clés
2. **Résilience**: Continue de fonctionner même si certaines clés sont épuisées
3. **Transparence**: Logs détaillés pour suivre l'utilisation des clés
4. **Simplicité**: Aucune modification du code existant nécessaire

## Obtenir des clés NewsAPI

1. Créez un compte sur https://newsapi.org/
2. Obtenez votre clé API gratuite (100 requêtes/jour)
3. Pour plus de requêtes, créez plusieurs comptes ou souscrivez à un plan payant
4. Ajoutez toutes vos clés dans le fichier `api_keys.csv`

## Amélioration: pageSize augmenté

Le système récupère maintenant jusqu'à **100 articles** par requête (au lieu de 20), pour garder plus de news et réduire le nombre d'appels API.

```python
'pageSize': 100  # Augmenté pour garder plus de news
```

## Dépannage

### Le système utilise toujours NEWSAPI_KEY

- Vérifiez que `api_keys.csv` existe dans le bon dossier
- Vérifiez le format CSV (colonne `newsapi_key`)
- Regardez les logs au démarrage

### Toutes les clés sont épuisées rapidement

- Vérifiez vos limites de taux sur NewsAPI
- Utilisez le cache (déjà implémenté) pour éviter les requêtes répétées
- Ajoutez plus de clés dans le CSV
- Considérez un plan payant NewsAPI pour plus de requêtes

### Erreurs d'import

Si vous voyez des erreurs d'import de `APIKeyRotator`:
- Vérifiez que `utils/__init__.py` existe
- Vérifiez que le chemin d'import est correct
