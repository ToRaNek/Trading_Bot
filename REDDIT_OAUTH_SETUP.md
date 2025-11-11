# Configuration Reddit OAuth pour Azure VPS

## Pourquoi?

Les datacenters (Azure, AWS, etc.) ont leurs IP bloquées par Reddit. L'authentification OAuth contourne ce blocage.

## Étapes

### 1. Créer une application Reddit

1. Va sur https://www.reddit.com/prefs/apps
2. Clique sur "create another app..." en bas
3. Remplis le formulaire:
   - **name**: `TradingBot` (ou un nom de ton choix)
   - **App type**: Sélectionne **"script"**
   - **description**: (optionnel) `Bot de trading automatique`
   - **about url**: (optionnel) laisse vide
   - **redirect uri**: Mets `http://localhost:8080`
4. Clique sur "create app"

### 2. Récupérer les credentials

Après création, tu verras:
```
personal use script
[CLIENT_ID]       <- Sous le nom de l'app (chaîne de ~14 caractères)

secret
[CLIENT_SECRET]   <- À côté de "secret" (chaîne de ~27 caractères)
```

### 3. Ajouter dans le fichier .env

Ouvre le fichier `.env` et remplis:

```env
REDDIT_CLIENT_ID=ton_client_id_ici
REDDIT_CLIENT_SECRET=ton_client_secret_ici
```

**Exemple:**
```env
REDDIT_CLIENT_ID=abc123DEF456ghi
REDDIT_CLIENT_SECRET=xyz789ABC123def456GHI789jkl
```

### 4. Redémarrer le bot

```bash
python main.py
```

## Vérification

Dans les logs, tu devrais voir:
```
[Reddit] ✅ Token OAuth obtenu avec succès
[Reddit] Session avec authentification OAuth
```

Au lieu de:
```
[Reddit] r/NVDA_Stock: Status 403
```

## Notes

- Les tokens OAuth sont **temporaires** (valides ~1h)
- Le bot les **renouvelle automatiquement**
- Fonctionne depuis n'importe quelle IP (y compris Azure)
- **Gratuit** et illimité (2000 requêtes / 10 min)

## Dépannage

### Erreur 401 Unauthorized
- Vérifie que les credentials sont corrects
- Assure-toi d'avoir choisi "script" comme type d'app

### Toujours Status 403
- Vérifie que le .env est dans le bon dossier
- Redémarre complètement le bot (pas juste !stop/!start)
- Vérifie les logs pour voir si le token est obtenu

### Pas de posts récupérés mais pas d'erreur 403
- Vérifie que les subreddits existent
- Certains subreddits peuvent être privés ou bannis
