# CI/CD - RAG Newsletter

Ce document décrit la configuration CI/CD pour le projet RAG Newsletter avec MLX et Apple Silicon.

## 🚀 Pipeline GitHub Actions

### Jobs de la CI/CD

1. **test-ubuntu** : Tests sur Ubuntu (compatibilité générale)
2. **test-macos** : Tests sur macOS avec MLX (Apple Silicon)
3. **integration-test** : Tests d'intégration avec Qdrant
4. **docker** : Build et push Docker (optionnel)

### Déclencheurs

- **Push** sur `main` et `develop`
- **Pull Request** vers `main`

## 🧪 Tests

### Tests Ubuntu
- ✅ Linting (flake8, black, isort)
- ✅ Tests unitaires (sans MLX)
- ✅ Vérifications de type (mypy)

### Tests macOS (Apple Silicon)
- ✅ Compatibilité MLX
- ✅ Tests avec MLXEmbeddingService
- ✅ Tests de base

### Tests d'intégration
- ✅ Connexion Qdrant
- ✅ Tests avec services externes
- ✅ Tests d'intégration

## 🛠️ Commandes locales

### Pré-commit
```bash
make pre-commit  # Formatage + linting + tests de fumée
```

### Tests locaux complets
```bash
make test-local  # Simule le CI localement
```

### Tests spécifiques
```bash
make test-smoke      # Tests de fumée
make test-mlx        # Tests MLX (macOS)
make test-integration # Tests d'intégration
```

## 📦 Dépendances

### Production
- `mlx` et `mlx-lm` (Apple Silicon uniquement)
- `torch` et `torchvision`
- `transformers`
- `qdrant-client`
- `pymupdf`, `pillow`

### Développement
- `pytest`, `pytest-asyncio`
- `black`, `isort`, `flake8`
- `mypy`

## 🔧 Configuration

### Variables d'environnement

```bash
# Tests
SKIP_MLX_TESTS=true          # Désactiver les tests MLX
QDRANT_URL=http://localhost:6333  # URL Qdrant

# Docker (optionnel)
REGISTRY_URL=your-registry.com
REGISTRY_USER=username
REGISTRY_PASSWORD=password
```

### Secrets GitHub (optionnel)

Pour le build Docker :
- `REGISTRY_URL`
- `REGISTRY_USER` 
- `REGISTRY_PASSWORD`

## 🐳 Docker

### Build local
```bash
make docker-build
```

### Démarrage services
```bash
make docker-run    # Démarrer Qdrant
make docker-stop   # Arrêter les services
```

## 📋 Checklist avant commit

- [ ] `make format-check` ✅
- [ ] `make lint` ✅
- [ ] `make test-smoke` ✅
- [ ] `make info` (vérifier l'environnement)

## 🔍 Debugging CI/CD

### Logs GitHub Actions
- Vérifier les logs dans l'onglet "Actions" de GitHub
- Les tests MLX ne s'exécutent que sur macOS
- Les tests d'intégration nécessitent Qdrant

### Tests locaux
```bash
# Simuler l'environnement CI
export SKIP_MLX_TESTS=true
make test-local
```

### Problèmes courants

1. **MLX non disponible sur Ubuntu**
   - Normal, MLX est uniquement pour Apple Silicon
   - Les tests MLX sont automatiquement skippés

2. **Qdrant non accessible**
   - Vérifier que Qdrant est démarré
   - Tests d'intégration skippés si Qdrant indisponible

3. **Tests de formatage échouent**
   - Exécuter `make format` pour corriger
   - Vérifier avec `make format-check`

## 🎯 Résultats attendus

### ✅ Succès
- Tous les tests passent (ou sont skippés selon la plateforme)
- Formatage et linting OK
- Build Docker réussi (si configuré)

### ❌ Échec
- Erreurs de syntaxe Python
- Tests critiques échouent
- Problèmes de formatage non résolus

## 📚 Ressources

- [GitHub Actions](https://docs.github.com/en/actions)
- [Poetry](https://python-poetry.org/docs/)
- [MLX](https://ml-explore.github.io/mlx/)
- [Qdrant](https://qdrant.tech/documentation/)
