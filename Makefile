# Makefile pour RAG Newsletter
.PHONY: help install test lint format clean docker-build docker-run

help: ## Afficher l'aide
	@echo "RAG Newsletter - Commandes disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installer les dépendances
	poetry install --with dev

test: ## Lancer les tests
	poetry run pytest src/rag_newsletter/tests/ -v

test-smoke: ## Lancer les tests de fumée
	poetry run pytest src/rag_newsletter/tests/test_ci_smoke.py -v

test-local: ## Lancer tous les tests locaux (comme le CI)
	./scripts/test-local.sh

lint: ## Vérifier le code avec flake8
	poetry run flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127

format: ## Formater le code avec black et isort
	poetry run black src/
	poetry run isort src/

format-check: ## Vérifier le formatage sans modifier
	poetry run black --check src/
	poetry run isort --check-only src/

type-check: ## Vérifier les types avec mypy
	poetry run mypy src/ --ignore-missing-imports

clean: ## Nettoyer les fichiers temporaires
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf dist/ build/ *.egg-info/

ingest: ## Lancer l'ingestion des documents
	PYTHONPATH=src poetry run python -m rag_newsletter --ingest --outdir downloads/

search: ## Lancer une recherche (remplacer QUERY par votre requête)
	@echo "Usage: make search QUERY='votre requête'"
	@if [ -z "$(QUERY)" ]; then echo "❌ Veuillez spécifier QUERY"; exit 1; fi
	PYTHONPATH=src poetry run python -m rag_newsletter --search "$(QUERY)"

docker-build: ## Construire l'image Docker
	docker build -f src/rag_newsletter/infra/Dockerfile -t rag-newsletter .

docker-run: ## Lancer le conteneur Docker
	docker-compose -f src/rag_newsletter/infra/docker-compose.yml up -d

docker-stop: ## Arrêter les conteneurs Docker
	docker-compose -f src/rag_newsletter/infra/docker-compose.yml down

dev: ## Mode développement (lancer tous les services)
	docker-compose -f src/rag_newsletter/infra/docker-compose.yml up -d qdrant
	@echo "✅ Qdrant démarré. Vous pouvez maintenant lancer l'ingestion."

# Commandes de test spécifiques
test-mlx: ## Tests MLX (macOS uniquement)
	poetry run pytest src/rag_newsletter/tests/ -v -k "mlx"

test-integration: ## Tests d'intégration
	poetry run pytest src/rag_newsletter/tests/ -v -k "integration"

# Commandes de déploiement
pre-commit: ## Préparation avant commit
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test-smoke

ci-local: ## Simuler le CI localement
	$(MAKE) test-local

# Informations sur le projet
info: ## Afficher les informations du projet
	@echo "📊 RAG Newsletter - Informations:"
	@echo "  🐍 Python: $(shell poetry run python --version)"
	@echo "  📦 Poetry: $(shell poetry --version)"
	@echo "  📁 Projet: $(shell pwd)"
	@echo "  🔧 MLX disponible: $(shell poetry run python -c 'import mlx.core; print("✅")' 2>/dev/null || echo "❌")"
	@echo "  🗄️  Qdrant: $(shell curl -s http://localhost:6333/health >/dev/null && echo "✅" || echo "❌")"
