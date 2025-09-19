#!/bin/bash

# Script d'installation pour RAG Newsletter Optimisé
# Optimisé pour Apple Silicon M4

set -e

echo "🚀 Installation de RAG Newsletter Optimisé"
echo "🍎 Optimisé pour Apple Silicon M4"
echo "=============================================="

# Vérifier le système
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Ce script est optimisé pour macOS"
    exit 1
fi

# Vérifier Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 requis"
    echo "📥 Installez avec: brew install python@3.11"
    exit 1
fi

echo "✅ Python 3.11 détecté"

# Vérifier Poetry
if ! command -v poetry &> /dev/null; then
    echo "📥 Installation de Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ Poetry disponible"

# Installation des dépendances
echo "📦 Installation des dépendances..."
poetry install

# Vérifier MLX
echo "🔍 Vérification de MLX..."
python3 -c "import mlx.core as mx; print('✅ MLX disponible')" || {
    echo "❌ MLX non disponible"
    echo "📥 Installation manuelle requise: pip install mlx mlx-lm"
    exit 1
}

# Vérifier PyTorch avec MPS
echo "🔍 Vérification de PyTorch MPS..."
python3 -c "import torch; print('✅ MPS disponible:', torch.backends.mps.is_available())" || {
    echo "❌ PyTorch MPS non disponible"
    exit 1
}

# Créer les répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p downloads logs

# Copier le fichier d'environnement
if [ ! -f .env ]; then
    echo "📝 Création du fichier .env..."
    cp env.example .env
    echo "⚠️  Configurez votre fichier .env avec vos credentials SharePoint"
fi

# Vérifier Docker pour Qdrant
echo "🔍 Vérification de Docker..."
if command -v docker &> /dev/null; then
    echo "✅ Docker disponible"
    
    # Démarrer Qdrant
    echo "🚀 Démarrage de Qdrant..."
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest || {
        echo "⚠️  Qdrant déjà en cours d'exécution ou erreur"
    }
    
    # Attendre que Qdrant soit prêt
    echo "⏳ Attente que Qdrant soit prêt..."
    sleep 5
    
    # Vérifier la santé de Qdrant
    if curl -s http://localhost:6333/health > /dev/null; then
        echo "✅ Qdrant opérationnel"
    else
        echo "⚠️  Qdrant non accessible"
    fi
else
    echo "⚠️  Docker non disponible - installez Docker pour utiliser Qdrant"
fi

# Test de démonstration
echo "🧪 Test de démonstration..."
if poetry run python demo.py > /dev/null 2>&1; then
    echo "✅ Démonstration réussie"
else
    echo "⚠️  Démonstration échouée - vérifiez les logs"
fi

echo ""
echo "🎉 Installation terminée!"
echo "=============================================="
echo "📖 Consultez le README.md pour l'utilisation"
echo "🚀 Commandes utiles:"
echo "   poetry run python -m rag_newsletter --help"
echo "   poetry run python demo.py"
echo "   poetry run python -m rag_newsletter --list-drives"
echo ""
echo "🔧 Configuration:"
echo "   Éditez le fichier .env avec vos credentials SharePoint"
echo "   Démarrez Qdrant: docker run -d -p 6333:6333 qdrant/qdrant:latest"
echo ""
echo "📊 Dashboard Qdrant: http://localhost:6333/dashboard"
