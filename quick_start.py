#!/usr/bin/env python3
"""
Script de démarrage rapide pour tester les optimisations RAG
Utilisation: python quick_start.py
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    print("🔍 Vérification des dépendances...")
    
    # Vérifier Python
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ requis")
        return False
    
    print(f"✅ Python {sys.version}")
    
    # Vérifier Poetry
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        print(f"✅ Poetry {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Poetry non installé")
        return False
    
    # Vérifier Docker
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        print(f"✅ Docker {result.stdout.split()[2]}")
    except FileNotFoundError:
        print("❌ Docker non installé")
        return False
    
    return True

def setup_environment():
    """Configure l'environnement"""
    print("\n🔧 Configuration de l'environnement...")
    
    # Créer le fichier .env s'il n'existe pas
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path("env.example")
        if env_example.exists():
            print("📝 Création du fichier .env à partir de env.example")
            env_file.write_text(env_example.read_text())
        else:
            print("⚠️  Fichier .env manquant, veuillez le créer manuellement")
    
    # Installer les dépendances
    print("📦 Installation des dépendances...")
    try:
        subprocess.run(["poetry", "install"], check=True)
        print("✅ Dépendances installées")
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation des dépendances")
        return False
    
    return True

def start_qdrant():
    """Démarre Qdrant"""
    print("\n🚀 Démarrage de Qdrant...")
    
    try:
        # Vérifier si Qdrant est déjà en cours d'exécution
        result = subprocess.run(
            ["curl", "-s", "http://localhost:6333/health"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Qdrant déjà en cours d'exécution")
            return True
        
        # Démarrer Qdrant avec Docker Compose
        print("🐳 Démarrage de Qdrant avec Docker Compose...")
        subprocess.run([
            "docker-compose", "-f", "src/rag_newsletter/infra/docker-compose.yml", 
            "up", "-d", "qdrant"
        ], check=True)
        
        print("✅ Qdrant démarré")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Erreur lors du démarrage de Qdrant")
        return False
    except FileNotFoundError:
        print("❌ Docker Compose non trouvé")
        return False

def test_basic_functionality():
    """Test des fonctionnalités de base"""
    print("\n🧪 Test des fonctionnalités de base...")
    
    try:
        # Test de la configuration
        print("🔧 Test de la configuration...")
        result = subprocess.run([
            "python", "-m", "rag_newsletter", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Module RAG fonctionnel")
        else:
            print("❌ Erreur dans le module RAG")
            return False
        
        # Test des optimisations
        print("🚀 Test des optimisations...")
        result = subprocess.run([
            "python", "-m", "rag_newsletter", "--search", "test", "--no-mlx"
        ], capture_output=True, text=True)
        
        print("✅ Test des optimisations terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def run_example():
    """Lance un exemple complet"""
    print("\n📚 Exemple d'utilisation...")
    
    # Exemple avec les optimisations
    print("🎯 Exemple avec optimisations M4:")
    print("python -m rag_newsletter --search 'sustainability goals' --mmr-lambda 0.7")
    print("\n🔍 Exemple avec recherche filtrée:")
    print("python -m rag_newsletter --search 'climate change' --no-mmr")
    print("\n📊 Test des performances:")
    print("python examples/test_optimizations.py")

def main():
    """Fonction principale"""
    print("🚀 Démarrage rapide RAG Newsletter avec optimisations Apple Silicon")
    print("=" * 70)
    
    # Vérifier les dépendances
    if not check_dependencies():
        print("\n❌ Dépendances manquantes. Veuillez les installer avant de continuer.")
        return 1
    
    # Configurer l'environnement
    if not setup_environment():
        print("\n❌ Erreur lors de la configuration de l'environnement.")
        return 1
    
    # Démarrer Qdrant
    if not start_qdrant():
        print("\n❌ Erreur lors du démarrage de Qdrant.")
        return 1
    
    # Tester les fonctionnalités
    if not test_basic_functionality():
        print("\n❌ Erreur lors du test des fonctionnalités.")
        return 1
    
    # Afficher les exemples
    run_example()
    
    print("\n✅ Démarrage rapide terminé avec succès!")
    print("\n📖 Prochaines étapes:")
    print("1. Configurez vos identifiants Azure dans le fichier .env")
    print("2. Téléchargez des documents: python -m rag_newsletter --download")
    print("3. Ingérez les documents: python -m rag_newsletter --ingest")
    print("4. Testez la recherche: python -m rag_newsletter --search 'votre requête'")
    print("5. Lancez les tests de performance: python examples/test_optimizations.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
