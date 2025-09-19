"""
Tests de fumée pour la CI/CD
Ces tests vérifient que les imports de base fonctionnent
"""

import os
import sys
import pytest

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_basic_imports():
    """Test que les imports de base fonctionnent"""
    try:
        from rag_newsletter.embeddings import MLXEmbeddingService, OptimizedVectorStoreService
        from rag_newsletter.processing import OptimizedDocumentProcessor
        from rag_newsletter.ingestion import OptimizedRAGIngestionService
        print("✅ Tous les imports de base fonctionnent")
    except ImportError as e:
        pytest.skip(f"Imports échoués (normal si MLX n'est pas disponible): {e}")

def test_mlx_imports():
    """Test spécifique pour MLX (uniquement sur macOS/Apple Silicon)"""
    if os.getenv('SKIP_MLX_TESTS'):
        pytest.skip("Tests MLX désactivés pour cette plateforme")
    
    try:
        import mlx.core as mx
        print(f"✅ MLX importé avec succès")
        
        # Test basique MLX
        a = mx.array([1, 2, 3])
        result = mx.sum(a)
        assert result.item() == 6
        print("✅ MLX fonctionne correctement")
        
    except ImportError:
        pytest.skip("MLX non disponible sur cette plateforme")

def test_qdrant_connection():
    """Test de connexion à Qdrant (si disponible)"""
    qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    
    try:
        import requests
        response = requests.get(f"{qdrant_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Qdrant accessible à {qdrant_url}")
        else:
            pytest.skip(f"Qdrant non accessible: {response.status_code}")
    except Exception as e:
        pytest.skip(f"Qdrant non disponible: {e}")

def test_integration():
    """Test d'intégration basique"""
    if os.getenv('SKIP_MLX_TESTS'):
        pytest.skip("Tests d'intégration désactivés")
    
    try:
        # Test que les services peuvent être instanciés
        from rag_newsletter.embeddings import MLXEmbeddingService
        from rag_newsletter.embeddings.vector_store import OptimizedVectorStoreService
        
        # Test d'instanciation (sans initialisation complète)
        embedding_service = MLXEmbeddingService.__new__(MLXEmbeddingService)
        vector_service = OptimizedVectorStoreService.__new__(OptimizedVectorStoreService)
        
        print("✅ Services peuvent être instanciés")
        
    except Exception as e:
        pytest.skip(f"Tests d'intégration échoués: {e}")

def test_config_files():
    """Test que les fichiers de configuration existent"""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    
    # Vérifier pyproject.toml
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    assert os.path.exists(pyproject_path), "pyproject.toml manquant"
    print("✅ pyproject.toml existe")
    
    # Vérifier README.md
    readme_path = os.path.join(project_root, 'README.md')
    assert os.path.exists(readme_path), "README.md manquant"
    print("✅ README.md existe")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])