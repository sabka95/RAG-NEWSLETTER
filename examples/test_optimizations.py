#!/usr/bin/env python3
"""
Script de test pour les optimisations RAG
Teste les performances avec différentes configurations sur Apple Silicon M4
"""

import time
import logging
from pathlib import Path
import sys

# Ajouter le chemin du module
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_newsletter.ingestion.rag_ingestion import RAGIngestionService
from rag_newsletter.configs.optimization_config import (
    M4_PERFORMANCE_CONFIG, 
    M4_BALANCED_CONFIG, 
    M4_MEMORY_CONFIG
)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_embedding_generation():
    """Test de performance de génération d'embeddings"""
    print("🚀 Test de performance - Génération d'embeddings")
    print("=" * 60)
    
    # Test avec différentes configurations
    configs = [
        ("Performance", M4_PERFORMANCE_CONFIG),
        ("Balanced", M4_BALANCED_CONFIG), 
        ("Memory Efficient", M4_MEMORY_CONFIG)
    ]
    
    for config_name, config in configs:
        print(f"\n📊 Configuration: {config_name}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            # Initialiser le service avec la configuration
            rag_service = RAGIngestionService(
                model_name=config.embedding.model_name,
                dimension=config.embedding.dimension,
                use_mlx=config.mlx.enabled,
                use_hnsw=config.hnsw.enabled,
                use_binary_quantization=config.binary_quantization.enabled,
                mmr_lambda=config.mmr.lambda_param
            )
            
            init_time = time.time() - start_time
            print(f"⏱️  Temps d'initialisation: {init_time:.2f}s")
            
            # Test de génération d'embedding pour une requête
            test_query = "sustainability and climate change initiatives"
            start_embedding = time.time()
            
            embedding = rag_service.embedding_service.embed_query(test_query)
            
            embedding_time = time.time() - start_embedding
            print(f"🔍 Temps d'embedding: {embedding_time:.3f}s")
            print(f"📏 Dimension: {len(embedding)}")
            print(f"🎯 MLX activé: {config.mlx.enabled}")
            print(f"🔢 Binary quantization: {config.binary_quantization.enabled}")
            
        except Exception as e:
            print(f"❌ Erreur avec {config_name}: {e}")

def benchmark_search_performance():
    """Test de performance de recherche"""
    print("\n🔍 Test de performance - Recherche vectorielle")
    print("=" * 60)
    
    try:
        # Utiliser la configuration équilibrée
        config = M4_BALANCED_CONFIG
        rag_service = RAGIngestionService(
            model_name=config.embedding.model_name,
            dimension=config.embedding.dimension,
            use_mlx=config.mlx.enabled,
            use_hnsw=config.hnsw.enabled,
            use_binary_quantization=config.binary_quantization.enabled,
            mmr_lambda=config.mmr.lambda_param
        )
        
        # Test de recherche avec MMR
        queries = [
            "sustainability goals and objectives",
            "financial performance and results", 
            "climate change mitigation strategies",
            "digital transformation initiatives",
            "employee engagement and development"
        ]
        
        print(f"🎯 Test avec {len(queries)} requêtes différentes")
        print("-" * 40)
        
        total_time = 0
        for i, query in enumerate(queries, 1):
            start_time = time.time()
            
            # Recherche avec MMR
            results = rag_service.search(query, k=5, use_mmr=True)
            
            search_time = time.time() - start_time
            total_time += search_time
            
            print(f"Query {i}: {search_time:.3f}s - {len(results)} résultats")
        
        avg_time = total_time / len(queries)
        print(f"\n📊 Temps moyen par requête: {avg_time:.3f}s")
        print(f"⚡ Requêtes par seconde: {1/avg_time:.1f}")
        
    except Exception as e:
        print(f"❌ Erreur lors du test de recherche: {e}")

def test_mmr_diversity():
    """Test de la diversité MMR"""
    print("\n🎯 Test de diversité MMR")
    print("=" * 60)
    
    try:
        config = M4_BALANCED_CONFIG
        rag_service = RAGIngestionService(
            model_name=config.embedding.model_name,
            dimension=config.embedding.dimension,
            use_mlx=config.mlx.enabled,
            use_hnsw=config.hnsw.enabled,
            use_binary_quantization=config.binary_quantization.enabled,
            mmr_lambda=config.mmr.lambda_param
        )
        
        query = "sustainability and environment"
        
        # Test avec différents paramètres lambda
        lambda_values = [0.0, 0.3, 0.7, 1.0]
        
        for lambda_val in lambda_values:
            print(f"\n🔧 Lambda = {lambda_val}")
            print("-" * 30)
            
            # Mettre à jour le lambda
            rag_service.mmr_lambda = lambda_val
            
            results = rag_service.search(query, k=5, use_mmr=True)
            
            # Analyser la diversité des sources
            sources = [r['metadata'].get('source_file', 'Unknown') for r in results]
            unique_sources = set(sources)
            
            print(f"📄 Sources trouvées: {len(unique_sources)}")
            print(f"📊 Diversité: {len(unique_sources)/len(results)*100:.1f}%")
            
            for j, result in enumerate(results, 1):
                source = result['metadata'].get('source_file', 'Unknown')
                score = result['score']
                print(f"  {j}. {source} (score: {score:.3f})")
                
    except Exception as e:
        print(f"❌ Erreur lors du test MMR: {e}")

def test_collection_stats():
    """Test des statistiques de collection"""
    print("\n📊 Test des statistiques de collection")
    print("=" * 60)
    
    try:
        config = M4_BALANCED_CONFIG
        rag_service = RAGIngestionService(
            model_name=config.embedding.model_name,
            dimension=config.embedding.dimension,
            use_mlx=config.mlx.enabled,
            use_hnsw=config.hnsw.enabled,
            use_binary_quantization=config.binary_quantization.enabled,
            mmr_lambda=config.mmr.lambda_param
        )
        
        stats = rag_service.get_collection_stats()
        
        if stats.get('status') == 'success':
            print("✅ Statistiques récupérées avec succès")
            
            coll_stats = stats['collection_stats']
            model_info = stats['model_info']
            search_config = stats['search_config']
            
            print(f"\n📈 Collection:")
            print(f"  - Vecteurs: {coll_stats.get('vectors_count', 'N/A')}")
            print(f"  - Points: {coll_stats.get('points_count', 'N/A')}")
            print(f"  - Indexés: {coll_stats.get('indexed_vectors_count', 'N/A')}")
            print(f"  - Statut: {coll_stats.get('status', 'N/A')}")
            
            print(f"\n🤖 Modèle:")
            print(f"  - Nom: {model_info.get('model_name', 'N/A')}")
            print(f"  - Dimension: {model_info.get('dimension', 'N/A')}")
            print(f"  - MLX: {model_info.get('use_mlx', 'N/A')}")
            print(f"  - Binary quantization: {model_info.get('binary_quantization', 'N/A')}")
            
            print(f"\n⚙️  Configuration:")
            print(f"  - MMR lambda: {search_config.get('mmr_lambda', 'N/A')}")
            print(f"  - HNSW: {search_config.get('use_hnsw', 'N/A')}")
            print(f"  - Binary quantization: {search_config.get('use_binary_quantization', 'N/A')}")
            
        else:
            print(f"❌ Erreur: {stats.get('message', 'Inconnue')}")
            
    except Exception as e:
        print(f"❌ Erreur lors du test des statistiques: {e}")

def main():
    """Fonction principale de test"""
    print("🧪 Tests d'optimisation RAG pour Apple Silicon M4")
    print("=" * 60)
    
    # Afficher la configuration par défaut
    print("🔧 Configuration par défaut:")
    M4_BALANCED_CONFIG.print_config()
    
    # Exécuter les tests
    try:
        benchmark_embedding_generation()
        benchmark_search_performance()
        test_mmr_diversity()
        test_collection_stats()
        
        print("\n✅ Tous les tests terminés avec succès!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()
