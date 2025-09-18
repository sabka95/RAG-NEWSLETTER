#!/usr/bin/env python3
"""
Test simple avec un seul document pour éviter les problèmes de mémoire
"""

import sys
import os
from pathlib import Path

# Ajouter le chemin du module
sys.path.append(str(Path(__file__).parent / "src"))

from rag_newsletter.ingestion.rag_ingestion import RAGIngestionService

def main():
    print("🧪 Test simple avec un seul document")
    print("=" * 50)
    
    # Utiliser le plus petit document
    test_file = "downloads/te_charte_hseq_en_09_21.pdf"
    
    if not os.path.exists(test_file):
        print(f"❌ Fichier de test non trouvé: {test_file}")
        return 1
    
    try:
        print(f"📄 Test avec: {test_file}")
        
        # Initialiser le service RAG
        rag_service = RAGIngestionService(
            model_name="marco/mcdse-2b-v1",
            dimension=1024,
            use_mlx=False,  # Désactiver MLX pour éviter les problèmes
            use_hnsw=True,
            use_binary_quantization=True,
            mmr_lambda=0.7
        )
        
        print("✅ Service RAG initialisé")
        
        # Ingérer un seul document
        result = rag_service.ingest_documents([test_file])
        
        print(f"📊 Résultat de l'ingestion:")
        print(f"  - Statut: {result['status']}")
        if result['status'] == 'success':
            print(f"  - Chunks: {result['total_chunks']}")
            print(f"  - Fichiers: {result['processed_files']}")
            print(f"  - Vecteurs: {result['vector_ids']}")
            
            # Test de recherche
            print("\n🔍 Test de recherche...")
            search_results = rag_service.search("safety", k=3)
            
            print(f"🎯 Résultats trouvés: {len(search_results)}")
            for i, result in enumerate(search_results, 1):
                print(f"\n--- Résultat {i} ---")
                print(f"Score: {result['score']:.3f}")
                print(f"Source: {result['metadata'].get('source_file', 'N/A')}")
                print(f"Page: {result['metadata'].get('page_number', 'N/A')}")
                print(f"Contenu: {result['content'][:100]}...")
            
            # Test des statistiques
            print("\n📊 Statistiques de la collection:")
            stats = rag_service.get_collection_stats()
            if stats.get('status') == 'success':
                coll_stats = stats['collection_stats']
                print(f"  - Points: {coll_stats.get('points_count', 'N/A')}")
                print(f"  - Indexés: {coll_stats.get('indexed_vectors_count', 'N/A')}")
                print(f"  - Statut: {coll_stats.get('status', 'N/A')}")
            
            print("\n✅ Test réussi !")
            return 0
        else:
            print(f"❌ Erreur: {result.get('message', 'Inconnue')}")
            return 1
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
