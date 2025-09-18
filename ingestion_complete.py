#!/usr/bin/env python3
"""
Script d'ingestion complète optimisé pour éviter les problèmes de mémoire
Traite les documents un par un avec gestion d'erreurs
"""

import sys
import os
from pathlib import Path
import logging

# Ajouter le chemin du module
sys.path.append(str(Path(__file__).parent / "src"))

from rag_newsletter.ingestion.rag_ingestion import RAGIngestionService

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("🚀 Ingestion complète optimisée")
    print("=" * 50)
    
    # Lister tous les fichiers PDF
    downloads_dir = Path("downloads")
    pdf_files = list(downloads_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ Aucun fichier PDF trouvé dans downloads/")
        return 1
    
    print(f"📄 Fichiers trouvés: {len(pdf_files)}")
    for i, file in enumerate(pdf_files, 1):
        print(f"  {i}. {file.name}")
    
    try:
        # Initialiser le service RAG une seule fois
        print("\n🔧 Initialisation du service RAG...")
        rag_service = RAGIngestionService(
            model_name="marco/mcdse-2b-v1",
            dimension=1024,
            use_mlx=False,  # Désactiver MLX pour éviter les problèmes
            use_hnsw=True,
            use_binary_quantization=True,
            mmr_lambda=0.7
        )
        print("✅ Service RAG initialisé")
        
        # Traiter chaque fichier individuellement
        total_chunks = 0
        successful_files = 0
        failed_files = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n📄 [{i}/{len(pdf_files)}] Traitement: {pdf_file.name}")
            
            try:
                # Ingérer un seul fichier
                result = rag_service.ingest_documents([str(pdf_file)])
                
                if result['status'] == 'success':
                    chunks = result['total_chunks']
                    total_chunks += chunks
                    successful_files += 1
                    print(f"✅ Succès: {chunks} chunks ajoutés")
                else:
                    print(f"❌ Erreur: {result.get('message', 'Inconnue')}")
                    failed_files.append(pdf_file.name)
                    
            except Exception as e:
                print(f"❌ Exception lors du traitement: {e}")
                failed_files.append(pdf_file.name)
                continue
        
        # Résumé final
        print(f"\n📊 Résumé de l'ingestion:")
        print(f"  - Fichiers traités avec succès: {successful_files}/{len(pdf_files)}")
        print(f"  - Total des chunks: {total_chunks}")
        print(f"  - Fichiers en erreur: {len(failed_files)}")
        
        if failed_files:
            print(f"\n❌ Fichiers en erreur:")
            for file in failed_files:
                print(f"  - {file}")
        
        # Test de recherche
        print(f"\n🔍 Test de recherche...")
        search_results = rag_service.search("sustainability", k=5)
        
        print(f"🎯 Résultats trouvés: {len(search_results)}")
        for i, result in enumerate(search_results, 1):
            print(f"\n--- Résultat {i} ---")
            print(f"Score: {result['score']:.3f}")
            print(f"Source: {result['metadata'].get('source_file', 'N/A')}")
            print(f"Page: {result['metadata'].get('page_number', 'N/A')}")
            print(f"Contenu: {result['content'][:100]}...")
        
        # Statistiques finales
        print(f"\n📊 Statistiques finales:")
        stats = rag_service.get_collection_stats()
        if stats.get('status') == 'success':
            coll_stats = stats['collection_stats']
            print(f"  - Points dans la collection: {coll_stats.get('points_count', 'N/A')}")
            print(f"  - Statut: {coll_stats.get('status', 'N/A')}")
        
        if successful_files == len(pdf_files):
            print(f"\n🎉 Ingestion complète réussie!")
            return 0
        else:
            print(f"\n⚠️  Ingestion partiellement réussie")
            return 1
            
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
