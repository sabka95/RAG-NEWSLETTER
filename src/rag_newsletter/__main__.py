import os, argparse, pathlib
from rag_newsletter.ingestion.sharepoint_client import make_client_from_env
from rag_newsletter.ingestion.rag_ingestion import RAGIngestionService
from dotenv import load_dotenv
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    load_dotenv() 
    p = argparse.ArgumentParser(description="RAG Newsletter - Importateur SharePoint avec RAG")
    p.add_argument("--drive", default=os.getenv("SP_DRIVE_NAME", "Documents"))
    p.add_argument("--drive-id")
    p.add_argument("--max", type=int, default=100)
    p.add_argument("--download", action="store_true", help="Télécharger les fichiers")
    p.add_argument("--ingest", action="store_true", help="Ingérer les fichiers dans le vector store")
    p.add_argument("--search", type=str, help="Rechercher dans les documents indexés")
    p.add_argument("--outdir", default="downloads", help="Répertoire de sortie")
    p.add_argument("--list-drives", action="store_true", help="Lister les drives disponibles")
    p.add_argument("--extensions", nargs="+", 
                   default=[".pdf", ".docx", ".pptx", ".xlsx", ".txt"],
                   help="Extensions de fichiers à importer")
    p.add_argument("--qdrant-url", default="http://localhost:6333", help="URL du serveur Qdrant")
    p.add_argument("--collection", default="rag_newsletter", help="Nom de la collection Qdrant")
    
    # Options avancées pour les optimisations
    p.add_argument("--model", default="marco/mcdse-2b-v1", help="Modèle d'embeddings à utiliser")
    p.add_argument("--dimension", type=int, default=1024, help="Dimension des embeddings")
    p.add_argument("--no-mlx", action="store_true", help="Désactiver MLX (utiliser PyTorch standard)")
    p.add_argument("--no-hnsw", action="store_true", help="Désactiver HNSW pour Qdrant")
    p.add_argument("--no-binary-quantization", action="store_true", help="Désactiver la binary quantization")
    p.add_argument("--mmr-lambda", type=float, default=0.7, help="Paramètre lambda pour MMR (0.0=diversité max, 1.0=pertinence max)")
    p.add_argument("--no-mmr", action="store_true", help="Désactiver MMR (recherche standard)")
    a = p.parse_args()

    try:
        sp = make_client_from_env()
        
        if a.list_drives:
            drives = sp.list_drives()
            print(f"Drives disponibles ({len(drives)}):")
            for i, drive in enumerate(drives, 1):
                print(f"  {i}. {drive['name']} (ID: {drive['id']})")
            return
        
        drive_id = a.drive_id
        if not drive_id:
            drive_id = sp.find_drive_id(a.drive)
            if not drive_id: 
                raise SystemExit(f"Drive '{a.drive}' introuvable. Utilise --drive-id ou ajuste SP_DRIVE_NAME.")

        # Lister les fichiers seulement si nécessaire
        if a.download or a.list_drives:
            files = sp.list_files(drive_id, exts=tuple(a.extensions))
            print(f"Fichiers trouvés: {len(files)}")
            
            # Afficher les fichiers (limités par --max)
            for f in files[:a.max]:
                size_mb = round(f.get('size', 0) / (1024 * 1024), 2)
                print(f"- {f['name']}  | {f['last_modified']} | {size_mb} MB")

        if a.download and files:
            print(f"\nTéléchargement des fichiers...")
            downloaded = sp.download_multiple(
                drive_id=drive_id,
                files=files,
                output_dir=a.outdir,
                max_files=a.max
            )
            
            summary = sp.get_download_summary(downloaded)
            print(f"\nRésumé du téléchargement:")
            print(f"- Fichiers téléchargés: {summary['total_files']}")
            print(f"- Taille totale: {summary['total_size_mb']} MB")
            print(f"- Extensions: {summary['extensions']}")
            print(f"- Répertoire: {pathlib.Path(a.outdir).absolute()}")
        
        if a.ingest:
            print(f"\nIngestion des documents dans le vector store optimisé...")
            rag_service = RAGIngestionService(
                qdrant_url=a.qdrant_url,
                collection_name=a.collection,
                model_name=a.model,
                dimension=a.dimension,
                use_mlx=not a.no_mlx,
                use_hnsw=not a.no_hnsw,
                use_binary_quantization=not a.no_binary_quantization,
                mmr_lambda=a.mmr_lambda
            )
            
            # Utiliser les fichiers téléchargés ou chercher dans le répertoire
            if a.download and 'downloaded' in locals():
                # Utiliser les fichiers qui viennent d'être téléchargés
                file_paths = [f["local_path"] for f in downloaded]
                print(f"Utilisation des {len(file_paths)} fichiers téléchargés")
            else:
                # Chercher les fichiers PDF dans le répertoire
                download_path = pathlib.Path(a.outdir)
                file_paths = list(download_path.glob("*.pdf"))
                file_paths = [str(f) for f in file_paths]
                print(f"Utilisation des {len(file_paths)} fichiers trouvés dans {a.outdir}")
            
            if not file_paths:
                print("Aucun fichier à ingérer")
                return 1
            
            result = rag_service.ingest_documents(file_paths)
            print(f"\nRésultat de l'ingestion:")
            print(f"- Statut: {result['status']}")
            if result['status'] == 'success':
                print(f"- Pages traitées: {result['total_chunks']}")
                print(f"- Fichiers traités: {result['processed_files']}")
                print(f"- IDs vectoriels: {result['vector_ids']}")
            else:
                print(f"- Erreur: {result['message']}")
        
        if a.search:
            print(f"\nRecherche optimisée: '{a.search}'")
            rag_service = RAGIngestionService(
                qdrant_url=a.qdrant_url,
                collection_name=a.collection,
                model_name=a.model,
                dimension=a.dimension,
                use_mlx=not a.no_mlx,
                use_hnsw=not a.no_hnsw,
                use_binary_quantization=not a.no_binary_quantization,
                mmr_lambda=a.mmr_lambda
            )
            
            # Afficher les options utilisées
            print(f"🔧 Configuration:")
            print(f"  - Modèle: {a.model}")
            print(f"  - MLX: {not a.no_mlx}")
            print(f"  - HNSW: {not a.no_hnsw}")
            print(f"  - Binary quantization: {not a.no_binary_quantization}")
            print(f"  - MMR lambda: {a.mmr_lambda}")
            print(f"  - MMR activé: {not a.no_mmr}")
            
            # Effectuer la recherche
            results = rag_service.search(a.search, k=5, use_mmr=not a.no_mmr)
            print(f"\n🎯 Résultats trouvés: {len(results)}")
            
            for i, result in enumerate(results, 1):
                print(f"\n--- Résultat {i} ---")
                print(f"Score: {result['score']:.3f}")
                print(f"Source: {result['metadata'].get('source_file', 'N/A')}")
                print(f"Page: {result['metadata'].get('page_number', 'N/A')}")
                print(f"Contenu: {result['content'][:200]}...")
            
            # Afficher les stats de la collection
            stats = rag_service.get_collection_stats()
            if stats.get('status') == 'success':
                print(f"\n📊 Statistiques de la collection:")
                coll_stats = stats['collection_stats']
                print(f"  - Vecteurs: {coll_stats.get('vectors_count', 'N/A')}")
                print(f"  - Points: {coll_stats.get('points_count', 'N/A')}")
                print(f"  - Indexés: {coll_stats.get('indexed_vectors_count', 'N/A')}")
                print(f"  - Statut: {coll_stats.get('status', 'N/A')}")
            
    except Exception as e:
        logger.error(f"Erreur: {e}")
        return 1

if __name__ == "__main__":
    main()
