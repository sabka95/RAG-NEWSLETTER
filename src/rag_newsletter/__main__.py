# =============================================================================
# RAG Newsletter - Point d'entrée principal
# =============================================================================
# Script principal pour l'ingestion et la gestion des documents RAG
# avec optimisations Apple Silicon et intégration SharePoint.
# =============================================================================

import argparse
import os
import pathlib

from dotenv import load_dotenv
from loguru import logger

from rag_newsletter.ingestion.rag_ingestion import OptimizedRAGIngestionService
from rag_newsletter.ingestion.sharepoint_client import make_client_from_env

# Configuration du logging avec loguru
logger.remove()  # Supprimer le handler par défaut
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
)


def main():
    """
    Point d'entrée principal du script RAG Newsletter.

    Ce script permet de gérer l'ingestion et la recherche de documents
    avec optimisations Apple Silicon et intégration SharePoint.

    Fonctionnalités disponibles:
    - Téléchargement de documents depuis SharePoint
    - Ingestion de documents dans le vector store Qdrant
    - Recherche sémantique avec MMR (Maximum Marginal Relevance)
    - Comparaison de documents
    - Filtrage par documents spécifiques
    - Statistiques de la collection

    Exemples d'utilisation:
        python -m rag_newsletter --download --max 50                    # Télécharger 50 documents
        python -m rag_newsletter --ingest --batch-size 5               # Ingérer avec des lots de 5
        python -m rag_newsletter --search "sustainability strategy"    # Rechercher des documents
        python -m rag_newsletter --search-mmr --lambda 0.5             # Recherche MMR avec diversité
        python -m rag_newsletter --compare doc1.pdf doc2.pdf           # Comparer deux documents
        python -m rag_newsletter --stats                                # Afficher les statistiques
    """
    # Charger les variables d'environnement depuis .env
    load_dotenv()

    # Configuration du parser d'arguments
    p = argparse.ArgumentParser(
        description="RAG Newsletter Optimisé - Importateur SharePoint avec MCDSE-2B + MLX"
    )

    # Arguments de configuration SharePoint
    p.add_argument(
        "--drive",
        default=os.getenv("SP_DRIVE_NAME", "Documents"),
        help="Nom du drive SharePoint à utiliser",
    )
    p.add_argument("--drive-id", help="ID du drive SharePoint (si spécifique)")
    p.add_argument(
        "--max", type=int, default=100, help="Nombre maximum de documents à traiter"
    )

    # Arguments d'actions principales
    p.add_argument(
        "--download",
        action="store_true",
        help="Télécharger les fichiers depuis SharePoint",
    )
    p.add_argument(
        "--ingest",
        action="store_true",
        help="Ingérer les fichiers dans le vector store optimisé",
    )
    p.add_argument("--search", type=str, help="Rechercher dans les documents indexés")
    p.add_argument(
        "--search-mmr",
        action="store_true",
        help="Utiliser la recherche MMR (Maximum Marginal Relevance)",
    )
    p.add_argument(
        "--lambda",
        type=float,
        default=0.7,
        help="Facteur de diversité pour MMR (0.0-1.0)",
    )
    p.add_argument(
        "--compare", nargs=2, metavar=("DOC1", "DOC2"), help="Comparer deux documents"
    )
    p.add_argument(
        "--filter-docs",
        nargs="+",
        help="Filtrer la recherche à des documents spécifiques",
    )
    p.add_argument(
        "--list-drives",
        action="store_true",
        help="Lister les drives SharePoint disponibles",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Afficher les statistiques de la collection",
    )

    # Arguments de configuration
    p.add_argument(
        "--outdir",
        default="downloads",
        help="Répertoire de sortie pour les téléchargements",
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdf", ".docx", ".pptx", ".xlsx", ".txt"],
        help="Extensions de fichiers à importer",
    )
    p.add_argument(
        "--qdrant-url", default="http://localhost:6333", help="URL du serveur Qdrant"
    )
    p.add_argument(
        "--collection", default="rag_newsletter", help="Nom de la collection Qdrant"
    )
    p.add_argument(
        "--model", default="marco/mcdse-2b-v1", help="Modèle d'embedding à utiliser"
    )
    p.add_argument(
        "--no-binary-quantization",
        action="store_true",
        help="Désactiver la quantization binaire",
    )
    p.add_argument(
        "--batch-size", type=int, default=10, help="Taille des lots pour l'ingestion"
    )

    # Parser les arguments
    a = p.parse_args()

    try:
        # Initialiser le service RAG optimisé avec les paramètres fournis
        rag_service = OptimizedRAGIngestionService(
            qdrant_url=a.qdrant_url,
            collection_name=a.collection,
            model_name=a.model,
            use_binary_quantization=not a.no_binary_quantization,
            use_mmr=True,  # Activer MMR pour la diversité des résultats
        )

        # Afficher les informations de configuration
        logger.info("🚀 RAG Newsletter Optimisé - Configuration:")
        logger.info(f"   📱 Modèle: {a.model}")
        logger.info(f"   🔗 Qdrant: {a.qdrant_url}")
        logger.info(f"   📚 Collection: {a.collection}")
        logger.info(f"   ⚡ Binary Quantization: {not a.no_binary_quantization}")
        logger.info("   🎯 MMR Search: True")
        logger.info("   🍎 Optimisé pour Apple Silicon M4")

        # Gestion des drives SharePoint
        if a.list_drives or a.download or a.ingest:
            # Initialiser le client SharePoint avec les variables d'environnement
            sp = make_client_from_env()

            # Lister les drives SharePoint disponibles
            if a.list_drives:
                drives = sp.list_drives()
                logger.info(f"📁 Drives disponibles ({len(drives)}):")
                for i, drive in enumerate(drives, 1):
                    logger.info(f"   {i}. {drive['name']} (ID: {drive['id']})")
                return

            # Résoudre l'ID du drive SharePoint
            drive_id = a.drive_id
            if not drive_id:
                drive_id = sp.find_drive_id(a.drive)
                if not drive_id:
                    raise SystemExit(
                        f"Drive '{a.drive}' introuvable. Utilise --drive-id ou ajuste SP_DRIVE_NAME."
                    )

            # Lister les fichiers seulement si nécessaire
            if a.download or a.ingest:
                files = sp.list_files(drive_id, exts=tuple(a.extensions))
                logger.info(f"📄 Fichiers trouvés: {len(files)}")

                # Afficher les fichiers (limités par --max)
                for f in files[: a.max]:
                    size_mb = round(f.get("size", 0) / (1024 * 1024), 2)
                    logger.info(
                        f"   - {f['name']}  | {f['last_modified']} | {size_mb} MB"
                    )

            # Télécharger les fichiers depuis SharePoint
            if a.download and files:
                logger.info("\n📥 Téléchargement des fichiers...")
                downloaded = sp.download_multiple(
                    drive_id=drive_id,
                    files=files,
                    output_dir=a.outdir,  # Répertoire de sortie
                    max_files=a.max,  # Limiter par --max
                )

                # Afficher le résumé du téléchargement
                summary = sp.get_download_summary(downloaded)
                logger.info("\n✅ Résumé du téléchargement:")
                logger.info(f"   📁 Fichiers téléchargés: {summary['total_files']}")
                logger.info(f"   💾 Taille totale: {summary['total_size_mb']} MB")
                logger.info(f"   📋 Extensions: {summary['extensions']}")
                logger.info(f"   📂 Répertoire: {pathlib.Path(a.outdir).absolute()}")

        # Ingestion optimisée des documents dans le vector store
        if a.ingest:
            logger.info("\n🚀 Ingestion optimisée des documents...")

            # Utiliser les fichiers téléchargés ou chercher dans le répertoire
            if a.download and "downloaded" in locals():
                # Utiliser les fichiers qui viennent d'être téléchargés
                file_paths = [f["local_path"] for f in downloaded]
                logger.info(
                    f"📁 Utilisation des {len(file_paths)} fichiers téléchargés"
                )
            else:
                # Chercher les fichiers PDF dans le répertoire
                download_path = pathlib.Path(a.outdir)
                file_paths = list(download_path.glob("*.pdf"))
                file_paths = [str(f) for f in file_paths]
                logger.info(
                    f"📁 Utilisation des {len(file_paths)} fichiers trouvés dans {a.outdir}"
                )

            if not file_paths:
                logger.warning("⚠️  Aucun fichier à ingérer")
                return 1

            # Ingérer les documents avec le service RAG optimisé
            result = rag_service.ingest_documents(
                file_paths=file_paths, batch_size=a.batch_size
            )

            # Afficher le résumé de l'ingestion
            logger.info("\n🎉 Résultat de l'ingestion optimisée:")
            logger.info(f"   📊 Statut: {result['status']}")
            if result["status"] == "success":
                logger.info(f"   📄 Pages traitées: {result['total_chunks']}")
                logger.info(f"   📁 Fichiers traités: {result['processed_files']}")
                logger.info(f"   🔢 IDs vectoriels: {result['vector_ids']}")
                logger.info(f"   ⚡ Optimisations: {result['optimizations']}")
            else:
                logger.error(f"   ❌ Erreur: {result['message']}")

        # Recherche optimisée dans les documents indexés
        if a.search:
            logger.info(f"\n🔍 Recherche optimisée: '{a.search}'")

            # Déterminer le type de recherche selon les paramètres
            if a.search_mmr:
                logger.info(f"🎯 Mode MMR avec lambda={getattr(a, 'lambda', 0.7)}")
                search_results = rag_service.search(
                    query=a.search,
                    k=5,
                    use_mmr=True,
                    lambda_mult=getattr(a, "lambda", 0.7),
                )
            elif a.filter_docs:
                logger.info(f"📋 Recherche filtrée aux documents: {a.filter_docs}")
                search_results = rag_service.search_with_document_filter(
                    query=a.search, document_names=a.filter_docs, k=5
                )
            else:
                logger.info("🔍 Recherche HNSW standard")
                search_results = rag_service.search(query=a.search, k=5)

            # Afficher les résultats de la recherche
            logger.info(f"\n✅ Résultats trouvés: {len(search_results)}")
            for i, result in enumerate(search_results, 1):
                logger.info(f"\n--- Résultat {i} ---")
                logger.info(f"   📊 Score: {result['score']:.3f}")
                logger.info(f"   📄 Source: {result['source']}")
                logger.info(f"   📃 Page: {result['page']}")
                logger.info(f"   🔢 Chunk: {result['chunk_index']}")
                logger.info(f"   📝 Contenu: {result['content'][:200]}...")

        # Comparaison de documents spécifiques
        if a.compare:
            doc1, doc2 = a.compare
            logger.info(f"\n🔄 Comparaison: {doc1} vs {doc2}")
            logger.info(f"   🔍 Requête: '{a.search or 'Analyse générale'}'")

            # Effectuer la comparaison des documents
            comparison_results = rag_service.compare_documents(
                query=a.search or "Analyse générale",
                document_pairs=[(doc1, doc2)],
                k_per_doc=3,
            )

            for comparison_key, results in comparison_results.items():
                logger.info(f"\n📊 Résultats de comparaison: {comparison_key}")
                logger.info(f"   📄 {doc1}: {len(results[doc1])} résultats")
                logger.info(f"   📄 {doc2}: {len(results[doc2])} résultats")

                if results["similarities"]:
                    logger.info(f"   ✅ Similarités: {results['similarities']}")
                if results["differences"]:
                    logger.info(f"   ⚠️  Différences: {results['differences']}")

        # Statistiques de la collection Qdrant
        if a.stats:
            logger.info("\n📊 Statistiques de la collection:")
            stats = rag_service.get_collection_stats()
            logger.info(f"   📚 Collection: {stats['collection_name']}")
            logger.info(f"   🔗 URL: {stats['qdrant_url']}")
            logger.info(f"   🤖 Modèle: {stats['model']}")
            logger.info(f"   ⚡ Optimisations: {stats['optimizations']}")

            if stats.get("collection_info"):
                info = stats["collection_info"]
                logger.info(f"   📄 Vecteurs: {info.get('vectors_count', 'N/A')}")
                logger.info(f"   📊 Points: {info.get('points_count', 'N/A')}")
                logger.info(f"   📈 Segments: {info.get('segments_count', 'N/A')}")
                logger.info(f"   ✅ Statut: {info.get('status', 'N/A')}")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
