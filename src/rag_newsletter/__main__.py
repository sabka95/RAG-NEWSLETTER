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
    - Statistiques de la collection
    
    Note: Pour la recherche et l'analyse, utilisez l'interface Streamlit:
        streamlit run src/rag_newsletter/ui/streamlit_app.py

    Exemples d'utilisation:
        python -m rag_newsletter --download --max 50      # Télécharger 50 documents
        python -m rag_newsletter --ingest --batch-size 5  # Ingérer avec des lots de 5
        python -m rag_newsletter --stats                  # Afficher les statistiques
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
            use_binary_quantization=not a.no_binary_quantization,
        )

        # Afficher les informations de configuration
        logger.info("🚀 RAG Newsletter Optimisé - Configuration:")
        logger.info(f"   📱 Modèle: marco/mcdse-2b-v1 (auto-détecté)")
        logger.info(f"   🔗 Qdrant: {a.qdrant_url}")
        logger.info(f"   📚 Collection: {a.collection}")
        logger.info(f"   ⚡ Binary Quantization: {not a.no_binary_quantization}")
        logger.info("   🎯 MMR Search: Always enabled")
        logger.info("   🍎 Platform: Auto-detected (Apple Silicon / Linux)")

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

        # Statistiques de la collection Qdrant
        if a.stats:
            logger.info("\n📊 Statistiques de la collection:")
            stats = rag_service.vector_store.get_collection_info()
            logger.info(f"   📚 Collection: {stats.get('name', 'N/A')}")
            logger.info(f"   📄 Vecteurs: {stats.get('vectors_count', 'N/A')}")
            logger.info(f"   📊 Points: {stats.get('points_count', 'N/A')}")
            logger.info(f"   📈 Segments: {stats.get('segments_count', 'N/A')}")
            logger.info(f"   ✅ Statut: {stats.get('status', 'N/A')}")
            logger.info(f"   💾 Taille disque: {stats.get('disk_data_size', 0) / 1024:.2f} KB")
            logger.info(f"   🧠 Taille RAM: {stats.get('ram_data_size', 0) / 1024:.2f} KB")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
