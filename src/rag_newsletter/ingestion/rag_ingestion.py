from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..embeddings import get_embedding_service, OptimizedVectorStoreService
from ..processing.document_processor import OptimizedDocumentProcessor


class OptimizedRAGIngestionService:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_newsletter",
        use_binary_quantization: bool = True,
    ):
        """
        Service principal d'ingestion RAG optimisé pour Apple Silicon

        Args:
            qdrant_url: URL du serveur Qdrant
            collection_name: Nom de la collection
            use_binary_quantization: Utiliser la quantization binaire
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.use_binary_quantization = use_binary_quantization

        # Initialiser les services optimisés
        logger.info("🚀 Initialisation des services RAG optimisés...")

        self.embedding_service = get_embedding_service()
        self.vector_store = OptimizedVectorStoreService(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            embedding_service=self.embedding_service,
            use_binary_quantization=use_binary_quantization,
        )
        self.document_processor = OptimizedDocumentProcessor()

        logger.info("✅ Services RAG optimisés initialisés avec succès!")

    def ingest_documents(
        self,
        file_paths: List[str],
        source_metadata: Optional[Dict] = None,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Ingère une liste de documents avec optimisations

        Args:
            file_paths: Liste des chemins vers les fichiers
            source_metadata: Métadonnées additionnelles
            batch_size: Taille des lots pour le traitement

        Returns:
            Résumé de l'ingestion
        """
        all_chunks = []
        processed_files = []

        logger.info(f"🚀 Début de l'ingestion optimisée de {len(file_paths)} fichiers")
        logger.info("⚙️  Configuration: MCDSE-2B + HNSW + Binary Quantization")

        # Traitement en lots pour optimiser la mémoire
        for batch_start in range(0, len(file_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(file_paths))
            batch_files = file_paths[batch_start:batch_end]

            logger.info(
                f"📦 Traitement du lot {batch_start//batch_size + 1}: fichiers {batch_start+1}-{batch_end}"
            )

            # Traitement en lot des documents
            batch_results = self.document_processor.process_multiple_pdfs(
                batch_files, source_metadata
            )

            # Collecter tous les documents
            for file_path, documents in batch_results.items():
                if documents:
                    all_chunks.extend(documents)
                    processed_files.append(
                        {
                            "file_path": file_path,
                            "chunks_count": len(documents),
                            "pages": len(
                                set(
                                    doc.metadata.get("page_number", 0)
                                    for doc in documents
                                )
                            ),
                        }
                    )
                    logger.info(
                        f"✅ {Path(file_path).name}: {len(documents)} documents"
                    )
                else:
                    logger.warning(f"⚠️  {Path(file_path).name}: aucun document généré")

        if not all_chunks:
            logger.warning("⚠️  Aucun chunk généré")
            return {"status": "error", "message": "Aucun chunk généré"}

        try:
            # Ajouter au vector store avec optimisations
            logger.info(
                f"💾 Ajout de {len(all_chunks)} chunks au vector store optimisé..."
            )

            ids = self.vector_store.add_documents(all_chunks)

            logger.info(
                f"✅ Embeddings générés et stockés: {len(ids)} vecteurs optimisés"
            )

            # Statistiques détaillées
            stats = self.document_processor.get_processing_stats(
                {
                    fp: [
                        doc
                        for doc in all_chunks
                        if doc.metadata.get("source_file") == Path(fp).name
                    ]
                    for fp in file_paths
                }
            )

            summary = {
                "status": "success",
                "total_chunks": len(all_chunks),
                "processed_files": len(processed_files),
                "files": processed_files,
                "vector_ids": len(ids),
                "optimizations": {
                    "model": "marco/mcdse-2b-v1",
                    "binary_quantization": self.use_binary_quantization,
                    "hnsw_indexing": True,
                    "mmr_search": True,
                },
                "statistics": stats,
            }

            logger.info("🎉 Ingestion optimisée terminée avec succès!")
            logger.info(
                f"📊 Résumé: {len(processed_files)} fichiers, {len(all_chunks)} chunks, {len(ids)} embeddings"
            )
            return summary

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout au vector store: {e}")
            return {"status": "error", "message": str(e)}

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de la collection

        Returns:
            Statistiques de la collection
        """
        try:
            collection_info = self.vector_store.get_collection_info()

            return {
                "collection_name": self.collection_name,
                "qdrant_url": self.qdrant_url,
                "model": "marco/mcdse-2b-v1",
                "optimizations": {
                    "binary_quantization": self.use_binary_quantization,
                    "hnsw_indexing": True,
                    "mmr_search": True,
                },
                "collection_info": collection_info,
            }
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des stats: {e}")
            return {}


# Alias pour la compatibilité
RAGIngestionService = OptimizedRAGIngestionService