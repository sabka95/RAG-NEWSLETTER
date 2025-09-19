from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..embeddings import MLXEmbeddingService, OptimizedVectorStoreService
from ..processing.document_processor import OptimizedDocumentProcessor


class OptimizedRAGIngestionService:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_newsletter",
        model_name: str = "marco/mcdse-2b-v1",
        use_binary_quantization: bool = True,
        use_mmr: bool = True,
    ):
        """
        Service principal d'ingestion RAG optimisé pour Apple Silicon

        Args:
            qdrant_url: URL du serveur Qdrant
            collection_name: Nom de la collection
            model_name: Nom du modèle d'embedding
            use_binary_quantization: Utiliser la quantization binaire
            use_mmr: Utiliser MMR pour la recherche
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = model_name
        self.use_binary_quantization = use_binary_quantization
        self.use_mmr = use_mmr

        # Initialiser les services optimisés
        logger.info("🚀 Initialisation des services RAG optimisés...")

        self.embedding_service = MLXEmbeddingService(model_name=model_name)
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
            logger.info(
                "⏳ Génération des embeddings MCDSE... (optimisé pour Apple Silicon)"
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
                    "model": self.model_name,
                    "binary_quantization": self.use_binary_quantization,
                    "hnsw_indexing": True,
                    "mmr_search": self.use_mmr,
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

    def search(
        self,
        query: str,
        k: int = 5,
        use_mmr: Optional[bool] = None,
        lambda_mult: float = 0.7,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Recherche dans les documents indexés avec optimisations

        Args:
            query: Requête de recherche
            k: Nombre de résultats
            use_mmr: Utiliser MMR (None = utiliser la config par défaut)
            lambda_mult: Facteur de diversité pour MMR
            filter: Filtres à appliquer

        Returns:
            Résultats de recherche optimisés
        """
        try:
            # Utiliser MMR si demandé ou configuré
            if (use_mmr is True) or (use_mmr is None and self.use_mmr):
                logger.info(f"🎯 Recherche MMR avec lambda={lambda_mult}")
                results = self.vector_store.mmr_search(
                    query=query, k=k, lambda_mult=lambda_mult, filter=filter
                )
            else:
                logger.info("🔍 Recherche HNSW standard")
                results = self.vector_store.similarity_search_with_score(
                    query=query, k=k, filter=filter
                )

            # Formater les résultats
            search_results = []
            for doc, score in results:
                search_results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                        "source": doc.metadata.get("source_file", "N/A"),
                        "page": doc.metadata.get("page_number", "N/A"),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                    }
                )

            logger.info(f"✅ Recherche terminée: {len(search_results)} résultats")
            return search_results

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            return []

    def search_with_document_filter(
        self, query: str, document_names: List[str], k: int = 5
    ) -> List[Dict]:
        """
        Recherche limitée à des documents spécifiques (mode "docs cités")

        Args:
            query: Requête de recherche
            document_names: Noms des documents à inclure
            k: Nombre de résultats

        Returns:
            Résultats filtrés par document
        """
        try:
            logger.info(f"📋 Recherche limitée aux documents: {document_names}")

            # Créer un filtre pour les documents spécifiques
            filter_dict = {"source_file": document_names}

            return self.search(query=query, k=k, filter=filter_dict)

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche filtrée: {e}")
            return []

    def compare_documents(
        self, query: str, document_pairs: List[Tuple[str, str]], k_per_doc: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Compare les réponses entre différents documents

        Args:
            query: Requête de comparaison
            document_pairs: Paires de documents à comparer
            k_per_doc: Nombre de résultats par document

        Returns:
            Résultats de comparaison par document
        """
        try:
            logger.info(f"🔄 Comparaison multi-docs pour: {query}")

            comparison_results = {}

            for doc1, doc2 in document_pairs:
                # Recherche dans chaque document
                results_doc1 = self.search_with_document_filter(
                    query=query, document_names=[doc1], k=k_per_doc
                )
                results_doc2 = self.search_with_document_filter(
                    query=query, document_names=[doc2], k=k_per_doc
                )

                comparison_results[f"{doc1}_vs_{doc2}"] = {
                    doc1: results_doc1,
                    doc2: results_doc2,
                    "query": query,
                    "similarities": self._analyze_similarities(
                        results_doc1, results_doc2
                    ),
                    "differences": self._analyze_differences(
                        results_doc1, results_doc2
                    ),
                }

            logger.info(
                f"✅ Comparaison terminée: {len(comparison_results)} paires analysées"
            )
            return comparison_results

        except Exception as e:
            logger.error(f"❌ Erreur lors de la comparaison: {e}")
            return {}

    def _analyze_similarities(
        self, results1: List[Dict], results2: List[Dict]
    ) -> List[str]:
        """Analyse les similarités entre deux ensembles de résultats"""
        similarities = []

        # Comparer les scores de similarité
        scores1 = [r["score"] for r in results1]
        scores2 = [r["score"] for r in results2]

        if scores1 and scores2:
            avg_score1 = sum(scores1) / len(scores1)
            avg_score2 = sum(scores2) / len(scores2)

            if abs(avg_score1 - avg_score2) < 0.1:
                similarities.append("Scores de pertinence similaires")

            # Analyser les contenus similaires
            contents1 = [r["content"][:100] for r in results1]
            contents2 = [r["content"][:100] for r in results2]

            for c1 in contents1:
                for c2 in contents2:
                    if c1.lower() in c2.lower() or c2.lower() in c1.lower():
                        similarities.append(f"Contenu similaire trouvé: {c1[:50]}...")

        return similarities

    def _analyze_differences(
        self, results1: List[Dict], results2: List[Dict]
    ) -> List[str]:
        """Analyse les différences entre deux ensembles de résultats"""
        differences = []

        # Analyser les différences de scores
        if results1 and results2:
            max_score1 = max(r["score"] for r in results1)
            max_score2 = max(r["score"] for r in results2)

            if abs(max_score1 - max_score2) > 0.2:
                differences.append(
                    f"Différence significative de pertinence: {max_score1:.3f} vs {max_score2:.3f}"
                )

            # Analyser les sources différentes
            sources1 = set(r["source"] for r in results1)
            sources2 = set(r["source"] for r in results2)

            if sources1 != sources2:
                differences.append(f"Sources différentes: {sources1} vs {sources2}")

        return differences

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
                "model": self.model_name,
                "optimizations": {
                    "binary_quantization": self.use_binary_quantization,
                    "hnsw_indexing": True,
                    "mmr_search": self.use_mmr,
                },
                "collection_info": collection_info,
            }
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des stats: {e}")
            return {}


# Alias pour la compatibilité
RAGIngestionService = OptimizedRAGIngestionService
