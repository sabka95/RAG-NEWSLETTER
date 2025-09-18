from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from ..embeddings import EmbeddingService, VectorStoreService
from ..processing.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class RAGIngestionService:
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "rag_newsletter",
                 model_name: str = "marco/mcdse-2b-v1",
                 dimension: int = 1024,
                 use_mlx: bool = True,
                 use_hnsw: bool = True,
                 use_binary_quantization: bool = True,
                 mmr_lambda: float = 0.7):
        """
        Service principal d'ingestion RAG avec optimisations avancées
        
        Args:
            qdrant_url: URL du serveur Qdrant
            collection_name: Nom de la collection
            model_name: Nom du modèle d'embeddings
            dimension: Dimension des embeddings
            use_mlx: Utiliser MLX pour l'optimisation Apple Silicon
            use_hnsw: Utiliser HNSW pour l'indexation Qdrant
            use_binary_quantization: Activer la quantization binaire
            mmr_lambda: Paramètre lambda pour MMR
        """
        logger.info("🚀 Initialisation du service RAG optimisé")
        logger.info(f"🍎 MLX: {use_mlx}")
        logger.info(f"🏗️  HNSW: {use_hnsw}")
        logger.info(f"🔢 Binary quantization: {use_binary_quantization}")
        logger.info(f"🎯 MMR lambda: {mmr_lambda}")
        
        self.embedding_service = EmbeddingService(
            model_name=model_name,
            dimension=dimension,
            use_mlx=use_mlx,
            binary_quantization=use_binary_quantization
        )
        
        self.vector_store = VectorStoreService(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            embedding_service=self.embedding_service,
            dimension=dimension,
            use_hnsw=use_hnsw,
            use_binary_quantization=use_binary_quantization,
            mmr_lambda=mmr_lambda
        )
        
        self.document_processor = DocumentProcessor()
        self.mmr_lambda = mmr_lambda
    
    def ingest_documents(self, file_paths: List[str], 
                        source_metadata: Dict = None) -> Dict[str, Any]:
        """
        Ingère une liste de documents
        
        Args:
            file_paths: Liste des chemins vers les fichiers
            source_metadata: Métadonnées additionnelles
            
        Returns:
            Résumé de l'ingestion
        """
        all_chunks = []
        processed_files = []
        
        logger.info(f"🚀 Début de l'ingestion de {len(file_paths)} fichiers")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"📄 [{i}/{len(file_paths)}] Traitement: {file_path}")
                
                # Traiter le document
                chunks = self.document_processor.process_pdf(
                    file_path, 
                    source_metadata
                )
                
                all_chunks.extend(chunks)
                processed_files.append({
                    "file_path": file_path,
                    "chunks_count": len(chunks)
                })
                
                logger.info(f"✅ Fichier traité: {len(chunks)} chunks générés")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'ingestion de {file_path}: {e}")
                continue
        
        if not all_chunks:
            logger.warning("⚠️  Aucun chunk généré")
            return {"status": "error", "message": "Aucun chunk généré"}
        
        try:
            # Ajouter au vector store
            logger.info(f"💾 Ajout de {len(all_chunks)} chunks au vector store...")
            logger.info("⏳ Génération des embeddings... (peut prendre du temps)")
            
            ids = self.vector_store.add_documents(all_chunks)
            
            logger.info(f"✅ Embeddings générés et stockés: {len(ids)} vecteurs")
            
            summary = {
                "status": "success",
                "total_chunks": len(all_chunks),
                "processed_files": len(processed_files),
                "files": processed_files,
                "vector_ids": len(ids)
            }
            
            logger.info(f"🎉 Ingestion terminée avec succès!")
            logger.info(f"📊 Résumé: {len(processed_files)} fichiers, {len(all_chunks)} chunks, {len(ids)} embeddings")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout au vector store: {e}")
            return {"status": "error", "message": str(e)}
    
    def search(self, query: str, k: int = 5, use_mmr: bool = True) -> List[Dict]:
        """
        Recherche dans les documents indexés avec options avancées
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats
            use_mmr: Utiliser MMR pour diversifier les résultats
            
        Returns:
            Résultats de recherche
        """
        try:
            if use_mmr:
                logger.info(f"🎯 Recherche MMR avec lambda={self.mmr_lambda}")
                results = self.vector_store.mmr_search(query, k=k)
            else:
                logger.info("🔍 Recherche standard")
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            search_results = []
            for doc, score in results:
                search_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            
            logger.info(f"✅ Recherche terminée: {len(search_results)} résultats")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            return []
    
    def search_with_filters(self, 
                           query: str, 
                           k: int = 5, 
                           source_files: Optional[List[str]] = None,
                           use_mmr: bool = True) -> List[Dict]:
        """
        Recherche avec filtres sur les documents sources
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats
            source_files: Liste des fichiers sources à filtrer
            use_mmr: Utiliser MMR pour diversifier
            
        Returns:
            Résultats de recherche filtrés
        """
        try:
            # Construire le filtre Qdrant
            filter_conditions = None
            if source_files:
                filter_conditions = {
                    "must": [
                        {
                            "key": "source_file",
                            "match": {"any": source_files}
                        }
                    ]
                }
                logger.info(f"🔍 Recherche filtrée sur {len(source_files)} fichiers")
            
            if use_mmr:
                results = self.vector_store.mmr_search(query, k=k, filter=filter_conditions)
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k, filter=filter_conditions)
            
            search_results = []
            for doc, score in results:
                search_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche filtrée: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la collection"""
        try:
            stats = self.vector_store.get_collection_info()
            return {
                "status": "success",
                "collection_stats": stats,
                "model_info": {
                    "model_name": self.embedding_service.model_name,
                    "dimension": self.embedding_service.dimension,
                    "use_mlx": self.embedding_service.use_mlx,
                    "binary_quantization": self.embedding_service.binary_quantization
                },
                "search_config": {
                    "mmr_lambda": self.mmr_lambda,
                    "use_hnsw": self.vector_store.use_hnsw,
                    "use_binary_quantization": self.vector_store.use_binary_quantization
                }
            }
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des stats: {e}")
            return {"status": "error", "message": str(e)}
