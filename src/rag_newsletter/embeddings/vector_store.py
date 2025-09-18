from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, QuantizationConfig, ScalarQuantization
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "rag_newsletter",
                 embedding_service=None,
                 dimension: int = 1024,
                 use_hnsw: bool = True,
                 use_binary_quantization: bool = True,
                 mmr_lambda: float = 0.7):
        """
        Service de gestion du vector store Qdrant avec optimisations avancées
        
        Args:
            qdrant_url: URL du serveur Qdrant
            collection_name: Nom de la collection
            embedding_service: Service d'embeddings
            dimension: Dimension des embeddings
            use_hnsw: Utiliser HNSW pour l'indexation
            use_binary_quantization: Activer la quantization binaire
            mmr_lambda: Paramètre lambda pour MMR (0.0 = diversité max, 1.0 = pertinence max)
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.dimension = dimension
        self.use_hnsw = use_hnsw
        self.use_binary_quantization = use_binary_quantization
        self.mmr_lambda = mmr_lambda
        
        self.client = None
        self.vector_store = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialise le client Qdrant avec optimisations"""
        try:
            logger.info(f"🔗 Connexion à Qdrant: {self.qdrant_url}")
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Créer la collection avec optimisations
            self._create_collection_if_not_exists()
            
            logger.info("✅ Client Qdrant initialisé avec succès")
            logger.info(f"📊 Dimension: {self.dimension}")
            logger.info(f"🏗️  HNSW: {self.use_hnsw}")
            logger.info(f"🔢 Binary quantization: {self.use_binary_quantization}")
            logger.info(f"🎯 MMR lambda: {self.mmr_lambda}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de Qdrant: {e}")
            raise
    
    def _create_collection_if_not_exists(self):
        """Crée la collection avec optimisations HNSW et quantization"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"🏗️  Création de la collection optimisée: {self.collection_name}")
                
                # Configuration HNSW
                hnsw_config = None
                if self.use_hnsw:
                    hnsw_config = HnswConfigDiff(
                        m=16,  # Nombre de connexions pour chaque nœud
                        ef_construct=200,  # Taille de la liste dynamique pendant la construction
                        full_scan_threshold=10000,  # Seuil pour passer en scan complet
                        max_indexing_threads=4,  # Nombre de threads pour l'indexation
                    )
                    logger.info("🏗️  Configuration HNSW activée")
                
                # Configuration de la quantization
                quantization_config = None
                if self.use_binary_quantization:
                    quantization_config = QuantizationConfig(
                        scalar=ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                quantile=0.99,
                                always_ram=True
                            )
                        )
                    )
                    logger.info("🔢 Binary quantization activée")
                
                # Créer la collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    ),
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=20000,
                        memmap_threshold=50000,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=2
                    )
                )
                logger.info(f"✅ Collection '{self.collection_name}' créée avec optimisations")
            else:
                logger.info(f"ℹ️  Collection '{self.collection_name}' existe déjà")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de la collection: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Ajoute des documents au vector store avec optimisations
        
        Args:
            documents: Liste de documents LangChain
            
        Returns:
            Liste des IDs des documents ajoutés
        """
        if not self.client:
            raise RuntimeError("Client Qdrant non initialisé")
        
        try:
            logger.info(f"📄 Ajout de {len(documents)} documents au vector store optimisé")
            
            # Générer les embeddings avec le modèle MCDSE
            logger.info("🖼️  Génération des embeddings avec MCDSE...")
            embeddings = self.embedding_service.embed_documents(documents)
            
            if not embeddings:
                logger.warning("⚠️  Aucun embedding généré")
                return []
            
            # Nettoyer les métadonnées pour Qdrant
            cleaned_metadata_list = []
            for doc in documents:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    # Exclure les données binaires mais garder les autres métadonnées
                    if key not in ['image_data', 'image_format'] and isinstance(value, (str, int, float, bool, list, dict)):
                        cleaned_metadata[key] = value
                cleaned_metadata_list.append(cleaned_metadata)
            
            # Préparer les points pour Qdrant
            points = []
            for i, (doc, embedding, metadata) in enumerate(zip(documents, embeddings, cleaned_metadata_list)):
                points.append({
                    "id": i,  # Utiliser un ID numérique au lieu d'une chaîne
                    "vector": embedding,
                    "payload": {
                        "page_content": doc.page_content,
                        **metadata
                    }
                })
            
            # Insérer dans Qdrant avec optimisations
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Forcer l'indexation pour HNSW
            if self.use_hnsw:
                self.client.update_collection(
                    collection_name=self.collection_name,
                    optimizer_config=models.OptimizersConfigDiff(
                        indexing_threshold=0  # Forcer l'indexation immédiate
                    )
                )
                logger.info("🏗️  Indexation HNSW forcée")
            
            ids = [str(i) for i in range(len(documents))]
            logger.info(f"✅ Documents ajoutés avec optimisations: {len(ids)} IDs")
            return ids
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout des documents: {e}")
            raise
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 5,
                                   filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Recherche de similarité avec scores et optimisations
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats à retourner
            filter: Filtres à appliquer
            
        Returns:
            Liste de tuples (document, score)
        """
        if not self.client:
            raise RuntimeError("Client Qdrant non initialisé")
        
        try:
            # Générer l'embedding de la requête
            query_embedding = self.embedding_service.embed_query(query)
            
            # Recherche dans Qdrant avec optimisations
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k * 2,  # Récupérer plus de résultats pour MMR
                with_payload=True,
                query_filter=filter
            )
            
            # Convertir en documents LangChain
            documents_with_scores = []
            for result in results:
                doc = Document(
                    page_content=result.payload.get("page_content", ""),
                    metadata={k: v for k, v in result.payload.items() if k != "page_content"}
                )
                documents_with_scores.append((doc, result.score))
            
            logger.info(f"🔍 Recherche terminée: {len(documents_with_scores)} résultats")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            raise
    
    def mmr_search(self, 
                   query: str, 
                   k: int = 5,
                   filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Recherche avec Maximum Marginal Relevance (MMR) pour diversité
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats à retourner
            filter: Filtres à appliquer
            
        Returns:
            Liste de tuples (document, score) diversifiés
        """
        try:
            logger.info(f"🎯 Recherche MMR avec lambda={self.mmr_lambda}")
            
            # Récupérer plus de résultats pour la diversité
            initial_results = self.similarity_search_with_score(
                query=query,
                k=k * 3,  # 3x plus de résultats pour la sélection MMR
                filter=filter
            )
            
            if not initial_results:
                return []
            
            # Appliquer MMR
            mmr_results = self._apply_mmr(initial_results, k)
            
            logger.info(f"🎯 MMR terminé: {len(mmr_results)} résultats diversifiés")
            return mmr_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche MMR: {e}")
            # Fallback vers recherche standard
            return self.similarity_search_with_score(query, k, filter)
    
    def _apply_mmr(self, results: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        """
        Applique l'algorithme MMR pour diversifier les résultats
        
        Args:
            results: Liste des résultats initiaux (document, score)
            k: Nombre de résultats finaux
            
        Returns:
            Liste diversifiée de résultats
        """
        if len(results) <= k:
            return results
        
        # Extraire les embeddings des documents pour le calcul MMR
        embeddings = []
        for doc, score in results:
            # Re-générer l'embedding pour le calcul de similarité
            embedding = self.embedding_service.embed_query(doc.page_content[:100])  # Limiter pour performance
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Initialiser avec le meilleur résultat
        selected_indices = [0]
        selected_embeddings = [embeddings[0]]
        
        # MMR pour les k-1 résultats restants
        for _ in range(k - 1):
            best_mmr_score = -float('inf')
            best_index = -1
            
            for i, (doc, score) in enumerate(results):
                if i in selected_indices:
                    continue
                
                # Calculer la similarité avec les documents déjà sélectionnés
                similarities = cosine_similarity([embeddings[i]], selected_embeddings)[0]
                max_similarity = np.max(similarities)
                
                # Score MMR: lambda * score_pertinence - (1-lambda) * max_similarity
                mmr_score = self.mmr_lambda * score - (1 - self.mmr_lambda) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_index = i
            
            if best_index != -1:
                selected_indices.append(best_index)
                selected_embeddings.append(embeddings[best_index])
        
        # Retourner les résultats sélectionnés
        mmr_results = [results[i] for i in sorted(selected_indices)]
        return mmr_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Retourne les informations sur la collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "hnsw_config": collection_info.config.hnsw_config,
                "quantization_config": collection_info.config.quantization_config
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos: {e}")
            return {}
    
    def delete_collection(self):
        """Supprime la collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️  Collection '{self.collection_name}' supprimée")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
            raise