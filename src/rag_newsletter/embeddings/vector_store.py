from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

class OptimizedVectorStoreService:
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "rag_newsletter",
                 embedding_service=None,
                 use_binary_quantization: bool = True,
                 hnsw_config: Optional[Dict] = None):
        """
        Service de gestion du vector store Qdrant optimisé pour Apple Silicon
        
        Args:
            qdrant_url: URL du serveur Qdrant
            collection_name: Nom de la collection
            embedding_service: Service d'embeddings MLX
            use_binary_quantization: Utiliser la quantization binaire pour économiser l'espace
            hnsw_config: Configuration HNSW personnalisée
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.use_binary_quantization = use_binary_quantization
        self.client = None
        self.vector_store = None
        
        # Configuration HNSW optimisée pour Apple Silicon
        self.hnsw_config = hnsw_config or {
            "m": 16,  # Nombre de connexions pour chaque nœud
            "ef_construct": 100,  # Taille de la liste dynamique pendant la construction
            "ef": 64,  # Taille de la liste dynamique pendant la recherche
            "full_scan_threshold": 10000,  # Seuil pour le scan complet
        }
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialise le client Qdrant avec optimisations"""
        try:
            logger.info(f"🔗 Connexion à Qdrant: {self.qdrant_url}")
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Créer la collection si elle n'existe pas
            self._create_collection_if_not_exists()
            
            logger.info("✅ Client Qdrant initialisé avec optimisations HNSW")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de Qdrant: {e}")
            raise
    
    def _create_collection_if_not_exists(self):
        """Crée la collection avec optimisations HNSW et binary quantization"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"🏗️  Création de la collection optimisée: {self.collection_name}")
                
                # Configuration des vecteurs
                vector_config = models.VectorParams(
                    size=1536,  # Taille des embeddings MCDSE
                    distance=models.Distance.COSINE,
                    on_disk=True,  # Stockage sur disque pour économiser la RAM
                )
                
                # Configuration HNSW
                hnsw_config = models.HnswConfigDiff(
                    m=self.hnsw_config["m"],
                    ef_construct=self.hnsw_config["ef_construct"],
                    full_scan_threshold=self.hnsw_config["full_scan_threshold"],
                    max_indexing_threads=0,  # Auto-détection du nombre de threads
                )
                
                # Configuration de la quantization binaire si activée
                quantization_config = None
                if self.use_binary_quantization:
                    quantization_config = models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(
                            always_ram=True,  # Garder en RAM pour de meilleures performances
                        )
                    )
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_config,
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                )
                
                logger.info(f"✅ Collection '{self.collection_name}' créée avec HNSW + Binary Quantization")
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
            logger.info(f"📚 Ajout de {len(documents)} documents au vector store optimisé")
            
            # Générer les embeddings avec le modèle MCDSE
            logger.info("🖼️  Génération des embeddings MCDSE...")
            embeddings = self.embedding_service.embed_documents(documents)
            
            # Nettoyer les métadonnées pour Qdrant
            cleaned_metadata_list = []
            for doc in documents:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    # Exclure les données binaires mais garder les autres métadonnées
                    if key not in ['image_data', 'image_format'] and isinstance(value, (str, int, float, bool, list, dict)):
                        cleaned_metadata[key] = value
                cleaned_metadata_list.append(cleaned_metadata)
            
            # Ajouter les embeddings avec optimisations
            points = []
            for i, (doc, embedding, metadata) in enumerate(zip(documents, embeddings, cleaned_metadata_list)):
                points.append({
                    "id": i + 1,  # IDs commencent à 1 (Qdrant n'accepte pas 0)
                    "vector": embedding,
                    "payload": {
                        "page_content": doc.page_content,
                        **metadata
                    }
                })
            
            # Vérifier qu'il y a des points à insérer
            if not points:
                logger.warning("Aucun point à insérer dans Qdrant")
                return []
            
            # Insérer dans Qdrant avec configuration optimisée
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,  # Attendre la confirmation
            )
            
            ids = [str(i + 1) for i in range(len(documents))]
            logger.info(f"✅ Documents ajoutés avec succès: {len(ids)} embeddings optimisés")
            return ids
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout des documents: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter: Optional[Dict] = None) -> List[Document]:
        """
        Recherche de similarité optimisée avec HNSW
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats à retourner
            filter: Filtres à appliquer
            
        Returns:
            Liste des documents les plus similaires
        """
        if not self.client:
            raise RuntimeError("Client Qdrant non initialisé")
        
        try:
            # Générer l'embedding de la requête
            query_embedding = self.embedding_service.embed_query(query)
            
            # Recherche optimisée avec HNSW
            search_params = models.SearchParams(
                hnsw_ef=self.hnsw_config["ef"],  # Utiliser la configuration HNSW
                exact=False,  # Utiliser HNSW au lieu du scan exact
            )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                search_params=search_params,
                query_filter=self._build_filter(filter) if filter else None,
            )
            
            # Convertir les résultats en documents LangChain
            documents = []
            for result in results:
                doc = Document(
                    page_content=result.payload.get("page_content", ""),
                    metadata={k: v for k, v in result.payload.items() if k != "page_content"}
                )
                documents.append(doc)
            
            logger.info(f"🔍 Recherche HNSW terminée: {len(documents)} résultats")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            raise
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 5,
                                   filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Recherche de similarité avec scores et optimisations HNSW
        
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
            
            # Recherche optimisée avec HNSW
            search_params = models.SearchParams(
                hnsw_ef=self.hnsw_config["ef"],
                exact=False,
            )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                search_params=search_params,
                query_filter=self._build_filter(filter) if filter else None,
            )
            
            # Convertir les résultats en documents LangChain avec scores
            documents_with_scores = []
            for result in results:
                doc = Document(
                    page_content=result.payload.get("page_content", ""),
                    metadata={k: v for k, v in result.payload.items() if k != "page_content"}
                )
                documents_with_scores.append((doc, result.score))
            
            logger.info(f"🔍 Recherche HNSW avec scores terminée: {len(documents_with_scores)} résultats")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche avec scores: {e}")
            raise
    
    def mmr_search(self, 
                   query: str, 
                   k: int = 5,
                   lambda_mult: float = 0.7,
                   filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Recherche Maximum Marginal Relevance (MMR) pour diversifier les résultats
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats à retourner
            lambda_mult: Facteur de diversité (0.0 = max diversité, 1.0 = max pertinence)
            filter: Filtres à appliquer
            
        Returns:
            Liste de tuples (document, score_mmr)
        """
        if not self.client:
            raise RuntimeError("Client Qdrant non initialisé")
        
        try:
            # Récupérer plus de résultats pour l'algorithme MMR
            fetch_k = min(k * 3, 50)  # Récupérer 3x plus de résultats
            
            # Recherche initiale avec HNSW
            search_params = models.SearchParams(
                hnsw_ef=self.hnsw_config["ef"],
                exact=False,
            )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=self.embedding_service.embed_query(query),
                limit=fetch_k,
                with_payload=True,
                search_params=search_params,
                query_filter=self._build_filter(filter) if filter else None,
            )
            
            if not results:
                return []
            
            # Convertir en documents avec scores
            candidate_docs = []
            candidate_scores = []
            for result in results:
                doc = Document(
                    page_content=result.payload.get("page_content", ""),
                    metadata={k: v for k, v in result.payload.items() if k != "page_content"}
                )
                candidate_docs.append(doc)
                candidate_scores.append(result.score)
            
            # Appliquer l'algorithme MMR
            selected_docs = self._apply_mmr(
                query_embedding=self.embedding_service.embed_query(query),
                candidate_docs=candidate_docs,
                candidate_scores=candidate_scores,
                k=k,
                lambda_mult=lambda_mult
            )
            
            logger.info(f"🎯 Recherche MMR terminée: {len(selected_docs)} résultats diversifiés")
            return selected_docs
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche MMR: {e}")
            raise
    
    def _apply_mmr(self, 
                   query_embedding: List[float], 
                   candidate_docs: List[Document], 
                   candidate_scores: List[float],
                   k: int, 
                   lambda_mult: float) -> List[Tuple[Document, float]]:
        """
        Applique l'algorithme Maximum Marginal Relevance
        
        Args:
            query_embedding: Embedding de la requête
            candidate_docs: Documents candidats
            candidate_scores: Scores de similarité initiaux
            k: Nombre de résultats à sélectionner
            lambda_mult: Facteur de diversité
            
        Returns:
            Liste des documents sélectionnés avec scores MMR
        """
        if len(candidate_docs) <= k:
            return list(zip(candidate_docs, candidate_scores))
        
        # Convertir en numpy pour les calculs
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # Calculer les embeddings des documents candidats
        doc_embeddings = []
        for doc in candidate_docs:
            # Re-générer l'embedding du document (ou utiliser le cache si disponible)
            doc_embedding = self._get_document_embedding(doc)
            # S'assurer que l'embedding est un array 1D
            if isinstance(doc_embedding, list):
                doc_embedding = np.array(doc_embedding)
            doc_embeddings.append(doc_embedding.flatten())
        
        doc_embeddings = np.array(doc_embeddings)
        
        # Algorithme MMR
        selected_indices = []
        remaining_indices = list(range(len(candidate_docs)))
        
        # Sélectionner le premier document (le plus pertinent)
        best_idx = np.argmax(candidate_scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Sélectionner les documents restants
        for _ in range(min(k - 1, len(remaining_indices))):
            mmr_scores = []
            
            for idx in remaining_indices:
                # Score de pertinence (similarité avec la requête)
                relevance = cosine_similarity(
                    query_vec, 
                    doc_embeddings[idx].reshape(1, -1)
                )[0][0]
                
                # Score de diversité (similarité maximale avec les documents déjà sélectionnés)
                if selected_indices:
                    max_similarity = max([
                        cosine_similarity(
                            doc_embeddings[idx].reshape(1, -1),
                            doc_embeddings[sel_idx].reshape(1, -1)
                        )[0][0]
                        for sel_idx in selected_indices
                    ])
                else:
                    max_similarity = 0
                
                # Score MMR
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
                mmr_scores.append(mmr_score)
            
            # Sélectionner le document avec le meilleur score MMR
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Retourner les documents sélectionnés avec leurs scores initiaux
        selected_docs = []
        for idx in selected_indices:
            selected_docs.append((candidate_docs[idx], candidate_scores[idx]))
        
        return selected_docs
    
    def _get_document_embedding(self, doc: Document) -> List[float]:
        """
        Récupère l'embedding d'un document (utilise le cache si disponible)
        
        Args:
            doc: Document LangChain
            
        Returns:
            Embedding du document
        """
        # Si l'embedding est déjà en cache dans les métadonnées
        if 'cached_embedding' in doc.metadata:
            return doc.metadata['cached_embedding']
        
        # Sinon, re-générer l'embedding
        # Note: Dans un système de production, vous voudriez utiliser un cache Redis
        return self.embedding_service.embed_documents([doc])[0]
    
    def _build_filter(self, filter_dict: Dict) -> models.Filter:
        """
        Construit un filtre Qdrant à partir d'un dictionnaire
        
        Args:
            filter_dict: Dictionnaire de filtres
            
        Returns:
            Filtre Qdrant
        """
        conditions = []
        
        for key, value in filter_dict.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=conditions)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur la collection
        
        Returns:
            Dictionnaire avec les informations de la collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos de collection: {e}")
            return {}

# Alias pour la compatibilité
VectorStoreService = OptimizedVectorStoreService