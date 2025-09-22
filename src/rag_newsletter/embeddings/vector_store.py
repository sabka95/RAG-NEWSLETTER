 # =============================================================================
# RAG Newsletter - Service de Vector Store Optimisé
# =============================================================================
# Service de gestion du vector store Qdrant optimisé pour Apple Silicon avec
# HNSW indexing, Binary Quantization et MMR search pour des performances maximales.
# =============================================================================

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant as LangChainQdrant
from .embedding_service import LangChainMLXEmbeddings
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sklearn.metrics.pairwise import cosine_similarity


class OptimizedVectorStoreService:
    """
    Service de gestion du vector store Qdrant optimisé pour Apple Silicon M4.

    Ce service utilise Qdrant comme base de données vectorielle avec des optimisations
    avancées pour les processeurs Apple Silicon :
    - HNSW indexing pour des recherches ultra-rapides
    - Binary Quantization pour économiser 75% d'espace de stockage
    - MMR (Maximum Marginal Relevance) pour diversifier les résultats
    - Configuration optimisée pour les processeurs M4

    Fonctionnalités clés:
    - Stockage et recherche d'embeddings vectoriels
    - Recherche de similarité avec scores
    - Recherche MMR pour diversifier les résultats
    - Filtrage par métadonnées
    - Gestion des collections optimisées

    Exemple d'utilisation:
        >>> service = OptimizedVectorStoreService(embedding_service=mlx_service)
        >>> service.add_documents(documents)
        >>> results = service.similarity_search("sustainability strategy", k=5)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_newsletter",
        embedding_service=None,
        use_binary_quantization: bool = True,
        hnsw_config: Optional[Dict] = None,
    ):
        """
        Initialise le service de vector store optimisé.

        Args:
            qdrant_url (str): URL du serveur Qdrant (défaut: "http://localhost:6333")
            collection_name (str): Nom de la collection Qdrant (défaut: "rag_newsletter")
            embedding_service: Service d'embeddings MLX pour générer les vecteurs
            use_binary_quantization (bool): Activer la quantization binaire pour économiser l'espace
            hnsw_config (Optional[Dict]): Configuration HNSW personnalisée

        Raises:
            RuntimeError: Si la connexion à Qdrant échoue
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.use_binary_quantization = use_binary_quantization
        self.client = None
        self.vector_store = None

        # Configuration HNSW optimisée pour Apple Silicon M4
        # HNSW (Hierarchical Navigable Small World) est un algorithme de recherche
        # vectorielle qui permet des recherches ultra-rapides même sur de gros volumes
        self.hnsw_config = hnsw_config or {
            "m": 16,  # Nombre de connexions pour chaque nœud (optimisé pour M4)
            "ef_construct": 100,  # Taille de la liste dynamique pendant la construction
            "ef": 64,  # Taille de la liste dynamique pendant la recherche
            "full_scan_threshold": 10000,  # Seuil pour le scan complet (performance)
        }

        # Initialiser la connexion à Qdrant
        self._initialize_client()

    def _initialize_client(self):
        """
        Initialise le client Qdrant avec optimisations pour Apple Silicon.

        Cette méthode établit la connexion à Qdrant et configure la collection
        avec les paramètres optimisés pour les processeurs Apple Silicon M4.

        Raises:
            RuntimeError: Si la connexion à Qdrant échoue
        """
        try:
            logger.info(f"🔗 Connexion à Qdrant: {self.qdrant_url}")
            self.client = QdrantClient(url=self.qdrant_url)

            # Créer la collection si elle n'existe pas avec les optimisations
            self._create_collection_if_not_exists()
            
            # Initialiser le vector store LangChain pour MMR
            self._initialize_langchain_vectorstore()

            logger.info("✅ Client Qdrant initialisé avec optimisations HNSW")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de Qdrant: {e}")
            raise RuntimeError(f"Impossible de se connecter à Qdrant: {e}")

    def _create_collection_if_not_exists(self):
        """
        Crée la collection Qdrant avec optimisations HNSW et Binary Quantization.

        Cette méthode crée une collection Qdrant optimisée pour Apple Silicon M4
        avec les paramètres suivants :
        - HNSW indexing pour des recherches ultra-rapides
        - Binary Quantization pour économiser 75% d'espace de stockage
        - Configuration optimisée pour les processeurs M4
        - Stockage sur disque pour économiser la RAM

        Raises:
            RuntimeError: Si la création de la collection échoue
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(
                    f"🏗️  Création de la collection optimisée: {self.collection_name}"
                )

                # Configuration des vecteurs pour MCDSE-2B-V1
                vector_config = models.VectorParams(
                    size=1536,  # Taille des embeddings MCDSE-2B-V1 (1536 dimensions)
                    distance=models.Distance.COSINE,  # Distance cosinus pour la similarité
                    on_disk=True,  # Stockage sur disque pour économiser la RAM
                )

                # Configuration HNSW optimisée pour Apple Silicon M4
                hnsw_config = models.HnswConfigDiff(
                    m=self.hnsw_config["m"],  # Connexions par nœud (optimisé pour M4)
                    ef_construct=self.hnsw_config[
                        "ef_construct"
                    ],  # Construction de l'index
                    full_scan_threshold=self.hnsw_config[
                        "full_scan_threshold"
                    ],  # Seuil de scan
                    max_indexing_threads=0,  # Auto-détection du nombre de threads (utilise tous les cores M4)
                )

                # Configuration de la quantization binaire si activée
                quantization_config = None
                if self.use_binary_quantization:
                    quantization_config = models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(
                            always_ram=True,  # Garder en RAM pour de meilleures performances
                        )
                    )

                # Créer la collection avec toutes les optimisations
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_config,
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                )

                logger.info(
                    f"✅ Collection '{self.collection_name}' créée avec HNSW + Binary Quantization"
                )
            else:
                logger.info(f"ℹ️  Collection '{self.collection_name}' existe déjà")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de la collection: {e}")
            raise RuntimeError(f"Impossible de créer la collection Qdrant: {e}")

    def _initialize_langchain_vectorstore(self):
        """
        Initialise le vector store LangChain pour MMR et autres fonctionnalités.
        
        Cette méthode crée un vector store LangChain qui utilise la collection Qdrant
        existante pour effectuer des recherches MMR et autres opérations LangChain.
        """
        try:
            # Créer un wrapper LangChain pour notre service MLX
            langchain_embeddings = LangChainMLXEmbeddings(self.embedding_service)
            
            # Créer un vector store LangChain à partir de notre client Qdrant
            self.langchain_vectorstore = LangChainQdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=langchain_embeddings
            )
            
            logger.info("✅ Vector store LangChain initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du vector store: {e}")
            # Ne pas lever d'exception car MMR est optionnel
            self.langchain_vectorstore = None

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Ajoute des documents au vector store avec optimisations pour Apple Silicon.

        Cette méthode traite une liste de documents LangChain, génère leurs embeddings
        avec le service MLX, et les stocke dans Qdrant avec les optimisations HNSW
        et Binary Quantization.

        Args:
            documents (List[Document]): Liste de documents LangChain à ajouter.
                                      Chaque document doit avoir 'image_data' dans ses métadonnées.

        Returns:
            List[str]: Liste des IDs des documents ajoutés dans Qdrant

        Raises:
            RuntimeError: Si le client Qdrant n'est pas initialisé ou si l'ajout échoue

        Exemple:
            >>> service = OptimizedVectorStoreService(embedding_service=mlx_service)
            >>> documents = [doc1, doc2, doc3]  # Documents avec image_data
            >>> ids = service.add_documents(documents)
            >>> print(f"Ajouté {len(ids)} documents")
        """
        if not self.client:
            raise RuntimeError(
                "Client Qdrant non initialisé. Appelez d'abord __init__()"
            )

        try:
            logger.info(
                f"📚 Ajout de {len(documents)} documents au vector store optimisé"
            )

            # Générer les embeddings avec le modèle MCDSE-2B-V1
            logger.info("🖼️  Génération des embeddings MCDSE...")
            embeddings = self.embedding_service.embed_documents(documents)

            # Nettoyer les métadonnées pour Qdrant
            # Qdrant ne peut pas stocker de données binaires dans les métadonnées
            cleaned_metadata_list = []
            for doc in documents:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    # Exclure les données binaires mais garder les autres métadonnées
                    if key not in ["image_data", "image_format"] and isinstance(
                        value, (str, int, float, bool, list, dict)
                    ):
                        cleaned_metadata[key] = value
                cleaned_metadata_list.append(cleaned_metadata)

            # Préparer les points pour l'insertion dans Qdrant
            points = []
            for i, (doc, embedding, metadata) in enumerate(
                zip(documents, embeddings, cleaned_metadata_list)
            ):
                points.append(
                    {
                        "id": i + 1,  # IDs commencent à 1 (Qdrant n'accepte pas 0)
                        "vector": embedding,  # Vecteur d'embedding de 1536 dimensions
                        "payload": {
                            "page_content": doc.page_content,  # Contenu textuel du document
                            **metadata,  # Métadonnées nettoyées
                        },
                    }
                )

            # Vérifier qu'il y a des points à insérer
            if not points:
                logger.warning("Aucun point à insérer dans Qdrant")
                return []

            # Insérer dans Qdrant avec configuration optimisée
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,  # Attendre la confirmation de l'insertion
            )

            ids = [str(i + 1) for i in range(len(documents))]
            logger.info(
                f"✅ Documents ajoutés avec succès: {len(ids)} embeddings optimisés"
            )
            return ids

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout des documents: {e}")
            raise RuntimeError(f"Impossible d'ajouter les documents: {e}")

    def _perform_search(
        self, query: str, k: int = 5, filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Effectue une recherche vectorielle HNSW et retourne les résultats avec scores.

        Cette méthode privée centralise la logique de recherche commune aux deux
        fonctions publiques similarity_search et similarity_search_with_score.

        Args:
            query (str): Requête textuelle à rechercher
            k (int): Nombre de résultats à retourner
            filter (Optional[Dict]): Filtres de métadonnées à appliquer

        Returns:
            List[Tuple[Document, float]]: Liste de tuples (document, score_de_similarité)

        Raises:
            RuntimeError: Si le client Qdrant n'est pas initialisé ou si la recherche échoue
        """
        if not self.client:
            raise RuntimeError(
                "Client Qdrant non initialisé. Appelez d'abord __init__()"
            )

        try:
            # Générer l'embedding de la requête avec le service MLX
            query_embedding = self.embedding_service.embed_query(query)

            # Recherche optimisée avec HNSW
            search_params = models.SearchParams(
                hnsw_ef=self.hnsw_config[
                    "ef"
                ],  # Utiliser la configuration HNSW optimisée
                exact=False,  # Utiliser HNSW au lieu du scan exact (plus rapide)
            )

            # Effectuer la recherche dans Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,  # Inclure les métadonnées
                search_params=search_params,
                query_filter=self._build_filter(filter) if filter else None,
            )

            # Convertir les résultats en documents LangChain avec scores
            documents_with_scores = []
            for result in results:
                doc = Document(
                    page_content=result.payload.get("page_content", ""),
                    metadata={
                        k: v for k, v in result.payload.items() if k != "page_content"
                    },
                )
                documents_with_scores.append((doc, result.score))

            return documents_with_scores

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            raise RuntimeError(f"Impossible d'effectuer la recherche: {e}")

    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict] = None,
        use_mmr: bool = False,
        lambda_mult: float = 0.7
    ) -> List[Document]:
        """
        Recherche de similarité optimisée avec HNSW et/ou MMR pour Apple Silicon.

        Cette méthode effectue une recherche vectorielle dans la collection Qdrant :
        - HNSW seul : Rapide et pertinent (par défaut)
        - HNSW + MMR : Rapide ET diversifié (recommandé)

        HNSW et MMR sont complémentaires :
        - HNSW : Indexation rapide pour trouver les candidats
        - MMR : Sélection intelligente pour diversifier les résultats

        Args:
            query (str): Requête textuelle à rechercher (ex: "sustainability strategy")
            k (int): Nombre de résultats à retourner (défaut: 5)
            filter (Optional[Dict]): Filtres de métadonnées à appliquer
            use_mmr (bool): Utiliser HNSW+MMR pour diversifier (défaut: False)
            lambda_mult (float): Facteur de diversité MMR (0.0 = max diversité, 1.0 = max pertinence)

        Returns:
            List[Document]: Liste des documents les plus similaires à la requête

        Raises:
            RuntimeError: Si le client Qdrant n'est pas initialisé ou si la recherche échoue

        Exemple:
            >>> service = OptimizedVectorStoreService(embedding_service=mlx_service)
            >>> # HNSW seul (rapide)
            >>> results = service.similarity_search("climate change", k=3)
            >>> # HNSW + MMR (rapide ET diversifié)
            >>> results = service.similarity_search("climate change", k=3, use_mmr=True, lambda_mult=0.5)
        """
        if use_mmr:
            # Utiliser HNSW + MMR (complémentaires) via LangChain
            if not self.langchain_vectorstore:
                raise RuntimeError("Vector store LangChain non initialisé pour MMR")
            
            try:
                # Utiliser MMR de LangChain directement
                documents = self.langchain_vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    lambda_mult=lambda_mult,
                    filter=filter
                )
                
                # Récupérer les métadonnées complètes pour chaque document
                documents_with_metadata = []
                for doc in documents:
                    doc_id = doc.metadata.get('_id')
                    if doc_id:
                        # Récupérer les métadonnées complètes depuis Qdrant
                        try:
                            point = self.client.retrieve(
                                collection_name=self.collection_name,
                                ids=[doc_id],
                                with_payload=True
                            )[0]
                            # Remplacer les métadonnées limitées par les complètes
                            doc.metadata = {k: v for k, v in point.payload.items() if k != "page_content"}
                        except Exception as e:
                            logger.warning(f"Impossible de récupérer les métadonnées pour ID {doc_id}: {e}")
                    documents_with_metadata.append(doc)
                
                logger.info(f"🎯 Recherche HNSW+MMR LangChain terminée: {len(documents_with_metadata)} résultats diversifiés")
                return documents_with_metadata
            except Exception as e:
                logger.error(f"❌ Erreur MMR LangChain, fallback vers HNSW seul: {e}")
                # Fallback vers HNSW seul en cas d'erreur MMR
                results = self._perform_search(query, k, filter)
                documents = [doc for doc, score in results]
                logger.info(f"🔍 Recherche HNSW seul (fallback) terminée: {len(documents)} résultats")
                return documents
        else:
            # Utiliser HNSW seul (rapide mais moins diversifié)
            results = self._perform_search(query, k, filter)
            documents = [doc for doc, score in results]
            logger.info(f"🔍 Recherche HNSW seul terminée: {len(documents)} résultats")
            return documents

    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict] = None,
        use_mmr: bool = False,
        lambda_mult: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """
        Recherche de similarité avec scores et optimisations HNSW ou MMR pour Apple Silicon.

        Cette méthode effectue une recherche vectorielle rapide et retourne les documents
        avec leurs scores de similarité, permettant d'évaluer la pertinence des résultats.

        Args:
            query (str): Requête textuelle à rechercher (ex: "sustainability strategy")
            k (int): Nombre de résultats à retourner (défaut: 5)
            filter (Optional[Dict]): Filtres de métadonnées à appliquer
            use_mmr (bool): Utiliser MMR pour diversifier les résultats (défaut: False)
            lambda_mult (float): Facteur de diversité pour MMR (0.0 = max diversité, 1.0 = max pertinence)

        Returns:
            List[Tuple[Document, float]]: Liste de tuples (document, score_de_similarité)
                                        Les scores sont entre 0 et 1 (1 = parfaitement similaire)

        Raises:
            RuntimeError: Si le client Qdrant n'est pas initialisé ou si la recherche échoue

        Exemple:
            >>> service = OptimizedVectorStoreService(embedding_service=mlx_service)
            >>> # Recherche HNSW avec scores
            >>> results = service.similarity_search_with_score("climate change", k=3)
            >>> # Recherche MMR avec scores
            >>> results = service.similarity_search_with_score("climate change", k=3, use_mmr=True)
        """
        if use_mmr:
            # Utiliser HNSW + MMR (complémentaires) via LangChain
            if not self.langchain_vectorstore:
                raise RuntimeError("Vector store LangChain non initialisé pour MMR")
            
            try:
                # Utiliser MMR de LangChain directement
                documents = self.langchain_vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    lambda_mult=lambda_mult,
                    filter=filter
                )
                
                # Récupérer les métadonnées complètes pour chaque document
                documents_with_metadata = []
                for doc in documents:
                    doc_id = doc.metadata.get('_id')
                    if doc_id:
                        # Récupérer les métadonnées complètes depuis Qdrant
                        try:
                            point = self.client.retrieve(
                                collection_name=self.collection_name,
                                ids=[doc_id],
                                with_payload=True
                            )[0]
                            # Remplacer les métadonnées limitées par les complètes
                            doc.metadata = {k: v for k, v in point.payload.items() if k != "page_content"}
                        except Exception as e:
                            logger.warning(f"Impossible de récupérer les métadonnées pour ID {doc_id}: {e}")
                    documents_with_metadata.append(doc)
                
                # MMR ne retourne pas de scores, on utilise 1.0 par défaut
                results = [(doc, 1.0) for doc in documents_with_metadata]
                logger.info(f"🎯 Recherche HNSW+MMR LangChain avec scores terminée: {len(results)} résultats diversifiés")
                return results
            except Exception as e:
                logger.error(f"❌ Erreur MMR LangChain, fallback vers HNSW seul: {e}")
                # Fallback vers HNSW seul en cas d'erreur MMR
                results = self._perform_search(query, k, filter)
                logger.info(f"🔍 Recherche HNSW seul (fallback) avec scores terminée: {len(results)} résultats")
                return results
        else:
            # Utiliser HNSW seul (rapide mais moins diversifié)
            results = self._perform_search(query, k, filter)
            logger.info(f"🔍 Recherche HNSW seul avec scores terminée: {len(results)} résultats")
            return results

    def _build_filter(self, filter_dict: Dict) -> models.Filter:
        """
        Construit un filtre Qdrant à partir d'un dictionnaire de conditions.

        Cette méthode convertit un dictionnaire Python en filtre Qdrant,
        permettant de filtrer les résultats de recherche par métadonnées.

        Args:
            filter_dict (Dict): Dictionnaire de filtres avec les clés suivantes :
                - str/int/float: Valeur exacte à matcher
                - list: Valeurs à matcher (OR)

        Returns:
            models.Filter: Filtre Qdrant configuré

        Exemple:
            >>> filter_dict = {
            ...     "source_file": "document.pdf",
            ...     "category": ["sustainability", "climate"]
            ... }
            >>> filter = service._build_filter(filter_dict)
        """
        conditions = []

        for key, value in filter_dict.items():
            if isinstance(value, list):
                # Filtre par valeurs multiples (OR)
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=value))
                )
            else:
                # Filtre par valeur exacte
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )

        return models.Filter(must=conditions)

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Retourne les informations détaillées sur la collection Qdrant.

        Cette méthode fournit des métriques utiles pour surveiller l'état
        et les performances de la collection vectorielle.

        Returns:
            Dict[str, Any]: Dictionnaire avec les informations de la collection :
                - name: Nom de la collection
                - vectors_count: Nombre total de vecteurs
                - indexed_vectors_count: Nombre de vecteurs indexés
                - points_count: Nombre total de points
                - segments_count: Nombre de segments
                - status: Statut de la collection
                - optimizer_status: Statut de l'optimiseur

        Raises:
            RuntimeError: Si la récupération des informations échoue
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
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


# =============================================================================
# Alias de compatibilité
# =============================================================================
# Alias pour maintenir la compatibilité avec l'ancienne API
# Permet d'utiliser 'VectorStoreService' au lieu de 'OptimizedVectorStoreService'
# =============================================================================
VectorStoreService = OptimizedVectorStoreService
