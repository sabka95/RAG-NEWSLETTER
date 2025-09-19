# =============================================================================
# RAG Newsletter - Service de Vector Store Optimisé
# =============================================================================
# Service de gestion du vector store Qdrant optimisé pour Apple Silicon avec
# HNSW indexing, Binary Quantization et MMR search pour des performances maximales.
# =============================================================================

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain.schema import Document
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

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Recherche de similarité optimisée avec HNSW pour Apple Silicon.

        Cette méthode effectue une recherche vectorielle rapide dans la collection
        Qdrant en utilisant l'algorithme HNSW optimisé pour les processeurs M4.

        Args:
            query (str): Requête textuelle à rechercher (ex: "sustainability strategy")
            k (int): Nombre de résultats à retourner (défaut: 5)
            filter (Optional[Dict]): Filtres de métadonnées à appliquer

        Returns:
            List[Document]: Liste des documents les plus similaires à la requête

        Raises:
            RuntimeError: Si le client Qdrant n'est pas initialisé ou si la recherche échoue

        Exemple:
            >>> service = OptimizedVectorStoreService(embedding_service=mlx_service)
            >>> results = service.similarity_search("climate change", k=3)
            >>> print(f"Trouvé {len(results)} documents similaires")
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

            # Convertir les résultats en documents LangChain
            documents = []
            for result in results:
                doc = Document(
                    page_content=result.payload.get("page_content", ""),
                    metadata={
                        k: v for k, v in result.payload.items() if k != "page_content"
                    },
                )
                documents.append(doc)

            logger.info(f"🔍 Recherche HNSW terminée: {len(documents)} résultats")
            return documents

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            raise RuntimeError(f"Impossible d'effectuer la recherche: {e}")

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Recherche de similarité avec scores et optimisations HNSW pour Apple Silicon.

        Cette méthode effectue une recherche vectorielle rapide et retourne les documents
        avec leurs scores de similarité, permettant d'évaluer la pertinence des résultats.

        Args:
            query (str): Requête textuelle à rechercher (ex: "sustainability strategy")
            k (int): Nombre de résultats à retourner (défaut: 5)
            filter (Optional[Dict]): Filtres de métadonnées à appliquer

        Returns:
            List[Tuple[Document, float]]: Liste de tuples (document, score_de_similarité)
                                        Les scores sont entre 0 et 1 (1 = parfaitement similaire)

        Raises:
            RuntimeError: Si le client Qdrant n'est pas initialisé ou si la recherche échoue

        Exemple:
            >>> service = OptimizedVectorStoreService(embedding_service=mlx_service)
            >>> results = service.similarity_search_with_score("climate change", k=3)
            >>> for doc, score in results:
            ...     print(f"Score: {score:.3f} - {doc.page_content[:50]}...")
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

            logger.info(
                f"🔍 Recherche HNSW avec scores terminée: {len(documents_with_scores)} résultats"
            )
            return documents_with_scores

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche avec scores: {e}")
            raise RuntimeError(f"Impossible d'effectuer la recherche avec scores: {e}")

    def mmr_search(
        self,
        query: str,
        k: int = 5,
        lambda_mult: float = 0.7,
        filter: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recherche MMR (Maximum Marginal Relevance) pour diversifier les résultats.

        Cette méthode utilise l'algorithme MMR pour sélectionner des documents
        qui sont à la fois pertinents par rapport à la requête et diversifiés
        entre eux, évitant la redondance dans les résultats.

        Args:
            query (str): Requête textuelle à rechercher (ex: "sustainability strategy")
            k (int): Nombre de résultats à retourner (défaut: 5)
            lambda_mult (float): Facteur de diversité (0.0 = max diversité, 1.0 = max pertinence)
            filter (Optional[Dict]): Filtres de métadonnées à appliquer

        Returns:
            List[Tuple[Document, float]]: Liste de tuples (document, score_mmr)
                                        Les scores MMR combinent pertinence et diversité

        Raises:
            RuntimeError: Si le client Qdrant n'est pas initialisé ou si la recherche échoue

        Exemple:
            >>> service = OptimizedVectorStoreService(embedding_service=mlx_service)
            >>> results = service.mmr_search("sustainability", k=3, lambda_mult=0.5)
            >>> for doc, score in results:
            ...     print(f"Score MMR: {score:.3f} - {doc.page_content[:50]}...")
        """
        if not self.client:
            raise RuntimeError(
                "Client Qdrant non initialisé. Appelez d'abord __init__()"
            )

        try:
            # Récupérer plus de résultats pour l'algorithme MMR
            # MMR a besoin de plus de candidats pour bien diversifier
            fetch_k = min(k * 3, 50)  # Récupérer 3x plus de résultats (max 50)

            # Recherche initiale avec HNSW
            search_params = models.SearchParams(
                hnsw_ef=self.hnsw_config[
                    "ef"
                ],  # Utiliser la configuration HNSW optimisée
                exact=False,  # Utiliser HNSW au lieu du scan exact (plus rapide)
            )

            # Effectuer la recherche dans Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=self.embedding_service.embed_query(query),
                limit=fetch_k,
                with_payload=True,  # Inclure les métadonnées
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
                    metadata={
                        k: v for k, v in result.payload.items() if k != "page_content"
                    },
                )
                candidate_docs.append(doc)
                candidate_scores.append(result.score)

            # Appliquer l'algorithme MMR pour diversifier les résultats
            selected_docs = self._apply_mmr(
                query_embedding=self.embedding_service.embed_query(query),
                candidate_docs=candidate_docs,
                candidate_scores=candidate_scores,
                k=k,
                lambda_mult=lambda_mult,
            )

            logger.info(
                f"🎯 Recherche MMR terminée: {len(selected_docs)} résultats diversifiés"
            )
            return selected_docs

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche MMR: {e}")
            raise RuntimeError(f"Impossible d'effectuer la recherche MMR: {e}")

    def _apply_mmr(
        self,
        query_embedding: List[float],
        candidate_docs: List[Document],
        candidate_scores: List[float],
        k: int,
        lambda_mult: float,
    ) -> List[Tuple[Document, float]]:
        """
        Applique l'algorithme MMR (Maximum Marginal Relevance) pour diversifier les résultats.

        L'algorithme MMR sélectionne des documents qui maximisent la pertinence
        par rapport à la requête tout en minimisant la redondance entre les résultats.

        Args:
            query_embedding (List[float]): Embedding de la requête (1536 dimensions)
            candidate_docs (List[Document]): Documents candidats à diversifier
            candidate_scores (List[float]): Scores de similarité initiaux des candidats
            k (int): Nombre de résultats à sélectionner
            lambda_mult (float): Facteur de diversité (0.0 = max diversité, 1.0 = max pertinence)

        Returns:
            List[Tuple[Document, float]]: Liste des documents sélectionnés avec scores MMR
                                        Les scores MMR combinent pertinence et diversité
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
                    query_vec, doc_embeddings[idx].reshape(1, -1)
                )[0][0]

                # Score de diversité (similarité maximale avec les documents déjà sélectionnés)
                if selected_indices:
                    max_similarity = max(
                        [
                            cosine_similarity(
                                doc_embeddings[idx].reshape(1, -1),
                                doc_embeddings[sel_idx].reshape(1, -1),
                            )[0][0]
                            for sel_idx in selected_indices
                        ]
                    )
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
        Récupère l'embedding d'un document en utilisant le cache si disponible.

        Cette méthode optimise les performances en évitant de re-générer
        les embeddings déjà calculés, ce qui est crucial pour l'algorithme MMR.

        Args:
            doc (Document): Document LangChain dont on veut l'embedding

        Returns:
            List[float]: Embedding du document (1536 dimensions)

        Note:
            Dans un système de production, un cache Redis serait plus approprié
            pour gérer les embeddings de manière distribuée.
        """
        # Si l'embedding est déjà en cache dans les métadonnées
        if "cached_embedding" in doc.metadata:
            return doc.metadata["cached_embedding"]

        # Sinon, re-générer l'embedding avec le service MLX
        # Note: Dans un système de production, vous voudriez utiliser un cache Redis
        return self.embedding_service.embed_documents([doc])[0]

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
