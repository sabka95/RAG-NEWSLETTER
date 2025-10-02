# =============================================================================
# RAG Newsletter - Service de Vector Store Optimisé
# =============================================================================
# Service de gestion du vector store Qdrant optimisé pour Apple Silicon avec
# HNSW indexing et MMR search pour des performances maximales.
# =============================================================================

from typing import Any, Dict, List, Optional
import numpy as np

from langchain.schema import Document
from loguru import logger

try:
    from langchain_community.vectorstores.utils import maximal_marginal_relevance
except ImportError:
    # Fallback pour versions anciennes de LangChain
    from langchain.vectorstores.utils import maximal_marginal_relevance
from qdrant_client import QdrantClient
from qdrant_client.http import models


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
            embedding_service: Service d'embeddings pour générer les vecteurs
            use_binary_quantization (bool): Activer la quantization binaire pour économiser l'espace
            hnsw_config (Optional[Dict]): Configuration HNSW personnalisée

        Raises:
            RuntimeError: Si la connexion à Qdrant échoue
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.use_binary_quantization = use_binary_quantization

        # Configuration HNSW optimisée pour Apple Silicon M4
        self.hnsw_config = hnsw_config or {
            "m": 32,  # Nombre de connexions bidirectionnelles (32 optimal pour M4)
            "ef_construct": 256,  # Paramètre de construction (plus haut = meilleure qualité)
            "max_indexing_threads": 8,  # Threads optimisés pour processeurs M4
        }

        # Initialisation des services
        self.client = None

        logger.info("🚀 Initialisation du service de vector store optimisé...")
        self._initialize_client()
        self._create_collection_if_not_exists()

        logger.info("✅ Service de vector store optimisé initialisé avec succès!")

    def _initialize_client(self):
        """Initialise le client Qdrant avec configuration optimisée."""
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                timeout=60,  # Timeout adapté pour les opérations lourdes
                prefer_grpc=False,  # HTTP plus stable pour les connexions locales
            )

            # Vérifier la connexion
            collections = self.client.get_collections()
            logger.info(f"✅ Connexión à Qdrant établie: {len(collections.collections)} collections")

        except Exception as e:
            logger.error(f"❌ Erreur de connexion à Qdrant: {e}")
            raise RuntimeError(f"Impossible de se connecter à Qdrant: {e}")

    def _create_collection_if_not_exists(self):
        """Crée la collection avec optimisations Apple Silicon si elle n'existe pas."""
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
                    m=self.hnsw_config["m"],
                    ef_construct=self.hnsw_config["ef_construct"],
                    full_scan_threshold=20000,  # Seuil pour basculer vers scan complet
                    max_indexing_threads=self.hnsw_config["max_indexing_threads"],
                )

                # Configuration de quantization binaire
                quantization_config = None
                if self.use_binary_quantization:
                    quantization_config = models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(always_ram=True)
                    )

                # Créer la collection avec toutes les optimisations
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_config,
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,  # Nombre de segments optimisé pour M4
                        max_segment_size=200000,  # Taille max des segments
                        memmap_threshold=50000,  # Seuil pour l'utilisation de memmap
                        indexing_threshold=20000,  # Seuil pour l'indexation
                        flush_interval_sec=10,  # Intervalle de flush optimisé
                        max_optimization_threads=4,  # Threads d'optimisation pour M4
                    ),
                )

                logger.info("✅ Collection créée avec optimisations Apple Silicon M4")
            else:
                logger.info(f"📂 Collection existante: {self.collection_name}")

        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de la collection: {e}")
            raise RuntimeError(f"Impossible de créer la collection: {e}")

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
            embeddings = self.embedding_service.embed_pdf_pages(documents)

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
                wait=True,  # Attendre la confirmation de l'insertion
                points=[
                    models.PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"],
                    )
                    for point in points
                ],
            )

            logger.info(
                f"✅ Documents stockés avec succès dans Qdrant: {len(points)} vecteurs"
            )
            return [str(point["id"]) for point in points]

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout des documents: {e}")
            raise RuntimeError(f"Impossible d'ajouter les documents: {e}")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None,
        lambda_mult: float = 0.7,
    ) -> List[tuple[Document, float]]:
        """
        Recherche de similarité avec MMR directe dans Qdrant.

        Args:
            query (str): Requête de recherche  
            k (int): Nombre de résultats à retourner (défaut: 5)
            filter (Optional[Dict]): Filtres de métadonnées à appliquer
            lambda_mult (float): Facteur de diversité MMR (défaut: 0.7, 1.0 = pas de diversité)

        Returns:
            List[Tuple[Document, float]]: Liste des (document, score) avec MMR appliqué

        Exemple:
            >>> results = service.similarity_search_with_score("climate change", k=3)
            >>> for doc, score in results:
            >>>     print(f"Score: {score:.3f} - {doc.page_content[:100]}...")
        """
        try:
            logger.info(f"🔍 Recherche vectorielle MMR: '{query}' (k={k}, lambda={lambda_mult})")

            # 1. Convertir le filtre si nécessaire
            qdrant_filter = self._convert_filter(filter) if filter else None
            
            # 2. Générer l'embedding de la requête
            query_embedding = self.embedding_service.embed_query(query)
            
            # 3. Rechercher plus de résultats que nécessaire pour avoir des options MMR
            search_k = min(k * 4, 50)  # Chercher 4x plus pour avoir du choix
            
            # 4. Recherche initiale dans Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_k,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=True,  # On a besoin des vecteurs pour MMR
            )
            
            if not search_results:
                logger.info("❌ Aucun résultat trouvé")
                return []
            
            # 5. Préparer les données pour l'algorithme MMR officiel de LangChain
            embeddings = [result.vector for result in search_results]
            documents = []
            scores = []
            
            for result in search_results:
                doc = Document(
                    page_content=result.payload.get("page_content", ""),
                    metadata=result.payload,
                )
                documents.append(doc)
                scores.append(result.score)
            
            # 6. Utiliser l'implémentation MMR officielle de LangChain
            # Convertir les embeddings en numpy arrays si nécessaire
            query_embedding_np = np.array(query_embedding) if not isinstance(query_embedding, np.ndarray) else query_embedding
            embeddings_np = [np.array(emb) if not isinstance(emb, np.ndarray) else emb for emb in embeddings]
            
            selected_indices = maximal_marginal_relevance(
                query_embedding=query_embedding_np,
                embedding_list=embeddings_np,
                lambda_mult=lambda_mult,
                k=k,
            )
            
            # 7. Construire les résultats finaux avec les scores originaux
            selected_docs = []
            for idx in selected_indices:
                doc = documents[idx]
                score = scores[idx]
                selected_docs.append((doc, score))
            
            logger.info(f"✅ Recherche MMR LangChain terminée: {len(selected_docs)} résultats")
            return selected_docs

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche MMR: {e}")
            raise RuntimeError(f"Échec de la recherche vectorielle: {e}")

    def _convert_filter(self, filter_dict: Dict) -> models.Filter:
        """
        Convertit un dictionnaire de filtres en format Qdrant.

        Args:
            filter_dict (Dict): Dictionnaire de filtres

        Returns:
            models.Filter: Filtre Qdrant
        """
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, str):
                conditions.append(
                    models.FieldCondition(
                        key=key, match=models.MatchValue(value=value)
                    )
                )
            elif isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key, match=models.MatchAny(any=value)
                    )
                )

        return models.Filter(must=conditions)

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur la collection.

        Returns:
            Dict[str, Any]: Informations de la collection
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
                "points_count": info.points_count,
                "vectors_count": info.points_count,
                "segments_count": info.segments_count if hasattr(info, 'segments_count') else 0,
                "disk_data_size": info.disk_data_size if hasattr(info, 'disk_data_size') else 0,
                "ram_data_size": info.ram_data_size if hasattr(info, 'ram_data_size') else 0,
                "config": {
                    "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 1536,
                    "distance": info.config.params.vectors.distance.value if hasattr(info.config.params, 'vectors') else "Cosine",
                    "hnsw_config": {
                        "m": info.config.hnsw_config.m if hasattr(info.config, 'hnsw_config') else 16,
                        "ef_construct": info.config.hnsw_config.ef_construct if hasattr(info.config, 'hnsw_config') else 100,
                        "max_indexing_threads": info.config.hnsw_config.max_indexing_threads if hasattr(info.config, 'hnsw_config') else 0,
                    },
                    "quantization_enabled": self.use_binary_quantization,
                },
            }
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'obtention des infos: {e}")
            return {}

    def delete_collection(self):
        """Supprime la collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"✅ Collection supprimée: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la suppression: {e}")


# =============================================================================
# Alias de compatibilité
# =============================================================================
# Alias pour maintenir la compatibilité avec l'ancienne API
# Permet d'utiliser 'VectorStoreService' au lieu de 'OptimizedVectorStoreService'
# =============================================================================
VectorStoreService = OptimizedVectorStoreService