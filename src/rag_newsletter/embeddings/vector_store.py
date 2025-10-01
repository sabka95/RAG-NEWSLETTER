 # =============================================================================
# RAG Newsletter - Service de Vector Store Optimisé
# =============================================================================
# Service de gestion du vector store Qdrant optimisé pour Apple Silicon avec
# HNSW indexing, Binary Quantization et MMR search pour des performances maximales.
# =============================================================================

from typing import Any, Dict, List, Optional, Tuple
import re
from collections import Counter
import math

import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant as LangChainQdrant
from .embedding_service import LangChainMLXEmbeddings
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sklearn.metrics.pairwise import cosine_similarity

# Import pour le reranking Cross-Encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers CrossEncoder non disponible")

# Import pour BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("⚠️ rank-bm25 non disponible, BM25 désactivé")


class BM25Service:
    """
    Service BM25 pour l'indexation hybride dense+sparse.
    
    BM25 est un algorithme de scoring de pertinence qui complète les embeddings
    denses en capturant les correspondances exactes de termes et la fréquence
    des mots-clés importants.
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, use_available: bool = True):
        """
        Initialise le service BM25.
        
        Args:
            k1 (float): Paramètre de saturation de la fréquence des termes (défaut: 1.2)
            b (float): Paramètre de normalisation de la longueur (défaut: 0.75)
            use_available (bool): Utiliser BM25 si disponible (défaut: True)
        """
        self.k1 = k1
        self.b = b
        self.use_bm25 = use_available and BM25_AVAILABLE
        self.bm25_model = None
        self.documents = []
        self.document_metadata = []
        
        if not self.use_bm25:
            logger.warning("⚠️ BM25 désactivé, recherche textuelle basique utilisée")
        else:
            logger.info(f"🔍 Service BM25 initialisé (k1={k1}, b={b})")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Ajoute des documents à l'index BM25.
        
        Args:
            documents (List[Dict]): Documents avec 'content' et métadonnées
            
        Returns:
            bool: True si succès, False sinon
        """
        if not self.use_bm25:
            logger.warning("⚠️ BM25 non disponible, documents non indexés")
            return False
        
        try:
            # Extraire les contenus textuels
            corpus = []
            metadata = []
            
            for doc in documents:
                content = doc.get("content", "")
                if content:
                    # Tokeniser le contenu
                    tokens = self._tokenize(content)
                    corpus.append(tokens)
                    metadata.append(doc)
            
            if not corpus:
                logger.warning("⚠️ Aucun contenu textuel trouvé dans les documents")
                return False
            
            # Créer le modèle BM25
            self.bm25_model = BM25Okapi(corpus, k1=self.k1, b=self.b)
            self.documents = corpus
            self.document_metadata = metadata
            
            logger.info(f"✅ {len(corpus)} documents indexés avec BM25")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur indexation BM25: {e}")
            return False
    
    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Recherche BM25 avec scores de pertinence.
        
        Args:
            query (str): Requête textuelle
            k (int): Nombre de résultats à retourner
            min_score (float): Score minimum pour filtrer les résultats
            
        Returns:
            List[Tuple[Dict, float]]: Documents avec scores BM25
        """
        if not self.use_bm25 or not self.bm25_model:
            logger.warning("⚠️ BM25 non disponible, recherche textuelle basique")
            logger.debug(f"BM25 disponible: {self.use_bm25}, Modèle: {self.bm25_model is not None}")
            return self._basic_text_search(query, k)
        
        try:
            # Debug: vérifier l'état de BM25
            logger.debug(f"BM25: {len(self.documents)} documents, {len(self.document_metadata)} métadonnées")
            
            # Tokeniser la requête
            query_tokens = self._tokenize(query)
            
            if not query_tokens:
                logger.warning("⚠️ Requête vide après tokenisation")
                return []
            
            # Calculer les scores BM25
            scores = self.bm25_model.get_scores(query_tokens)
            
            # Créer les résultats avec scores
            results = []
            for i, (score, metadata) in enumerate(zip(scores, self.document_metadata)):
                if score >= min_score:
                    results.append((metadata, float(score)))
            
            # Trier par score décroissant
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Retourner les k meilleurs
            final_results = results[:k]
            
            logger.info(f"🔍 Recherche BM25: {len(final_results)} résultats (score min: {min_score})")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche BM25: {e}")
            return self._basic_text_search(query, k)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenise le texte pour BM25.
        
        Args:
            text (str): Texte à tokeniser
            
        Returns:
            List[str]: Liste de tokens
        """
        if not text:
            return []
        
        # Nettoyer et normaliser le texte
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Diviser en tokens
        tokens = text.split()
        
        # Filtrer les tokens trop courts
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def _basic_text_search(self, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        Recherche textuelle basique (fallback).
        
        Args:
            query (str): Requête textuelle
            k (int): Nombre de résultats
            
        Returns:
            List[Tuple[Dict, float]]: Résultats avec scores basiques
        """
        if not self.document_metadata:
            return []
        
        query_lower = query.lower()
        results = []
        
        for doc in self.document_metadata:
            content = doc.get("content", "").lower()
            
            # Score basique basé sur la fréquence des mots
            query_words = query_lower.split()
            matches = sum(1 for word in query_words if word in content)
            score = matches / len(query_words) if query_words else 0.0
            
            if score > 0:
                results.append((doc, score))
        
        # Trier par score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du service BM25.
        
        Returns:
            Dict[str, Any]: Statistiques du service
        """
        return {
            "bm25_available": self.use_bm25,
            "documents_indexed": len(self.documents),
            "model_initialized": self.bm25_model is not None,
            "parameters": {
                "k1": self.k1,
                "b": self.b
            }
        }


class HybridSearchService:
    """
    Service de recherche hybride combinant embeddings denses et BM25.
    
    Combine les résultats des embeddings vectoriels (dense) et BM25 (sparse)
    pour améliorer la pertinence globale de la recherche.
    """
    
    def __init__(self, vector_service, bm25_service: BM25Service, alpha: float = 0.7):
        """
        Initialise le service de recherche hybride.
        
        Args:
            vector_service: Service d'embeddings vectoriels
            bm25_service (BM25Service): Service BM25
            alpha (float): Poids des embeddings vs BM25 (0.0 = BM25 seul, 1.0 = embeddings seul)
        """
        self.vector_service = vector_service
        self.bm25_service = bm25_service
        self.alpha = alpha  # Poids des embeddings (1-alpha = poids BM25)
        
        logger.info(f"🔀 Service de recherche hybride initialisé (alpha={alpha})")
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 5, 
        use_mmr: bool = True,
        lambda_mult: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Recherche hybride combinant embeddings et BM25.
        
        Args:
            query (str): Requête textuelle
            k (int): Nombre de résultats finaux
            use_mmr (bool): Utiliser MMR pour les embeddings
            lambda_mult (float): Facteur de diversité MMR
            
        Returns:
            List[Dict]: Documents avec scores hybrides
        """
        try:
            logger.info(f"🔀 Recherche hybride: '{query[:50]}...'")
            
            # 1. Recherche vectorielle (embeddings)
            vector_results = []
            if self.vector_service:
                try:
                    # Utiliser similarity_search_with_score pour obtenir les scores
                    vector_results_langchain = self.vector_service.similarity_search_with_score(
                        query=query,
                        k=k * 2, # Récupérer plus de candidats pour la fusion
                        use_mmr=use_mmr,
                        lambda_mult=lambda_mult
                    )
                    
                    # Convertir en format standard
                    for doc, score in vector_results_langchain:
                        vector_results.append({
                            "document": doc,
                            "score": float(score),  # Score réel
                            "source": "vector"
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Erreur recherche vectorielle: {e}")
                    # Fallback vers recherche simple
                    try:
                        vector_results_langchain = self.vector_service.similarity_search(
                            query=query,
                            k=k * 2,
                            use_mmr=use_mmr,
                            lambda_mult=lambda_mult
                        )
                        # Convertir en format avec scores
                        for doc in vector_results_langchain:
                            vector_results.append({
                                "document": doc,
                                "score": 1.0,  # Score par défaut
                                "source": "vector"
                            })
                    except Exception as e2:
                        logger.warning(f"⚠️ Erreur recherche vectorielle fallback: {e2}")
            
            # 2. Recherche BM25 (sparse)
            bm25_results = []
            if self.bm25_service:
                try:
                    bm25_docs_with_scores = self.bm25_service.search(query, k=k*2)
                    
                    # Normaliser les scores BM25
                    if bm25_docs_with_scores:
                        max_score = max(score for _, score in bm25_docs_with_scores)
                        for doc, score in bm25_docs_with_scores:
                            normalized_score = score / max_score if max_score > 0 else 0.0
                            bm25_results.append({
                                "document": doc,
                                "score": normalized_score,
                                "source": "bm25"
                            })
                except Exception as e:
                    logger.warning(f"⚠️ Erreur recherche BM25: {e}")
            
            # 3. Combiner les résultats
            combined_results = self._combine_results(vector_results, bm25_results, k)
            
            logger.info(f"✅ Recherche hybride terminée: {len(combined_results)} résultats")
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche hybride: {e}")
            # Fallback vers recherche vectorielle seule
            if self.vector_service:
                try:
                    return self.vector_service.search(query, k, use_mmr, lambda_mult)
                except:
                    return []
            return []
    
    def _combine_results(
        self, 
        vector_results: List[Dict], 
        bm25_results: List[Dict], 
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine les résultats vectoriels et BM25.
        
        Args:
            vector_results (List[Dict]): Résultats des embeddings
            bm25_results (List[Dict]): Résultats BM25
            k (int): Nombre de résultats finaux
            
        Returns:
            List[Dict]: Résultats combinés
        """
        # Créer un dictionnaire pour combiner les scores
        combined_scores = {}
        
        # Ajouter les scores vectoriels
        for result in vector_results:
            doc_id = self._get_document_id(result["document"])
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "document": result["document"],
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "combined_score": 0.0
                }
            combined_scores[doc_id]["vector_score"] = result["score"]
        
        # Ajouter les scores BM25
        for result in bm25_results:
            doc_id = self._get_document_id(result["document"])
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "document": result["document"],
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "combined_score": 0.0
                }
            combined_scores[doc_id]["bm25_score"] = result["score"]
        
        # Calculer les scores combinés
        for doc_id, scores in combined_scores.items():
            vector_score = scores["vector_score"]
            bm25_score = scores["bm25_score"]
            
            # Score hybride pondéré
            combined_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            scores["combined_score"] = combined_score
        
        # Trier par score combiné et retourner les k meilleurs
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # Retourner les documents avec métadonnées de scoring
        final_results = []
        for result in sorted_results[:k]:
            doc = result["document"]
            
            # Convertir Document LangChain en dictionnaire si nécessaire
            if hasattr(doc, 'page_content'):
                # C'est un objet Document LangChain
                doc_dict = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "0"),
                    "_hybrid_score": result["combined_score"],
                    "_vector_score": result["vector_score"],
                    "_bm25_score": result["bm25_score"],
                    **doc.metadata
                }
            else:
                # C'est déjà un dictionnaire
                doc_dict = doc.copy()
                doc_dict["_hybrid_score"] = result["combined_score"]
                doc_dict["_vector_score"] = result["vector_score"]
                doc_dict["_bm25_score"] = result["bm25_score"]
            
            final_results.append(doc_dict)
        
        return final_results
    
    def _get_document_id(self, document) -> str:
        """
        Génère un ID unique pour un document.
        
        Args:
            document: Document (Dict ou Document LangChain)
            
        Returns:
            str: ID unique du document
        """
        # Gérer les objets Document LangChain
        if hasattr(document, 'metadata'):
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", "0")
        else:
            # C'est un dictionnaire
            source = document.get("source", "unknown")
            page = document.get("page", "0")
        
        return f"{source}_{page}"


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
        use_reranking: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_bm25: bool = True,
        hybrid_alpha: float = 0.7,
    ):
        """
        Initialise le service de vector store optimisé.

        Args:
            qdrant_url (str): URL du serveur Qdrant (défaut: "http://localhost:6333")
            collection_name (str): Nom de la collection Qdrant (défaut: "rag_newsletter")
            embedding_service: Service d'embeddings MLX pour générer les vecteurs
            use_binary_quantization (bool): Activer la quantization binaire pour économiser l'espace
            hnsw_config (Optional[Dict]): Configuration HNSW personnalisée
            use_reranking (bool): Activer le reranking Cross-Encoder (défaut: True)
            reranker_model (str): Modèle Cross-Encoder à utiliser (défaut: ms-marco-MiniLM-L-6-v2)
            use_bm25 (bool): Activer BM25 pour recherche hybride (défaut: True)
            hybrid_alpha (float): Poids des embeddings vs BM25 (défaut: 0.7)

        Raises:
            RuntimeError: Si la connexion à Qdrant échoue
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.use_binary_quantization = use_binary_quantization
        self.use_reranking = use_reranking
        self.reranker_model = reranker_model
        self.use_bm25 = use_bm25 and BM25_AVAILABLE
        self.hybrid_alpha = hybrid_alpha
        self.client = None
        self.vector_store = None
        self.reranker = None
        self.bm25_service = None
        self.hybrid_service = None

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
        
        # Initialiser le reranker si activé
        if self.use_reranking:
            self._initialize_reranker()
        
        # Initialiser BM25 si activé
        if self.use_bm25:
            self._initialize_bm25()

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

    def _initialize_reranker(self):
        """
        Initialise le Cross-Encoder pour le reranking.
        
        Cette méthode charge le modèle Cross-Encoder qui sera utilisé pour
        reranker les résultats de recherche et améliorer la pertinence.
        """
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("⚠️ Cross-Encoder non disponible, reranking désactivé")
            self.use_reranking = False
            return
        
        try:
            logger.info(f"🔄 Chargement du reranker Cross-Encoder: {self.reranker_model}")
            self.reranker = CrossEncoder(self.reranker_model)
            logger.info("✅ Reranker Cross-Encoder initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du reranker: {e}")
            logger.warning("⚠️ Reranking désactivé")
            self.use_reranking = False
            self.reranker = None

    def _initialize_bm25(self):
        """
        Initialise le service BM25 pour la recherche hybride.
        
        Cette méthode crée le service BM25 et le service de recherche hybride
        pour combiner les embeddings denses avec la recherche textuelle sparse.
        """
        try:
            logger.info("🔍 Initialisation du service BM25...")
            self.bm25_service = BM25Service()
            self.hybrid_service = HybridSearchService(
                vector_service=self,
                bm25_service=self.bm25_service,
                alpha=self.hybrid_alpha
            )
            
            # Initialiser BM25 avec les documents existants
            self._load_existing_documents_to_bm25()
            
            logger.info("✅ Service BM25 initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation BM25: {e}")
            self.use_bm25 = False
            self.bm25_service = None
            self.hybrid_service = None

    def _load_existing_documents_to_bm25(self):
        """
        Charge les documents existants dans BM25.
        
        Cette méthode récupère tous les documents du vector store et les indexe
        avec BM25 pour permettre la recherche hybride.
        """
        try:
            if not self.bm25_service:
                return
                
            logger.info("📚 Chargement des documents existants pour BM25...")
            
            # Récupérer tous les points de la collection
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Limite raisonnable
                with_payload=True
            )[0]
            
            if not points:
                logger.warning("⚠️ Aucun document trouvé dans la collection")
                return
            
            # Convertir les points en documents BM25
            bm25_docs = []
            for point in points:
                payload = point.payload
                if payload and 'page_content' in payload:
                    bm25_doc = {
                        'content': payload['page_content'],
                        'source': payload.get('source', 'Document inconnu'),
                        'page': payload.get('page', 'N/A')
                    }
                    bm25_docs.append(bm25_doc)
            
            if bm25_docs:
                # Indexer avec BM25
                success = self.bm25_service.add_documents(bm25_docs)
                if success:
                    logger.info(f"✅ {len(bm25_docs)} documents chargés dans BM25")
                else:
                    logger.warning("⚠️ Échec chargement documents BM25")
            else:
                logger.warning("⚠️ Aucun document valide trouvé pour BM25")
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement documents BM25: {e}")

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
            
            # Indexer aussi avec BM25 si activé
            if self.use_bm25 and self.bm25_service:
                try:
                    # Convertir les documents LangChain en format BM25
                    bm25_docs = []
                    for doc in documents:
                        bm25_doc = {
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page", "0"),
                            **doc.metadata
                        }
                        bm25_docs.append(bm25_doc)
                    
                    # Indexer avec BM25
                    bm25_success = self.bm25_service.add_documents(bm25_docs)
                    if bm25_success:
                        logger.info(f"✅ Documents indexés avec BM25: {len(bm25_docs)}")
                    else:
                        logger.warning("⚠️ Échec indexation BM25")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur indexation BM25: {e}")
            
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

    def search_with_reranking(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict] = None,
        rerank_candidates: int = 20
    ) -> List[Tuple[Document, float]]:
        """
        Recherche avec reranking Cross-Encoder pour une pertinence optimale.
        
        Cette méthode combine HNSW + MMR pour la récupération initiale, puis utilise
        un Cross-Encoder pour reranker les candidats et améliorer la pertinence.
        
        Args:
            query (str): Requête textuelle à rechercher
            k (int): Nombre de résultats finaux à retourner
            filter (Optional[Dict]): Filtres de métadonnées à appliquer
            rerank_candidates (int): Nombre de candidats à reranker (défaut: 20)
            
        Returns:
            List[Tuple[Document, float]]: Documents rerankés avec scores de pertinence
            
        Raises:
            RuntimeError: Si le reranker n'est pas disponible
        """
        if not self.use_reranking or not self.reranker:
            logger.warning("⚠️ Reranking non disponible, utilisation de la recherche standard")
            return self.similarity_search_with_score(query, k, filter)
        
        try:
            logger.info(f"🎯 Recherche avec reranking Cross-Encoder: '{query[:50]}...'")
            
            # 1. Récupération initiale avec plus de candidats
            logger.info(f"🔍 Récupération de {rerank_candidates} candidats...")
            initial_results = self._perform_search(query, rerank_candidates, filter)
            
            if not initial_results:
                logger.warning("⚠️ Aucun candidat trouvé pour le reranking")
                return []
            
            # 2. Préparer les paires (query, document) pour le reranking
            query_doc_pairs = []
            documents = []
            
            for doc, score in initial_results:
                query_doc_pairs.append([query, doc.page_content])
                documents.append((doc, score))
            
            # 3. Reranking avec Cross-Encoder
            logger.info(f"🔄 Reranking de {len(query_doc_pairs)} candidats...")
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # 4. Combiner les scores et trier
            reranked_results = []
            for i, (doc, original_score) in enumerate(documents):
                rerank_score = float(rerank_scores[i])
                # Combiner le score original et le score de reranking
                combined_score = (original_score + rerank_score) / 2
                reranked_results.append((doc, combined_score))
            
            # 5. Trier par score combiné et retourner les k meilleurs
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            final_results = reranked_results[:k]
            
            logger.info(f"✅ Reranking terminé: {len(final_results)} résultats optimisés")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du reranking: {e}")
            logger.warning("⚠️ Fallback vers recherche standard")
            return self.similarity_search_with_score(query, k, filter)
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict] = None,
        use_mmr: bool = True,
        lambda_mult: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Recherche hybride combinant embeddings denses et BM25.
        
        Cette méthode combine les résultats des embeddings vectoriels (dense)
        et BM25 (sparse) pour améliorer la pertinence globale.
        
        Args:
            query (str): Requête textuelle à rechercher
            k (int): Nombre de résultats à retourner
            filter (Optional[Dict]): Filtres de métadonnées à appliquer
            use_mmr (bool): Utiliser MMR pour les embeddings
            lambda_mult (float): Facteur de diversité MMR
            
        Returns:
            List[Dict]: Documents avec scores hybrides
            
        Raises:
            RuntimeError: Si le service hybride n'est pas disponible
        """
        if not self.use_bm25 or not self.hybrid_service:
            logger.warning("⚠️ Recherche hybride non disponible, fallback vers recherche standard")
            return self.similarity_search(query, k, filter, use_mmr, lambda_mult)
        
        try:
            logger.info(f"🔀 Recherche hybride: '{query[:50]}...'")
            
            # Utiliser le service hybride
            results = self.hybrid_service.hybrid_search(
                query=query,
                k=k,
                use_mmr=use_mmr,
                lambda_mult=lambda_mult
            )
            
            logger.info(f"✅ Recherche hybride terminée: {len(results)} résultats")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche hybride: {e}")
            logger.warning("⚠️ Fallback vers recherche standard")
            return self.similarity_search(query, k, filter, use_mmr, lambda_mult)

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
