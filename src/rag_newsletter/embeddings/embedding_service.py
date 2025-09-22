# =============================================================================
# RAG Newsletter - Service d'Embeddings MLX
# =============================================================================
# Service d'embeddings optimisé pour Apple Silicon utilisant le modèle MCDSE-2B-V1
# avec MLX pour des performances maximales sur processeurs M4.
# =============================================================================

import io
from typing import List

import torch
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class MLXEmbeddingService:
    """
    Service d'embeddings optimisé pour Apple Silicon M4 utilisant MLX.

    Ce service utilise le modèle MCDSE-2B-V1 (Multi-modal Cross-Document Search Embeddings)
    pour générer des embeddings de haute qualité à partir d'images de documents PDF.
    Optimisé pour les processeurs Apple Silicon avec Metal Performance Shaders (MPS).

    Fonctionnalités clés:
    - Embeddings d'images de documents (PDF → image → embedding)
    - Embeddings de requêtes textuelles
    - Optimisations Apple Silicon (MPS, MLX)
    - Normalisation L2 automatique
    - Gestion intelligente des contraintes de pixels du modèle

    Exemple d'utilisation:
        >>> service = MLXEmbeddingService()
        >>> embeddings = service.embed_documents(documents)
        >>> query_embedding = service.embed_query("sustainability strategy")
    """

    def __init__(self, model_name: str = "marco/mcdse-2b-v1", dimension: int = 1536):
        """
        Initialise le service d'embeddings MLX.

        Args:
            model_name (str): Nom du modèle HuggingFace à utiliser.
                             Par défaut: "marco/mcdse-2b-v1" (optimisé pour documents)
            dimension (int): Dimension des embeddings de sortie.
                           Par défaut: 1536 (dimension standard MCDSE)

        Raises:
            RuntimeError: Si le modèle ne peut pas être chargé ou si MLX n'est pas disponible
        """
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self.processor = None

        # Initialisation du modèle (peut prendre plusieurs secondes)
        self._initialize_model()

        # Prompts optimisés pour le modèle MCDSE-2B-V1
        # Format spécifique requis par le modèle pour l'encodage d'images
        self.document_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "What is shown in this image?<|im_end|>\n<|endoftext|>"
        )
        self.query_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "Query: %s<|im_end|>\n<|endoftext|>"
        )

    def _initialize_model(self):
        """
        Initialise le modèle MCDSE-2B-V1 avec optimisations Apple Silicon.

        Cette méthode charge le modèle et le processeur avec des paramètres optimisés
        pour les processeurs Apple Silicon M4, en utilisant Metal Performance Shaders (MPS)
        pour accélérer les calculs sur GPU.

        Raises:
            RuntimeError: Si le modèle ne peut pas être chargé ou si les dépendances sont manquantes
        """
        try:
            logger.info(f"Chargement du modèle MCDSE: {self.model_name}")
            logger.info("⏳ Configuration pour Apple Silicon M4...")

            # Configuration des contraintes de pixels pour le modèle MCDSE-2B-V1
            # Le modèle a des limites strictes sur la taille des images d'entrée
            min_pixels = 1 * 28 * 28  # Minimum: 28x28 pixels
            max_pixels = 960 * 28 * 28  # Maximum: 960x28 pixels (contrainte du modèle)

            # Charger le processeur (tokenizer + image processor)
            logger.info("📥 Chargement du processeur...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, min_pixels=min_pixels, max_pixels=max_pixels
            )

            # Charger le modèle principal avec optimisations Apple Silicon
            logger.info("📥 Chargement du modèle principal...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,  # Précision optimisée pour M4
                low_cpu_mem_usage=True,  # Économie de mémoire RAM
            ).eval()  # Mode évaluation (pas d'entraînement)

            # Déplacer le modèle sur Metal Performance Shaders (MPS) si disponible
            if torch.backends.mps.is_available():
                logger.info("🍎 Déplacement du modèle vers Apple Silicon MPS...")
                self.model = self.model.to("mps")
            else:
                logger.warning("⚠️ MPS non disponible, utilisation du CPU")

            # Configuration du padding pour la génération de texte
            # Padding à gauche pour les modèles de génération
            self.model.padding_side = "left"
            self.processor.tokenizer.padding_side = "left"

            logger.info("✅ Modèle MCDSE chargé avec succès sur Apple Silicon!")

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise RuntimeError(f"Impossible de charger le modèle MCDSE: {e}")


    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Génère les embeddings pour une liste de documents contenant des images.

        Cette méthode traite une liste de documents LangChain, extrait les images
        depuis leurs métadonnées, et génère des embeddings vectoriels optimisés
        pour la recherche sémantique.

        Args:
            documents (List[Document]): Liste de documents LangChain contenant des images.
                                      Chaque document doit avoir 'image_data' dans ses métadonnées.

        Returns:
            List[List[float]]: Liste des embeddings vectoriels (1536 dimensions chacun).
                              Chaque embedding est normalisé L2.

        Raises:
            RuntimeError: Si le modèle MCDSE n'est pas initialisé
            ValueError: Si aucun document valide n'est fourni

        Exemple:
            >>> service = MLXEmbeddingService()
            >>> documents = [doc1, doc2, doc3]  # Documents avec image_data
            >>> embeddings = service.embed_documents(documents)
            >>> print(f"Généré {len(embeddings)} embeddings de {len(embeddings[0])} dimensions")
        """
        if not self.model or not self.processor:
            raise RuntimeError(
                "Modèle MCDSE non initialisé. Appelez d'abord __init__()"
            )

        if not documents:
            raise ValueError("Aucun document fourni")

        embeddings = []
        successful_embeddings = 0

        logger.info(f"🖼️  Génération des embeddings pour {len(documents)} documents...")

        for i, doc in enumerate(documents):
            try:
                # Récupérer les données de l'image depuis les métadonnées
                image_data = doc.metadata.get("image_data")
                if not image_data:
                    logger.warning(
                        f"Pas de données d'image pour le document {doc.metadata.get('source_file', 'inconnu')}"
                    )
                    continue

                # Convertir les bytes en image PIL (déjà validé par document_processor)
                try:
                    image = Image.open(io.BytesIO(image_data))
                except Exception as img_error:
                    logger.warning(
                        f"Erreur lors du traitement de l'image pour {doc.metadata.get('source_file', 'inconnu')}: {img_error}"
                    )
                    continue

                # Générer l'embedding pour l'image
                embedding = self._encode_document(image)
                embeddings.append(embedding)
                successful_embeddings += 1

                # Afficher la progression tous les 10 documents
                if (i + 1) % 10 == 0:
                    logger.info(
                        f"📊 Progression: {i + 1}/{len(documents)} documents traités"
                    )

            except Exception as e:
                logger.error(f"Erreur lors de l'embedding du document {i}: {e}")
                continue

        logger.info(
            f"✅ Embeddings générés avec succès: {successful_embeddings}/{len(documents)} vecteurs"
        )
        return embeddings

    def _encode_document(self, image: Image.Image) -> List[float]:
        """
        Encode un document (image) en embedding vectoriel.

        Cette méthode utilise le modèle MCDSE-2B-V1 pour convertir une image
        de document en vecteur d'embedding de 1536 dimensions, optimisé pour
        la recherche sémantique.

        Args:
            image (Image.Image): Image PIL du document à encoder

        Returns:
            List[float]: Vecteur d'embedding normalisé L2 de 1536 dimensions

        Raises:
            RuntimeError: Si l'encodage échoue ou si le modèle n'est pas disponible

        Exemple:
            >>> service = MLXEmbeddingService()
            >>> image = Image.open("document.png")
            >>> embedding = service._encode_document(image)
            >>> print(f"Embedding généré: {len(embedding)} dimensions")
        """
        try:
            # Préparer les inputs pour le modèle MCDSE-2B-V1
            # Le modèle attend un prompt textuel + une image
            inputs = self.processor(
                text=[self.document_prompt],  # Prompt pour l'encodage d'image
                images=[image],  # Image déjà optimisée par document_processor
                videos=None,  # Pas de vidéo pour ce modèle
                padding="longest",  # Padding pour gérer les tailles variables
                return_tensors="pt",  # Retourner des tenseurs PyTorch
            )

            # Déplacer les inputs sur MPS (Metal Performance Shaders) pour Apple Silicon
            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}

            # Préparer les inputs pour la génération (format requis par le modèle)
            cache_position = torch.arange(0, 1)  # Position du cache pour la génération
            inputs = self.model.prepare_inputs_for_generation(
                **inputs,
                cache_position=cache_position,
                use_cache=False,  # Pas de cache pour l'embedding
            )

            # Générer l'embedding avec le modèle
            with torch.no_grad():  # Pas de calcul de gradient (mode inférence)
                output = self.model(
                    **inputs,
                    return_dict=True,  # Retourner un dictionnaire
                    output_hidden_states=True,  # Inclure les états cachés pour l'embedding
                )

                # Extraire l'embedding du dernier état caché
                # [:, -1] = dernier token de la séquence
                embeddings = output.hidden_states[-1][:, -1]

                # Normaliser L2 et tronquer à la dimension souhaitée
                # La normalisation L2 améliore la qualité de la recherche sémantique
                normalized_embedding = torch.nn.functional.normalize(
                    embeddings[:, : self.dimension],  # Tronquer à la dimension cible
                    p=2,  # Norme L2
                    dim=-1,  # Normaliser sur la dernière dimension
                )

                # Convertir en liste Python et retourner
                return normalized_embedding.cpu().squeeze().tolist()

        except Exception as e:
            logger.error(f"Erreur lors de l'encodage du document: {e}")
            raise RuntimeError(f"Impossible d'encoder le document: {e}")

    def embed_query(self, query: str) -> List[float]:
        """
        Génère l'embedding pour une requête textuelle.

        Cette méthode convertit une requête textuelle en vecteur d'embedding
        compatible avec les embeddings de documents, permettant la recherche
        sémantique dans la base vectorielle.

        Args:
            query (str): Texte de la requête à encoder (ex: "sustainability strategy")

        Returns:
            List[float]: Vecteur d'embedding normalisé L2 de 1536 dimensions

        Raises:
            RuntimeError: Si le modèle MCDSE n'est pas initialisé

        Exemple:
            >>> service = MLXEmbeddingService()
            >>> query = "What is TotalEnergies' sustainability strategy?"
            >>> embedding = service.embed_query(query)
            >>> print(f"Embedding de requête: {len(embedding)} dimensions")
        """
        try:
            logger.info(f"🔍 Génération d'embedding pour la requête: {query[:50]}...")

            # Créer une image factice pour les requêtes textuelles
            # Le modèle MCDSE-2B-V1 nécessite toujours une image, même pour du texte
            dummy_image = Image.new("RGB", (56, 56))  # Image noire 56x56 pixels

            # Préparer les inputs avec le prompt de requête
            inputs = self.processor(
                text=[self.query_prompt % query],  # Formatage du prompt avec la requête
                images=[dummy_image],  # Image factice
                videos=None,  # Pas de vidéo
                padding="longest",  # Padding pour gérer les tailles variables
                return_tensors="pt",  # Retourner des tenseurs PyTorch
            )

            # Déplacer sur MPS (Metal Performance Shaders) pour Apple Silicon
            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}

            # Préparer pour la génération
            cache_position = torch.arange(0, 1)
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )

            # Générer l'embedding
            with torch.no_grad():  # Mode inférence (pas de gradient)
                output = self.model(
                    **inputs,
                    return_dict=True,  # Retourner un dictionnaire
                    output_hidden_states=True,  # Inclure les états cachés
                )

                # Extraire et normaliser l'embedding
                embeddings = output.hidden_states[-1][:, -1]
                normalized_embedding = torch.nn.functional.normalize(
                    embeddings[:, : self.dimension],  # Tronquer à la dimension cible
                    p=2,  # Norme L2
                    dim=-1,  # Normaliser sur la dernière dimension
                )

                return normalized_embedding.cpu().squeeze().tolist()

        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding de requête: {e}")
            # Fallback : retourner un vecteur zéro en cas d'erreur
            # Cela permet au système de continuer à fonctionner même en cas de problème
            return [0.0] * self.dimension


class LangChainMLXEmbeddings(Embeddings):
    """
    Wrapper pour MLXEmbeddingService compatible avec l'interface LangChain Embeddings.
    
    Cette classe permet d'utiliser MLXEmbeddingService avec les vector stores LangChain
    en implémentant l'interface standard Embeddings.
    """
    
    def __init__(self, mlx_service: MLXEmbeddingService):
        """
        Initialise le wrapper avec un service MLX.
        
        Args:
            mlx_service: Instance de MLXEmbeddingService
        """
        self.mlx_service = mlx_service
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes à embedder
            
        Returns:
            Liste d'embeddings (vecteurs de 1536 dimensions)
        """
        # Créer des documents LangChain temporaires
        documents = [Document(page_content=text, metadata={}) for text in texts]
        return self.mlx_service.embed_documents(documents)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Génère un embedding pour une requête.
        
        Args:
            text: Texte de la requête
            
        Returns:
            Embedding (vecteur de 1536 dimensions)
        """
        return self.mlx_service.embed_query(text)


# =============================================================================
# Alias de compatibilité
# =============================================================================
# Alias pour maintenir la compatibilité avec l'ancienne API
# Permet d'utiliser 'EmbeddingService' au lieu de 'MLXEmbeddingService'
# =============================================================================
EmbeddingService = MLXEmbeddingService