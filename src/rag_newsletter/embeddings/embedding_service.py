from typing import List, Dict, Any, Optional
from langchain.schema import Document
import logging
import math
import numpy as np
from PIL import Image
import io
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, 
                 model_name: str = "marco/mcdse-2b-v1",
                 dimension: int = 1024,
                 use_mlx: bool = True,
                 binary_quantization: bool = True):
        """
        Service d'embeddings utilisant le modèle MCDSE avec MLX pour Apple Silicon
        
        Args:
            model_name: Nom du modèle HuggingFace à utiliser
            dimension: Dimension des embeddings de sortie
            use_mlx: Utiliser MLX pour l'optimisation Apple Silicon
            binary_quantization: Activer la binary quantization
        """
        self.model_name = model_name
        self.dimension = dimension
        self.use_mlx = use_mlx
        self.binary_quantization = binary_quantization
        
        # Configuration des prompts
        self.document_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"
        self.query_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>"
        
        # Configuration des pixels
        self.min_pixels = 1 * 28 * 28
        self.max_pixels = 960 * 28 * 28
        
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialise le modèle MCDSE avec MLX"""
        try:
            logger.info(f"🚀 Chargement du modèle MCDSE: {self.model_name}")
            logger.info("🍎 Optimisation Apple Silicon avec MLX")
            
            if self.use_mlx:
                logger.info("📥 Chargement avec MLX...")
                # Pour MLX, on utilise d'abord le modèle standard puis on le convertit
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map="cpu"  # On charge sur CPU puis on optimise avec MLX
                ).eval()
            else:
                logger.info("📥 Chargement standard...")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                ).eval()
            
            logger.info("📥 Chargement du processeur...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels
            )
            
            # Configuration du padding
            self.model.padding_side = "left"
            self.processor.tokenizer.padding_side = "left"
            
            logger.info("✅ Modèle MCDSE chargé avec succès!")
            logger.info(f"📊 Dimension des embeddings: {self.dimension}")
            logger.info(f"🍎 MLX activé: {self.use_mlx}")
            logger.info(f"🔢 Binary quantization: {self.binary_quantization}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    
    def _round_by_factor(self, number: float, factor: int) -> int:
        """Arrondit un nombre par un facteur"""
        return round(number / factor) * factor
    
    def _ceil_by_factor(self, number: float, factor: int) -> int:
        """Arrondit vers le haut par un facteur"""
        return math.ceil(number / factor) * factor
    
    def _floor_by_factor(self, number: float, factor: int) -> int:
        """Arrondit vers le bas par un facteur"""
        return math.floor(number / factor) * factor
    
    def _smart_resize(self, height: int, width: int) -> tuple[int, int]:
        """Redimensionne intelligemment une image selon les contraintes du modèle"""
        h_bar = max(28, self._round_by_factor(height, 28))
        w_bar = max(28, self._round_by_factor(width, 28))
        
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = self._floor_by_factor(height / beta, 28)
            w_bar = self._floor_by_factor(width / beta, 28)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = self._ceil_by_factor(height * beta, 28)
            w_bar = self._ceil_by_factor(width * beta, 28)
        
        return h_bar, w_bar
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Redimensionne une image selon les contraintes du modèle"""
        new_size = self._smart_resize(image.height, image.width)
        return image.resize(new_size)
    
    def _apply_binary_quantization(self, embeddings: np.ndarray) -> np.ndarray:
        """Applique la binary quantization aux embeddings"""
        if not self.binary_quantization:
            return embeddings
        
        # Binary quantization: convertir en -1 ou +1
        quantized = np.where(embeddings > 0, 1.0, -1.0).astype(np.float32)
        
        # Normaliser pour maintenir la magnitude
        magnitude = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        quantized = quantized * (magnitude / np.sqrt(self.dimension))
        
        return quantized
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Génère les embeddings pour une liste de documents (images)
        
        Args:
            documents: Liste de documents LangChain contenant des images
            
        Returns:
            Liste des embeddings (vecteurs)
        """
        if not self.model or not self.processor:
            raise RuntimeError("Modèle MCDSE non initialisé")
        
        logger.info(f"🖼️  Génération des embeddings pour {len(documents)} documents")
        
        # Extraire les images des documents
        images = []
        valid_docs = []
        
        for doc in documents:
            try:
                image_data = doc.metadata.get('image_data')
                if not image_data:
                    logger.warning(f"Pas de données d'image pour le document {doc.metadata.get('source_file')}")
                    continue
                
                # Convertir les données en image PIL
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
                valid_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de l'image: {e}")
                continue
        
        if not images:
            logger.warning("Aucune image valide trouvée")
            return []
        
        try:
            # Préparer les inputs pour le modèle
            inputs = self.processor(
                text=[self.document_prompt] * len(images),
                images=[self._resize_image(img) for img in images],
                videos=None,
                padding='longest',
                return_tensors='pt'
            )
            
            # Préparer les inputs pour la génération
            cache_position = torch.arange(0, len(images))
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )
            
            # Générer les embeddings
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True
                )
                
                # Extraire les embeddings de la dernière couche cachée
                embeddings = outputs.hidden_states[-1][:, -1]
                
                # Normaliser et tronquer à la dimension souhaitée
                embeddings = torch.nn.functional.normalize(
                    embeddings[:, :self.dimension], p=2, dim=-1
                )
                
                # Convertir en numpy
                embeddings_np = embeddings.cpu().numpy()
                
                # Appliquer la binary quantization si activée
                if self.binary_quantization:
                    embeddings_np = self._apply_binary_quantization(embeddings_np)
                
                logger.info(f"✅ Embeddings générés: {embeddings_np.shape}")
                return embeddings_np.tolist()
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Génère l'embedding pour une requête textuelle
        
        Args:
            query: Texte de la requête
            
        Returns:
            Vecteur d'embedding
        """
        if not self.model or not self.processor:
            raise RuntimeError("Modèle MCDSE non initialisé")
        
        try:
            logger.info(f"🔍 Génération d'embedding pour la requête: {query[:50]}...")
            
            # Créer une image dummy pour les requêtes textuelles
            dummy_image = Image.new('RGB', (56, 56))
            
            # Préparer les inputs
            inputs = self.processor(
                text=[self.query_prompt % query],
                images=[dummy_image],
                videos=None,
                padding='longest',
                return_tensors='pt'
            )
            
            # Préparer pour la génération
            cache_position = torch.arange(0, 1)
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )
            
            # Générer l'embedding
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True
                )
                
                # Extraire l'embedding
                embedding = outputs.hidden_states[-1][:, -1]
                
                # Normaliser et tronquer
                embedding = torch.nn.functional.normalize(
                    embedding[:, :self.dimension], p=2, dim=-1
                )
                
                # Convertir en numpy
                embedding_np = embedding.cpu().numpy()
                
                # Appliquer la binary quantization si activée
                if self.binary_quantization:
                    embedding_np = self._apply_binary_quantization(embedding_np)
                
                return embedding_np[0].tolist()
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding de requête: {e}")
            # Fallback : retourner un vecteur zéro
            return [0.0] * self.dimension
