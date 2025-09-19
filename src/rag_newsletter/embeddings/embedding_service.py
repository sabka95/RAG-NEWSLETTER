from typing import List, Dict, Any
from langchain.schema import Document
import logging
import math
import io
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch
from loguru import logger

class MLXEmbeddingService:
    def __init__(self, model_name: str = "marco/mcdse-2b-v1", dimension: int = 1536):
        """
        Service d'embeddings utilisant MLX avec le modèle MCDSE-2B-V1
        
        Args:
            model_name: Nom du modèle HuggingFace à utiliser
            dimension: Dimension des embeddings de sortie
        """
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self.processor = None
        self._initialize_model()
        
        # Prompts pour l'encodage
        self.document_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"
        self.query_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>"
    
    def _initialize_model(self):
        """Initialise le modèle MCDSE avec MLX"""
        try:
            logger.info(f"Chargement du modèle MCDSE: {self.model_name}")
            logger.info("⏳ Configuration pour Apple Silicon M4...")
            
            # Configuration pour Apple Silicon
            min_pixels = 1 * 28 * 28
            max_pixels = 960 * 28 * 28
            
            # Charger le processeur
            logger.info("📥 Chargement du processeur...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            
            # Charger le modèle avec optimisations Apple Silicon
            logger.info("📥 Chargement du modèle principal...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            ).eval()
            
            # Déplacer le modèle sur MPS après chargement
            if torch.backends.mps.is_available():
                logger.info("🍎 Déplacement du modèle vers Apple Silicon MPS...")
                self.model = self.model.to('mps')
            
            # Configuration du padding
            self.model.padding_side = "left"
            self.processor.tokenizer.padding_side = "left"
            
            logger.info("✅ Modèle MCDSE chargé avec succès sur Apple Silicon!")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    
    def _smart_resize(self, height: int, width: int, max_pixels: int = 960 * 28 * 28, min_pixels: int = 1 * 28 * 28) -> tuple[int, int]:
        """Redimensionne intelligemment une image selon les contraintes du modèle"""
        def round_by_factor(number: float, factor: int) -> int:
            return round(number / factor) * factor
        
        def ceil_by_factor(number: float, factor: int) -> int:
            return math.ceil(number / factor) * factor
        
        def floor_by_factor(number: float, factor: int) -> int:
            return math.floor(number / factor) * factor
        
        h_bar = max(28, round_by_factor(height, 28))
        w_bar = max(28, round_by_factor(width, 28))
        
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, 28)
            w_bar = floor_by_factor(width / beta, 28)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, 28)
            w_bar = ceil_by_factor(width * beta, 28)
        
        return h_bar, w_bar
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Redimensionne une image selon les contraintes du modèle"""
        new_size = self._smart_resize(image.height, image.width)
        return image.resize(new_size)
    
    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimise une image pour l'embedding (alias pour _resize_image)"""
        return self._resize_image(image)
    
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
        
        embeddings = []
        
        logger.info(f"🖼️  Génération des embeddings pour {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            try:
                # Récupérer les données de l'image depuis les métadonnées
                image_data = doc.metadata.get('image_data')
                if not image_data:
                    logger.warning(f"Pas de données d'image pour le document {doc.metadata.get('source_file')}")
                    continue
                
                # Convertir les données en image PIL
                try:
                    if isinstance(image_data, bytes):
                        image = Image.open(io.BytesIO(image_data))
                    else:
                        image = Image.open(image_data)
                    
                    # Vérifier que l'image est valide
                    if image.size[0] == 0 or image.size[1] == 0:
                        logger.warning(f"Image invalide pour le document {doc.metadata.get('source_file')}")
                        continue
                        
                except Exception as img_error:
                    logger.warning(f"Erreur lors du traitement de l'image pour {doc.metadata.get('source_file')}: {img_error}")
                    continue
                
                # Générer l'embedding pour l'image
                embedding = self._encode_document(image)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Progression: {i + 1}/{len(documents)} documents traités")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'embedding du document {i}: {e}")
                continue
        
        logger.info(f"✅ Embeddings générés avec succès: {len(embeddings)} vecteurs")
        return embeddings
    
    def _encode_document(self, image: Image.Image) -> List[float]:
        """
        Encode un document (image) en embedding
        
        Args:
            image: Image PIL du document
            
        Returns:
            Vecteur d'embedding normalisé
        """
        try:
            # Préparer les inputs pour le modèle
            inputs = self.processor(
                text=[self.document_prompt],
                images=[self._resize_image(image)],
                videos=None,
                padding='longest',
                return_tensors='pt'
            )
            
            # Déplacer sur MPS (Metal Performance Shaders) pour Apple Silicon
            if torch.backends.mps.is_available():
                inputs = {k: v.to('mps') for k, v in inputs.items()}
            
            # Préparer les inputs pour la génération
            cache_position = torch.arange(0, 1)
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )
            
            # Générer l'embedding
            with torch.no_grad():
                output = self.model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True
                )
                
                # Extraire l'embedding du dernier état caché
                embeddings = output.hidden_states[-1][:, -1]
                
                # Normaliser et tronquer à la dimension souhaitée
                normalized_embedding = torch.nn.functional.normalize(
                    embeddings[:, :self.dimension], p=2, dim=-1
                )
                
                return normalized_embedding.cpu().squeeze().tolist()
                
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage du document: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Génère l'embedding pour une requête textuelle
        
        Args:
            query: Texte de la requête
            
        Returns:
            Vecteur d'embedding normalisé
        """
        try:
            logger.info(f"🔍 Génération d'embedding pour la requête: {query[:50]}...")
            
            # Créer une image factice pour les requêtes textuelles
            dummy_image = Image.new('RGB', (56, 56))
            
            # Préparer les inputs
            inputs = self.processor(
                text=[self.query_prompt % query],
                images=[dummy_image],
                videos=None,
                padding='longest',
                return_tensors='pt'
            )
            
            # Déplacer sur MPS si disponible
            if torch.backends.mps.is_available():
                inputs = {k: v.to('mps') for k, v in inputs.items()}
            
            # Préparer pour la génération
            cache_position = torch.arange(0, 1)
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )
            
            # Générer l'embedding
            with torch.no_grad():
                output = self.model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True
                )
                
                # Extraire et normaliser l'embedding
                embeddings = output.hidden_states[-1][:, -1]
                normalized_embedding = torch.nn.functional.normalize(
                    embeddings[:, :self.dimension], p=2, dim=-1
                )
                
                return normalized_embedding.cpu().squeeze().tolist()
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding de requête: {e}")
            # Fallback : retourner un vecteur zéro
            return [0.0] * self.dimension
    
    def embed_queries_batch(self, queries: List[str]) -> List[List[float]]:
        """
        Génère les embeddings pour plusieurs requêtes en batch
        
        Args:
            queries: Liste des requêtes textuelles
            
        Returns:
            Liste des vecteurs d'embedding
        """
        try:
            logger.info(f"🔍 Génération d'embeddings batch pour {len(queries)} requêtes...")
            
            # Créer des images factices pour toutes les requêtes
            dummy_images = [Image.new('RGB', (56, 56)) for _ in queries]
            
            # Préparer les inputs en batch
            inputs = self.processor(
                text=[self.query_prompt % q for q in queries],
                images=dummy_images,
                videos=None,
                padding='longest',
                return_tensors='pt'
            )
            
            # Déplacer sur MPS si disponible
            if torch.backends.mps.is_available():
                inputs = {k: v.to('mps') for k, v in inputs.items()}
            
            # Préparer pour la génération
            cache_position = torch.arange(0, inputs['input_ids'].shape[0])
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )
            
            # Générer les embeddings en batch
            with torch.no_grad():
                output = self.model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True
                )
                
                # Extraire et normaliser les embeddings
                embeddings = output.hidden_states[-1][:, -1]
                normalized_embeddings = torch.nn.functional.normalize(
                    embeddings[:, :self.dimension], p=2, dim=-1
                )
                
                return normalized_embeddings.cpu().tolist()
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings batch: {e}")
            # Fallback : retourner des vecteurs zéro
            return [[0.0] * self.dimension for _ in queries]

# Alias pour la compatibilité
EmbeddingService = MLXEmbeddingService
