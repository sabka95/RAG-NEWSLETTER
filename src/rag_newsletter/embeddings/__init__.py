"""
Module d'embeddings optimisé pour Apple Silicon avec MLX et MCDSE
"""

from .embedding_service import EmbeddingService
from .vector_store import VectorStoreService

__all__ = [
    "EmbeddingService",
    "VectorStoreService"
]

__version__ = "0.2.0"