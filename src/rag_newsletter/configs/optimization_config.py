"""
Configuration des optimisations pour le système RAG
Optimisé pour Apple Silicon M4 avec GPU 10 cœurs
"""

from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class MLXConfig:
    """Configuration MLX pour Apple Silicon"""
    enabled: bool = True
    memory_efficient: bool = True
    compile: bool = True
    cache_models: bool = True

@dataclass
class HNSWConfig:
    """Configuration HNSW pour Qdrant"""
    enabled: bool = True
    m: int = 16  # Nombre de connexions pour chaque nœud
    ef_construct: int = 200  # Taille de la liste dynamique pendant la construction
    full_scan_threshold: int = 10000  # Seuil pour passer en scan complet
    max_indexing_threads: int = 4  # Nombre de threads pour l'indexation

@dataclass
class BinaryQuantizationConfig:
    """Configuration de la binary quantization"""
    enabled: bool = True
    scalar_type: str = "int8"
    quantile: float = 0.99
    always_ram: bool = True

@dataclass
class MMRConfig:
    """Configuration MMR (Maximum Marginal Relevance)"""
    enabled: bool = True
    lambda_param: float = 0.7  # 0.0 = diversité max, 1.0 = pertinence max
    max_candidates: int = 50  # Nombre max de candidats pour la sélection MMR

@dataclass
class EmbeddingConfig:
    """Configuration des embeddings"""
    model_name: str = "marco/mcdse-2b-v1"
    dimension: int = 1024
    batch_size: int = 8  # Optimisé pour M4
    max_pixels: int = 960 * 28 * 28
    min_pixels: int = 1 * 28 * 28

@dataclass
class PerformanceConfig:
    """Configuration des performances"""
    # Optimisations CPU/GPU pour M4
    cpu_threads: int = 8  # Utiliser tous les cœurs de performance
    gpu_memory_limit: int = 8192  # 8GB pour le GPU M4
    
    # Optimisations mémoire
    chunk_size: int = 1024
    max_concurrent_downloads: int = 4
    
    # Cache
    enable_embedding_cache: bool = True
    cache_size_mb: int = 512

class OptimizationConfig:
    """Configuration complète des optimisations"""
    
    def __init__(self, 
                 platform: str = "apple_silicon_m4",
                 workload: str = "balanced"):
        """
        Initialise la configuration selon la plateforme et la charge de travail
        
        Args:
            platform: Plateforme cible ("apple_silicon_m4", "general")
            workload: Type de charge ("performance", "balanced", "memory_efficient")
        """
        self.platform = platform
        self.workload = workload
        
        # Configuration par défaut
        self.mlx = MLXConfig()
        self.hnsw = HNSWConfig()
        self.binary_quantization = BinaryQuantizationConfig()
        self.mmr = MMRConfig()
        self.embedding = EmbeddingConfig()
        self.performance = PerformanceConfig()
        
        # Ajustements selon la plateforme
        self._configure_for_platform()
        self._configure_for_workload()
    
    def _configure_for_platform(self):
        """Configure selon la plateforme"""
        if self.platform == "apple_silicon_m4":
            # Optimisations spécifiques M4
            self.mlx.enabled = True
            self.mlx.memory_efficient = True
            self.performance.cpu_threads = 8
            self.performance.gpu_memory_limit = 8192
            
            # HNSW optimisé pour M4
            self.hnsw.m = 16
            self.hnsw.ef_construct = 200
            self.hnsw.max_indexing_threads = 4
            
            # Embeddings optimisés
            self.embedding.batch_size = 8
            self.embedding.dimension = 1024
            
        elif self.platform == "general":
            # Configuration générale
            self.mlx.enabled = False
            self.performance.cpu_threads = 4
            self.performance.gpu_memory_limit = 4096
            self.embedding.batch_size = 4
    
    def _configure_for_workload(self):
        """Configure selon la charge de travail"""
        if self.workload == "performance":
            # Priorité aux performances
            self.hnsw.ef_construct = 300
            self.mmr.max_candidates = 100
            self.performance.max_concurrent_downloads = 8
            
        elif self.workload == "memory_efficient":
            # Priorité à l'efficacité mémoire
            self.hnsw.ef_construct = 100
            self.mmr.max_candidates = 20
            self.embedding.batch_size = 2
            self.performance.max_concurrent_downloads = 2
            self.performance.cache_size_mb = 256
            
        elif self.workload == "balanced":
            # Équilibre performance/mémoire
            self.hnsw.ef_construct = 200
            self.mmr.max_candidates = 50
            self.embedding.batch_size = 4
            self.performance.max_concurrent_downloads = 4
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Retourne la configuration sous forme de dictionnaire"""
        return {
            "platform": self.platform,
            "workload": self.workload,
            "mlx": {
                "enabled": self.mlx.enabled,
                "memory_efficient": self.mlx.memory_efficient,
                "compile": self.mlx.compile,
                "cache_models": self.mlx.cache_models
            },
            "hnsw": {
                "enabled": self.hnsw.enabled,
                "m": self.hnsw.m,
                "ef_construct": self.hnsw.ef_construct,
                "full_scan_threshold": self.hnsw.full_scan_threshold,
                "max_indexing_threads": self.hnsw.max_indexing_threads
            },
            "binary_quantization": {
                "enabled": self.binary_quantization.enabled,
                "scalar_type": self.binary_quantization.scalar_type,
                "quantile": self.binary_quantization.quantile,
                "always_ram": self.binary_quantization.always_ram
            },
            "mmr": {
                "enabled": self.mmr.enabled,
                "lambda_param": self.mmr.lambda_param,
                "max_candidates": self.mmr.max_candidates
            },
            "embedding": {
                "model_name": self.embedding.model_name,
                "dimension": self.embedding.dimension,
                "batch_size": self.embedding.batch_size,
                "max_pixels": self.embedding.max_pixels,
                "min_pixels": self.embedding.min_pixels
            },
            "performance": {
                "cpu_threads": self.performance.cpu_threads,
                "gpu_memory_limit": self.performance.gpu_memory_limit,
                "chunk_size": self.performance.chunk_size,
                "max_concurrent_downloads": self.performance.max_concurrent_downloads,
                "enable_embedding_cache": self.performance.enable_embedding_cache,
                "cache_size_mb": self.performance.cache_size_mb
            }
        }
    
    def print_config(self):
        """Affiche la configuration"""
        print("🔧 Configuration des optimisations RAG")
        print(f"📱 Plateforme: {self.platform}")
        print(f"⚖️  Charge de travail: {self.workload}")
        print()
        
        print("🍎 MLX (Apple Silicon):")
        print(f"  - Activé: {self.mlx.enabled}")
        print(f"  - Mémoire efficace: {self.mlx.memory_efficient}")
        print(f"  - Compilation: {self.mlx.compile}")
        print()
        
        print("🏗️  HNSW (Index vectoriel):")
        print(f"  - Activé: {self.hnsw.enabled}")
        print(f"  - Connexions (m): {self.hnsw.m}")
        print(f"  - EF construct: {self.hnsw.ef_construct}")
        print(f"  - Threads indexation: {self.hnsw.max_indexing_threads}")
        print()
        
        print("🔢 Binary Quantization:")
        print(f"  - Activé: {self.binary_quantization.enabled}")
        print(f"  - Type: {self.binary_quantization.scalar_type}")
        print(f"  - Quantile: {self.binary_quantization.quantile}")
        print()
        
        print("🎯 MMR (Diversité):")
        print(f"  - Activé: {self.mmr.enabled}")
        print(f"  - Lambda: {self.mmr.lambda_param}")
        print(f"  - Candidats max: {self.mmr.max_candidates}")
        print()
        
        print("📊 Performance:")
        print(f"  - Threads CPU: {self.performance.cpu_threads}")
        print(f"  - Limite GPU: {self.performance.gpu_memory_limit}MB")
        print(f"  - Batch size: {self.embedding.batch_size}")
        print(f"  - Téléchargements concurrents: {self.performance.max_concurrent_downloads}")

# Configurations prédéfinies
M4_PERFORMANCE_CONFIG = OptimizationConfig("apple_silicon_m4", "performance")
M4_BALANCED_CONFIG = OptimizationConfig("apple_silicon_m4", "balanced")
M4_MEMORY_CONFIG = OptimizationConfig("apple_silicon_m4", "memory_efficient")
GENERAL_CONFIG = OptimizationConfig("general", "balanced")

# Configuration par défaut pour M4
DEFAULT_CONFIG = M4_BALANCED_CONFIG
