"""
Configuration optimisée pour Apple Silicon M4
"""

# Configuration MLX pour Apple Silicon
MLX_CONFIG = {
    "device": "mps",  # Metal Performance Shaders
    "enable_gpu": True,
    "memory_fraction": 0.8,  # Utiliser 80% de la mémoire GPU
    "precision": "bfloat16",  # Précision optimisée pour M4
}

# Configuration du modèle MCDSE-2B-V1
MCDSE_CONFIG = {
    "model_name": "marco/mcdse-2b-v1",
    "max_pixels": 960 * 28 * 28,  # Limite du modèle
    "min_pixels": 1 * 28 * 28,
    "image_size": (56, 56),  # Taille par défaut pour les requêtes
    "dpi": 150,  # Résolution optimale pour M4
}

# Configuration HNSW optimisée pour Apple Silicon
HNSW_CONFIG = {
    "m": 16,  # Connexions par nœud (optimisé pour M4)
    "ef_construct": 100,  # Construction de l'index
    "ef": 64,  # Recherche
    "full_scan_threshold": 10000,  # Seuil pour scan complet
    "max_indexing_threads": 0,  # Auto-détection (utilise tous les cores M4)
}

# Configuration Binary Quantization
BINARY_QUANTIZATION_CONFIG = {
    "enabled": True,
    "always_ram": True,  # Garder en RAM pour M4 (beaucoup de RAM)
    "compression_ratio": 0.75,  # Réduction de 75% de l'espace
}

# Configuration MMR (Maximum Marginal Relevance)
MMR_CONFIG = {
    "enabled": True,
    "default_lambda": 0.7,  # Équilibre diversité/pertinence
    "lambda_range": (0.0, 1.0),  # Plage de valeurs
    "fetch_multiplier": 3,  # Multiplier k pour l'algorithme MMR
}

# Configuration du processeur de documents
DOCUMENT_PROCESSOR_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "batch_size": 10,  # Optimisé pour M4
    "max_concurrent_files": 4,  # Limite de concurrence
}

# Configuration des performances
PERFORMANCE_CONFIG = {
    "cache_embeddings": True,
    "use_batch_processing": True,
    "parallel_downloads": True,
    "max_memory_usage": "8GB",  # Limite pour M4
}

# Configuration du logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    "rotation": "100 MB",
    "retention": "7 days",
}

# Configuration Qdrant
QDRANT_CONFIG = {
    "url": "http://localhost:6333",
    "grpc_port": 6334,
    "collection_name": "rag_newsletter",
    "vector_size": 1536,  # Taille des embeddings MCDSE
    "distance": "Cosine",
    "on_disk": True,  # Stockage sur disque pour économiser la RAM
}

# Configuration SharePoint
SHAREPOINT_CONFIG = {
    "max_files_per_batch": 50,
    "supported_extensions": [".pdf", ".docx", ".pptx", ".xlsx", ".txt"],
    "max_file_size": "100MB",
    "download_timeout": 300,  # 5 minutes
    "retry_attempts": 3,
}

# Configuration des optimisations spécifiques M4
M4_OPTIMIZATIONS = {
    "use_metal_shaders": True,
    "optimize_for_neural_engine": True,
    "memory_pressure_handling": True,
    "thermal_throttling_protection": True,
    "cpu_cores_utilization": 10,  # 10 cores M4
    "gpu_cores_utilization": 10,  # 10 cores GPU M4
}

# Configuration des benchmarks
BENCHMARK_CONFIG = {
    "target_ingestion_speed": "2.2 docs/sec",  # Objectif M4
    "target_search_latency": "50ms",  # Objectif HNSW
    "target_memory_usage": "8GB",  # Limite M4
    "target_accuracy": 0.95,  # 95% de précision
}

def get_optimized_config():
    """Retourne la configuration optimisée complète"""
    return {
        "mlx": MLX_CONFIG,
        "mcdse": MCDSE_CONFIG,
        "hnsw": HNSW_CONFIG,
        "binary_quantization": BINARY_QUANTIZATION_CONFIG,
        "mmr": MMR_CONFIG,
        "document_processor": DOCUMENT_PROCESSOR_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "qdrant": QDRANT_CONFIG,
        "sharepoint": SHAREPOINT_CONFIG,
        "m4_optimizations": M4_OPTIMIZATIONS,
        "benchmarks": BENCHMARK_CONFIG,
    }

def validate_config():
    """Valide la configuration pour Apple Silicon M4"""
    import platform
    import torch
    
    # Vérifier le système
    if platform.system() != "Darwin":
        raise RuntimeError("Cette configuration est optimisée pour macOS")
    
    # Vérifier la disponibilité de MPS
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal Performance Shaders non disponible")
    
    # Vérifier MLX
    try:
        import mlx.core as mx
        logger.info("✅ MLX disponible")
    except ImportError:
        raise RuntimeError("MLX non installé")
    
    logger.info("✅ Configuration Apple Silicon M4 validée")
    return True
