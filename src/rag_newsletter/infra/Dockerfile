# Dockerfile optimisé pour Apple Silicon avec MLX
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="RAG Newsletter Team"
LABEL description="RAG Newsletter with MLX optimizations for Apple Silicon"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY pyproject.toml poetry.lock ./

# Installation de Poetry
RUN pip install poetry==1.7.1

# Configuration de Poetry
RUN poetry config virtualenvs.create false

# Installation des dépendances Python
RUN poetry install --no-dev --no-interaction --no-ansi

# Installation spécifique de MLX pour Linux (fallback)
# Note: MLX est optimisé pour macOS, mais on peut installer une version compatible
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    sentence-transformers

# Copie du code source
COPY src/ ./src/
COPY downloads/ ./downloads/
COPY examples/ ./examples/

# Création des répertoires nécessaires
RUN mkdir -p /app/models /app/logs /app/cache

# Configuration des permissions
RUN chmod +x src/rag_newsletter/__main__.py

# Variables d'environnement pour les optimisations
ENV MLX_ENABLED=false  # Désactivé par défaut dans Docker (optimisé pour macOS)
ENV HNSW_ENABLED=true
ENV BINARY_QUANTIZATION=true
ENV MMR_LAMBDA=0.7
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_LEVEL=INFO

# Exposition des ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Point d'entrée
ENTRYPOINT ["python", "-m", "rag_newsletter"]

# Commande par défaut
CMD ["--help"]
