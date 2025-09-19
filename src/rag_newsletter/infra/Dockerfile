# Dockerfile optimisé pour Apple Silicon M4 avec MLX
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="RAG Newsletter Team"
LABEL description="RAG Newsletter avec MLX et MCDSE-2B optimisé pour Apple Silicon"

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
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry
RUN pip install poetry==1.7.1

# Configuration de Poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers de configuration
COPY pyproject.toml poetry.lock* ./

# Installation des dépendances Python
RUN poetry install --no-dev && rm -rf $POETRY_CACHE_DIR

# Copie du code source
COPY src/ ./src/
COPY downloads/ ./downloads/

# Création du répertoire pour les logs
RUN mkdir -p /app/logs

# Exposition des ports
EXPOSE 8501 6333

# Variables d'environnement pour MLX
ENV MLX_ENABLE_GPU=1
ENV MLX_DEVICE=mps

# Commande par défaut
CMD ["poetry", "run", "python", "-m", "rag_newsletter", "--help"]
