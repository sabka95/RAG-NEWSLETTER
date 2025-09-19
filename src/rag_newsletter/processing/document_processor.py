from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
from langchain.schema import Document
import logging
import io
import math
from datetime import datetime
from PIL import Image, ImageOps
from loguru import logger

class OptimizedDocumentProcessor:
    def __init__(self, 
                 dpi: int = 150,
                 max_pixels: int = 960 * 28 * 28,
                 min_pixels: int = 1 * 28 * 28,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Processeur de documents optimisé pour le modèle MCDSE-2B-V1
        
        Args:
            dpi: Résolution pour la conversion PDF vers image
            max_pixels: Nombre maximum de pixels pour une image (contrainte du modèle)
            min_pixels: Nombre minimum de pixels pour une image
            chunk_size: Taille des chunks de texte
            chunk_overlap: Chevauchement entre les chunks
        """
        self.dpi = dpi
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _smart_resize(self, height: int, width: int) -> tuple[int, int]:
        """
        Redimensionne intelligemment une image selon les contraintes du modèle MCDSE
        
        Args:
            height: Hauteur originale
            width: Largeur originale
            
        Returns:
            Tuple (nouvelle_hauteur, nouvelle_largeur)
        """
        def round_by_factor(number: float, factor: int) -> int:
            return round(number / factor) * factor
        
        def ceil_by_factor(number: float, factor: int) -> int:
            return math.ceil(number / factor) * factor
        
        def floor_by_factor(number: float, factor: int) -> int:
            return math.floor(number / factor) * factor
        
        h_bar = max(28, round_by_factor(height, 28))
        w_bar = max(28, round_by_factor(width, 28))
        
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = floor_by_factor(height / beta, 28)
            w_bar = floor_by_factor(width / beta, 28)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, 28)
            w_bar = ceil_by_factor(width * beta, 28)
        
        return h_bar, w_bar
    
    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """
        Optimise une image pour le modèle MCDSE
        
        Args:
            image: Image PIL
            
        Returns:
            Image optimisée
        """
        # Redimensionner intelligemment
        new_size = self._smart_resize(image.height, image.width)
        optimized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Améliorer le contraste et la netteté si nécessaire
        if optimized_image.mode != 'RGB':
            optimized_image = optimized_image.convert('RGB')
        
        # Appliquer une légère amélioration du contraste
        optimized_image = ImageOps.autocontrast(optimized_image, cutoff=1)
        
        return optimized_image
    
    def _extract_text_from_page(self, page) -> str:
        """
        Extrait le texte d'une page PDF
        
        Args:
            page: Page PyMuPDF
            
        Returns:
            Texte extrait
        """
        try:
            # Extraire le texte avec métadonnées
            text_dict = page.get_text("dict")
            text_content = ""
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_content += span["text"] + " "
                        text_content += "\n"
            
            return text_content.strip()
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction du texte: {e}")
            return ""
    
    def _chunk_text(self, text: str, source_file: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Découpe le texte en chunks avec métadonnées
        
        Args:
            text: Texte à découper
            source_file: Nom du fichier source
            page_num: Numéro de page
            
        Returns:
            Liste des chunks avec métadonnées
        """
        if not text.strip():
            return []
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "start_word": i,
                    "end_word": min(i + self.chunk_size, len(words)),
                    "source_file": source_file,
                    "page_number": page_num,
                })
        
        return chunks
    
    def process_pdf(self, file_path: str, source_metadata: Optional[Dict] = None) -> List[Document]:
        """
        Traite un fichier PDF en optimisant pour le modèle MCDSE
        
        Args:
            file_path: Chemin vers le fichier PDF
            source_metadata: Métadonnées additionnelles
            
        Returns:
            Liste des documents (une page = un document avec image optimisée)
        """
        try:
            logger.info(f"📄 Traitement optimisé du fichier PDF: {file_path}")
            
            # Ouvrir le PDF avec PyMuPDF
            doc = fitz.open(file_path)
            documents = []
            
            for page_num in range(len(doc)):
                try:
                    # Récupérer la page
                    page = doc[page_num]
                    
                    # Extraire le texte de la page
                    page_text = self._extract_text_from_page(page)
                    
                    # Convertir la page en image avec optimisations
                    mat = fitz.Matrix(self.dpi/72, self.dpi/72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)  # Pas d'alpha pour optimiser
                    
                    # Convertir en PIL Image
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Optimiser l'image pour le modèle MCDSE
                    optimized_image = self._optimize_image(pil_image)
                    
                    # Convertir l'image optimisée en bytes
                    img_buffer = io.BytesIO()
                    optimized_image.save(img_buffer, format='PNG', optimize=True)
                    optimized_img_data = img_buffer.getvalue()
                    
                    # Créer les chunks de texte si nécessaire
                    text_chunks = self._chunk_text(page_text, Path(file_path).name, page_num + 1)
                    
                    if text_chunks:
                        # Créer un document par chunk
                        for chunk in text_chunks:
                            document = Document(
                                page_content=chunk["text"],
                                metadata={
                                    "source_file": Path(file_path).name,
                                    "file_type": "pdf",
                                    "page_number": page_num + 1,
                                    "total_pages": len(doc),
                                    "chunk_index": chunk["chunk_index"],
                                    "start_word": chunk["start_word"],
                                    "end_word": chunk["end_word"],
                                    "image_data": optimized_img_data,
                                    "image_format": "png",
                                    "dpi": self.dpi,
                                    "image_size": optimized_image.size,
                                    "processing_timestamp": datetime.now().isoformat(),
                                }
                            )
                            
                            # Ajouter les métadonnées source si fournies
                            if source_metadata:
                                document.metadata.update(source_metadata)
                            
                            documents.append(document)
                    else:
                        # Créer un document sans texte si la page est vide
                        document = Document(
                            page_content=f"Page {page_num + 1} du document {Path(file_path).name}",
                            metadata={
                                "source_file": Path(file_path).name,
                                "file_type": "pdf",
                                "page_number": page_num + 1,
                                "total_pages": len(doc),
                                "chunk_index": 0,
                                "image_data": optimized_img_data,
                                "image_format": "png",
                                "dpi": self.dpi,
                                "image_size": optimized_image.size,
                                "processing_timestamp": datetime.now().isoformat(),
                            }
                        )
                        
                        if source_metadata:
                            document.metadata.update(source_metadata)
                        
                        documents.append(document)
                    
                    logger.info(f"✅ Page {page_num + 1} optimisée: {len(text_chunks)} chunks, image {optimized_image.size}")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lors du traitement de la page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
            logger.info(f"🎉 Fichier traité avec succès: {len(documents)} documents générés")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du PDF {file_path}: {e}")
            raise
    
    def process_multiple_pdfs(self, file_paths: List[str], source_metadata: Optional[Dict] = None) -> Dict[str, List[Document]]:
        """
        Traite plusieurs fichiers PDF en parallèle (optimisé pour Apple Silicon)
        
        Args:
            file_paths: Liste des chemins vers les fichiers PDF
            source_metadata: Métadonnées additionnelles
            
        Returns:
            Dictionnaire {file_path: [documents]}
        """
        results = {}
        
        logger.info(f"📚 Traitement en lot de {len(file_paths)} fichiers PDF")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"[{i}/{len(file_paths)}] Traitement: {Path(file_path).name}")
                documents = self.process_pdf(file_path, source_metadata)
                results[file_path] = documents
                
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de {file_path}: {e}")
                results[file_path] = []
        
        total_documents = sum(len(docs) for docs in results.values())
        logger.info(f"🎯 Traitement en lot terminé: {total_documents} documents générés")
        
        return results
    
    def get_processing_stats(self, results: Dict[str, List[Document]]) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le traitement
        
        Args:
            results: Résultats du traitement
            
        Returns:
            Statistiques de traitement
        """
        total_files = len(results)
        total_documents = sum(len(docs) for docs in results.values())
        successful_files = sum(1 for docs in results.values() if docs)
        
        # Statistiques par fichier
        file_stats = []
        for file_path, documents in results.items():
            if documents:
                total_chunks = sum(doc.metadata.get('chunk_index', 0) + 1 for doc in documents)
                pages = len(set(doc.metadata.get('page_number', 0) for doc in documents))
                file_stats.append({
                    "file": Path(file_path).name,
                    "documents": len(documents),
                    "pages": pages,
                    "chunks": total_chunks,
                })
        
        return {
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": total_files - successful_files,
            "total_documents": total_documents,
            "file_details": file_stats,
        }

# Alias pour la compatibilité
DocumentProcessor = OptimizedDocumentProcessor