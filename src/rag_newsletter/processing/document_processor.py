from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from langchain.schema import Document
import logging
import io
from PIL import Image

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, dpi: int = 150):
        """
        Processeur de documents pour le RAG avec DSE (Document Screenshot Embedding)
        
        Args:
            dpi: Résolution pour la conversion PDF vers image
        """
        self.dpi = dpi
    
    def process_pdf(self, file_path: str, source_metadata: Dict = None) -> List[Document]:
        """
        Traite un fichier PDF en convertissant chaque page en image
        
        Args:
            file_path: Chemin vers le fichier PDF
            source_metadata: Métadonnées additionnelles
            
        Returns:
            Liste des documents (une page = un document avec image)
        """
        try:
            logger.info(f"Traitement du fichier PDF: {file_path}")
            
            # Ouvrir le PDF avec PyMuPDF
            doc = fitz.open(file_path)
            documents = []
            
            for page_num in range(len(doc)):
                # Récupérer la page
                page = doc[page_num]
                
                # Convertir la page en image
                mat = fitz.Matrix(self.dpi/72, self.dpi/72)  # 72 DPI par défaut
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Créer un document LangChain avec l'image
                document = Document(
                    page_content=f"Page {page_num + 1} du document {Path(file_path).name}",
                    metadata={
                        "source_file": Path(file_path).name,
                        "file_type": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": len(doc),
                        "image_data": img_data,  # Données de l'image
                        "image_format": "png",
                        "dpi": self.dpi
                    }
                )
                
                # Ajouter les métadonnées source si fournies
                if source_metadata:
                    document.metadata.update(source_metadata)
                
                documents.append(document)
                logger.info(f"Page {page_num + 1} convertie en image")
            
            doc.close()
            logger.info(f"Fichier traité: {len(documents)} pages converties en images")
            return documents
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du PDF {file_path}: {e}")
            raise
