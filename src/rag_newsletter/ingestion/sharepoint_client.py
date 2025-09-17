from __future__ import annotations
import os, time, requests
from typing import Optional, List, Dict
from urllib.parse import urlparse
import msal
import logging
import pathlib
import json
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH = "https://graph.microsoft.com/v1.0"

class GraphTokenProvider:
    def __init__(self, tenant: str, client_id: str, secret: str):
        self.app = msal.ConfidentialClientApplication(
            client_id, authority=f"https://login.microsoftonline.com/{tenant}",
            client_credential=secret
        )
        self._tok, self._exp = None, 0

    def get(self) -> str:
        if not self._tok or time.time() > self._exp - 60:
            logger.info("Acquiring new token...")
            res = self.app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
            if "access_token" not in res:
                logger.error(f"Auth error: {res}")
                raise RuntimeError(f"Auth error: {res}")
            self._tok = res["access_token"]
            self._exp = time.time() + int(res.get("expires_in", 3600))
            logger.info("Token acquired successfully")
        return self._tok

class GraphClient:
    def __init__(self, tp: GraphTokenProvider): self.tp = tp; self.s = requests.Session()
    def get_json(self, url: str) -> dict:
        r = self.s.get(url, headers={"Authorization": f"Bearer {self.tp.get()}"}, timeout=30); r.raise_for_status(); return r.json()
    def stream(self, url: str):
        r = self.s.get(url, headers={"Authorization": f"Bearer {self.tp.get()}"}, timeout=60, stream=True); r.raise_for_status(); return r

class SharePointClient:
    def __init__(self, graph: GraphClient, site_url: Optional[str]=None, site_id: Optional[str]=None):
        self.g = graph
        self.site_id = site_id or self._resolve_site_id(site_url)

    def _resolve_site_id(self, site_url: str) -> str:
        u = urlparse(site_url)
        return self.g.get_json(f"{GRAPH}/sites/{u.netloc}:{u.path}")["id"]

    def list_drives(self) -> List[Dict]:
        return self.g.get_json(f"{GRAPH}/sites/{self.site_id}/drives").get("value", [])

    def find_drive_id(self, name: str) -> Optional[str]:
        for d in self.list_drives():
            if d.get("name","").casefold() == name.casefold(): return d["id"]
        return None

    def list_files(self, drive_id: str, folder_id: Optional[str]=None, exts=(".pdf",".docx",".pptx",".xlsx",".txt")) -> List[Dict]:
        url = f"{GRAPH}/drives/{drive_id}/root/children" if not folder_id else f"{GRAPH}/drives/{drive_id}/items/{folder_id}/children"
        logger.info(f"Listing files from: {url}")
        data = self.g.get_json(url)
        out: List[Dict] = []
        
        for it in data.get("value", []):
            if "folder" in it:
                logger.info(f"Exploring folder: {it['name']}")
                out += self.list_files(drive_id, it["id"], exts)
            elif "file" in it and it["name"].lower().endswith(exts):
                file_info = {
                    "id": it["id"], 
                    "name": it["name"], 
                    "last_modified": it.get("lastModifiedDateTime"), 
                    "web_url": it.get("webUrl"),
                    "size": it.get("size", 0)
                }
                out.append(file_info)
                logger.info(f"Found file: {it['name']} ({file_info['size']} bytes)")
        
        return out

    def download(self, drive_id: str, item_id: str, dest_path: str) -> str:
        logger.info(f"Downloading file {item_id} to {dest_path}")
        with self.g.stream(f"{GRAPH}/drives/{drive_id}/items/{item_id}/content") as r, open(dest_path, "wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
        logger.info(f"Download completed: {dest_path}")
        return dest_path
    
    def download_multiple(self, drive_id: str, files: List[Dict], output_dir: str = "downloads", 
                         max_files: Optional[int] = None) -> List[Dict]:
        """
        Télécharge plusieurs fichiers en une fois
        
        Args:
            drive_id: ID du drive SharePoint
            files: Liste des fichiers à télécharger
            output_dir: Répertoire de destination
            max_files: Nombre maximum de fichiers à télécharger
            
        Returns:
            Liste des fichiers téléchargés avec métadonnées
        """
        # Créer le répertoire de sortie
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Limiter le nombre de fichiers si spécifié
        if max_files:
            files = files[:max_files]
        
        downloaded_files = []
        
        for i, file_info in enumerate(files, 1):
            try:
                logger.info(f"[{i}/{len(files)}] Downloading: {file_info['name']}")
                
                # Créer un nom de fichier sécurisé
                safe_name = self._sanitize_filename(file_info['name'])
                dest_path = output_path / safe_name
                
                # Télécharger le fichier
                self.download(drive_id, file_info['id'], str(dest_path))
                
                # Ajouter les métadonnées
                file_metadata = {
                    **file_info,
                    'local_path': str(dest_path),
                    'downloaded_at': datetime.now().isoformat(),
                    'safe_name': safe_name
                }
                downloaded_files.append(file_metadata)
                
            except Exception as e:
                logger.error(f"Error downloading {file_info['name']}: {e}")
                continue
        
        # Sauvegarder les métadonnées
        metadata_file = output_path / "download_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(downloaded_files, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Download completed: {len(downloaded_files)} files to {output_path}")
        return downloaded_files
    
    def _sanitize_filename(self, filename: str) -> str:
        """Nettoie le nom de fichier pour éviter les caractères problématiques"""
        import re
        # Remplacer les caractères problématiques
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limiter la longueur
        if len(safe_name) > 200:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:200-len(ext)] + ext
        return safe_name
    
    def get_download_summary(self, downloaded_files: List[Dict]) -> Dict:
        """Retourne un résumé des téléchargements"""
        if not downloaded_files:
            return {"message": "Aucun fichier téléchargé"}
        
        total_size = sum(f.get('size', 0) for f in downloaded_files)
        extensions = {}
        for f in downloaded_files:
            ext = pathlib.Path(f['name']).suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        return {
            "total_files": len(downloaded_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "extensions": extensions,
            "files": [f['name'] for f in downloaded_files]
        }

def make_client_from_env() -> SharePointClient:
    tp = GraphTokenProvider(os.environ["AZURE_TENANT_ID"], os.environ["AZURE_CLIENT_ID"], os.environ["AZURE_CLIENT_SECRET"])
    gc = GraphClient(tp)
    site_id = os.getenv("SP_SITE_ID")
    if site_id:
        return SharePointClient(gc, site_id=site_id)
    return SharePointClient(gc, site_url=os.environ["SP_SITE_URL"])
