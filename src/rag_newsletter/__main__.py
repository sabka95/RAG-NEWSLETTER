import os, argparse, pathlib
from rag_newsletter.ingestion.sharepoint_client import make_client_from_env
from dotenv import load_dotenv

def main():
    load_dotenv() 
    p = argparse.ArgumentParser(description="RAG Newsletter - Importateur SharePoint")
    p.add_argument("--drive", default=os.getenv("SP_DRIVE_NAME", "Documents"))
    p.add_argument("--drive-id")
    p.add_argument("--max", type=int, default=100)
    p.add_argument("--download", action="store_true", help="Télécharger les fichiers")
    p.add_argument("--outdir", default="downloads", help="Répertoire de sortie")
    p.add_argument("--list-drives", action="store_true", help="Lister les drives disponibles")
    p.add_argument("--extensions", nargs="+", 
                   default=[".pdf", ".docx", ".pptx", ".xlsx", ".txt"],
                   help="Extensions de fichiers à importer")
    a = p.parse_args()

    try:
        sp = make_client_from_env()
        
        if a.list_drives:
            drives = sp.list_drives()
            print(f"Drives disponibles ({len(drives)}):")
            for i, drive in enumerate(drives, 1):
                print(f"  {i}. {drive['name']} (ID: {drive['id']})")
            return
        
        drive_id = a.drive_id
        if not drive_id:
            drive_id = sp.find_drive_id(a.drive)
            if not drive_id: 
                raise SystemExit(f"Drive '{a.drive}' introuvable. Utilise --drive-id ou ajuste SP_DRIVE_NAME.")

        files = sp.list_files(drive_id, exts=tuple(a.extensions))
        print(f"Fichiers trouvés: {len(files)}")
        
        # Afficher les fichiers (limités par --max)
        for f in files[:a.max]:
            size_mb = round(f.get('size', 0) / (1024 * 1024), 2)
            print(f"- {f['name']}  | {f['last_modified']} | {size_mb} MB")

        if a.download and files:
            print(f"\nTéléchargement des fichiers...")
            downloaded = sp.download_multiple(
                drive_id=drive_id,
                files=files,
                output_dir=a.outdir,
                max_files=a.max
            )
            
            summary = sp.get_download_summary(downloaded)
            print(f"\nRésumé du téléchargement:")
            print(f"- Fichiers téléchargés: {summary['total_files']}")
            print(f"- Taille totale: {summary['total_size_mb']} MB")
            print(f"- Extensions: {summary['extensions']}")
            print(f"- Répertoire: {pathlib.Path(a.outdir).absolute()}")
            
    except Exception as e:
        print(f"Erreur: {e}")
        return 1

if __name__ == "__main__":
    main()
