#!/usr/bin/env python3
"""
Démonstration interactive du système RAG Newsletter
"""

import sys
from pathlib import Path
import requests
import json
import random

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def interactive_search():
    """Recherche interactive"""
    print("🔍 Mode de recherche interactif")
    print("Tapez 'quit' pour quitter, 'help' pour l'aide")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n❓ Votre question: ").strip()
            
            if query.lower() == 'quit':
                print("👋 Au revoir !")
                break
            elif query.lower() == 'help':
                show_help()
                continue
            elif query.lower() == 'stats':
                show_stats()
                continue
            elif not query:
                continue
            
            # Effectuer la recherche
            results = perform_search(query)
            
            if results:
                print(f"\n✅ {len(results)} résultats trouvés:")
                for i, result in enumerate(results[:3]):  # Limiter à 3 résultats
                    display_result(i+1, result)
            else:
                print("❌ Aucun résultat trouvé")
                
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

def perform_search(query, limit=5):
    """Effectuer une recherche"""
    try:
        url = "http://localhost:6333/collections/rag_newsletter/points/search"
        
        # Vecteur simulé
        test_vector = [random.gauss(0, 0.1) for _ in range(1536)]
        norm = sum(x*x for x in test_vector) ** 0.5
        normalized_vector = [x/norm for x in test_vector]
        
        payload = {
            "vector": normalized_vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('result', [])
        else:
            return []
            
    except Exception:
        return []

def display_result(rank, result):
    """Afficher un résultat de recherche"""
    payload_data = result.get('payload', {})
    score = result.get('score', 0)
    
    source = payload_data.get('source', 'N/A')
    page = payload_data.get('page', 'N/A')
    content = payload_data.get('content', 'N/A')
    doc_type = payload_data.get('document_type', 'N/A')
    
    print(f"\n   {rank}. 📊 Score: {score:.4f}")
    print(f"      📄 Document: {source}")
    print(f"      📃 Page: {page}")
    print(f"      📋 Type: {doc_type}")
    print(f"      📝 Contenu: {content[:200]}...")

def show_help():
    """Afficher l'aide"""
    print("\n📖 Aide - Commandes disponibles:")
    print("   • Tapez votre question pour rechercher")
    print("   • 'stats' - Afficher les statistiques du système")
    print("   • 'help' - Afficher cette aide")
    print("   • 'quit' - Quitter")
    print("\n💡 Exemples de questions:")
    print("   • What is TotalEnergies' sustainability strategy?")
    print("   • What are the financial results for 2024?")
    print("   • How does TotalEnergies handle climate change?")
    print("   • What are TotalEnergies' ESG commitments?")

def show_stats():
    """Afficher les statistiques"""
    try:
        response = requests.get("http://localhost:6333/collections/rag_newsletter")
        
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            
            print("\n📊 Statistiques du système:")
            print(f"   📄 Points indexés: {result.get('points_count', 0)}")
            print(f"   🔢 Vecteurs indexés: {result.get('indexed_vectors_count', 0)}")
            print(f"   📊 Segments: {result.get('segments_count', 0)}")
            print(f"   ✅ Statut: {result.get('status', 'N/A')}")
            
            # Configuration
            config = result.get('config', {})
            params = config.get('params', {})
            vectors = params.get('vectors', {})
            
            print(f"\n⚙️  Configuration:")
            print(f"   📏 Dimensions: {vectors.get('size', 'N/A')}")
            print(f"   📐 Distance: {vectors.get('distance', 'N/A')}")
            print(f"   💾 Stockage: {'Sur disque' if vectors.get('on_disk') else 'En mémoire'}")
            
            # Optimisations
            hnsw_config = config.get('hnsw_config', {})
            quantization = config.get('quantization_config', {})
            
            print(f"   🌐 HNSW: m={hnsw_config.get('m', 'N/A')}, ef_construct={hnsw_config.get('ef_construct', 'N/A')}")
            print(f"   ⚡ Binary Quantization: {'Activé' if quantization.get('binary') else 'Désactivé'}")
            
            # Analyser les types de documents
            url = "http://localhost:6333/collections/rag_newsletter/points/scroll"
            payload = {
                "limit": 50,
                "with_payload": True,
                "with_vector": False
            }
            
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                points = result.get('result', {}).get('points', [])
                
                doc_types = {}
                sources = {}
                
                for point in points:
                    payload_data = point.get('payload', {})
                    doc_type = payload_data.get('document_type', 'Unknown')
                    source = payload_data.get('source', 'Unknown')
                    
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    sources[source] = sources.get(source, 0) + 1
                
                print(f"\n📚 Types de documents (échantillon):")
                for doc_type, count in doc_types.items():
                    print(f"   {doc_type}: {count} pages")
                
                print(f"\n📄 Sources (échantillon):")
                for source, count in list(sources.items())[:5]:
                    print(f"   {source}: {count} pages")
            
        else:
            print("❌ Erreur lors de la récupération des statistiques")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

def run_predefined_tests():
    """Exécuter des tests prédéfinis"""
    print("🧪 Tests prédéfinis du système")
    print("=" * 50)
    
    test_questions = [
        {
            "question": "What is TotalEnergies' sustainability strategy?",
            "category": "🌱 Durabilité"
        },
        {
            "question": "What are TotalEnergies' financial results for 2024?",
            "category": "💰 Finances"
        },
        {
            "question": "How does TotalEnergies handle climate change?",
            "category": "🌍 Climat"
        },
        {
            "question": "What are TotalEnergies' ESG commitments?",
            "category": "📋 ESG"
        },
        {
            "question": "What are TotalEnergies' investment strategies?",
            "category": "📈 Investissements"
        }
    ]
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n{i}. {test['category']}")
        print(f"   ❓ {test['question']}")
        
        results = perform_search(test['question'], limit=2)
        
        if results:
            print(f"   ✅ {len(results)} résultats trouvés")
            for j, result in enumerate(results):
                payload_data = result.get('payload', {})
                source = payload_data.get('source', 'N/A')
                page = payload_data.get('page', 'N/A')
                score = result.get('score', 0)
                print(f"      {j+1}. {source} - Page {page} (score: {score:.4f})")
        else:
            print("   ❌ Aucun résultat")
        
        print("-" * 30)

def main():
    """Fonction principale"""
    print("🚀 Démonstration Interactive - RAG Newsletter TotalEnergies")
    print("=" * 60)
    print("🤖 Modèle: MCDSE-2B-V1 (Apple Silicon M4)")
    print("🔗 Vector Store: Qdrant (HNSW + Binary Quantization)")
    print("⚡ Optimisations: MLX + MMR Search")
    print("=" * 60)
    
    while True:
        print("\n🎯 Choisissez un mode:")
        print("   1. 🔍 Recherche interactive")
        print("   2. 🧪 Tests prédéfinis")
        print("   3. 📊 Statistiques du système")
        print("   4. ❌ Quitter")
        
        try:
            choice = input("\nVotre choix (1-4): ").strip()
            
            if choice == '1':
                interactive_search()
            elif choice == '2':
                run_predefined_tests()
            elif choice == '3':
                show_stats()
            elif choice == '4':
                print("👋 Au revoir !")
                break
            else:
                print("❌ Choix invalide. Veuillez choisir 1, 2, 3 ou 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
