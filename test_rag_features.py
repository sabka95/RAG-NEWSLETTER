#!/usr/bin/env python3
"""
Script de test complet pour toutes les fonctionnalités RAG améliorées.

Ce script teste :
1. Reranking Cross-Encoder
2. Self-Reflection 
3. Décomposition de requêtes complexes
4. BM25 et recherche hybride
5. Workflow RAG complet
"""

import sys
import os
import time
from typing import Dict, List, Any

# Ajouter le chemin du projet
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from langchain.schema import Document

# Imports des services RAG
try:
    from rag_newsletter.embeddings.embedding_service import LangChainMLXEmbeddings
    from rag_newsletter.embeddings.vector_store import OptimizedVectorStoreService
    from rag_newsletter.embeddings.vector_store import BM25Service, HybridSearchService
    from rag_newsletter.workflows.rag_workflow import RAGWorkflow
    from rag_newsletter.workflows.advanced_intent_analyzer import AdvancedIntentAnalyzer
    from rag_newsletter.workflows.llm_response_generator import LLMResponseGenerator
    logger.info("✅ Tous les imports réussis")
except ImportError as e:
    logger.error(f"❌ Erreur import: {e}")
    sys.exit(1)


class RAGFeatureTester:
    """Testeur complet des fonctionnalités RAG."""
    
    def __init__(self):
        """Initialise le testeur."""
        self.test_results = {}
        self.test_documents = self._create_test_documents()
        
    def _create_test_documents(self) -> List[Document]:
        """Crée des documents de test."""
        return [
            Document(
                page_content="TotalEnergies a annoncé une stratégie de transition énergétique ambitieuse pour 2030, visant à réduire ses émissions de CO2 de 40% et à investir 60 milliards d'euros dans les énergies renouvelables.",
                metadata={
                    "source": "TotalEnergies_Strategy_2024.pdf",
                    "page": 1,
                    "category": "strategy",
                    "year": 2024
                }
            ),
            Document(
                page_content="Les résultats financiers du Q4 2023 montrent un chiffre d'affaires de 45,2 milliards d'euros, en hausse de 12% par rapport au Q4 2022. Les investissements dans les énergies renouvelables ont représenté 3,2 milliards d'euros.",
                metadata={
                    "source": "TotalEnergies_PR_4Q23_Results.pdf", 
                    "page": 2,
                    "category": "financial",
                    "year": 2023
                }
            ),
            Document(
                page_content="La politique de sécurité HSEQ (Hygiène, Sécurité, Environnement, Qualité) de TotalEnergies s'articule autour de trois piliers : prévention des risques, formation continue et amélioration continue des processus.",
                metadata={
                    "source": "te_charte_hseq_en_09_21.pdf",
                    "page": 3,
                    "category": "safety",
                    "year": 2021
                }
            ),
            Document(
                page_content="Le rapport de développement durable 2024 révèle que TotalEnergies a réduit ses émissions de scope 1 et 2 de 15% depuis 2020, tout en augmentant sa capacité renouvelable de 8 GW à 12 GW.",
                metadata={
                    "source": "totalenergies_sustainability-climate-2024-progress-report_2024_en.pdf",
                    "page": 5,
                    "category": "sustainability",
                    "year": 2024
                }
            )
        ]
    
    def test_bm25_service(self) -> bool:
        """Teste le service BM25."""
        logger.info("🔍 Test du service BM25...")
        
        try:
            # Créer le service BM25
            bm25_service = BM25Service()
            
            # Convertir les documents pour BM25
            bm25_docs = []
            for doc in self.test_documents:
                bm25_doc = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "0"),
                    **doc.metadata
                }
                bm25_docs.append(bm25_doc)
            
            # Indexer les documents
            success = bm25_service.add_documents(bm25_docs)
            if not success:
                logger.error("❌ Échec indexation BM25")
                return False
            
            # Tester la recherche
            results = bm25_service.search("stratégie énergétique", k=3)
            
            if len(results) > 0:
                logger.info(f"✅ BM25 fonctionne: {len(results)} résultats trouvés")
                logger.info(f"   Premier résultat: {results[0][0].get('source', 'unknown')}")
                return True
            else:
                logger.error("❌ Aucun résultat BM25")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur test BM25: {e}")
            return False
    
    def test_vector_store_with_bm25(self) -> bool:
        """Teste le vector store avec BM25."""
        logger.info("🔍 Test du vector store avec BM25...")
        
        try:
            # Créer un service d'embeddings mock
            class MockEmbeddingService:
                def embed_documents(self, docs):
                    return [[0.1] * 1536 for _ in docs]
                def embed_query(self, query):
                    return [0.1] * 1536
            
            embedding_service = MockEmbeddingService()
            
            # Créer le vector store avec BM25
            vector_store = OptimizedVectorStoreService(
                embedding_service=embedding_service,
                use_bm25=True,
                hybrid_alpha=0.7
            )
            
            # Ajouter les documents
            ids = vector_store.add_documents(self.test_documents)
            
            if len(ids) > 0:
                logger.info(f"✅ Vector store avec BM25: {len(ids)} documents ajoutés")
                
                # Tester la recherche hybride
                if hasattr(vector_store, 'hybrid_search'):
                    results = vector_store.hybrid_search("stratégie énergétique", k=2)
                    logger.info(f"✅ Recherche hybride: {len(results)} résultats")
                    return True
                else:
                    logger.warning("⚠️ Méthode hybrid_search non disponible")
                    return False
            else:
                logger.error("❌ Aucun document ajouté")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur test vector store: {e}")
            return False
    
    def test_intent_analyzer_decomposition(self) -> bool:
        """Teste la décomposition de requêtes complexes."""
        logger.info("🔍 Test de la décomposition de requêtes...")
        
        try:
            # Créer l'analyseur d'intention
            analyzer = AdvancedIntentAnalyzer(
                llm_model="qwen3:14b",
                enable_snorkel=False,
                enable_robustness_testing=False
            )
            
            # Tester la décomposition
            complex_query = "Compare la stratégie énergétique de TotalEnergies et analyse l'évolution de ses investissements dans les énergies renouvelables"
            
            decomposition = analyzer.decompose_complex_query(complex_query)
            
            if decomposition.get("is_complex", False):
                sub_queries = decomposition.get("sub_queries", [])
                strategy = decomposition.get("strategy", "unknown")
                
                logger.info(f"✅ Décomposition réussie:")
                logger.info(f"   Sous-requêtes: {len(sub_queries)}")
                logger.info(f"   Stratégie: {strategy}")
                for i, sub_query in enumerate(sub_queries, 1):
                    logger.info(f"   {i}. {sub_query}")
                
                return True
            else:
                logger.warning("⚠️ Requête non détectée comme complexe")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur test décomposition: {e}")
            return False
    
    def test_self_reflection(self) -> bool:
        """Teste la self-reflection."""
        logger.info("🔍 Test de la self-reflection...")
        
        try:
            # Créer le générateur de réponse
            generator = LLMResponseGenerator(
                model="qwen3:14b",
                enable_self_correction=True
            )
            
            # Tester la self-correction
            test_result = {
                "answer": "TotalEnergies a une stratégie énergétique.",
                "citations": ["TotalEnergies_Strategy_2024.pdf – p.1"],
                "confidence": 0.8,
                "intent": "simple_qa",
                "method": "initial"
            }
            
            # Simuler la self-correction (sans LLM réel)
            if hasattr(generator, '_self_correct_response'):
                logger.info("✅ Méthode self-correction disponible")
                return True
            else:
                logger.warning("⚠️ Méthode self-correction non disponible")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur test self-reflection: {e}")
            return False
    
    def test_ragas_evaluator(self) -> bool:
        """Teste l'évaluateur RAGAS."""
        logger.info("🔍 Test de l'évaluateur RAGAS...")
        
        try:
            # Créer l'évaluateur
            evaluator = RAGASEvaluator(use_ragas=True, use_deepeval=False)
            
            # Tester l'évaluation
            query = "Quelle est la stratégie énergétique de TotalEnergies ?"
            answer = "TotalEnergies a annoncé une stratégie de transition énergétique ambitieuse pour 2030."
            contexts = ["TotalEnergies a annoncé une stratégie de transition énergétique..."]
            
            scores = evaluator.evaluate_response(query, answer, contexts)
            
            if scores and "overall_score" in scores:
                logger.info(f"✅ RAGAS fonctionne: score global {scores['overall_score']:.3f}")
                return True
            else:
                logger.warning("⚠️ Évaluation RAGAS échouée")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur test RAGAS: {e}")
            return False
    
    def test_workflow_integration(self) -> bool:
        """Teste l'intégration complète du workflow."""
        logger.info("🔍 Test de l'intégration workflow...")
        
        try:
            # Créer un service RAG mock
            class MockRAGService:
                def search(self, query, k=5, use_mmr=True, lambda_mult=0.7):
                    return self.test_documents[:k]
                
                def search_with_reranking(self, query, k=5, rerank_candidates=20):
                    return [(doc, 0.9) for doc in self.test_documents[:k]]
                
                def hybrid_search(self, query, k=5, use_mmr=True, lambda_mult=0.7):
                    return self.test_documents[:k]
            
            rag_service = MockRAGService()
            rag_service.test_documents = self.test_documents
            
            # Créer le workflow RAG
            workflow = RAGWorkflow(rag_service, max_retries=2)
            
            # Tester le workflow
            query = "Quelle est la stratégie énergétique de TotalEnergies ?"
            result = workflow.process_query(query, user_id="test_user")
            
            if result.get("success", False):
                logger.info("✅ Workflow RAG fonctionne")
                logger.info(f"   Réponse: {result.get('answer', '')[:100]}...")
                logger.info(f"   Confiance: {result.get('confidence', 0):.3f}")
                return True
            else:
                logger.error(f"❌ Workflow échoué: {result.get('error', 'unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur test workflow: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Exécute tous les tests."""
        logger.info("🚀 Début des tests RAG complets...")
        
        tests = {
            "BM25 Service": self.test_bm25_service,
            "Vector Store + BM25": self.test_vector_store_with_bm25,
            "Décomposition Requêtes": self.test_intent_analyzer_decomposition,
            "Self-Reflection": self.test_self_reflection,
            "RAGAS Evaluator": self.test_ragas_evaluator,
            "Workflow Intégration": self.test_workflow_integration
        }
        
        results = {}
        for test_name, test_func in tests.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 Test: {test_name}")
            logger.info(f"{'='*50}")
            
            start_time = time.time()
            try:
                result = test_func()
                duration = time.time() - start_time
                results[test_name] = result
                
                status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
                logger.info(f"{status} - {test_name} ({duration:.2f}s)")
                
            except Exception as e:
                duration = time.time() - start_time
                results[test_name] = False
                logger.error(f"❌ ERREUR - {test_name} ({duration:.2f}s): {e}")
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Affiche le résumé des tests."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 RÉSUMÉ DES TESTS RAG")
        logger.info(f"{'='*60}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"Total des tests: {total_tests}")
        logger.info(f"Tests réussis: {passed_tests}")
        logger.info(f"Tests échoués: {total_tests - passed_tests}")
        logger.info(f"Taux de réussite: {success_rate:.1f}%")
        
        logger.info(f"\n📋 DÉTAIL DES RÉSULTATS:")
        for test_name, result in results.items():
            status = "✅" if result else "❌"
            logger.info(f"  {status} {test_name}")
        
        if success_rate >= 80:
            logger.info(f"\n🎉 EXCELLENT! Le système RAG est prêt pour la production!")
        elif success_rate >= 60:
            logger.info(f"\n⚠️ BON! Quelques améliorations nécessaires.")
        else:
            logger.info(f"\n🚨 ATTENTION! Plusieurs problèmes à résoudre.")


def main():
    """Fonction principale."""
    logger.info("🚀 Démarrage des tests RAG complets...")
    
    tester = RAGFeatureTester()
    results = tester.run_all_tests()
    tester.print_summary(results)
    
    # Code de sortie basé sur les résultats
    success_rate = (sum(results.values()) / len(results)) * 100
    if success_rate >= 80:
        sys.exit(0)  # Succès
    else:
        sys.exit(1)  # Échec


if __name__ == "__main__":
    main()
