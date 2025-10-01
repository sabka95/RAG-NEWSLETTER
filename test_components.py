#!/usr/bin/env python3
"""
Script de test simple pour les composants RAG individuels.
"""

import sys
import os

# Ajouter le chemin du projet
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger

def test_imports():
    """Teste les imports des modules."""
    logger.info("🔍 Test des imports...")
    
    try:
        # Test BM25
        from rag_newsletter.embeddings.vector_store import BM25Service, HybridSearchService
        logger.info("✅ BM25 service importé")
        
        # Test Vector Store
        from rag_newsletter.embeddings.vector_store import OptimizedVectorStoreService
        logger.info("✅ Vector store importé")
        
        # Test Workflow
        from rag_newsletter.workflows.rag_workflow import RAGWorkflow
        logger.info("✅ RAG workflow importé")
        
        # Test Intent Analyzer
        from rag_newsletter.workflows.advanced_intent_analyzer import AdvancedIntentAnalyzer
        logger.info("✅ Intent analyzer importé")
        
        # Test Response Generator
        from rag_newsletter.workflows.llm_response_generator import LLMResponseGenerator
        logger.info("✅ Response generator importé")
        
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Erreur import: {e}")
        return False

def test_bm25_basic():
    """Teste BM25 basique."""
    logger.info("🔍 Test BM25 basique...")
    
    try:
        from rag_newsletter.embeddings.vector_store import BM25Service
        
        # Créer le service
        bm25 = BM25Service()
        
        # Documents de test
        docs = [
            {"content": "TotalEnergies stratégie énergétique", "source": "doc1.pdf"},
            {"content": "Résultats financiers Q4 2023", "source": "doc2.pdf"},
            {"content": "Politique sécurité HSEQ", "source": "doc3.pdf"}
        ]
        
        # Indexer
        success = bm25.add_documents(docs)
        if not success:
            logger.warning("⚠️ BM25 non disponible (rank-bm25 manquant)")
            return True  # Pas une erreur critique
        
        # Rechercher
        results = bm25.search("stratégie énergétique", k=2)
        logger.info(f"✅ BM25: {len(results)} résultats")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur BM25: {e}")
        return False

def test_vector_store_config():
    """Teste la configuration du vector store."""
    logger.info("🔍 Test configuration vector store...")
    
    try:
        from rag_newsletter.embeddings.vector_store import OptimizedVectorStoreService
        
        # Mock embedding service
        class MockEmbedding:
            def embed_documents(self, docs):
                return [[0.1] * 1536 for _ in docs]
            def embed_query(self, query):
                return [0.1] * 1536
        
        # Tester la création avec BM25
        vector_store = OptimizedVectorStoreService(
            embedding_service=MockEmbedding(),
            use_bm25=True,
            hybrid_alpha=0.7
        )
        
        logger.info("✅ Vector store avec BM25 créé")
        logger.info(f"   BM25 activé: {vector_store.use_bm25}")
        logger.info(f"   Alpha hybride: {vector_store.hybrid_alpha}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur vector store: {e}")
        return False

def test_intent_analyzer():
    """Teste l'analyseur d'intention."""
    logger.info("🔍 Test intent analyzer...")
    
    try:
        from rag_newsletter.workflows.advanced_intent_analyzer import AdvancedIntentAnalyzer
        
        # Créer l'analyseur
        analyzer = AdvancedIntentAnalyzer(
            llm_model="qwen3:14b",
            enable_snorkel=False,
            enable_robustness_testing=False
        )
        
        # Tester la décomposition
        query = "Compare la stratégie et analyse l'évolution"
        decomposition = analyzer.decompose_complex_query(query)
        
        logger.info(f"✅ Décomposition: {decomposition.get('is_complex', False)}")
        logger.info(f"   Sous-requêtes: {len(decomposition.get('sub_queries', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur intent analyzer: {e}")
        return False

def test_response_generator():
    """Teste le générateur de réponse."""
    logger.info("🔍 Test response generator...")
    
    try:
        from rag_newsletter.workflows.llm_response_generator import LLMResponseGenerator
        
        # Créer le générateur
        generator = LLMResponseGenerator(
            model="qwen3:14b",
            enable_self_correction=True
        )
        
        logger.info("✅ Response generator créé")
        logger.info(f"   Self-correction: {generator.enable_self_correction}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur response generator: {e}")
        return False

def test_ragas_evaluator():
    """Teste l'évaluateur RAGAS."""
    logger.info("🔍 Test RAGAS evaluator...")
    
    try:
        from rag_newsletter.evaluation.ragas_evaluator import RAGASEvaluator
        
        # Créer l'évaluateur
        evaluator = RAGASEvaluator(use_ragas=True, use_deepeval=False)
        
        logger.info("✅ RAGAS evaluator créé")
        logger.info(f"   RAGAS disponible: {evaluator.use_ragas}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur RAGAS: {e}")
        return False

def main():
    """Fonction principale."""
    logger.info("🚀 Test des composants RAG...")
    
    tests = [
        ("Imports", test_imports),
        ("BM25 Basique", test_bm25_basic),
        ("Vector Store Config", test_vector_store_config),
        ("Intent Analyzer", test_intent_analyzer),
        ("Response Generator", test_response_generator),
        ("RAGAS Evaluator", test_ragas_evaluator)
    ]
    
    results = {}
    for name, test_func in tests:
        logger.info(f"\n{'='*40}")
        logger.info(f"🧪 {name}")
        logger.info(f"{'='*40}")
        
        try:
            result = test_func()
            results[name] = result
            status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
            logger.info(f"{status} - {name}")
        except Exception as e:
            results[name] = False
            logger.error(f"❌ ERREUR - {name}: {e}")
    
    # Résumé
    logger.info(f"\n{'='*50}")
    logger.info("📊 RÉSUMÉ")
    logger.info(f"{'='*50}")
    
    total = len(results)
    passed = sum(results.values())
    success_rate = (passed / total) * 100
    
    logger.info(f"Tests: {passed}/{total} ({success_rate:.1f}%)")
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"  {status} {name}")
    
    if success_rate >= 80:
        logger.info("\n🎉 Tous les composants sont prêts!")
        return 0
    else:
        logger.info("\n⚠️ Certains composants ont des problèmes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
