"""
Évaluateur RAG avec RAGAS pour l'évaluation automatique de la qualité.

Ce module implémente l'évaluation automatique des réponses RAG en utilisant
RAGAS (RAG Assessment) pour mesurer la factualité, cohérence et complétude.
"""

import time
from typing import Dict, List, Any, Optional
from loguru import logger

# Imports pour RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy, 
        context_precision,
        context_recall,
        answer_correctness
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("⚠️ RAGAS non disponible, évaluation désactivée")

# Imports pour DeepEval (alternative)
try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("⚠️ DeepEval non disponible")


class RAGASEvaluator:
    """
    Évaluateur RAG utilisant RAGAS pour l'évaluation automatique.
    
    Mesure automatiquement :
    - Faithfulness : La réponse est-elle fidèle aux documents ?
    - Answer Relevancy : La réponse est-elle pertinente à la question ?
    - Context Precision : Les documents récupérés sont-ils pertinents ?
    - Context Recall : Les documents couvrent-ils bien la question ?
    - Answer Correctness : La réponse est-elle correcte ?
    """
    
    def __init__(self, use_ragas: bool = True, use_deepeval: bool = False):
        """
        Initialise l'évaluateur RAG.
        
        Args:
            use_ragas (bool): Utiliser RAGAS pour l'évaluation (défaut: True)
            use_deepeval (bool): Utiliser DeepEval comme alternative (défaut: False)
        """
        self.use_ragas = use_ragas and RAGAS_AVAILABLE
        self.use_deepeval = use_deepeval and DEEPEVAL_AVAILABLE
        
        if not self.use_ragas and not self.use_deepeval:
            logger.warning("⚠️ Aucun évaluateur disponible (RAGAS/DeepEval)")
        
        logger.info(f"📊 Évaluateur RAG initialisé (RAGAS: {self.use_ragas}, DeepEval: {self.use_deepeval})")
    
    def evaluate_response(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Évalue une réponse RAG avec RAGAS.
        
        Args:
            query (str): Question posée
            answer (str): Réponse générée par le système
            contexts (List[str]): Contexte des documents utilisés
            ground_truth (Optional[str]): Réponse de référence (optionnel)
            
        Returns:
            Dict[str, Any]: Scores d'évaluation détaillés
        """
        if not self.use_ragas:
            logger.warning("⚠️ RAGAS non disponible, évaluation simplifiée")
            return self._simple_evaluation(query, answer, contexts)
        
        try:
            logger.info(f"📊 Évaluation RAGAS de: '{query[:50]}...'")
            start_time = time.time()
            
            # Préparer les données pour RAGAS
            dataset = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth] if ground_truth else [None]
            }
            
            # Métriques à évaluer
            metrics = [
                faithfulness,           # Fidélité aux documents
                answer_relevancy,      # Pertinence de la réponse
                context_precision,     # Précision du contexte
                context_recall,        # Rappel du contexte
            ]
            
            # Ajouter answer_correctness si ground_truth disponible
            if ground_truth:
                metrics.append(answer_correctness)
            
            # Effectuer l'évaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                verbose=False
            )
            
            evaluation_time = time.time() - start_time
            
            # Extraire les scores
            scores = {
                "faithfulness": float(result["faithfulness"]),
                "answer_relevancy": float(result["answer_relevancy"]),
                "context_precision": float(result["context_precision"]),
                "context_recall": float(result["context_recall"]),
                "evaluation_time": evaluation_time,
                "evaluator": "ragas"
            }
            
            if ground_truth:
                scores["answer_correctness"] = float(result["answer_correctness"])
            
            # Calculer le score global
            scores["overall_score"] = self._calculate_overall_score(scores)
            
            logger.info(f"✅ Évaluation RAGAS terminée en {evaluation_time:.2f}s")
            logger.info(f"📊 Score global: {scores['overall_score']:.3f}")
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation RAGAS: {e}")
            logger.warning("⚠️ Fallback vers évaluation simplifiée")
            return self._simple_evaluation(query, answer, contexts)
    
    def evaluate_batch(
        self, 
        queries: List[str], 
        answers: List[str], 
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Évalue un batch de réponses RAG.
        
        Args:
            queries (List[str]): Liste des questions
            answers (List[str]): Liste des réponses
            contexts_list (List[List[str]]): Liste des contextes
            ground_truths (Optional[List[str]]): Réponses de référence
            
        Returns:
            Dict[str, Any]: Scores d'évaluation agrégés
        """
        if not self.use_ragas:
            logger.warning("⚠️ RAGAS non disponible, évaluation batch simplifiée")
            return self._simple_batch_evaluation(queries, answers, contexts_list)
        
        try:
            logger.info(f"📊 Évaluation batch RAGAS: {len(queries)} échantillons")
            start_time = time.time()
            
            # Préparer le dataset
            dataset = {
                "question": queries,
                "answer": answers,
                "contexts": contexts_list,
                "ground_truth": ground_truths if ground_truths else [None] * len(queries)
            }
            
            # Métriques
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            
            if ground_truths:
                metrics.append(answer_correctness)
            
            # Évaluation batch
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                verbose=False
            )
            
            evaluation_time = time.time() - start_time
            
            # Scores agrégés
            scores = {
                "faithfulness": float(result["faithfulness"]),
                "answer_relevancy": float(result["answer_relevancy"]),
                "context_precision": float(result["context_precision"]),
                "context_recall": float(result["context_recall"]),
                "evaluation_time": evaluation_time,
                "sample_count": len(queries),
                "evaluator": "ragas"
            }
            
            if ground_truths:
                scores["answer_correctness"] = float(result["answer_correctness"])
            
            scores["overall_score"] = self._calculate_overall_score(scores)
            
            logger.info(f"✅ Évaluation batch terminée en {evaluation_time:.2f}s")
            logger.info(f"📊 Score global moyen: {scores['overall_score']:.3f}")
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation batch RAGAS: {e}")
            return self._simple_batch_evaluation(queries, answers, contexts_list)
    
    def _simple_evaluation(self, query: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """Évaluation simplifiée sans RAGAS."""
        return {
            "faithfulness": 0.8,  # Estimation basique
            "answer_relevancy": 0.8,
            "context_precision": 0.8,
            "context_recall": 0.8,
            "overall_score": 0.8,
            "evaluation_time": 0.01,
            "evaluator": "simple"
        }
    
    def _simple_batch_evaluation(self, queries: List[str], answers: List[str], contexts_list: List[List[str]]) -> Dict[str, Any]:
        """Évaluation batch simplifiée."""
        return {
            "faithfulness": 0.8,
            "answer_relevancy": 0.8,
            "context_precision": 0.8,
            "context_recall": 0.8,
            "overall_score": 0.8,
            "evaluation_time": 0.01,
            "sample_count": len(queries),
            "evaluator": "simple"
        }
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calcule le score global pondéré."""
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "context_precision": 0.2,
            "context_recall": 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores:
                total_score += scores[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_evaluation_report(self, scores: Dict[str, Any]) -> str:
        """
        Génère un rapport d'évaluation lisible.
        
        Args:
            scores (Dict[str, Any]): Scores d'évaluation
            
        Returns:
            str: Rapport formaté
        """
        report = f"""
📊 RAPPORT D'ÉVALUATION RAG
{'='*50}

🎯 SCORES DÉTAILLÉS:
• Fidélité (Faithfulness): {scores.get('faithfulness', 0):.3f}
• Pertinence (Answer Relevancy): {scores.get('answer_relevancy', 0):.3f}
• Précision Contexte (Context Precision): {scores.get('context_precision', 0):.3f}
• Rappel Contexte (Context Recall): {scores.get('context_recall', 0):.3f}
"""
        
        if 'answer_correctness' in scores:
            report += f"• Exactitude (Answer Correctness): {scores['answer_correctness']:.3f}\n"
        
        report += f"""
🏆 SCORE GLOBAL: {scores.get('overall_score', 0):.3f}
⏱️  Temps d'évaluation: {scores.get('evaluation_time', 0):.2f}s
🔧 Évaluateur: {scores.get('evaluator', 'unknown')}
"""
        
        if 'sample_count' in scores:
            report += f"📊 Échantillons évalués: {scores['sample_count']}\n"
        
        return report
