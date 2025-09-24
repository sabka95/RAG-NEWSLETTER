"""
Workflow RAG avec LLM (Mistral 7B) pour une vraie compréhension sémantique.

Ce module remplace l'analyse par regex par une compréhension intelligente
utilisant Mistral 7B via Ollama pour l'analyse d'intention et la génération de réponses.
"""

import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from .security_filter import SecurityFilter
from .advanced_intent_analyzer import AdvancedIntentAnalyzer, QueryIntent
from .llm_response_generator import LLMResponseGenerator


class RAGState(TypedDict):
    """État du workflow RAG avec LLM."""
    
    # Input
    query: str
    user_id: Optional[str]
    session_id: Optional[str]
    
    # Security & Intent Analysis
    security_result: Dict[str, Any]
    intent_analysis: Dict[str, Any]
    
    # Document Retrieval
    retrieved_documents: List[Dict]
    filtered_documents: List[Dict]
    
    # Response Generation
    draft_answer: str
    final_answer: str
    citations: List[str]
    validation_result: Dict[str, Any]
    
    # Error Handling
    retry_count: int
    error_message: Optional[str]
    
    # Metadata
    processing_time: float
    confidence_score: float
    workflow_version: str


class RAGWorkflow:
    """
    Workflow RAG avec LLM (Mistral 7B) pour une compréhension sémantique intelligente.
    
    Remplace l'analyse par regex par une vraie compréhension utilisant :
    - LLMIntentAnalyzer : Analyse d'intention avec Mistral 7B
    - LLMResponseGenerator : Génération de réponses avec Mistral 7B
    """
    
    def __init__(self, rag_service, max_retries: int = 3, llm_model: str = "llama3.1:8b"):
        """
        Initialise le workflow RAG avec LLM avancé.
        
        Args:
            rag_service: Service RAG existant
            max_retries (int): Nombre maximum de tentatives
            llm_model (str): Modèle LLM à utiliser (défaut: llama3.1:8b)
        """
        self.rag_service = rag_service
        self.max_retries = max_retries
        self.llm_model = llm_model
        
        # Initialiser les composants
        self.security_filter = SecurityFilter()
        self.intent_analyzer = AdvancedIntentAnalyzer(
            llm_model=llm_model,
            enable_snorkel=True,
            enable_robustness_testing=True
        )
        self.response_generator = LLMResponseGenerator(model=llm_model, fallback_to_basic=True)
        
        # Créer le workflow
        self.workflow = self._create_workflow()
        
        logger.info(f"🚀 Workflow RAG avec LLM initialisé: {llm_model}")
    
    def _create_workflow(self) -> StateGraph:
        """Crée le workflow LangGraph avec LLM."""
        
        workflow = StateGraph(RAGState)
        
        # Ajouter les nœuds
        workflow.add_node("security", self._security_node)
        workflow.add_node("intent", self._intent_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("comparison", self._comparison_node)
        workflow.add_node("generation", self._generation_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Définir les transitions
        workflow.add_conditional_edges(
            "security",
            self._route_after_security,
            {
                "approved": "intent",
                "blocked": "error_handler",
                "clarification": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "intent",
            self._route_after_intent,
            {
                "comparison": "comparison",
                "standard": "retrieval",
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("retrieval", "generation")
        workflow.add_edge("comparison", "generation")
        
        workflow.add_conditional_edges(
            "generation",
            self._route_after_generation,
            {
                "validate": "validation",
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "validation",
            self._route_after_validation,
            {
                "retry": "generation",
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "error_handler",
            self._route_after_error,
            {
                "complete": END,
                "retry": "intent"
            }
        )
        
        # Définir le point d'entrée
        workflow.set_entry_point("security")
        
        # Compiler avec checkpointing
        return workflow.compile(checkpointer=MemorySaver())
    
    def _security_node(self, state: RAGState) -> RAGState:
        """Nœud de filtrage de sécurité."""
        logger.info("🔒 Filtrage de sécurité...")
        
        try:
            security_result = self.security_filter.filter_query(state["query"])
            state["security_result"] = security_result
            
            if security_result["status"] == "blocked":
                logger.warning(f"🚨 Requête bloquée: {security_result['reason']}")
                state["final_answer"] = f"🚨 Requête bloquée pour des raisons de sécurité: {security_result.get('message', 'Contenu inapproprié détecté')}"
                state["confidence_score"] = 0.0
                state["error_message"] = f"Security blocked: {security_result['reason']}"
            elif security_result["status"] == "needs_clarification":
                logger.info(f"⚠️ Clarification nécessaire: {security_result['message']}")
                state["final_answer"] = f"Veuillez préciser votre question. {security_result['message']}"
                state["confidence_score"] = 0.0
                state["error_message"] = "Clarification needed"
            else:
                logger.info("✅ Requête approuvée par le filtre de sécurité")
            
        except Exception as e:
            logger.error(f"❌ Erreur filtrage sécurité: {e}")
            state["final_answer"] = "Une erreur s'est produite lors du filtrage de sécurité."
            state["confidence_score"] = 0.0
            state["error_message"] = f"Security error: {e}"
        
        return state
    
    def _intent_node(self, state: RAGState) -> RAGState:
        """Nœud d'analyse d'intention avec LLM."""
        logger.info("🧠 Analyse d'intention avec LLM...")
        
        try:
            # Utiliser le LLM pour analyser l'intention
            intent_analysis = self.intent_analyzer.analyze_intent(state["query"])
            state["intent_analysis"] = intent_analysis
            
            intent = intent_analysis.get("intent", QueryIntent.SIMPLE_QA)
            confidence = intent_analysis.get("confidence", 0.0)
            method = intent_analysis.get("analysis_method", "unknown")
            
            logger.info(f"🎯 Intention détectée: {intent.value} (confiance: {confidence:.2f}, méthode: {method})")
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse d'intention: {e}")
            state["error_message"] = f"Erreur analyse d'intention: {e}"
            state["intent_analysis"] = {
                "intent": QueryIntent.SIMPLE_QA,
                "confidence": 0.0,
                "error": str(e)
            }
        
        return state
    
    def _retrieval_node(self, state: RAGState) -> RAGState:
        """Nœud de récupération de documents."""
        logger.info("📚 Récupération de documents...")
        
        try:
            intent_analysis = state.get("intent_analysis", {})
            intent = intent_analysis.get("intent", QueryIntent.SIMPLE_QA)
            
            # Récupération automatique avec HNSW + MMR
            logger.info("🔍 Recherche automatique (HNSW + MMR)")
            
            # Ajuster les paramètres selon l'intention
            if intent == QueryIntent.EVOLUTION_ANALYSIS:
                # Plus de documents pour l'analyse temporelle
                k = 10
                lambda_mult = 0.4  # Plus de diversité pour capturer différentes périodes
            elif intent == QueryIntent.COMPARISON:
                k = 8
                lambda_mult = 0.5
            elif intent == QueryIntent.COMPLEX_AGGREGATION:
                # Maximum de documents pour l'agrégation complexe
                k = 12
                lambda_mult = 0.3  # Maximum de diversité pour couvrir tous les aspects
            else:
                k = 5
                lambda_mult = 0.7
            
            documents = self.rag_service.search(
                state["query"], 
                k=k, 
                use_mmr=True,
                lambda_mult=lambda_mult
            )
            
            state["retrieved_documents"] = documents
            logger.info(f"✅ {len(documents)} documents récupérés")
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération: {e}")
            state["retrieved_documents"] = []
            state["error_message"] = f"Erreur récupération: {e}"
        
        return state
    
    def _comparison_node(self, state: RAGState) -> RAGState:
        """Nœud de comparaison de documents."""
        logger.info("🔄 Comparaison de documents...")
        
        try:
            # Recherche automatique avec plus de diversité pour la comparaison
            logger.info("🔍 Recherche automatique pour comparaison (HNSW + MMR)")
            documents = self.rag_service.search(
                state["query"], 
                k=10, 
                use_mmr=True,
                lambda_mult=0.5  # Plus de diversité pour la comparaison
            )
            state["retrieved_documents"] = documents
            logger.info(f"✅ Recherche terminée: {len(documents)} documents récupérés")
            
        except Exception as e:
            logger.error(f"❌ Erreur comparaison: {e}")
            state["retrieved_documents"] = []
            state["error_message"] = f"Erreur comparaison: {e}"
        
        return state
    
    def _generation_node(self, state: RAGState) -> RAGState:
        """Nœud de génération de réponse avec LLM."""
        logger.info("✍️ Génération de réponse avec LLM...")
        
        try:
            # Si on a déjà une réponse (sécurité), on la garde
            if state.get("final_answer"):
                logger.info("✅ Réponse de sécurité conservée")
                return state
            
            documents = state.get("retrieved_documents", [])
            intent_analysis = state.get("intent_analysis", {})
            
            if not documents:
                state["final_answer"] = "Aucun document pertinent trouvé pour répondre à votre question."
                state["confidence_score"] = 0.0
                return state
            
            # Utiliser le LLM pour générer une réponse intelligente
            intent = intent_analysis.get("intent", QueryIntent.SIMPLE_QA)
            confidence = intent_analysis.get("confidence", 0.0)
            
            llm_result = self.response_generator.generate_answer(
                query=state["query"],
                documents=documents,
                intent=intent.value,
                confidence=confidence
            )
            
            state["draft_answer"] = llm_result["answer"]
            state["citations"] = llm_result["citations"]
            state["confidence_score"] = llm_result["confidence"]
            
            method = llm_result.get("method", "unknown")
            generation_time = llm_result.get("generation_time", 0.0)
            
            logger.info(f"✅ Réponse générée (méthode: {method}, temps: {generation_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"❌ Erreur génération: {e}")
            state["final_answer"] = "Une erreur s'est produite lors de la génération de la réponse."
            state["confidence_score"] = 0.0
            state["error_message"] = f"Erreur génération: {e}"
        
        return state
    
    def _validation_node(self, state: RAGState) -> RAGState:
        """Nœud de validation des citations."""
        logger.info("✅ Validation des citations...")
        
        try:
            draft_answer = state.get("draft_answer", "")
            citations = state.get("citations", [])
            
            # Validation basique des citations
            if citations and len(citations) > 0:
                state["final_answer"] = draft_answer
                state["validation_result"] = {"status": "valid", "citations_count": len(citations)}
                logger.info("✅ Citations validées")
            else:
                # Retry si pas de citations
                retry_count = state.get("retry_count", 0)
                if retry_count < self.max_retries:
                    state["retry_count"] = retry_count + 1
                    logger.warning(f"⚠️ Citations invalides, retry {retry_count + 1}")
                else:
                    state["final_answer"] = draft_answer
                    state["validation_result"] = {"status": "invalid", "reason": "no_citations"}
                    logger.warning("⚠️ Citations invalides, retry max atteint")
            
        except Exception as e:
            logger.error(f"❌ Erreur validation: {e}")
            state["final_answer"] = state.get("draft_answer", "Erreur de validation.")
            state["error_message"] = f"Erreur validation: {e}"
        
        return state
    
    def _error_handler_node(self, state: RAGState) -> RAGState:
        """Nœud de gestion d'erreurs."""
        logger.error("🚨 Gestion d'erreur...")
        
        try:
            error_message = state.get("error_message", "Erreur inconnue")
            retry_count = state.get("retry_count", 0)
            security_result = state.get("security_result", {})
            
            # Gestion des erreurs de sécurité
            if security_result.get("status") == "blocked":
                state["final_answer"] = "Désolé, votre requête ne peut pas être traitée pour des raisons de sécurité."
                state["confidence_score"] = 0.0
                logger.warning("🚨 Requête bloquée par le filtre de sécurité")
            
            elif security_result.get("status") == "needs_clarification":
                state["final_answer"] = f"Veuillez préciser votre question. {security_result.get('message', '')}"
                state["confidence_score"] = 0.0
                logger.info("⚠️ Clarification nécessaire")
            
            # Gestion des erreurs de retry
            elif retry_count >= self.max_retries:
                state["final_answer"] = "Désolé, nous avons rencontré des difficultés techniques. Veuillez réessayer plus tard."
                state["confidence_score"] = 0.0
                logger.error(f"❌ Nombre maximum de tentatives atteint ({retry_count})")
            
            # Gestion des erreurs générales
            else:
                state["final_answer"] = f"Une erreur s'est produite lors du traitement de votre requête. {error_message}"
                state["confidence_score"] = 0.0
                logger.error(f"❌ Erreur générale: {error_message}")
            
            # Nettoyer les champs d'erreur
            state["error_message"] = None
            
        except Exception as e:
            logger.error(f"❌ Erreur dans le gestionnaire d'erreurs: {e}")
            state["final_answer"] = "Une erreur critique s'est produite. Veuillez contacter le support technique."
            state["confidence_score"] = 0.0
        
        return state
    
    # Méthodes de routage
    def _route_after_security(self, state: RAGState) -> str:
        """Route après le filtrage de sécurité."""
        security_result = state.get("security_result", {})
        status = security_result.get("status", "approved")
        
        if status == "blocked":
            return "blocked"
        elif status == "needs_clarification":
            return "clarification"
        else:
            return "approved"
    
    def _route_after_intent(self, state: RAGState) -> str:
        """Route après l'analyse d'intention."""
        intent_analysis = state.get("intent_analysis", {})
        
        if intent_analysis.get("error"):
            return "error"
        
        intent = intent_analysis.get("intent", QueryIntent.SIMPLE_QA)
        
        if intent == QueryIntent.COMPARISON:
            return "comparison"
        else:
            return "standard"
    
    def _route_after_generation(self, state: RAGState) -> str:
        """Route après la génération."""
        if state.get("final_answer"):
            return "complete"
        else:
            return "validate"
    
    def _route_after_validation(self, state: RAGState) -> str:
        """Route après la validation."""
        validation_result = state.get("validation_result", {})
        
        if validation_result.get("status") == "valid":
            return "complete"
        else:
            retry_count = state.get("retry_count", 0)
            if retry_count < self.max_retries:
                return "retry"
            else:
                return "complete"
    
    def _route_after_error(self, state: RAGState) -> str:
        """Route après la gestion d'erreur."""
        security_result = state.get("security_result", {})
        
        # Si c'est un blocage de sécurité, arrêter le workflow
        if security_result.get("status") == "blocked":
            return "complete"
        elif security_result.get("status") == "needs_clarification":
            return "complete"
        else:
            # Pour les autres erreurs, essayer de continuer
            return "complete"
    
    def process_query(self, query: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        Traite une requête avec le workflow LLM.
        
        Args:
            query (str): Requête utilisateur
            user_id (str): ID utilisateur (optionnel)
            session_id (str): ID de session (optionnel)
            
        Returns:
            Dict[str, Any]: Résultat du traitement
        """
        start_time = time.time()
        
        # Générer un ID de session unique si non fourni
        if not session_id:
            session_id = f"llm_session_{int(time.time())}_{user_id or 'anonymous'}"
        
        initial_state = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "security_result": {},
            "intent_analysis": {},
            "retrieved_documents": [],
            "filtered_documents": [],
            "draft_answer": "",
            "final_answer": "",
            "citations": [],
            "validation_result": {},
            "retry_count": 0,
            "error_message": None,
            "processing_time": 0.0,
            "confidence_score": 0.0,
            "workflow_version": "2.0.0-llm"
        }
        
        try:
            logger.info(f"🚀 Début du traitement LLM: '{query[:50]}...' (session: {session_id})")
            
            # Configuration pour le checkpointing
            config = {"configurable": {"thread_id": session_id}}
            
            # Exécuter le workflow avec checkpointing
            result = self.workflow.invoke(initial_state, config=config)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Traitement LLM terminé en {processing_time:.2f}s (session: {session_id})")
            
            # Avec checkpointing, le résultat final est dans result
            state_values = result
            
            return {
                "success": True,
                "answer": state_values.get("final_answer", ""),
                "confidence": state_values.get("confidence_score", 0.0),
                "processing_time": processing_time,
                "intent": state_values.get("intent_analysis", {}).get("intent", {}).value if state_values.get("intent_analysis", {}).get("intent") else "unknown",
                "documents_retrieved": len(state_values.get("retrieved_documents", [])),
                "workflow_version": state_values.get("workflow_version", "2.0.0-llm"),
                "session_id": session_id,
                "citations": state_values.get("citations", []),
                "validation_result": state_values.get("validation_result", {}),
                "llm_model": self.llm_model
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Erreur workflow LLM: {e} (session: {session_id})")
            
            return {
                "success": False,
                "error": str(e),
                "answer": "Une erreur s'est produite lors du traitement de votre requête.",
                "processing_time": processing_time,
                "confidence": 0.0,
                "session_id": session_id,
                "llm_model": self.llm_model
            }
    
    # Méthodes de gestion des sessions (simplifiées)
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Récupère l'état d'une session."""
        return {
            "session_id": session_id,
            "exists": True,
            "current_state": {},
            "next_nodes": [],
            "is_complete": True,
            "note": "LLM Enhanced Workflow - Session management simplified"
        }
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Récupère l'historique d'une session."""
        return [{
            "step": 0,
            "state": {},
            "next_nodes": [],
            "note": "LLM Enhanced Workflow - History management simplified"
        }]
