"""
Générateur de réponses LLM (Qwen3 14B local).

Ce module utilise Qwen3 14B via Ollama pour générer des réponses intelligentes
et cohérentes basées sur les documents récupérés et l'intention détectée.
"""

import json
import re
import time
from typing import Dict, List, Any, Optional
from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("⚠️ Ollama non disponible, utilisation du mode mock")

class LLMResponseGenerator:
    """
    Générateur de réponses utilisant Qwen3 14B pour reformuler intelligemment
    les contextes récupérés en réponses cohérentes et naturelles.
    """
    
    def __init__(self, model: str = "qwen3:14b", fallback_to_basic: bool = True, enable_self_correction: bool = True):
        """
        Initialise le générateur LLM.
        
        Args:
            model (str): Modèle Ollama à utiliser
            fallback_to_basic (bool): Utiliser la génération basique en cas d'erreur LLM
            enable_self_correction (bool): Activer le self-correction loop (défaut: True)
        """
        self.model = model
        self.fallback_to_basic = fallback_to_basic
        self.enable_self_correction = enable_self_correction
        self.ollama_available = OLLAMA_AVAILABLE
        
        logger.info(f"🤖 Générateur de réponses LLM initialisé: {model} (fallback: {fallback_to_basic}, self-correction: {enable_self_correction})")
    
    def generate_answer(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        intent: str,
        confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Génère une réponse intelligente basée sur les documents et l'intention.
        
        Args:
            query (str): Requête utilisateur
            documents (List[Dict]): Documents récupérés
            intent (str): Intention détectée
            confidence (float): Score de confiance de l'intention
            
        Returns:
            Dict[str, Any]: Réponse générée avec métadonnées
        """
        try:
            logger.info(f"🤖 Génération de réponse LLM pour intention: {intent}")
            
            # Vérifier si on a des documents
            if not documents:
                return self._generate_no_documents_response(query, intent, confidence)
            
            # Essayer d'abord avec le LLM
            if self.ollama_available:
                try:
                    result = self._generate_with_llm(query, documents, intent, confidence)
                    
                    # Self-correction si activé
                    if self.enable_self_correction and result.get("method") == "llm":
                        result = self._self_correct_response(result, query, documents, intent, confidence)
                    
                    return result
                except Exception as e:
                    logger.warning(f"⚠️ Erreur génération LLM: {e}")
                    if self.fallback_to_basic:
                        logger.info("🔄 Basculement vers la génération basique")
                        return self._generate_basic_response(query, documents, intent, confidence)
                    else:
                        raise
            else:
                logger.info("🔄 LLM non disponible, utilisation de la génération basique")
                return self._generate_basic_response(query, documents, intent, confidence)
            
        except Exception as e:
            logger.error(f"❌ Erreur génération de réponse: {e}")
            return self._generate_error_response(query, documents, intent, confidence)
    
    def _generate_with_llm(self, query: str, documents: List[Dict], intent: str, confidence: float) -> Dict[str, Any]:
        """Génère une réponse en utilisant Qwen3 14B."""
        logger.info(f"🤖 Génération LLM de: '{query[:50]}...'")
        
        # Construire le prompt selon l'intention
        prompt = self._build_response_prompt(query, documents, intent)
        
        # Appeler Qwen3 14B
        start_time = time.time()
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.2,  # Température optimisée pour Qwen3
                "num_predict": 3000,  # Plus de tokens pour Qwen3
                "num_ctx": 8192,  # Contexte large pour Qwen3
                "stop": ["\n\n---", "###", "**Note:**", "---END---"]  # Tokens d'arrêt
            }
        )
        llm_time = time.time() - start_time
        
        # Extraire la réponse et les citations
        answer = response['response'].strip()
        citations = self._extract_citations_from_response(answer, documents)
        
        result = {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "intent": intent,
            "provider": "qwen3_14b",
            "model": self.model,
            "generation_time": llm_time,
            "method": "llm"
        }
        
        logger.info(f"✅ Réponse LLM générée en {llm_time:.2f}s")
        return result
    
    def _build_response_prompt(self, query: str, documents: List[Dict], intent: str) -> str:
        """Construit le prompt pour la génération de réponse."""
        
        # Contexte de base
        context_parts = [
            "Tu es un assistant IA expert en analyse de documents d'entreprise.",
            "Ton rôle est de fournir des réponses précises et bien citées basées sur les documents fournis.",
            "",
            "**IMPORTANT:** Pour les citations, utilise TOUJOURS le nom complet du fichier (ex: TotalEnergies_PR_4Q23_Results.pdf) et non pas 'Document 1' ou 'Document 2'.",
            "",
            f"**QUESTION UTILISATEUR:** {query}",
            f"**INTENTION DÉTECTÉE:** {intent}",
            "",
            "**DOCUMENTS PERTINENTS:**"
        ]
        
        # Ajouter les documents (limiter à 5 pour éviter les tokens excessifs)
        for i, doc in enumerate(documents[:5], 1):
            source = doc.get("source", "Document inconnu")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")
            
            # Limiter le contenu pour éviter les tokens excessifs
            content_limited = content[:800] + "..." if len(content) > 800 else content
            
            context_parts.extend([
                f"",
                f"**Document {i}: {source} – Page {page}**",
                f"**Nom du fichier pour citation: {source}**",
                f"{content_limited}"
            ])
        
        # Détecter la langue de la requête et ajuster les instructions
        query_language = self._detect_query_language(query)
        if query_language == "english":
            context_parts.extend([
                "",
                "**IMPORTANT:** Respond in English. The user asked in English, so provide your answer in English.",
                ""
            ])
        elif query_language == "french":
            context_parts.extend([
                "",
                "**IMPORTANT:** Réponds en français. L'utilisateur a posé sa question en français, donc fournis ta réponse en français.",
                ""
            ])
        
        # Instructions spécifiques selon l'intention
        if intent == "comparison":
            context_parts.extend([
                "",
                "**INSTRUCTIONS SPÉCIFIQUES:**",
                "- Compare les documents de manière structurée et claire",
                "- Identifie les similarités et différences clés",
                "- Organise ta réponse en sections claires :",
                "  * **Similarités** : Points communs entre les documents",
                "  * **Différences** : Points de divergence",
                "  * **Conclusion** : Synthèse de la comparaison",
                "- Cite chaque information avec le format exact: [NOM_COMPLET_DU_FICHIER.pdf – p.X]",
                "- Utilise un ton professionnel et analytique",
                "- Assure-toi de fournir une réponse complète et détaillée"
            ])
        elif intent == "financial_analysis":
            context_parts.extend([
                "",
                "**INSTRUCTIONS SPÉCIFIQUES:**",
                "- Analyse les aspects financiers mentionnés dans les documents",
                "- Extrais les chiffres, budgets, coûts et indicateurs clés",
                "- Structure ta réponse par catégories financières",
                "- Cite chaque information avec le format exact: [NOM_COMPLET_DU_FICHIER.pdf – p.X]",
                "- Utilise un ton professionnel et précis"
            ])
        elif intent == "status_check":
            context_parts.extend([
                "",
                "**INSTRUCTIONS SPÉCIFIQUES:**",
                "- Vérifie l'état d'avancement mentionné dans les documents",
                "- Identifie les objectifs, réalisations et prochaines étapes",
                "- Structure ta réponse par statut (Réalisé, En cours, À venir)",
                "- Cite chaque information avec le format exact: [NOM_COMPLET_DU_FICHIER.pdf – p.X]",
                "- Utilise un ton informatif et clair"
            ])
        elif intent == "evolution_analysis":
            context_parts.extend([
                "",
                "**INSTRUCTIONS SPÉCIFIQUES:**",
                "- Analyse l'évolution temporelle des éléments mentionnés",
                "- Identifie les changements, tendances et progressions dans le temps",
                "- Structure ta réponse par période ou par évolution",
                "- Compare les données disponibles entre différentes années",
                "- Si des données historiques manquent, indique-le clairement",
                "- Cite chaque information avec le format exact: [NOM_COMPLET_DU_FICHIER.pdf – p.X]",
                "- Utilise un ton analytique et temporel"
            ])
        elif intent == "complex_aggregation":
            context_parts.extend([
                "",
                "**INSTRUCTIONS SPÉCIFIQUES POUR AGRÉGATION COMPLEXE:**",
                "- Synthétise et agrège les informations de tous les documents pertinents",
                "- Donne une vue d'ensemble complète et structurée de la stratégie/approche",
                "- Organise ta réponse par thèmes ou catégories logiques (ex: Objectifs, Approche, Résultats, Perspectives)",
                "- Intègre les informations de plusieurs sources de manière cohérente",
                "- Fournis une analyse globale avec des conclusions et recommandations",
                "- Structure ta réponse avec des sections claires et des sous-titres",
                "- Cite chaque information avec le format exact: [NOM_COMPLET_DU_FICHIER.pdf – p.X]",
                "- Utilise un ton synthétique, professionnel et stratégique",
                "- Assure-toi de couvrir tous les aspects importants mentionnés dans les documents"
            ])
        else:
            context_parts.extend([
                "",
                "**INSTRUCTIONS SPÉCIFIQUES:**",
                "- Réponds de manière claire et structurée à la question",
                "- Synthétise les informations pertinentes des documents",
                "- Cite chaque information avec le format exact: [NOM_COMPLET_DU_FICHIER.pdf – p.X]",
                "- Si l'information n'est pas dans les documents, indique-le clairement",
                "- Utilise un ton professionnel et précis"
            ])
        
        context_parts.extend([
            "",
            "**FORMAT DE RÉPONSE ATTENDU:**",
            "- Réponse structurée et claire",
            "- Citations précises au format [NOM_COMPLET_DU_FICHIER.pdf – p.X]",
            "- Ton professionnel adapté au contexte d'entreprise",
            "- Réponse complète et détaillée (pas de réponse tronquée)",
            "",
            "**IMPORTANT:** Génère une réponse complète. Ne t'arrête pas au milieu d'une phrase.",
            "",
            "**RÉPONSE:**"
        ])
        
        return "\n".join(context_parts)
    
    def _extract_citations_from_response(self, response: str, documents: List[Dict]) -> List[str]:
        """Extrait les citations de la réponse générée."""
        citations = []
        
        # Rechercher les citations au format [NomDuDocument – p.X]
        citation_pattern = r'\[([^\]]+ – p\.\d+)\]'
        found_citations = re.findall(citation_pattern, response)
        citations.extend(found_citations)
        
        # Si aucune citation trouvée, ajouter les sources des documents utilisés
        if not citations:
            for doc in documents[:3]:
                source = doc.get("source", "Document inconnu")
                page = doc.get("page", "N/A")
                citations.append(f"{source} – p.{page}")
        
        return list(set(citations))  # Supprimer les doublons
    
    def _generate_basic_response(self, query: str, documents: List[Dict], intent: str, confidence: float) -> Dict[str, Any]:
        """Génère une réponse basique (fallback)."""
        logger.info("📝 Génération de réponse basique")
        
        if intent == "comparison":
            answer = f"**Comparaison basée sur votre question:** '{query}'\n\n"
            answer += "Voici une analyse comparative des documents trouvés :\n\n"
            
            # Grouper par document
            doc_groups = {}
            for doc in documents:
                source = doc.get("source", "Document inconnu")
                if source not in doc_groups:
                    doc_groups[source] = []
                doc_groups[source].append(doc)
            
            for doc_name, docs in doc_groups.items():
                answer += f"**{doc_name}** :\n"
                for doc in docs[:2]:
                    page = doc.get("page", "N/A")
                    content = doc.get("content", "")[:200] + "..."
                    answer += f"  - p.{page}: {content}\n"
                answer += "\n"
            
            answer += "Cette comparaison montre les points clés identifiés dans les documents analysés."
            
        elif intent == "financial_analysis":
            answer = f"**Analyse financière pour:** '{query}'\n\n"
            answer += "Voici les informations financières pertinentes trouvées :\n\n"
            
            for doc in documents[:3]:
                source = doc.get("source", "Document inconnu")
                page = doc.get("page", "N/A")
                content = doc.get("content", "")[:200]
                answer += f"**{source} – p.{page}** : {content}...\n\n"
            
            answer += "Ces informations fournissent un aperçu des aspects financiers mentionnés dans les documents."
            
        else:
            answer = f"**Réponse à votre question:** '{query}'\n\n"
            answer += "Basé sur l'analyse des documents, voici la réponse :\n\n"
            
            for i, doc in enumerate(documents[:3], 1):
                source = doc.get("source", "Document inconnu")
                page = doc.get("page", "N/A")
                content = doc.get("content", "")[:200]
                answer += f"**{source} – p.{page}** : {content}...\n\n"
            
            answer += "Cette réponse synthétise les informations pertinentes trouvées dans les documents."
        
        # Extraire les citations
        citations = []
        for doc in documents[:3]:
            source = doc.get("source", "Document inconnu")
            page = doc.get("page", "N/A")
            citations.append(f"{source} – p.{page}")
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence * 0.8,  # Réduire légèrement la confiance
            "intent": intent,
            "provider": "basic_fallback",
            "model": "basic",
            "generation_time": 0.01,
            "method": "basic"
        }
    
    def _generate_no_documents_response(self, query: str, intent: str, confidence: float) -> Dict[str, Any]:
        """Génère une réponse quand aucun document n'est trouvé."""
        answer = f"**Réponse à votre question:** '{query}'\n\n"
        answer += "Aucun document pertinent n'a été trouvé pour répondre à votre question. "
        answer += "Veuillez reformuler votre requête ou vérifier que les documents contiennent les informations recherchées."
        
        return {
            "answer": answer,
            "citations": [],
            "confidence": 0.0,
            "intent": intent,
            "provider": "no_documents",
            "model": "none",
            "generation_time": 0.01,
            "method": "no_documents"
        }
    
    def _generate_error_response(self, query: str, documents: List[Dict], intent: str, confidence: float) -> Dict[str, Any]:
        """Génère une réponse d'erreur."""
        answer = f"**Réponse à votre question:** '{query}'\n\n"
        answer += "Une erreur s'est produite lors de la génération de la réponse. "
        answer += "Voici les informations disponibles dans les documents :\n\n"
        
        for i, doc in enumerate(documents[:3], 1):
            source = doc.get("source", "Document inconnu")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")[:200] + "..."
            answer += f"{i}. **{source} – p.{page}**\n{content}\n\n"
        
        citations = [f"{doc.get('source', 'Document inconnu')} – p.{doc.get('page', 'N/A')}" 
                    for doc in documents[:3]]
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence * 0.3,  # Réduire fortement la confiance
            "intent": intent,
            "provider": "error_fallback",
            "model": "error",
            "generation_time": 0.01,
            "method": "error"
        }
    
    def test_llm_connection(self) -> bool:
        """Teste la connexion au LLM."""
        try:
            if not self.ollama_available:
                return False
            
            response = ollama.generate(
                model=self.model,
                prompt="Test de connexion. Réponds juste 'OK'.",
                options={"num_predict": 10}
            )
            
            logger.info("✅ Connexion LLM testée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur test connexion LLM: {e}")
            return False
    
    def _self_correct_response(
        self, 
        initial_result: Dict[str, Any], 
        query: str, 
        documents: List[Dict], 
        intent: str, 
        confidence: float
    ) -> Dict[str, Any]:
        """
        Self-correction loop utilisant un LLM critique pour améliorer la réponse.
        
        Cette méthode utilise Qwen2.5 14B comme critique pour analyser et améliorer
        la réponse initiale, puis génère une version corrigée.
        
        Args:
            initial_result (Dict[str, Any]): Résultat initial de la génération
            query (str): Question posée
            documents (List[Dict]): Documents utilisés
            intent (str): Intention détectée
            confidence (float): Score de confiance
            
        Returns:
            Dict[str, Any]: Résultat corrigé ou initial si pas d'amélioration
        """
        if not self.ollama_available:
            logger.warning("⚠️ LLM non disponible pour self-correction")
            return initial_result
        
        try:
            logger.info("🔄 Début du self-correction loop...")
            
            # 1. Analyser la réponse initiale avec un critique LLM
            critique_result = self._critique_response(initial_result, query, documents, intent)
            
            # 2. Si le critique suggère des améliorations, générer une version corrigée
            if critique_result.get("needs_improvement", False):
                logger.info("🔧 Améliorations suggérées, génération de la version corrigée...")
                corrected_result = self._generate_corrected_response(
                    initial_result, critique_result, query, documents, intent, confidence
                )
                
                # 3. Comparer et choisir la meilleure version
                if self._is_corrected_better(initial_result, corrected_result):
                    logger.info("✅ Version corrigée adoptée")
                    return corrected_result
                else:
                    logger.info("ℹ️ Version initiale conservée")
                    return initial_result
            else:
                logger.info("✅ Réponse initiale validée par le critique")
                return initial_result
                
        except Exception as e:
            logger.error(f"❌ Erreur self-correction: {e}")
            logger.warning("⚠️ Retour à la réponse initiale")
            return initial_result
    
    def _critique_response(
        self, 
        result: Dict[str, Any], 
        query: str, 
        documents: List[Dict], 
        intent: str
    ) -> Dict[str, Any]:
        """
        Utilise un LLM critique pour analyser la qualité de la réponse.
        
        Args:
            result (Dict[str, Any]): Résultat à critiquer
            query (str): Question posée
            documents (List[Dict]): Documents utilisés
            intent (str): Intention détectée
            
        Returns:
            Dict[str, Any]: Analyse critique avec suggestions d'amélioration
        """
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        
        # Construire le prompt de critique
        critique_prompt = self._build_critique_prompt(query, answer, documents, intent, citations)
        
        try:
            # Appeler le LLM critique
            response = ollama.generate(
                model=self.model,
                prompt=critique_prompt,
                options={
                    "temperature": 0.1,  # Très conservateur pour la critique
                    "num_predict": 1000,
                    "stop": ["\n\n---", "###", "**Note:**", "---END---"]
                }
            )
            
            critique_text = response['response'].strip()
            
            # Analyser la critique
            needs_improvement = "AMÉLIORATION NÉCESSAIRE" in critique_text.upper()
            suggestions = self._extract_suggestions(critique_text)
            
            return {
                "critique_text": critique_text,
                "needs_improvement": needs_improvement,
                "suggestions": suggestions,
                "critique_quality": self._assess_critique_quality(critique_text)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur critique LLM: {e}")
            return {
                "critique_text": "",
                "needs_improvement": False,
                "suggestions": [],
                "critique_quality": 0.0
            }
    
    def _build_critique_prompt(self, query: str, answer: str, documents: List[Dict], intent: str, citations: List[str]) -> str:
        """Construit le prompt pour le critique LLM."""
        
        context_parts = [
            "Tu es un expert critique en analyse de réponses RAG.",
            "Ton rôle est d'analyser la qualité d'une réponse générée et de suggérer des améliorations.",
            "",
            f"**QUESTION:** {query}",
            f"**INTENTION:** {intent}",
            "",
            "**RÉPONSE À ANALYSER:**",
            answer,
            "",
            "**CITATIONS UTILISÉES:**",
            "\n".join([f"- {citation}" for citation in citations]),
            "",
            "**DOCUMENTS DISPONIBLES:**"
        ]
        
        # Ajouter les documents (limités)
        for i, doc in enumerate(documents[:3], 1):
            source = doc.get("source", "Document inconnu")
            content = doc.get("content", "")[:300] + "..."
            context_parts.extend([
                f"",
                f"**Document {i}: {source}**",
                f"{content}"
            ])
        
        context_parts.extend([
            "",
            "**INSTRUCTIONS DE CRITIQUE:**",
            "Analyse cette réponse selon les critères suivants :",
            "",
            "1. **FIDÉLITÉ AUX DOCUMENTS** :",
            "   - La réponse est-elle basée sur les documents fournis ?",
            "   - Y a-t-il des informations non supportées par les sources ?",
            "   - Les citations sont-elles correctes et pertinentes ?",
            "",
            "2. **PERTINENCE À LA QUESTION** :",
            "   - La réponse répond-elle directement à la question ?",
            "   - Y a-t-il des éléments hors-sujet ?",
            "   - La réponse est-elle complète ?",
            "",
            "3. **QUALITÉ DE LA RÉPONSE** :",
            "   - La réponse est-elle claire et bien structurée ?",
            "   - Le ton est-il professionnel et adapté ?",
            "   - Y a-t-il des répétitions ou redondances ?",
            "",
            "4. **ADÉQUATION À L'INTENTION** :",
            "   - La réponse correspond-elle au type d'intention détecté ?",
            "   - La structure est-elle adaptée (comparaison, analyse, etc.) ?",
            "",
            "**FORMAT DE RÉPONSE ATTENDU:**",
            "Commence par :",
            "- **ÉVALUATION GLOBALE** : [BONNE/AMÉLIORATION NÉCESSAIRE]",
            "- **POINTS FORTS** : [liste des points positifs]",
            "- **POINTS À AMÉLIORER** : [liste des problèmes identifiés]",
            "- **SUGGESTIONS** : [recommandations concrètes]",
            "",
            "**RÉPONSE:**"
        ])
        
        return "\n".join(context_parts)
    
    def _extract_suggestions(self, critique_text: str) -> List[str]:
        """Extrait les suggestions d'amélioration du texte de critique."""
        suggestions = []
        lines = critique_text.split('\n')
        
        in_suggestions = False
        for line in lines:
            if "SUGGESTIONS" in line.upper() or "SUGGESTION" in line.upper():
                in_suggestions = True
                continue
            elif in_suggestions and line.strip():
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    suggestions.append(line.strip()[1:].strip())
                elif line.strip() and not line.strip().startswith('**'):
                    suggestions.append(line.strip())
        
        return suggestions[:5]  # Limiter à 5 suggestions
    
    def _assess_critique_quality(self, critique_text: str) -> float:
        """Évalue la qualité de la critique (0.0 à 1.0)."""
        quality_indicators = [
            "ÉVALUATION GLOBALE" in critique_text,
            "POINTS FORTS" in critique_text,
            "POINTS À AMÉLIORER" in critique_text,
            "SUGGESTIONS" in critique_text,
            len(critique_text) > 200  # Critique substantielle
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def _generate_corrected_response(
        self, 
        initial_result: Dict[str, Any], 
        critique: Dict[str, Any], 
        query: str, 
        documents: List[Dict], 
        intent: str, 
        confidence: float
    ) -> Dict[str, Any]:
        """Génère une version corrigée de la réponse."""
        
        # Construire le prompt de correction
        correction_prompt = self._build_correction_prompt(
            query, initial_result, critique, documents, intent
        )
        
        try:
            # Générer la réponse corrigée
            response = ollama.generate(
                model=self.model,
                prompt=correction_prompt,
                options={
                    "temperature": 0.2,
                    "num_predict": 2500,
                    "num_ctx": 8192,
                    "stop": ["\n\n---", "###", "**Note:**", "---END---"]
                }
            )
            
            corrected_answer = response['response'].strip()
            corrected_citations = self._extract_citations_from_response(corrected_answer, documents)
            
            return {
                "answer": corrected_answer,
                "citations": corrected_citations,
                "confidence": confidence * 1.1,  # Légèrement augmenter la confiance
                "intent": intent,
                "provider": "qwen3_14b_corrected",
                "model": self.model,
                "generation_time": initial_result.get("generation_time", 0.0) + 2.0,  # Ajouter temps de correction
                "method": "llm_corrected",
                "correction_applied": True,
                "original_answer": initial_result.get("answer", ""),
                "critique_suggestions": critique.get("suggestions", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération corrigée: {e}")
            return initial_result
    
    def _build_correction_prompt(
        self, 
        query: str, 
        initial_result: Dict[str, Any], 
        critique: Dict[str, Any], 
        documents: List[Dict], 
        intent: str
    ) -> str:
        """Construit le prompt pour la génération de la réponse corrigée."""
        
        context_parts = [
            "Tu es un expert en génération de réponses RAG.",
            "Tu dois améliorer une réponse existante en tenant compte des critiques et suggestions.",
            "",
            f"**QUESTION:** {query}",
            f"**INTENTION:** {intent}",
            "",
            "**RÉPONSE INITIALE À AMÉLIORER:**",
            initial_result.get("answer", ""),
            "",
            "**CRITIQUE ET SUGGESTIONS:**",
            critique.get("critique_text", ""),
            "",
            "**DOCUMENTS DISPONIBLES:**"
        ]
        
        # Ajouter les documents
        for i, doc in enumerate(documents[:5], 1):
            source = doc.get("source", "Document inconnu")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")[:400] + "..."
            
            context_parts.extend([
                f"",
                f"**Document {i}: {source} – Page {page}**",
                f"{content}"
            ])
        
        context_parts.extend([
            "",
            "**INSTRUCTIONS POUR LA CORRECTION:**",
            "1. Garde les éléments positifs de la réponse initiale",
            "2. Corrige les problèmes identifiés dans la critique",
            "3. Applique les suggestions d'amélioration",
            "4. Assure-toi que la réponse est fidèle aux documents",
            "5. Utilise des citations précises au format [NOM_DU_FICHIER.pdf – p.X]",
            "6. Maintiens un ton professionnel et structuré",
            "",
            "**RÉPONSE CORRIGÉE:**"
        ])
        
        return "\n".join(context_parts)
    
    def _is_corrected_better(self, initial: Dict[str, Any], corrected: Dict[str, Any]) -> bool:
        """Détermine si la version corrigée est meilleure que l'originale."""
        
        # Critères simples pour comparer
        initial_len = len(initial.get("answer", ""))
        corrected_len = len(corrected.get("answer", ""))
        
        initial_citations = len(initial.get("citations", []))
        corrected_citations = len(corrected.get("citations", []))
        
        # La version corrigée est meilleure si :
        # 1. Elle a plus de citations
        # 2. Elle n'est pas trop courte (au moins 80% de la longueur originale)
        # 3. Elle a été marquée comme corrigée
        
        return (
            corrected_citations >= initial_citations and
            corrected_len >= initial_len * 0.8 and
            corrected.get("correction_applied", False)
        )
    
    def _detect_query_language(self, query: str) -> str:
        """
        Détecte la langue de la requête pour adapter la réponse.
        
        Args:
            query (str): Requête utilisateur
            
        Returns:
            str: 'english', 'french', ou 'unknown'
        """
        # Mots-clés anglais communs
        english_indicators = [
            'what', 'how', 'when', 'where', 'why', 'which', 'who',
            'are', 'is', 'was', 'were', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'about'
        ]
        
        # Mots-clés français communs
        french_indicators = [
            'quoi', 'comment', 'quand', 'où', 'pourquoi', 'lequel', 'qui',
            'sont', 'est', 'était', 'étaient', 'avoir', 'a', 'avait',
            'sera', 'serait', 'pourrait', 'devrait', 'peut', 'peut-être',
            'le', 'la', 'les', 'un', 'une', 'et', 'ou', 'mais', 'dans', 'sur', 'à',
            'pour', 'de', 'avec', 'par', 'depuis', 'sur', 'elle', 'il', 'ils', 'elles',
            'comment', 'pourquoi', 'quand', 'où', 'que', 'qui', 'quoi', 'dont',
            'aborde-t-elle', 'comment', 'pourquoi', 'quand', 'où'
        ]
        
        query_lower = query.lower()
        
        # Compter les indicateurs
        english_count = sum(1 for word in english_indicators if word in query_lower)
        french_count = sum(1 for word in french_indicators if word in query_lower)
        
        # Détection basée sur les mots-clés
        if english_count > french_count and english_count > 0:
            return "english"
        elif french_count > english_count and french_count > 0:
            return "french"
        else:
            # Détection par caractères spéciaux
            if any(char in query for char in 'àâäéèêëïîôöùûüÿç'):
                return "french"
            else:
                return "unknown"
