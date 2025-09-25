"""
Générateur de réponses LLM (Llama 3.1 8B local).

Ce module utilise Llama 3.1 8B via Ollama pour générer des réponses intelligentes
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
    Générateur de réponses utilisant Llama 3.1 8B pour reformuler intelligemment
    les contextes récupérés en réponses cohérentes et naturelles.
    """
    
    def __init__(self, model: str = "llama3.1:8b", fallback_to_basic: bool = True):
        """
        Initialise le générateur LLM.
        
        Args:
            model (str): Modèle Ollama à utiliser
            fallback_to_basic (bool): Utiliser la génération basique en cas d'erreur LLM
        """
        self.model = model
        self.fallback_to_basic = fallback_to_basic
        self.ollama_available = OLLAMA_AVAILABLE
        
        logger.info(f"🤖 Générateur de réponses LLM initialisé: {model} (fallback: {fallback_to_basic})")
    
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
                    return self._generate_with_llm(query, documents, intent, confidence)
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
        """Génère une réponse en utilisant Llama 3.1 8B."""
        logger.info(f"🤖 Génération LLM de: '{query[:50]}...'")
        
        # Construire le prompt selon l'intention
        prompt = self._build_response_prompt(query, documents, intent)
        
        # Appeler Llama 3.1 8B
        start_time = time.time()
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.3,  # Température modérée pour la créativité
                "num_predict": 2000,  # Augmenter la limite pour les réponses longues
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
            "provider": "llama3.1_8b",
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
