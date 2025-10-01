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

try:
    from langchain.llms import Ollama as LangChainOllama
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict
    LANGCHAIN_AVAILABLE = True
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LANGGRAPH_AVAILABLE = False
    logger.warning("⚠️ LangChain/LangGraph non disponible")

# State pour LangGraph self-reflection
class ReflectionState(TypedDict):
    query: str
    documents: List[Dict]
    intent: str
    initial_answer: str
    critique_text: str
    improved_answer: str
    iteration: int
    max_iterations: int

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
        
        # Initialiser LangGraph pour self-reflection si disponible
        self.reflection_graph = None
        if LANGGRAPH_AVAILABLE and enable_self_correction:
            try:
                self.reflection_graph = self._create_reflection_graph()
                logger.info("✅ LangGraph self-reflection initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Erreur initialisation LangGraph: {e}")
                self.reflection_graph = None
        
        logger.info(f"🤖 Générateur de réponses LLM initialisé: {model} (fallback: {fallback_to_basic}, self-correction: {enable_self_correction})")
    
    def _create_reflection_graph(self):
        """Crée le graph LangGraph pour la self-reflection."""
        if not LANGGRAPH_AVAILABLE:
            return None
            
        # Créer le graph
        workflow = StateGraph(ReflectionState)
        
        # Ajouter les nodes
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("improve", self._improve_node)
        
        # Définir les edges
        workflow.set_entry_point("critique")
        workflow.add_conditional_edges(
            "critique",
            self._should_improve,
            {
                "improve": "improve",
                "end": END
            }
        )
        workflow.add_edge("improve", "critique")
        
        return workflow.compile()
    
    def _critique_node(self, state: ReflectionState) -> ReflectionState:
        """Node de critique de la réponse."""
        critique_prompt = self._build_critique_prompt_langgraph(state)
        
        try:
            response = ollama.generate(
                model="llama3.1:8b",
                prompt=critique_prompt,
                options={"temperature": 0.1, "num_predict": 2000}
            )
            state["critique_text"] = response['response'].strip()
            logger.info(f"🔍 Critique générée (itération {state['iteration']})")
            logger.info(f"📋 CONTENU DE LA CRITIQUE: {state['critique_text']}")
        except Exception as e:
            logger.error(f"❌ Erreur critique: {e}")
            state["critique_text"] = "Erreur lors de la critique"
        
        return state
    
    def _improve_node(self, state: ReflectionState) -> ReflectionState:
        """Node d'amélioration de la réponse."""
        improvement_prompt = self._build_improvement_prompt_langgraph(state)
        
        try:
            response = ollama.generate(
                model="llama3.1:8b",
                prompt=improvement_prompt,
                options={"temperature": 0.2, "num_predict": 4000}
            )
            # Nettoyer la réponse pour supprimer le processus de pensée
            cleaned_response = self._clean_response(response['response'].strip())
            state["improved_answer"] = cleaned_response
            logger.info(f"✨ Réponse améliorée (itération {state['iteration']})")
            # Incrémenter l'itération après amélioration
            state["iteration"] += 1
        except Exception as e:
            logger.error(f"❌ Erreur amélioration: {e}")
            state["improved_answer"] = state.get("initial_answer", "")
        
        return state
    
    def _should_improve(self, state: ReflectionState) -> str:
        """Détermine si on doit améliorer la réponse ou arrêter."""
        # Vérifier si on a atteint le maximum d'itérations
        if state["iteration"] >= state["max_iterations"]:
            logger.info("🛑 Maximum d'itérations atteint")
            return "end"
        
        # Vérifier si la critique demande une amélioration
        critique_upper = state["critique_text"].upper()
        if "AMÉLIORATION NÉCESSAIRE" in critique_upper:
            logger.info("🔄 Amélioration nécessaire, continuation")
            return "improve"
        elif "TROP GÉNÉRALE" in critique_upper or "TROP GÉNÉRAL" in critique_upper:
            logger.info("🔄 Réponse trop générale, amélioration nécessaire")
            return "improve"
        elif "MANQUE DE DÉTAILS" in critique_upper:
            logger.info("🔄 Manque de détails, amélioration nécessaire")
            return "improve"
        elif "COURTE" in critique_upper:
            logger.info("🔄 Réponse trop courte, amélioration nécessaire")
            return "improve"
        elif "RÉPONSE SATISFAISANTE" in critique_upper:
            logger.info("✅ Réponse satisfaisante, arrêt immédiat (pas d'amélioration inutile)")
            return "end"
        else:
            # Par défaut, si pas explicitement satisfaisant, on arrête aussi
            logger.info("✅ Critique neutre, arrêt")
            return "end"
    
    def _build_critique_prompt_langgraph(self, state: ReflectionState) -> str:
        """Construit le prompt de critique pour LangGraph."""
        # Utiliser improved_answer si disponible et non vide, sinon initial_answer
        improved = state.get("improved_answer", "")
        initial = state.get("initial_answer", "")
        answer = improved if improved else initial
        
        # Convertir le HTML en texte brut pour que le critique puisse bien lire les chiffres
        answer_text = re.sub(r'<[^>]+>', ' ', answer)  # Supprimer les balises HTML
        answer_text = re.sub(r'\s+', ' ', answer_text).strip()  # Nettoyer les espaces multiples
        
        logger.info(f"📄 TEXTE COMPLET envoyé au critique (longueur: {len(answer_text)} caractères):")
        logger.info(f"{answer_text}")
        
        return f"""Tu es un expert critique en analyse de réponses RAG.

QUESTION: {state['query']}
INTENTION: {state['intent']}

RÉPONSE À ANALYSER:
{answer_text}

DOCUMENTS DISPONIBLES:
{self._format_documents_for_prompt(state['documents'])}

Analyse cette réponse selon ces critères STRICTS :
1. **DÉTAIL** : La réponse contient-elle des chiffres, dates, exemples concrets ?
2. **SPÉCIFICITÉ** : Va-t-elle au-delà des généralités ?
3. **EXHAUSTIVITÉ** : Couvre-t-elle tous les aspects de la question ?
4. **PRÉCISION** : Utilise-t-elle les informations spécifiques des documents ?

Si la réponse est trop générale, courte ou manque de détails spécifiques, réponds "AMÉLIORATION NÉCESSAIRE".
Sinon, réponds "RÉPONSE SATISFAISANTE".

RÉPONSE:"""

    def _build_improvement_prompt_langgraph(self, state: ReflectionState) -> str:
        """Construit le prompt d'amélioration pour LangGraph."""
        return f"""Tu es un expert en génération de réponses RAG.

QUESTION: {state['query']}
INTENTION: {state['intent']}

RÉPONSE INITIALE:
{state.get('initial_answer', '')}

CRITIQUE:
{state['critique_text']}

DOCUMENTS DISPONIBLES:
{self._format_documents_for_prompt(state['documents'])}

IMPORTANT : Améliore la réponse en tenant compte de la critique. 

EXIGENCES STRICTES :
- Fournis une réponse DÉTAILLÉE et SPÉCIFIQUE
- Inclus des chiffres, dates, exemples concrets des documents
- Va au-delà des généralités
- Utilise les informations précises des documents fournis
- Structure la réponse avec des paragraphes clairs et des titres en HTML (utilise <h2> pour les titres principaux et <h3> pour les sous-titres, <p> pour les paragraphes, <strong> pour le gras - NE PAS utiliser ** ou ##)
- NE PAS mentionner les noms de fichiers dans la réponse
- NE PAS ajouter de section "Sources" ou de citations dans la réponse
- NE PAS ajouter de métacommentaires sur la qualité de la réponse
- Commence directement par le contenu principal

RÉPONSE AMÉLIORÉE:"""

    def _format_documents_for_prompt(self, documents: List[Dict]) -> str:
        """Formate les documents pour le prompt."""
        formatted = []
        for i, doc in enumerate(documents[:3], 1):
            source = doc.get("source", "Document inconnu")
            content = doc.get("content", "")[:300] + "..."
            formatted.append(f"Document {i}: {source}\n{content}")
        return "\n\n".join(formatted)
    
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
        
       # Appeler Qwen 2.5 avec structured output
        start_time = time.time()
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.4,  # Température optimale pour réponses exhaustives et créatives
                "num_predict": 4000,  # 4000 tokens pour réponses très longues (~3000 mots max)
                "num_ctx": 8192,  # Contexte large
                "stop": ["<think>", "\n\n---", "###", "**Note:**", "---END---"]  # Tokens d'arrêt
            }
        )
        llm_time = time.time() - start_time
        
        # Parser la réponse structurée
        parsed_response = self._parse_structured_response(response['response'])
        
        # Extraire la réponse et les citations
        answer = parsed_response.get('answer', response['response'].strip())
        citations = parsed_response.get('citations', [])
        
        # Nettoyer la réponse pour supprimer le processus de pensée
        answer = self._clean_response(answer)
        
        # Si pas de citations dans le JSON, les extraire de la réponse
        if not citations:
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
            "Ton rôle est de fournir des réponses précises basées sur les documents fournis.",
            "",
            "**EXIGENCES STRICTES - RÉPONSE EXHAUSTIVE OBLIGATOIRE:**",
            "- LONGUEUR MINIMALE ABSOLUE : 800-1000 mots (sauf si l'utilisateur demande explicitement une réponse concise)",
            "- Fournis une réponse TRÈS DÉTAILLÉE et SPÉCIFIQUE avec PLUSIEURS chiffres, dates et exemples concrets",
            "- Va au-delà des généralités : extrais TOUTES les informations pertinentes des documents",
            "- MINIMUM 2-4 sections différentes couvrant TOUS les aspects en profondeur",
            "- Inclus obligatoirement : contexte détaillé, objectifs multiples, actions concrètes nombreuses, résultats chiffrés variés, perspectives, exemples détaillés, conclusion approfondie",
            "- Pour CHAQUE affirmation, fournis des CHIFFRES et DONNÉES PRÉCISES des documents",
            "- N'hésite PAS à être verbeux et exhaustif - c'est REQUIS",
            "- Structure la réponse avec des titres en HTML : <h2> pour titres principaux, <h3> pour sous-titres",
            "- Utilise <p> pour les paragraphes et <strong> pour mettre en gras les chiffres et éléments importants",
            "- NE PAS utiliser ** ou ## pour le formatage Markdown",
            "- NE PAS mentionner les noms de fichiers ou documents dans ta réponse",
            "- Commence directement par le contenu principal sans métacommentaires",
            "- Les citations seront gérées séparément",
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
            content_limited = content[:2000] + "..." if len(content) > 2000 else content
            
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
                "- Base-toi sur les documents fournis pour ta réponse avec des détails spécifiques, chiffres et exemples",
                "- NE PAS mentionner les noms de fichiers ou documents dans ta réponse",
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
                "- Base-toi sur les documents fournis pour ta réponse avec des détails spécifiques, chiffres et exemples",
                "- NE PAS mentionner les noms de fichiers ou documents dans ta réponse",
                "- Utilise un ton professionnel et précis"
            ])
        elif intent == "status_check":
            context_parts.extend([
                "",
                "**INSTRUCTIONS SPÉCIFIQUES:**",
                "- Vérifie l'état d'avancement mentionné dans les documents",
                "- Identifie les objectifs, réalisations et prochaines étapes",
                "- Structure ta réponse par statut (Réalisé, En cours, À venir)",
                "- Base-toi sur les documents fournis pour ta réponse avec des détails spécifiques, chiffres et exemples",
                "- NE PAS mentionner les noms de fichiers ou documents dans ta réponse",
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
                "- Base-toi sur les documents fournis pour ta réponse avec des détails spécifiques, chiffres et exemples",
                "- NE PAS mentionner les noms de fichiers ou documents dans ta réponse",
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
                "- Base-toi sur les documents fournis pour ta réponse avec des détails spécifiques, chiffres et exemples",
                "- NE PAS mentionner les noms de fichiers ou documents dans ta réponse",
                "- Utilise un ton synthétique, professionnel et stratégique",
                "- Assure-toi de couvrir tous les aspects importants mentionnés dans les documents"
            ])
        else:
            context_parts.extend([
                "",
                "**INSTRUCTIONS SPÉCIFIQUES:**",
                "- Réponds de manière claire et structurée à la question",
                "- Synthétise les informations pertinentes des documents",
                "- Base-toi sur les documents fournis pour ta réponse avec des détails spécifiques, chiffres et exemples",
                "- NE PAS mentionner les noms de fichiers ou documents dans ta réponse",
                "- Si l'information n'est pas dans les documents, indique-le clairement",
                "- Utilise un ton professionnel et précis"
            ])
        
        context_parts.extend([
            "",
            "**IMPORTANT:** Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après.",
            "Commence directement par {{ et termine par }}.",
            "NE PAS utiliser de balises <think> ou de mode de réflexion.",
            "NE PAS mentionner les noms de fichiers ou documents dans ta réponse.",
            "",
            "**FORMAT JSON STRICT:**",
            "{{",
            '    "answer": "<h2>Titre Principal</h2><p>Réponse complète et détaillée avec chiffres, dates et exemples concrets.</p><h3>Sous-section</h3><p>Informations spécifiques basées sur les documents avec <strong>éléments importants</strong> en gras.</p>",',
            '    "citations": ["NOM_DU_FICHIER.pdf – p.X", "AUTRE_FICHIER.pdf – p.Y"],',
            '    "confidence": 0.95,',
            '    "reasoning": "Explication brève de la réponse"',
            "}}",
            "",
            "**RÉPONSE:**"
        ])
        
        return "\n".join(context_parts)
    
    def _clean_response(self, response: str) -> str:
        """Nettoie la réponse pour supprimer le processus de pensée."""
        
        # Phrases qui indiquent un processus de pensée
        reflection_phrases = [
            "Okay, let's tackle this step by step",
            "First, I need to understand",
            "I need to analyze",
            "Let me break this down",
            "The user wants me to",
            "I should",
            "I must",
            "I will",
            "Step by step",
            "Let's tackle",
            "I understand",
            "I need to",
            "Let me",
            "First, I",
            "I should",
            "I must",
            "I will",
            "The user's instructions",
            "Based on the critique",
            "The critique points out",
            "I need to ensure",
            "Putting it all together",
            "I should also check",
            "I need to make sure",
            "Looking at the documents",
            "However, the filenames give",
            "Since the actual content",
            "I need to infer",
            "Usually, energy strategies",
            "The sustainability reports would likely",
            "However, without specific details",
            "The user's instructions say",
            # Patterns spécifiques LangGraph
            "Voici une réponse améliorée",
            "Cette réponse améliorée répond aux exigences",
            "Cette réponse répond aux critères",
            "Cette réponse répond aux exigences strictes",
            "Cette réponse est améliorée",
            "Cette réponse est détaillée",
            "Cette réponse est spécifique",
            "Cette réponse est exhaustive",
            "Cette réponse est précise",
            "Cette réponse inclut des chiffres",
            "Cette réponse va au-delà des généralités"
        ]
        
        # Trouver la première phrase de réflexion
        for phrase in reflection_phrases:
            if phrase.lower() in response.lower():
                # Trouver la position de cette phrase
                pos = response.lower().find(phrase.lower())
                if pos != -1:
                    # Couper la réponse avant cette phrase
                    response = response[:pos].strip()
                    break
        
        # Nettoyer les phrases qui commencent par des mots de réflexion
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Vérifier si la ligne commence par un mot de réflexion
                starts_with_reflection = any(
                    line.lower().startswith(word.lower()) 
                    for word in ["okay,", "first,", "i need", "let me", "i should", "i must", "i will", "step by", "let's", "i understand", "the user", "based on", "i need to", "putting it"]
                )
                
                if not starts_with_reflection:
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _parse_structured_response(self, response: str) -> Dict[str, Any]:
        """Parse une réponse structurée JSON du LLM."""
        try:
            import json
            
            # Chercher du JSON dans la réponse
            json_str = ""
            lines = response.split('\n')
            in_json = False
            
            for line in lines:
                if '{' in line and not in_json:
                    in_json = True
                    json_str = line
                elif in_json:
                    json_str += '\n' + line
                    if '}' in line:
                        break
            
            if json_str:
                # Nettoyer le JSON
                json_str = json_str.strip()
                if json_str.startswith('```json'):
                    json_str = json_str[7:]
                if json_str.endswith('```'):
                    json_str = json_str[:-3]
                
                # Remplacer les accolades simples par des doubles
                json_str = json_str.replace('{{', '{').replace('}}', '}')
                
                # Parser le JSON
                data = json.loads(json_str)
                return {
                    'answer': data.get('answer', ''),
                    'citations': data.get('citations', []),
                    'confidence': data.get('confidence', 0.8),
                    'reasoning': data.get('reasoning', '')
                }
            else:
                # Fallback vers réponse libre
                return {
                    'answer': response,
                    'citations': [],
                    'confidence': 0.5,
                    'reasoning': 'Réponse libre'
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur parsing réponse structurée: {e}")
            return {
                'answer': response,
                'citations': [],
                'confidence': 0.5,
                'reasoning': 'Fallback'
            }
    
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
    
    def _self_correct_response_custom(
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
            logger.info(f"📊 RÉPONSE INITIALE COMPLÈTE À ANALYSER:")
            logger.info(f"'{initial_result.get('answer', '')}'")
            
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
    
    def _self_correct_response(
        self, 
        initial_result: Dict[str, Any], 
        query: str, 
        documents: List[Dict], 
        intent: str, 
        confidence: float
    ) -> Dict[str, Any]:
        """
        Self-correction loop utilisant LangChain SelfReflectionLLM.
        
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
        
        if not self.reflection_graph:
            logger.warning("⚠️ LangGraph self-reflection non disponible, utilisation de la méthode custom")
            return self._self_correct_response_custom(initial_result, query, documents, intent, confidence)
        
        try:
            logger.info("🔄 Début du self-correction loop avec LangGraph...")
            logger.info(f"📊 RÉPONSE INITIALE À ANALYSER: {initial_result.get('answer', '')}")
            
            # Initialiser l'état pour LangGraph
            initial_state = ReflectionState(
                query=query,
                documents=documents,
                intent=intent,
                initial_answer=initial_result.get("answer", ""),
                critique_text="",
                improved_answer="",
                iteration=0,
                max_iterations=2  # Maximum 2 itérations
            )
            
            # Exécuter le graph LangGraph
            final_state = self.reflection_graph.invoke(initial_state)
            
            # Récupérer la réponse finale (améliorée si disponible, sinon initiale)
            improved_answer = final_state.get("improved_answer", "")
            
            # Si improved_answer est vide, c'est que la critique était satisfaisante dès le début
            if not improved_answer:
                logger.info("✅ Réponse initiale validée par le critique (aucune amélioration nécessaire)")
                return initial_result
            
            # Debug : afficher les réponses pour comparaison
            logger.info(f"🔍 DEBUG LANGGRAPH:")
            logger.info(f"📝 RÉPONSE INITIALE: {initial_result.get('answer', '')}...")
            logger.info(f"📝 RÉPONSE AMÉLIORÉE: {improved_answer}...")
            logger.info(f"📝 SONT IDENTIQUES: {improved_answer == initial_result.get('answer', '')}")
            
            # Vérifier si la réponse a été réellement améliorée
            if improved_answer != initial_result.get("answer", ""):
                logger.info("✅ Version améliorée par LangGraph self-reflection")
                
                # Extraire les citations de la réponse améliorée
                improved_citations = self._extract_citations_from_response(improved_answer, documents)
            
                return {
                        "answer": improved_answer,
                        "citations": improved_citations,
                        "confidence": confidence * 1.1,
                    "intent": intent,
                        "provider": "qwen2.5_14b_langgraph_reflected",
                    "model": self.model,
                        "generation_time": initial_result.get("generation_time", 0.0) + 3.0,
                        "method": "langgraph_self_reflection",
                        "reflection_applied": True,
                    "correction_applied": True,
                    "original_answer": initial_result.get("answer", ""),
                        "improvement_notes": f"Amélioré par LangGraph ({final_state.get('iteration', 0)} itérations)"
                }

            else:
                logger.info("ℹ️ Version initiale conservée par LangGraph")
                return initial_result
            
        except Exception as e:
            logger.error(f"❌ Erreur self-correction LangGraph: {e}")
            # Fallback vers méthode custom
            return self._self_correct_response_custom(initial_result, query, documents, intent, confidence)
    
    def _build_reflection_context(self, query: str, documents: List[Dict], intent: str) -> str:
        """Construit le contexte pour la self-reflection LangChain."""
        context_parts = [
            f"Question: {query}",
            f"Intention: {intent}",
            "",
            "Documents pertinents:"
        ]
        
        for i, doc in enumerate(documents[:3], 1):
            source = doc.get("source", "Document inconnu")
            content = doc.get("content", "")[:300] + "..."
            context_parts.extend([
                f"",
                f"Document {i}: {source}",
                f"{content}"
        ])
        
        return "\n".join(context_parts)
    
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
