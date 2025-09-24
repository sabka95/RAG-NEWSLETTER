"""
Analyseur d'intention basé sur LLM (Mistral 7B local).

Ce module remplace l'analyse par regex par une vraie compréhension sémantique
utilisant Mistral 7B via Ollama pour analyser l'intention des requêtes.
"""

import json
import re
import time
from typing import Dict, List, Any, Optional
from enum import Enum
from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("⚠️ Ollama non disponible, utilisation du mode mock")

class QueryIntent(Enum):
    """Types d'intentions de requête supportées."""
    
    SIMPLE_QA = "simple_qa"                    # Question-réponse simple
    COMPARISON = "comparison"                   # Comparaison de documents
    STATUS_CHECK = "status_check"              # Vérification d'état/avancement
    FINANCIAL_ANALYSIS = "financial_analysis"  # Analyse financière
    EVOLUTION_ANALYSIS = "evolution_analysis"  # Analyse d'évolution temporelle
    COMPLEX_AGGREGATION = "complex_aggregation" # Agrégation complexe multi-docs

class LLMIntentAnalyzer:
    """
    Analyseur d'intention utilisant Mistral 7B pour une vraie compréhension sémantique.
    
    Remplace l'analyse par regex par une compréhension intelligente des requêtes
    utilisant un LLM local via Ollama.
    """
    
    def __init__(self, model: str = "mistral:7b", fallback_to_regex: bool = True):
        """
        Initialise l'analyseur LLM.
        
        Args:
            model (str): Modèle Ollama à utiliser
            fallback_to_regex (bool): Utiliser les regex en cas d'erreur LLM
        """
        self.model = model
        self.fallback_to_regex = fallback_to_regex
        self.ollama_available = OLLAMA_AVAILABLE
        
        # Initialiser le fallback regex si nécessaire
        if self.fallback_to_regex:
            self._initialize_regex_fallback()
        
        logger.info(f"🤖 Analyseur LLM initialisé: {model} (fallback: {fallback_to_regex})")
    
    def _initialize_regex_fallback(self):
        """Initialise les patterns regex en fallback."""
        self.intent_patterns = {
            QueryIntent.SIMPLE_QA: [
                r"quels?|quelle|comment|pourquoi|où|quand|qui",
                r"explique|décris|raconte|donne|montre",
                r"c'est quoi|qu'est-ce que|définition"
            ],
            QueryIntent.COMPARISON: [
                r"compare|comparer|comparaison",
                r"différence|différences|différent",
                r"vs|versus|contre",
                r"même|identique|similaire"
            ],
            QueryIntent.FINANCIAL_ANALYSIS: [
                r"budget|coût|prix|€|\$|%",
                r"investissement|financement",
                r"rentabilité|profit|bénéfice"
            ],
            QueryIntent.STATUS_CHECK: [
                r"état|avancement|progrès|statut",
                r"réalisé|terminé|en cours",
                r"objectif|but|target"
            ],
            QueryIntent.EVOLUTION_ANALYSIS: [
                r"évolution|évolué|changement|changements",
                r"depuis|depuis l'année|depuis 20\d{2}",
                r"progression|tendance|historique",
                r"comment ont évolué|comment a évolué|comment ont changé",
                r"depuis 2020|depuis 2021|depuis 2022|depuis 2023",
                r"au fil des années|dans le temps|temporel",
                r"comment.*évolu.*depuis|comment.*changé.*depuis"
            ],
            QueryIntent.COMPLEX_AGGREGATION: [
                r"synthèse|synthétise|résume|résumé",
                r"vue d'ensemble|état complet|situation globale",
                r"tous les|ensemble des|global|complet",
                r"donne-moi un état|résume tout|synthétise tout",
                r"donnez-moi une vue|donne-moi une vue|vue d'ensemble complète",
                r"état des lieux|bilan complet|panorama|aperçu général",
                r"récapitulatif|récap|récapituler|faire le point",
                r"stratégie complète|approche globale|vision d'ensemble",
                r"tout ce qui concerne|toutes les informations|informations complètes",
                r"comprehensive|overview|complete picture|full picture",
                r"^donnez-moi une vue d'ensemble complète|^donne-moi une vue d'ensemble complète",
                r"vue d'ensemble complète de la stratégie|vue d'ensemble complète de l'approche"
            ],
        }
    
    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyse l'intention d'une requête en utilisant Mistral 7B.
        
        Args:
            query (str): Requête utilisateur
            
        Returns:
            Dict[str, Any]: Analyse d'intention avec métadonnées
        """
        if not query or not query.strip():
            return self._create_error_result("Requête vide")
        
        try:
            # Essayer d'abord avec le LLM
            if self.ollama_available:
                try:
                    return self._analyze_with_llm(query)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur LLM: {e}")
                    if self.fallback_to_regex:
                        logger.info("🔄 Basculement vers l'analyse regex")
                        return self._analyze_with_regex(query)
                    else:
                        raise
            else:
                logger.info("🔄 LLM non disponible, utilisation des regex")
                return self._analyze_with_regex(query)
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse d'intention: {e}")
            return self._create_error_result(f"Erreur d'analyse: {e}")
    
    def _analyze_with_llm(self, query: str) -> Dict[str, Any]:
        """Analyse l'intention en utilisant Mistral 7B."""
        logger.info(f"🤖 Analyse LLM de: '{query[:50]}...'")
        
        # Construire le prompt pour l'analyse d'intention
        prompt = self._build_intent_prompt(query)
        
        # Appeler Mistral 7B
        start_time = time.time()
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Faible température pour la cohérence
                "num_predict": 200,  # Limiter la réponse
                "stop": ["\n\n", "---", "###"]  # Arrêter sur ces tokens
            }
        )
        llm_time = time.time() - start_time
        
        # Parser la réponse
        result = self._parse_llm_response(response['response'], query)
        result['analysis_method'] = 'llm'
        result['llm_time'] = llm_time
        result['model'] = self.model
        
        logger.info(f"✅ Analyse LLM terminée en {llm_time:.2f}s: {result['intent'].value}")
        return result
    
    def _build_intent_prompt(self, query: str) -> str:
        """Construit le prompt pour l'analyse d'intention."""
        return f"""Tu es un expert en analyse d'intention de requêtes pour un système RAG d'entreprise.

Analyse cette requête et détermine son intention principale :

REQUÊTE: "{query}"

INTENTIONS POSSIBLES:
- simple_qa: Question-réponse simple sur un sujet spécifique
- comparison: Comparaison entre documents, rapports, ou données
- financial_analysis: Analyse financière, budgets, coûts, investissements
- status_check: Vérification d'état, avancement, progression d'un projet
- evolution_analysis: Analyse d'évolution temporelle, changements dans le temps (MOT-CLÉS: "évolution", "depuis", "changement", "progression temporelle")
- complex_aggregation: Agrégation complexe d'informations de plusieurs sources (MOT-CLÉS: "vue d'ensemble", "synthèse", "stratégie complète", "état des lieux")

RÈGLES IMPORTANTES:
1. Si la requête contient "vue d'ensemble complète" → complex_aggregation
2. Si la requête contient "stratégie complète" → complex_aggregation  
3. Si la requête contient "synthèse" ou "résumé global" → complex_aggregation
4. Si la requête contient "évolution" + "depuis" + année → evolution_analysis
5. Si la requête contient "comment ont évolué" → evolution_analysis

EXEMPLES:
- "Donnez-moi une vue d'ensemble complète de la stratégie" → complex_aggregation
- "Comment ont évolué les objectifs depuis 2020" → evolution_analysis
- "Synthétisez la situation globale" → complex_aggregation

Réponds UNIQUEMENT au format JSON suivant :
{{
    "intent": "intention_detectee",
    "confidence": 0.95,
    "entities": ["entité1", "entité2"],
    "reasoning": "Explication courte de pourquoi cette intention"
}}

Réponse JSON:"""
    
    def _parse_llm_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse la réponse JSON du LLM."""
        try:
            # Nettoyer la réponse
            response = response.strip()
            
            # Extraire le JSON de la réponse
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                raise ValueError("Pas de JSON trouvé dans la réponse")
            
            # Valider et normaliser les données
            intent_str = data.get("intent", "simple_qa")
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                logger.warning(f"⚠️ Intention inconnue: {intent_str}, utilisation de simple_qa")
                intent = QueryIntent.SIMPLE_QA
            
            confidence = float(data.get("confidence", 0.8))
            entities = data.get("entities", [])
            reasoning = data.get("reasoning", "")
            
            return {
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "reasoning": reasoning,
                "query_length": len(original_query.split()),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur parsing réponse LLM: {e}")
            logger.debug(f"Réponse brute: {response}")
            raise
    
    
    def _analyze_with_regex(self, query: str) -> Dict[str, Any]:
        """Analyse l'intention en utilisant les regex (fallback)."""
        logger.info(f"🔍 Analyse regex de: '{query[:50]}...'")
        
        query_lower = query.lower()
        
        # Détection de l'intention avec regex
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                score += len(matches)
            if score > 0:
                intent_scores[intent] = score
        
        # Intention avec le score le plus élevé
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(0.8, 0.5 + (intent_scores[primary_intent] * 0.1))
        else:
            primary_intent = QueryIntent.SIMPLE_QA
            confidence = 0.6
        
        # Extraction des entités basique
        entities = self._extract_basic_entities(query_lower)
        
        return {
            "intent": primary_intent,
            "confidence": confidence,
            "entities": entities,
            "reasoning": f"Analyse regex - score: {intent_scores.get(primary_intent, 0)}",
            "query_length": len(query.split()),
            "analysis_timestamp": time.time(),
            "analysis_method": "regex"
        }
    
    def _extract_basic_entities(self, query: str) -> List[str]:
        """Extraction basique d'entités avec regex."""
        entities = []
        
        # Années
        years = re.findall(r'\b(20\d{2})\b', query)
        entities.extend(years)
        
        # Nombres avec unités
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(€|\$|%|euros?|dollars?)\b', query)
        entities.extend([f"{num} {unit}" for num, unit in numbers])
        
        # Mots-clés importants
        keywords = re.findall(r'\b(budget|coût|investissement|objectif|projet|rapport)\b', query)
        entities.extend(keywords)
        
        return list(set(entities))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Crée un résultat d'erreur."""
        return {
            "intent": QueryIntent.SIMPLE_QA,
            "confidence": 0.0,
            "entities": [],
            "reasoning": f"Erreur: {error_message}",
            "query_length": 0,
            "analysis_timestamp": time.time(),
            "analysis_method": "error",
            "error": error_message
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
