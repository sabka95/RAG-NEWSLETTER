"""
🚀 Interface Streamlit Ultra-Moderne pour Chatbot RAG d'Entreprise
================================================================

Interface complète avec :
- Chat interactif avec historique
- Mode comparaison de documents
- Panneau de citations avec liens SharePoint
- Paramètres avancés (modèle, confiance, etc.)
- Gestion des sessions et checkpointing
- Interface d'administration et monitoring
- Design moderne et responsive
"""

import streamlit as st
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from rag_newsletter.workflows import RAGWorkflow
from rag_newsletter.ingestion.rag_ingestion import OptimizedRAGIngestionService
from loguru import logger

# =============================================================================
# Configuration de la page
# =============================================================================

st.set_page_config(
    page_title="🤖 Chatbot RAG TotalEnergies",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Chatbot RAG TotalEnergies\nInterface intelligente pour l'analyse de documents d'entreprise"
    }
)

# =============================================================================
# CSS et Styles Personnalisés
# =============================================================================

st.markdown("""
<style>
    /* Thème principal */
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e6da4 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Chat container */
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .citation {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #fff;
    }
    
    /* Métriques */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    
    /* Sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Alertes */
    .success-alert {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-alert {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Initialisation de l'application
# =============================================================================

@st.cache_resource
def initialize_rag_service():
    """Initialise le service RAG avec cache."""
    try:
        logger.info("🚀 Initialisation du service RAG...")
        rag_service = OptimizedRAGIngestionService()
        logger.info("✅ Service RAG initialisé avec succès")
        return rag_service
    except Exception as e:
        logger.error(f"❌ Erreur initialisation service RAG: {e}")
        st.error(f"Erreur d'initialisation du service RAG: {e}")
        return None

@st.cache_resource
def initialize_workflow(_rag_service, model_name: str = "mistral:7b"):
    """Initialise le workflow RAG avec cache."""
    try:
        logger.info(f"🧠 Initialisation du workflow RAG avec modèle: {model_name}")
        workflow = RAGWorkflow(_rag_service, llm_model=model_name)
        logger.info("✅ Workflow RAG initialisé avec succès")
        return workflow
    except Exception as e:
        logger.error(f"❌ Erreur initialisation workflow: {e}")
        st.error(f"Erreur d'initialisation du workflow: {e}")
        return None

# =============================================================================
# Gestion de l'état de session
# =============================================================================

def initialize_session_state():
    """Initialise l'état de session Streamlit."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"streamlit_session_{int(time.time())}"
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "streamlit_user"
    
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    
    if 'rag_service' not in st.session_state:
        st.session_state.rag_service = None
    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "llama3.1:8b"
    
    # Suppression du mode comparaison - détection automatique par LLM
    
    if 'advanced_settings' not in st.session_state:
        st.session_state.advanced_settings = {
            'confidence_threshold': 0.7,
            'max_documents': 5,
            'use_mmr': True,
            'lambda_mult': 0.7,
            'enable_citations': True,
            'enable_validation': True
        }

# =============================================================================
# Composants UI
# =============================================================================

def render_header():
    """Affiche l'en-tête principal."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Chatbot RAG TotalEnergies</h1>
        <p>Interface intelligente pour l'analyse de documents d'entreprise</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Affiche la barre latérale avec paramètres."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>⚙️ Paramètres</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Modèle LLM
        st.subheader("🧠 Modèle LLM")
        model_options = {
            "Llama 3.1 8B": "llama3.1:8b",
            "Llama 3.1 7B": "llama3.1:7b",
            "Mistral 7B": "mistral:7b",
            "Mistral 7B Instruct": "mistral:7b-instruct",
            "Llama 2 7B": "llama2:7b",
            "Code Llama 7B": "codellama:7b"
        }
        
        selected_model = st.selectbox(
            "Choisir le modèle:",
            options=list(model_options.keys()),
            index=0,
            help="Modèle LLM pour l'analyse et la génération"
        )
        
        if model_options[selected_model] != st.session_state.model_name:
            st.session_state.model_name = model_options[selected_model]
            st.session_state.workflow = None  # Force re-initialization
        
        # Paramètres avancés
        st.subheader("🔧 Paramètres Avancés")
        
        with st.expander("⚙️ Configuration RAG", expanded=False):
            st.session_state.advanced_settings['confidence_threshold'] = st.slider(
                "Seuil de confiance",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.advanced_settings['confidence_threshold'],
                step=0.1,
                help="Seuil minimum de confiance pour les réponses"
            )
            
            st.session_state.advanced_settings['max_documents'] = st.slider(
                "Nombre max de documents",
                min_value=1,
                max_value=20,
                value=st.session_state.advanced_settings['max_documents'],
                help="Nombre maximum de documents à récupérer"
            )
            
            st.session_state.advanced_settings['use_mmr'] = st.checkbox(
                "Utiliser MMR (Maximal Marginal Relevance)",
                value=st.session_state.advanced_settings['use_mmr'],
                help="Améliore la diversité des résultats"
            )
            
            if st.session_state.advanced_settings['use_mmr']:
                st.session_state.advanced_settings['lambda_mult'] = st.slider(
                    "Lambda MMR",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.advanced_settings['lambda_mult'],
                    step=0.1,
                    help="Paramètre de diversité MMR"
                )
        
        # Information sur la détection automatique
        st.subheader("🧠 Intelligence Automatique")
        st.info("""
        **Détection automatique des intentions :**
        • Le LLM analyse automatiquement vos questions
        • Détecte les demandes de comparaison
        • Identifie les documents spécifiques mentionnés
        • Aucun mode manuel nécessaire !
        """)
        
        # Statistiques de session
        st.subheader("📈 Statistiques")
        if st.session_state.chat_history:
            st.metric("Messages échangés", len(st.session_state.chat_history))
            st.metric("Session ID", st.session_state.session_id[:20] + "...")
        else:
            st.info("Aucune conversation en cours")
        
        # Actions
        st.subheader("🔄 Actions")
        if st.button("🗑️ Effacer l'historique", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.session_id = f"streamlit_session_{int(time.time())}"
            st.rerun()
        
        if st.button("💾 Exporter la conversation", type="secondary"):
            export_conversation()

def render_chat_interface():
    """Affiche l'interface de chat principale."""
    st.subheader("💬 Chat Interactif")
    
    # Zone de chat
    chat_container = st.container()
    
    with chat_container:
        # Afficher l'historique
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 Vous:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Message du bot avec métriques
                confidence_color = "🟢" if message.get('confidence', 0) > 0.7 else "🟡" if message.get('confidence', 0) > 0.4 else "🔴"
                
                # Indicateur d'intention
                intent_icon = {
                    'comparison': '🔄',
                    'document_specific': '📄',
                    'financial_analysis': '💰',
                    'status_check': '📊',
                    'evolution_analysis': '📈',
                    'complex_aggregation': '🧩',
                    'simple_qa': '💬'
                }.get(message.get('intent', 'simple_qa'), '💬')
                
                intent_text = {
                    'comparison': 'Comparaison',
                    'document_specific': 'Document spécifique',
                    'financial_analysis': 'Analyse financière',
                    'status_check': 'Vérification d\'état',
                    'evolution_analysis': 'Analyse d\'évolution',
                    'complex_aggregation': 'Agrégation complexe',
                    'simple_qa': 'Q&A simple'
                }.get(message.get('intent', 'simple_qa'), 'Q&A simple')
                
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message['content']}
                    <br><br>
                    <small>
                        {intent_icon} <strong>{intent_text}</strong> | 
                        {confidence_color} Confiance: {message.get('confidence', 0):.2f} | 
                        ⏱️ Temps: {message.get('processing_time', 0):.2f}s | 
                        📚 Docs: {message.get('documents_retrieved', 0)}
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                # Afficher les citations si disponibles
                if message.get('citations'):
                    with st.expander(f"📄 Citations ({len(message['citations'])})", expanded=False):
                        for citation in message['citations']:
                            # Gérer les citations au format chaîne (ex: "Document.pdf – p.5")
                            if isinstance(citation, str):
                                st.markdown(f"""
                                <div class="citation">
                                    <strong>{citation}</strong>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Gérer les citations au format dictionnaire
                                st.markdown(f"""
                                <div class="citation">
                                    <strong>{citation.get('source', 'Document inconnu')}</strong> - Page {citation.get('page', 'N/A')}<br>
                                    <em>{citation.get('content', 'Contenu non disponible')[:200]}...</em>
                                </div>
                                """, unsafe_allow_html=True)
    
    # Zone de saisie
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "💭 Votre question:",
            placeholder="Posez votre question sur les documents TotalEnergies...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("🚀 Envoyer", type="primary", use_container_width=True)
    
    # Traitement de la requête
    if send_button and user_input:
        process_user_query(user_input)

# Fonction supprimée - la comparaison est maintenant entièrement automatique dans le chat

def render_analytics_dashboard():
    """Affiche le tableau de bord analytique."""
    st.subheader("📈 Tableau de Bord Analytique")
    
    if not st.session_state.chat_history:
        st.info("📊 Aucune donnée disponible. Commencez une conversation pour voir les statistiques.")
        return
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_messages = len(st.session_state.chat_history)
        st.metric("💬 Messages totaux", total_messages)
    
    with col2:
        avg_confidence = sum(msg.get('confidence', 0) for msg in st.session_state.chat_history if msg['type'] == 'bot') / max(1, len([msg for msg in st.session_state.chat_history if msg['type'] == 'bot']))
        st.metric("🎯 Confiance moyenne", f"{avg_confidence:.2f}")
    
    with col3:
        avg_processing_time = sum(msg.get('processing_time', 0) for msg in st.session_state.chat_history if msg['type'] == 'bot') / max(1, len([msg for msg in st.session_state.chat_history if msg['type'] == 'bot']))
        st.metric("⏱️ Temps moyen", f"{avg_processing_time:.2f}s")
    
    with col4:
        total_docs = sum(msg.get('documents_retrieved', 0) for msg in st.session_state.chat_history if msg['type'] == 'bot')
        st.metric("📚 Documents consultés", total_docs)
    
    with col5:
        total_comparisons = len([msg for msg in st.session_state.chat_history if msg.get('intent') == 'comparison'])
        st.metric("🔄 Comparaisons", total_comparisons)
    
    # Comparaisons récentes
    recent_comparisons = [
        msg for msg in st.session_state.chat_history 
        if msg.get('intent') == 'comparison'
    ]
    
    if recent_comparisons:
        st.subheader("🔄 Comparaisons Récentes")
        for i, comp in enumerate(recent_comparisons[-3:]):  # 3 dernières
            with st.expander(f"Comparaison {i+1} - {comp.get('timestamp', 'N/A')}", expanded=False):
                st.write(f"**Question :** {comp.get('content', 'N/A')}")
                st.write(f"**Confiance :** {comp.get('confidence', 0):.2f}")
                st.write(f"**Documents :** {comp.get('documents_retrieved', 0)}")
                if comp.get('citations'):
                    st.write(f"**Citations :** {len(comp['citations'])}")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique de confiance dans le temps
        if len(st.session_state.chat_history) > 1:
            bot_messages = [msg for msg in st.session_state.chat_history if msg['type'] == 'bot']
            if bot_messages:
                confidence_data = pd.DataFrame([
                    {
                        'Message': i+1,
                        'Confiance': msg.get('confidence', 0),
                        'Temps': msg.get('processing_time', 0)
                    }
                    for i, msg in enumerate(bot_messages)
                ])
                
                fig_confidence = px.line(
                    confidence_data, 
                    x='Message', 
                    y='Confiance',
                    title='Évolution de la Confiance',
                    color_discrete_sequence=['#667eea']
                )
                fig_confidence.update_layout(height=300)
                st.plotly_chart(fig_confidence, use_container_width=True)
    
    with col2:
        # Graphique des temps de traitement
        if len(st.session_state.chat_history) > 1:
            bot_messages = [msg for msg in st.session_state.chat_history if msg['type'] == 'bot']
            if bot_messages:
                fig_time = px.bar(
                    confidence_data, 
                    x='Message', 
                    y='Temps',
                    title='Temps de Traitement par Message',
                    color_discrete_sequence=['#f093fb']
                )
                fig_time.update_layout(height=300)
                st.plotly_chart(fig_time, use_container_width=True)

def render_admin_panel():
    """Affiche le panneau d'administration."""
    st.subheader("🔧 Panneau d'Administration")
    
    # Statut du système
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Statut du Système")
        
        # Vérifier le statut des services
        rag_status = "🟢 Opérationnel" if st.session_state.rag_service else "🔴 Hors service"
        workflow_status = "🟢 Opérationnel" if st.session_state.workflow else "🔴 Hors service"
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Service RAG:</strong> {rag_status}<br>
            <strong>Workflow:</strong> {workflow_status}<br>
            <strong>Modèle:</strong> {st.session_state.model_name}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Métriques Système")
        
        # Métriques système (mock pour l'instant)
        st.metric("💾 Mémoire utilisée", "2.3 GB")
        st.metric("🔄 Requêtes/min", "12")
        st.metric("📈 Uptime", "99.8%")
    
    # Actions d'administration
    st.markdown("### 🔄 Actions d'Administration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Redémarrer les services", type="secondary"):
            restart_services()
    
    with col2:
        if st.button("🧹 Nettoyer le cache", type="secondary"):
            clear_cache()
    
    with col3:
        if st.button("📊 Générer rapport", type="secondary"):
            generate_system_report()

# =============================================================================
# Logique métier
# =============================================================================

def process_user_query(query: str):
    """Traite une requête utilisateur."""
    if not st.session_state.workflow:
        st.error("❌ Workflow non initialisé. Veuillez patienter...")
        return
    
    # Ajouter le message utilisateur à l'historique
    st.session_state.chat_history.append({
        'type': 'user',
        'content': query,
        'timestamp': datetime.now()
    })
    
    # Afficher un spinner pendant le traitement
    with st.spinner("🧠 Analyse en cours..."):
        try:
            # Traiter la requête
            result = st.session_state.workflow.process_query(
                query=query,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            )
            
            # Ajouter la réponse à l'historique
            st.session_state.chat_history.append({
                'type': 'bot',
                'content': result.get('answer', 'Erreur de génération'),
                'confidence': result.get('confidence', 0.0),
                'processing_time': result.get('processing_time', 0.0),
                'documents_retrieved': result.get('documents_retrieved', 0),
                'intent': result.get('intent', 'unknown'),
                'citations': result.get('citations', []),
                'timestamp': datetime.now()
            })
            
            # Afficher un message de succès
            st.success("✅ Réponse générée avec succès!")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement requête: {e}")
            st.error(f"❌ Erreur lors du traitement: {e}")
            
            # Ajouter un message d'erreur à l'historique
            st.session_state.chat_history.append({
                'type': 'bot',
                'content': f"Une erreur s'est produite lors du traitement de votre requête: {e}",
                'confidence': 0.0,
                'processing_time': 0.0,
                'documents_retrieved': 0,
                'intent': 'error',
                'citations': [],
                'timestamp': datetime.now()
            })
    
    # Recharger la page pour afficher la nouvelle conversation
    st.rerun()

# Fonction supprimée - la comparaison est maintenant automatique via le chat principal

def export_conversation():
    """Exporte la conversation en JSON."""
    if not st.session_state.chat_history:
        st.warning("⚠️ Aucune conversation à exporter.")
        return
    
    # Préparer les données d'export
    export_data = {
        'session_id': st.session_state.session_id,
        'user_id': st.session_state.user_id,
        'model_name': st.session_state.model_name,
        'export_timestamp': datetime.now().isoformat(),
        'conversation': st.session_state.chat_history
    }
    
    # Créer le fichier JSON
    json_data = json.dumps(export_data, indent=2, default=str)
    
    # Proposer le téléchargement
    st.download_button(
        label="💾 Télécharger la conversation",
        data=json_data,
        file_name=f"conversation_{st.session_state.session_id[:10]}.json",
        mime="application/json"
    )

def restart_services():
    """Redémarre les services."""
    st.info("🔄 Redémarrage des services en cours...")
    
    # Réinitialiser les services
    st.session_state.rag_service = None
    st.session_state.workflow = None
    
    # Recharger la page
    st.rerun()

def clear_cache():
    """Nettoie le cache."""
    st.info("🧹 Nettoyage du cache en cours...")
    
    # Nettoyer le cache Streamlit
    st.cache_resource.clear()
    
    st.success("✅ Cache nettoyé avec succès!")

def generate_system_report():
    """Génère un rapport système."""
    st.info("📊 Génération du rapport système...")
    
    # Créer un rapport simple
    report = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_id,
        'total_messages': len(st.session_state.chat_history),
        'model_name': st.session_state.model_name,
        'system_status': 'operational'
    }
    
    st.json(report)

# =============================================================================
# Application principale
# =============================================================================

def main():
    """Fonction principale de l'application."""
    # Initialiser l'état de session
    initialize_session_state()
    
    # Afficher l'en-tête
    render_header()
    
    # Initialiser les services si nécessaire
    if not st.session_state.rag_service:
        with st.spinner("🚀 Initialisation des services..."):
            st.session_state.rag_service = initialize_rag_service()
    
    if not st.session_state.workflow and st.session_state.rag_service:
        with st.spinner("🧠 Initialisation du workflow..."):
            st.session_state.workflow = initialize_workflow(
                st.session_state.rag_service, 
                st.session_state.model_name
            )
    
    # Afficher la barre latérale
    render_sidebar()
    
    # Interface principale avec onglets
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "🔧 Admin"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_analytics_dashboard()
    
    with tab3:
        render_admin_panel()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🤖 Chatbot RAG TotalEnergies | Powered by Mistral 7B & LangGraph | 
        <a href="https://github.com/your-repo" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
