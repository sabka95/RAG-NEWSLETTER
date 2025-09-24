# 🔒 Sécurité Avancée - RAG Newsletter

## Architecture de Sécurité Multi-Couches

Le système RAG Newsletter implémente une architecture de sécurité multi-couches pour protéger contre les attaques et garantir la conformité :

### 🛡️ Couches de Sécurité

#### 1. **Couche Regex (Rapide)**
- Détection basique des patterns d'injection
- Filtrage du contenu inapproprié
- Normalisation des requêtes
- **Performance** : < 1ms par requête

#### 2. **Couche LLM (Précis)**
- Détection sémantique avec Llama2 7B
- Analyse intelligente des tentatives de manipulation
- Détection des injections subtiles
- **Performance** : ~100ms par requête

#### 3. **Couche Guardrails AI (Complet)**
- Détection PII (données personnelles)
- Filtrage du contenu toxique
- Validation complète des requêtes
- **Performance** : ~50ms par requête

#### 4. **Couche Red Teaming (Robustesse)**
- Tests de robustesse automatisés
- Simulation d'attaques adversaires
- Détection des vulnérabilités
- **Performance** : ~20ms par requête

## 🚀 Utilisation

### Configuration Basique
```python
from rag_newsletter.workflows.security_filter import SecurityFilter

# Filtre avec toutes les couches
filter = SecurityFilter(
    enable_llm_detection=True,
    enable_guardrails=True,
    enable_red_teaming=True
)

# Test d'une requête
result = filter.filter_query("Quels sont les objectifs 2025?")
print(result["status"])  # "approved"
```

### Configuration Optimisée pour Production
```python
# Configuration équilibrée (recommandée)
filter = SecurityFilter(
    enable_llm_detection=True,    # Détection sémantique
    enable_guardrails=True,       # Protection complète
    enable_red_teaming=False      # Désactivé en production
)
```

## 🧪 Tests de Sécurité

### Tests Locaux
```bash
# Exécuter tous les tests de sécurité
./scripts/test-security.sh

# Ou directement avec Python
poetry run python test_security_advanced.py
```

### Tests CI/CD
Les tests de sécurité sont automatiquement exécutés :
- **À chaque push/PR** : Tests de base
- **Quotidiennement** : Tests complets de red teaming
- **Sur main** : Tests de performance et robustesse

### Red Teaming Automatisé
```python
from test_security_advanced import SecurityRedTeam

red_team = SecurityRedTeam()
filter = SecurityFilter(enable_red_teaming=True)

# Simulation d'attaques
results = red_team.run_attack_simulation(filter)
print(f"Efficacité: {results['effectiveness_rate']:.1f}%")
```

## 📊 Métriques de Sécurité

### Indicateurs Clés
- **Taux de blocage des attaques** : > 90%
- **Taux d'approbation des requêtes légitimes** : > 95%
- **Temps de traitement moyen** : < 100ms
- **Faux positifs** : < 5%

### Monitoring
```python
# Statistiques du filtre
stats = filter.get_security_stats()
print(f"Patterns d'injection: {stats['injection_patterns_count']}")
print(f"Patterns inappropriés: {stats['inappropriate_patterns_count']}")
```

## 🔧 Configuration Avancée

### Personnalisation des Patterns
```python
# Ajouter des patterns personnalisés
filter.injection_patterns.extend([
    r"custom\s+injection\s+pattern",
    r"autre\s+pattern\s+malveillant"
])
```

### Configuration des Seuils
```python
# Ajuster les seuils de détection
filter.llm_confidence_threshold = 0.8
filter.similarity_threshold = 0.7
```

## 🚨 Gestion des Incidents

### Logs de Sécurité
```python
# Les tentatives d'attaque sont automatiquement loggées
logger.warning(f"🚨 Prompt injection détecté: {pattern}")
logger.warning(f"🚨 Contenu inapproprié détecté: {pattern}")
```

### Alertes
- **Niveau 1** : Tentatives d'injection basiques
- **Niveau 2** : Attaques sophistiquées détectées par LLM
- **Niveau 3** : Vulnérabilités découvertes par red teaming

## 📈 Amélioration Continue

### Mise à Jour des Patterns
1. Analyser les logs d'attaque
2. Identifier les nouveaux patterns
3. Mettre à jour les regex et prompts LLM
4. Tester avec red teaming

### Formation du Modèle
```python
# Améliorer la détection LLM avec des exemples
training_examples = [
    ("Ignore previous instructions", "malicious"),
    ("Quels sont les objectifs?", "benign")
]
```

## 🔐 Conformité et Audit

### RGPD
- Détection automatique des données personnelles
- Logs anonymisés des tentatives d'accès
- Rétention limitée des données de sécurité

### Audit de Sécurité
- Rapports automatiques générés
- Métriques de performance trackées
- Historique des tentatives d'attaque

## 🆘 Support et Dépannage

### Problèmes Courants

#### LLM Detection Non Disponible
```bash
# Vérifier Ollama
ollama list
ollama pull llama2:7b
```

#### Guardrails AI Non Disponible
```bash
# Installer Guardrails
poetry add guardrails-ai
```

#### Performance Lente
```python
# Désactiver les couches lourdes
filter = SecurityFilter(
    enable_llm_detection=False,  # Plus rapide
    enable_guardrails=True,      # Garder la protection
    enable_red_teaming=False     # Désactiver en production
)
```

### Contact Support
- **Issues** : GitHub Issues
- **Documentation** : Wiki du projet
- **Email** : security@rag-newsletter.com

---

**🔒 Sécurité RAG Newsletter** - Protection multi-couches pour l'entreprise
