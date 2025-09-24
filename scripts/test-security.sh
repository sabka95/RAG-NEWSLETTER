#!/bin/bash
# =============================================================================
# Script de Test de Sécurité Avancé
# =============================================================================
# Exécute les tests de sécurité multi-couches avec red teaming automatisé
# =============================================================================

set -e  # Arrêter en cas d'erreur

echo "🔒 Tests de Sécurité Avancés - Red Teaming Automatisé"
echo "====================================================="

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Erreur: pyproject.toml non trouvé. Exécutez ce script depuis la racine du projet."
    exit 1
fi

# Vérifier que Poetry est installé
if ! command -v poetry &> /dev/null; then
    echo "❌ Erreur: Poetry n'est pas installé. Installez Poetry d'abord."
    exit 1
fi

echo "✅ Vérifications préliminaires réussies"

# Installer les dépendances
echo "📦 Installation des dépendances..."
poetry install --with dev

# Exécuter les tests de sécurité
echo "🔒 Exécution des tests de sécurité..."
poetry run python test_security_advanced.py

# Vérifier que le rapport a été généré
if [ -f "security_test_report.json" ]; then
    echo "✅ Rapport de sécurité généré: security_test_report.json"
    
    # Afficher un résumé du rapport
    echo "📊 Résumé du rapport de sécurité:"
    poetry run python -c "
import json
with open('security_test_report.json', 'r') as f:
    report = json.load(f)

print(f'🛡️ Efficacité contre les attaques: {report[\"attack_simulation\"][\"effectiveness_rate\"]:.1f}%')
print(f'✅ Taux d\'approbation des requêtes légitimes: {report[\"benign_queries\"][\"approval_rate\"]:.1f}%')
print(f'🔒 Couches de sécurité actives: {len([k for k, v in report[\"security_layers\"].items() if v == \"active\"])}/4')

if report['recommendations']:
    print('💡 Recommandations:')
    for rec in report['recommendations']:
        print(f'   • {rec}')
"
else
    echo "⚠️ Avertissement: Rapport de sécurité non généré"
fi

echo "🎉 Tests de sécurité terminés!"
echo "📋 Consultez security_test_report.json pour le rapport détaillé"
