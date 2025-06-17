# 📊 Dashboard interactif de scoring crédit & veille technique NLP

**Projet 8 – Mastère Spécialisé Data Science – OpenClassrooms**  
**Client : Prêt à dépenser (société financière)**  
**Rôle : Data Scientist / Développeur Dashboard / Veille IA**

---

## 🎯 Objectif du projet

Après la mise en production d’un modèle de scoring crédit (voir Projet 7), **Prêt à dépenser** souhaite :
- Offrir une **interface explicative** du modèle aux chargés de relation client
- Permettre aux clients finaux de **comprendre les décisions de crédit**
- Mener une **veille technologique** sur les modèles NLP récents (ex. : DeBERTa)

---

## 🖥️ Partie 1 – Dashboard interactif de scoring

### ✅ Objectifs fonctionnels

Le dashboard développé avec **Streamlit** permet de :

- ✅ Visualiser le **score de solvabilité** et sa **probabilité** :
  - Indique si la prédiction est proche ou éloignée du seuil d’acceptation
  - Compréhension visuelle adaptée aux non-experts

- ✅ Présenter les **informations descriptives clés d’un client** (âge, revenus, type de prêt...)

- ✅ Comparer un client à l’ensemble ou à un **sous-groupe de clients similaires** :
  - Interface par **filtres dynamiques** (ex. : groupe de revenus, statut familial, etc.)
  - Graphiques interactifs comparatifs

- ✅ Assurer l’**accessibilité** du contenu graphique (conformité **WCAG** niveau AA) :
  - Couleurs perceptibles pour daltoniens
  - Utilisation de **descriptions textuelles**, contraste suffisant

- ✅ Être **accessible à distance** via **déploiement Cloud (Streamlit Cloud)**

- 🔁 Optionnel : possibilité de **modifier un profil client ou d’en saisir un nouveau** pour recalculer le score et la probabilité (simulation en direct)

---

### 📦 Fonctionnalités principales

| Fonction                             | Implémentation |
|--------------------------------------|----------------|
| Score de prédiction                  | API FastAPI hébergée sur AWS ECS |
| Graphique SHAP local(client) /global        | SHAP + Pyplot/Seaborn |
| Statistiques descriptives client     | Pandas + Streamlit display |
| Comparaison client/population        | Filtrage interactif + graphiques |
| Accessibilité WCAG                   | Contraste élevé + palettes perceptibles |
| Simulation de scénario               | Widgets d’entrée + appel API |

---

## 📚 Partie 2 – Veille technique : DeBERTa vs BERT

### 🔬 Objectif :
Comparer une approche **state-of-the-art NLP (DeBERTa)** à celle utilisée précédemment (BERT) pour la **classification de descriptions produit** (jeu de données du projet OC-P6).

### 📄 Contenu de l’analyse :
- Revue de l’architecture DeBERTa (Decoding-enhanced BERT with disentangled attention)
- Comparaison expérimentale sur un **sous-ensemble**
- Évaluation des performances (ARI score, matrice de confusion et visualisation t-SNE)
- Etude des Feature importance locale/globale
- Limites et améliorations

### 📁 Livrables NLP :
- 📓 Notebook comparatif 
- 📄 Note méthodologique 

---

## 🔧 Compétences mobilisées

| Domaine              | Compétence                                                                  |
|----------------------|------------------------------------------------------------------------------|
| Modélisation          | Comparaison fine de modèles NLP supervisés (DeBERTa vs BERT)               |
| Explicabilité         | SHAP & LIME pour expliquer les prédictions (globalement et localement)     |
| MLOps                | Appels d’API, intégration backend/Streamlit                                 |
| Visualisation         | UX/UI claire pour non-techniciens (chargés de relation client)             |
| Veille IA             | Rédaction d’une note de synthèse sur un modèle récent                      |
| Communication         | Présentation orale & support technique                                     |

---

## 🙋‍♂️ Réalisé par

**Maodo FALL**  
*Data Scientist*

---



