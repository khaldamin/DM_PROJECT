---
marp: true
theme: default
paginate: true
header: "Cupid AI — Data Mining & Matchmaking Prediction"
footer: "DM_PROJECT — Soutenance de Projet Machine Learning"
style: |
  section {
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 40px;
  }
  h1 {
    color: #e74c3c;
  }
  h2 {
    color: #2c3e50;
  }
  .highlight {
    background-color: #f8f9fa;
    border-left: 4px solid #e74c3c;
    padding: 10px;
  }
---

# 💘 Cupid Matchmaking
### Prédiction de la Compatibilité Relationnelle & Segmentation des Personnalités
**Projet de Data Mining & Machine Learning**

*Auteurs : Équipe DM_PROJECT*  
*Date : Novembre 2026*  
*Dépôt GitHub : [github.com/khaldamin/DM_PROJECT](https://github.com/khaldamin/DM_PROJECT)*

---

## 🎯 1. Contexte & Problématique

- **Constat** : Les plateformes de rencontre modernes (Tinder, Bumble, OkCupid) traitent des millions d'interactions quotidiennes.
- **Le Défi** : Peut-on modéliser et prédire la compatibilité humaine à partir de traits psychologiques et démographiques ?
- **Objectifs du Projet** :
  1. **Supervisé** : Prédire si deux personnes sont compatibles (`compatible` $\in \{0, 1\}$).
  2. **Non Supervisé** : Segmenter les profils en archétypes psychologiques (Clustering).
  3. **Explicabilité (XAI)** : Comprendre *pourquoi* deux profils matchent grâce à SHAP.
  4. **Déploiement** : Créer une application web temps réel sous Flask.

---

## 📊 2. Données & Exploration (EDA)

- **Dataset** : 100 000 paires de profils (OkCupid synthetic dataset).
- **Attributs par profil (Partenaire A & B)** :
  - *Démographiques* : Âge, niveau d'études (1-4), localisation.
  - *Traits Big 5* : Ouverture, Extraversion, Amabilité, Conscience professionnelle.
  - *Habitudes de vie* : Chronotype (matin/soir), Spontanéité, Expressivité émotionnelle, Ambition, Langage de l'amour, Métier.
- **Variable Cible** : `compatible` (1 = match réussi, 0 = échec).
- **Équilibre des classes** : Distribution naturelle préservée par échantillonnage stratifié 80% Train / 20% Test.

---

## ⚙️ 3. Prétraitement & Feature Engineering

### Prévention absolue du Data Leakage :
- Le préprocesseur est ajusté **uniquement** sur le jeu d'entraînement (`X_train`).
- À l'intérieur de la validation croisée, chaque fold ré-entraîne un préprocesseur isolé via `Pipeline`.

### 16 Caractéristiques d'Interaction Créées :
- Écarts absolus sur tous les traits : `diff_openness`, `diff_extraversion`, etc.
- **Distance euclidienne psychologique** : $\sqrt{\sum (a_i - b_i)^2}$.
- **Indice d'harmonie Big 5** : Stabilité conjointe du couple.
- **Matchs catégoriels** : `same_location`, `same_love_language`, `same_career_field`.
- **Dimension finale** : **51 features** après One-Hot Encoding et StandardScaler.

---

## 🤖 4. Modèles Supervisés & Méthodologie

Quatre familles d'algorithmes comparées selon le paradigme OOP (`BaseModel`) :

1. **Régression Logistique** : Modèle linéaire de référence avec régularisation $L_2$.
2. **Arbre de Décision (CART)** : Modèle non linéaire interpretable (`max_depth=12`).
3. **Random Forest** : Ensemble par bagging de 100 arbres réduisant la variance.
4. **XGBoost (Extreme Gradient Boosting)** : Boosting par gradient avec fonction de perte logistique.
5. **XGBoost Tuné** : Optimisation des hyperparamètres par `RandomizedSearchCV` sur 5 dimensions.

---

## 📈 5. Résultats & Benchmark Comparatif

Évaluation sur le jeu de test indépendant ($N=20\,000$) :

| Modèle | Accuracy | Précision | Rappel | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.6821 | 0.6453 | 0.5912 | 0.6170 | 0.7312 |
| Decision Tree | 0.5866 | 0.5172 | 0.5391 | 0.5279 | 0.6032 |
| Random Forest | 0.7045 | 0.6821 | 0.6201 | 0.6495 | 0.7721 |
| XGBoost | 0.7138 | 0.6934 | 0.6388 | 0.6650 | 0.7891 |
| **XGBoost Tuné** 🏆 | **0.7265** | **0.7042** | **0.6242** | **0.6618** | **0.7969** |

> **Vainqueur** : **XGBoost Tuné** surpasse tous les modèles avec un **ROC-AUC de ~0.80**.

---

## 🔍 6. Explicabilité des Décisions (SHAP XAI)

- **Constat** : Un modèle boîte noire est inacceptable dans les relations humaines.
- **Global SHAP Attribution** :
  - Facteur n°1 de friction : `diff_career_ambition` (divergence d'ambition professionnelle).
  - Facteurs d'harmonie : `same_love_language`, `personality_euclidean_dist`, `same_location`.
- **Local SHAP Waterfall** :
  - Pour chaque prédiction individuelle, l'algorithme génère la cascade exacte des gains/pertes de probabilité.

---

## 🔮 7. Clustering & Segmentation des Personnalités

Mise en œuvre des **3 grands paradigmes d'apprentissage non supervisé** :

1. **K-Means ($k=4$)** :
   - Sélectionné via la **Méthode du Coude (Inertie WCSS)** et l'**Analyse de Silhouette**.
2. **Clustering Hiérarchique Agglomératif (Ward)** :
   - Visualisation de l'arbre généalogique des profils par **Dendrogramme**.
3. **DBSCAN (Density-Based)** :
   - Analyse du $k$-distance pour fixer $\epsilon = 1.85$.
   - Identification et isolation des **profils atypiques (outliers / bruit = 0.15%)**.

---

## 👥 8. Les 4 Archétypes Découverts

| Cluster | Archétype | Part | Profil Psychologique |
|:---:|---|:---:|---|
| **0** | **The Grounded Stabilizers** | 23.3% | Haute amabilité & conscience professionnelle, stabilité |
| **1** | **The Dynamic Visionaries** | 25.7% | Forte ouverture d'esprit, extraversion & ambition |
| **2** | **The Spontaneous Socializers** | 25.9% | Spontanéité élevée, énergie sociale immédiate |
| **3** | **The Independent Thinkers** | 25.1% | Tempérament calme, analytique et forte autonomie |

---

## 🧩 9. Matrice d'Affinité Croisée (Pont Non-Supervisé / Supervisé)

*Quel archétype s'entend le mieux avec quel autre ?*

- **Constat marquant** :
  - **Independent Thinkers × Independent Thinkers** : **51.3 %** de taux de réussite (plus haut score de compatibilité du dataset !).
  - **Grounded Stabilizers × Grounded Stabilizers** : **49.0 %** de succès.
  - Les opposés extrêmes (ex: Stabilizers calmes vs Visionaries très ambitieux) présentent un taux de friction accru (~40.5 %).

---

## 💻 10. Déploiement & Application Web Flask

- **Architecture** : Serveur Flask REST + Interface Glassmorphism moderne (HTML5/CSS3/JS).
- **Fonctionnalités** :
  - Saisie complète des 2 profils ou sélection de **Presets rapides** (*Perfect Harmony*, *Balanced Duo*, *Opposites Attract*, *High Friction*).
  - Calcul du score de compatibilité (0-100%), décision binaire, estimation de longévité en mois, et score d'harmonie Big 5.
  - Endpoints `/api/predict`, `/api/health`, `/api/model-info`.
- **Qualité logicielle** : **36 tests unitaires passants à 100 %**.

---

## ⚖️ 11. Limites & Éthique du Projet

1. **Données Synthétiques** : Les interactions humaines réelles intègrent l'attirance physique, le timing de vie et la communication non-verbale.
2. **Biais Algorithmique** : Risque d'homophilie excessive (le modèle peut avoir tendance à privilégier les profils trop similaires).
3. **Performance Réelle** : Un F1 de ~0.66 reflète la complexité intrinsèque de l'amour : une part d'imprévisibilité reste irréductible.

---

## 🏁 12. Conclusion & Bilan

- **Réalisations** :
  - Pipeline de bout en bout respectant l'état de l'art du Data Mining.
  - Double approche réussie : **Classification Supervisée** (XGBoost 0.80 AUC) + **Clustering Non Supervisé** (4 archétypes).
  - Couche XAI SHAP pour une transparence totale des décisions.
  - Code documenté, testé (36/36 tests) et disponible en Open Source sur GitHub.

**Merci pour votre attention ! Place aux questions.**

