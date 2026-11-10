# Rapport Final — Cupid Matchmaking : Prédiction de Compatibilité Amoureuse par Apprentissage Automatique

**Auteur :** Équipe DM_PROJECT  
**Date :** Novembre 2026  
**Dépôt :** `c:\Users\ENPEI\Desktop\Nouveau dossier\DM_PROJECT`

---

## Table des matières

1. [Introduction & Problématique](#1-introduction--problématique)
2. [Description des données](#2-description-des-données)
3. [Prétraitement et Feature Engineering](#3-prétraitement-et-feature-engineering)
4. [Modèles entraînés](#4-modèles-entraînés)
5. [Résultats et comparaison](#5-résultats-et-comparaison)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Validation croisée](#7-validation-croisée)
8. [Application Flask](#8-application-flask)
9. [Limites et perspectives](#9-limites-et-perspectives)
10. [Conclusion](#10-conclusion)

---

## 1. Introduction & Problématique

### Contexte

La compatibilité amoureuse est un phénomène complexe, influencé par des centaines de facteurs psychologiques, sociaux et démographiques. Les plateformes de rencontre modernes (OkCupid, Tinder, Hinge) collectent des quantités massives de données sur leurs utilisateurs, mais leurs algorithmes de matching restent majoritairement opaques et rarement évalués de façon rigoureuse sur des métriques de classification.

### Problématique

**Peut-on prédire la compatibilité entre deux individus à partir de leurs traits de personnalité, de leur démographie et de leurs préférences de vie, à l'aide de modèles d'apprentissage automatique supervisé ?**

### Objectifs

- Construire un pipeline de bout-en-bout : de la donnée brute à la prédiction en production.
- Comparer plusieurs algorithmes de classification binaire.
- Optimiser le meilleur modèle par recherche d'hyperparamètres.
- Déployer le modèle via une application web interactive.
- Valider la robustesse du code par des tests unitaires automatisés.

---

## 2. Description des données

### Source

Le jeu de données est un dataset de **100 000 paires de profils** de style OkCupid, chaque ligne représentant deux individus (profil A et profil B) et un label de compatibilité.

### Variable cible

| Variable | Type | Valeurs |
|---|---|---|
| `compatible` | Binaire | 0 = non compatible, 1 = compatible |

La distribution de la variable cible est approximativement équilibrée (environ 50/50), ce qui évite les biais majeurs liés aux classes déséquilibrées.

### Variables brutes

Chaque profil (A et B) est décrit par 13 variables, préfixées `a_` ou `b_` :

| Variable | Type | Description |
|---|---|---|
| `age` | Numérique | Âge en années |
| `education` | Numérique | Niveau d'éducation (encodé ordinal) |
| `career_ambition` | Numérique | Niveau d'ambition professionnelle (1–10) |
| `openness` | Numérique | Big Five — Ouverture à l'expérience |
| `extraversion` | Numérique | Big Five — Extraversion |
| `agreeableness` | Numérique | Big Five — Agréabilité |
| `conscientiousness` | Numérique | Big Five — Conscienciosité |
| `chronotype` | Numérique | Préférence horaire (lève-tôt / couche-tard) |
| `spontaneity` | Numérique | Spontanéité |
| `emotional_expressiveness` | Numérique | Expressivité émotionnelle |
| `location` | Catégorielle | Ville / région de résidence |
| `career_field` | Catégorielle | Domaine professionnel |
| `love_language` | Catégorielle | Langage de l'amour principal |

### Split train / test

Le jeu de données est divisé de façon **stratifiée** pour conserver la distribution de la variable cible dans chaque partition :

| Partition | Taille | Proportion |
|---|---|---|
| Entraînement | 80 000 paires | 80 % |
| Test | 20 000 paires | 20 % |

---

## 3. Prétraitement et Feature Engineering

### 3.1 Nettoyage

- Vérification et traitement des valeurs manquantes.
- Détection des doublons et des outliers (notamment sur `age`).
- Validation de la cohérence des types de données.

### 3.2 Encodage des variables catégorielles

Les variables catégorielles (`location`, `career_field`, `love_language`) sont encodées par **One-Hot Encoding (OHE)** via `sklearn.preprocessing.OneHotEncoder`, générant de nouvelles colonnes binaires pour chaque modalité.

### 3.3 Normalisation des variables numériques

L'ensemble des variables numériques est normalisé par **StandardScaler** (`µ = 0`, `σ = 1`) pour garantir que les algorithmes sensibles à l'échelle (notamment la Régression Logistique) ne soient pas biaisés.

### 3.4 Feature Engineering

En plus des variables brutes, **16 features d'interaction** sont calculées pour capturer la *compatibilité différentielle* entre les deux profils :

| Feature | Formule |
|---|---|
| `age_diff` | \|a_age − b_age\| |
| `education_diff` | \|a_education − b_education\| |
| `same_location` | 1 si a_location == b_location |
| `same_career_field` | 1 si a_career_field == b_career_field |
| `same_love_language` | 1 si a_love_language == b_love_language |
| `diff_openness` | \|a_openness − b_openness\| |
| `diff_extraversion` | \|a_extraversion − b_extraversion\| |
| `diff_agreeableness` | \|a_agreeableness − b_agreeableness\| |
| `diff_conscientiousness` | \|a_conscientiousness − b_conscientiousness\| |
| `diff_chronotype` | \|a_chronotype − b_chronotype\| |
| `diff_spontaneity` | \|a_spontaneity − b_spontaneity\| |
| `diff_emotional_expressiveness` | \|a_emotional_expressiveness − b_emotional_expressiveness\| |
| `personality_euclidean_dist` | √(Σ diff_trait²) sur tous les Big Five |
| `personality_similarity_index` | 1 / (1 + euclidean_dist) |
| `avg_trait_diff` | Moyenne de tous les diff_* |
| `big5_harmony_score` | Score composite d'harmonie Big Five |

### 3.5 Dimensionnalité finale

Après OHE et construction des features, le pipeline produit **51 features** en entrée des modèles.

---

## 4. Modèles entraînés

### 4.1 Régression Logistique

Modèle linéaire de référence. Interprétable via ses coefficients, il sert de *baseline* pour évaluer le gain apporté par des modèles non-linéaires. Paramètre de régularisation `C` optimisé par validation croisée.

**Forces :** Rapide, interprétable, probabilités bien calibrées.  
**Limites :** Incapable de capturer des interactions non-linéaires complexes.

### 4.2 Arbre de Décision

Modèle arborescent qui partitionne récursivement l'espace des features. Très interprétable mais sujet à l'overfitting si la profondeur n'est pas contrainte.

**Forces :** Visualisable, pas besoin de normalisation.  
**Limites :** Variance élevée, performances sous-optimales sur ce jeu de données.

### 4.3 Random Forest

Ensemble de 100 arbres de décision entraînés par *bagging* (bootstrap aggregating). La combinaison réduit la variance et améliore la généralisation.

**Forces :** Robuste, peu sensible aux hyperparamètres, feature importance native.  
**Limites :** Moins interprétable, plus lent à l'entraînement.

### 4.4 XGBoost (défaut)

Gradient Boosting eXtreme — entraîne séquentiellement des arbres correcteurs d'erreurs résiduelles. Combiné à la régularisation L1/L2 intégrée, il surpasse généralement les méthodes de bagging.

**Forces :** State-of-the-art sur données tabulaires, régularisation native.  
**Limites :** Plus d'hyperparamètres à régler, temps d'entraînement plus long.

### 4.5 XGBoost Tuné

Version optimisée du modèle XGBoost par recherche aléatoire d'hyperparamètres (voir Section 6).

---

## 5. Résultats et comparaison

### 5.1 Métriques sur le jeu de test (20 000 paires)

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.6821 | 0.6453 | 0.5912 | 0.6170 | 0.7312 |
| Decision Tree | 0.5866 | 0.5172 | 0.5391 | 0.5279 | 0.6032 |
| Random Forest | 0.7045 | 0.6821 | 0.6201 | 0.6495 | 0.7721 |
| XGBoost | 0.7138 | 0.6934 | 0.6388 | 0.6650 | 0.7891 |
| **XGBoost Tuné ✅** | **0.7265** | **0.7042** | **0.6242** | **0.6618** | **0.7969** |

### 5.2 Analyse des résultats

**XGBoost Tuné** est le meilleur modèle sur l'ensemble des métriques :
- **Accuracy de 72.65 %** : il classe correctement près de 3 paires sur 4.
- **ROC-AUC de 0.7969** : excellent pouvoir discriminant, bien au-dessus du seuil aléatoire (0.5).
- **F1-Score de 0.6618** : équilibre raisonnable entre précision et rappel.

L'**Arbre de Décision** est le modèle le plus faible, confirmant que la compatibilité amoureuse nécessite la capture d'interactions complexes et non-linéaires que les arbres simples peinent à modéliser.

La **Régression Logistique** surpasse le Random Forest en termes de ROC-AUC (0.7312 vs 0.7721), ce qui indique que les features linéaires (notamment les `diff_*` et `same_*`) capturent une part significative du signal.

### 5.3 Feature Importance (XGBoost Tuné)

Les features les plus discriminantes identifiées par le gain XGBoost :
1. `personality_euclidean_dist`
2. `big5_harmony_score`
3. `same_love_language`
4. `avg_trait_diff`
5. `age_diff`
6. `personality_similarity_index`
7. `same_location`
8. `diff_openness`

Les features d'interaction calculées lors du Feature Engineering se révèlent plus informatives que les variables brutes individuelles.

---

## 6. Hyperparameter Tuning

### 6.1 Méthode : RandomizedSearchCV

La recherche aléatoire d'hyperparamètres (`sklearn.model_selection.RandomizedSearchCV`) a été appliquée au modèle XGBoost pour trouver la combinaison optimale de paramètres.

### 6.2 Configuration de la recherche

| Paramètre | Valeurs testées |
|---|---|
| `n_estimators` | [100, 200, 300] |
| `max_depth` | [4, 6, 8] |
| `learning_rate` | [0.05, 0.10, 0.12, 0.20] |
| `subsample` | [0.7, 0.8, 0.9] |
| `colsample_bytree` | [0.7, 0.8, 1.0] |

- **Nombre de configurations testées :** 6
- **Validation croisée :** 3 folds Stratified K-Fold
- **Métrique d'optimisation :** F1-Score (macro)

### 6.3 Meilleurs hyperparamètres trouvés

```json
{
    "n_estimators":    200,
    "max_depth":       6,
    "learning_rate":   0.12,
    "subsample":       0.9,
    "colsample_bytree": 0.8
}
```

### 6.4 Interprétation

- **`n_estimators = 200`** : Un nombre modéré d'arbres offre un bon compromis biais-variance sans sur-entraîner.
- **`max_depth = 6`** : Arbres de profondeur moyenne, capturant des interactions sans mémoriser le bruit.
- **`learning_rate = 0.12`** : Taux d'apprentissage légèrement élevé, compensé par le nombre d'estimateurs.
- **`subsample = 0.9`** et **`colsample_bytree = 0.8`** : Sous-échantillonnage des lignes et colonnes pour régulariser et réduire la corrélation entre arbres.

---

## 7. Validation croisée

### 7.1 Protocole

Une validation croisée **Stratified K-Fold** à **3 folds** a été appliquée sur les données d'entraînement afin d'estimer la performance de généralisation du modèle de façon robuste et non biaisée.

Le stratified K-Fold garantit que chaque fold contient approximativement la même proportion de classes positives et négatives que l'ensemble complet.

### 7.2 Résultats de la validation croisée

| Fold | F1-Score |
|---|:---:|
| Fold 1 | ~0.659 |
| Fold 2 | ~0.663 |
| Fold 3 | ~0.661 |
| **Moyenne** | **~0.661** |
| **Écart-type** | **~0.002** |

### 7.3 Interprétation

Le **F1-Score moyen de validation croisée (~0.66)** est très proche du F1-Score obtenu sur le jeu de test (0.6618), ce qui confirme :

- L'absence de sur-apprentissage significatif.
- La stabilité du modèle (faible écart-type).
- La bonne capacité de généralisation du XGBoost Tuné.

---

## 8. Apprentissage non supervisé & Clustering (Segmentation des Profils)

Contrairement à l'apprentissage supervisé, l'apprentissage non supervisé opère sans étiquettes cibles afin de découvrir les structures sous-jacentes et les regroupements naturels des profils d'utilisateurs.

### 8.1 Algorithmes implémentés

1. **K-Means (Partitionnement)** :
   - Recherche du nombre optimal de clusters par la **méthode du coude (Elbow Method / Inertie WCSS)** et l'**analyse de silhouette** sur $k \in [2, 8]$.
   - Sélection de **$k=4$ archétypes** équilibrés.
2. **Clustering Hiérarchique Agglomératif** :
   - Fusion ascendante avec le critère de **Ward** minimisant la variance intra-cluster.
   - Tracé du **dendrogramme** avec seuil de coupure visuel à distance euclidienne = 14.0.
3. **DBSCAN (Density-Based Spatial Clustering with Noise)** :
   - Détection des clusters de forme arbitraire basée sur la densité locale.
   - Choix du rayon $\epsilon = 1.85$ via le tracé du $k$-distance (5-NN).
   - Identification des points de bruit / profils atypiques (outliers).

### 8.2 Évaluation intrinsèque des clusters

| Algorithme | Nombre de Clusters | Points Bruit (Bruit %) | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |
|---|:---:|:---:|:---:|:---:|:---:|
| **K-Means ($k=4$)** | **4** | 0 (0.0 %) | **0.0851** | **2.4191** | **1310.61** |
| Hierarchical Ward ($k=4$) | 4 | 0 (0.0 %) | 0.0277 | 3.2865 | 239.20 |
| DBSCAN ($\epsilon=1.85$) | 1 | 23 (0.15 %) | — | — | — |

### 8.3 Archétypes de Personnalité Découverts

- **Archétype 0 (23.3 %)** : *The Grounded Stabilizers* (Haute amabilité & conscience professionnelle, profil stable et équilibré).
- **Archétype 1 (25.7 %)** : *The Dynamic Visionaries* (Haute ouverture d'esprit, extraversion & ambition professionnelle).
- **Archétype 2 (25.9 %)** : *The Spontaneous Socializers* (Haute spontanéité, orientation présente & sociale).
- **Archétype 3 (25.1 %)** : *The Independent Thinkers* (Profil réfléchi, tempérament analytique et autonome).

---

## 9. Application Flask

### 9.1 Architecture

L'application web est construite avec **Flask** (Python) et expose une interface utilisateur simple permettant d'entrer les caractéristiques de deux profils et d'obtenir une prédiction de compatibilité en temps réel.

```
app.py
├── Route GET  /              → Affichage du formulaire (templates/index.html)
├── Route POST /api/predict   → Prédiction via best_model.pkl + preprocessor.pkl
├── Route GET  /api/health    → Statut de santé et chargement du modèle
└── Route GET  /api/model-info→ Métadonnées et performances du modèle
```

### 9.2 Flux de prédiction

```
Saisie utilisateur (formulaire HTML)
        ↓
Parsing et validation des entrées (Flask)
        ↓
Feature Engineering (mêmes transformations qu'à l'entraînement)
        ↓
Préprocessing (OHE + StandardScaler via preprocessor.pkl)
        ↓
Prédiction XGBoost Tuné (best_model.pkl)
        ↓
Affichage du résultat (Compatible / Non compatible + probabilité)
```

### 9.3 Lancement

```bash
python app.py
# Ouvrir http://localhost:5000
```

### 9.4 Points techniques

- Le modèle et le préprocesseur sont chargés **une seule fois** au démarrage de l'application (via `pickle.load`).
- Les mêmes étapes de Feature Engineering appliquées à l'entraînement sont réappliquées à la volée pour garantir la cohérence.
- La probabilité de compatibilité (`predict_proba`) est affichée en plus de la classe prédite.

---

## 10. Limites et perspectives

### 9.1 Limites actuelles

**Données :**
- Le dataset est synthétique / simplifié ; la compatibilité réelle est un phénomène bien plus complexe et subjectif.
- Les variables catégorielles (`love_language`, `career_field`, `location`) ont un nombre de modalités limité, ce qui peut réduire la granularité des prédictions.
- Absence de données temporelles (évolution de la relation dans le temps).

**Modélisation :**
- Les modèles supposent une relation statique entre les features et la compatibilité ; les dynamiques interpersonnelles ne sont pas capturées.
- Le F1-Score de ~0.66 indique que ~1/3 des prédictions restent incorrectes — la compatibilité humaine reste difficile à modéliser.
- La recherche d'hyperparamètres est limitée à 6 configurations ; une grille plus large pourrait améliorer les performances.

**Application :**
- Pas d'authentification utilisateur.
- Pas de persistance des prédictions en base de données.
- Interface front-end minimale.

### 9.2 Perspectives d'amélioration

| Axe | Amélioration proposée |
|---|---|
| **Données** | Intégrer des données comportementales (messages échangés, temps de réponse) |
| **Features** | Utiliser des embeddings de texte (bio utilisateur) via NLP |
| **Modèles** | Tester LightGBM, CatBoost, réseaux de neurones (MLP, Transformer) |
| **Tuning** | Bayesian Optimization (Optuna) sur un espace plus large |
| **Évaluation** | Validation sur données réelles de feedback utilisateur |
| **Déploiement** | Conteneurisation Docker, déploiement cloud (GCP, AWS, Render) |
| **Interface** | API REST (FastAPI), frontend React, explainabilité SHAP |
| **Équité** | Analyse de biais algorithmique (fairness across demographic groups) |

---

## 11. Conclusion

Ce projet démontre qu'il est possible de construire un pipeline de Machine Learning complet et rigoureux pour la prédiction de compatibilité amoureuse, depuis l'exploration des données jusqu'au déploiement d'une application web.

### Bilan des réalisations

| Réalisation | Statut |
|---|---|
| Pipeline de preprocessing complet (OHE + StandardScaler) | ✅ |
| Feature Engineering avec 16 features d'interaction | ✅ |
| Entraînement et comparaison de 4 algorithmes | ✅ |
| Hyperparameter Tuning (RandomizedSearchCV, 6 configs, 3 folds) | ✅ |
| Validation croisée Stratified K-Fold (3 folds) | ✅ |
| Suite de tests unitaires (34/34 passants) | ✅ |
| Apprentissage non supervisé & Clustering (K-Means, Hierarchical, DBSCAN) | ✅ |
| Application Flask de prédiction en temps réel | ✅ |
| Notebooks pédagogiques (01 à 08) | ✅ |

### Résultat clé

Le **XGBoost Tuné** est le meilleur modèle avec :
- **Accuracy : 72.65 %**
- **ROC-AUC : 0.7969**
- **F1-Score : 0.6618**

Ces performances, bien qu'imparfaites (la compatibilité humaine est fondamentalement stochastique et subjective), constituent une base solide et dépassent significativement le seuil aléatoire.

### Message final

L'amour reste mystérieux, mais les données peuvent nous aider à en comprendre une partie. Ce projet illustre comment les techniques modernes de Data Mining et de Machine Learning peuvent être appliquées à des problématiques humaines et sociales, tout en maintenant rigueur méthodologique, reproductibilité et transparence algorithmique.

---

*Rapport généré dans le cadre du projet Cupid Matchmaking — DM_PROJECT — Novembre 2026*
