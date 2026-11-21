# 🎓 Guide Complet & Script de Soutenance — Cupid Matchmaking AI

**Durée cible :** 15 à 20 minutes + 10 minutes de Q&A  
**Projet :** Data Mining & Machine Learning Cupid Matchmaking  
**Auteurs :** Équipe DM_PROJECT  

---

## ⏱️ Découpage du Temps & Sommaire

| # | Titre du Slide / Section | Durée recommandée | Objectif clé |
|---|---|:---:|---|
| **1** | Introduction & Contexte du Projet | 1 min 30 | Poser la problématique humaine et algorithmique |
| **2** | Description des Données & Analyse Exploratoire (EDA) | 2 min | Montrer la compréhension du jeu de données (100k paires) |
| **3** | Feature Engineering & Prévention du Data Leakage | 2 min 30 | Expliquer les 73 variables et la rigueur de validation |
| **4** | Benchmark des Modèles Supervisés & Tuning | 2 min 30 | Justifier le choix du meilleur modèle (XGBoost) |
| **5** | Étude d'Ablation & Courbes d'Apprentissage | 1 min 30 | Prouver la valeur ajoutée du feature engineering |
| **6** | Robustesse, Calibration & Tests Statistiques | 2 min | Démontrer la rigueur scientifique (Wilcoxon, Brier) |
| **7** | Apprentissage Non Supervisé : Découverte des Archétypes | 2 min 30 | Présenter K-Means, CAH, DBSCAN et les 4 tempéraments |
| **8** | Matrice d'Affinité Croisée & Longévité | 1 min 30 | Lier clustering non supervisé et prédiction de couple |
| **9** | Explicabilité en Direct (XAI via SHAP) | 2 min | Transparence algorithmique (Beeswarm & Waterfall) |
| **10** | Démonstration de l'Application Web Flask | 2 min | Montrer le produit fini en live (`/`, `/dashboard`, `/archetypes`) |
| **11** | Architecture Logicielle, CI/CD & Tests | 1 min | 76/76 tests unitaires et intégration |
| **12** | Conclusion & Perspectives d'Avenir | 1 min | Bilan académique et extensions futures |

---

## 📑 Détail Slide par Slide (Script Oral & Supports)

### Slide 1 : Introduction & Problématique
- **Titre :** *Peut-on prédire la compatibilité amoureuse et segmenter les tempéraments humains grâce au Machine Learning ?*
- **Message clé :** Les algorithmes traditionnels se limitent souvent à des filtres par critères superficiels (âge, ville). Notre projet propose une double approche : **modélisation prédictive supervisée** + **profilage psychologique non supervisé (Big 5)**, avec une explicabilité totale (XAI).
- **Script oral :**
  > *"Monsieur le Président, mesdames et messieurs les membres du jury, nous vous présentons aujourd'hui le projet Cupid Matchmaking. Notre objectif a été de concevoir un système de bout en bout capable non seulement de prédire la compatibilité durable entre deux individus sur 100 000 paires réelles, mais surtout de comprendre et d'expliquer les dynamiques sous-jacentes du couple grâce à l'IA explicable."*

---

### Slide 2 : Compréhension des Données & Profiling
- **Titre :** *Le Jeu de Données Cupid : 100 000 Paires de Profils*
- **Chiffres clés :** 100 000 observations, 30 attributs bruts, équilibre parfait des classes (50.0% compatibles / 50.0% incompatibles).
- **Dimensions mesurées :**
  - Démographiques : âge (18–55 ans), niveau d'éducation (1–5), localisation géographique.
  - Psychologiques : Modèle Big 5 (Ouverture, Extraversion, Agréabilité, Conscienciosité), Ambition de carrière.
  - Rythme de vie & Affectif : Chronotype (matinal vs couche-tard), Spontanéité, Expressivité émotionnelle, Langage de l'amour (*Quality Time, Words of Affirmation, Acts of Service...*).
- **Script oral :**
  > *"Le jeu de données comprend 100 000 paires anonymisées. L'analyse exploratoire dans notre Notebook 01 montre qu'aucun trait pris isolément ne suffit à prédire l'issue d'une relation. C'est l'interaction et la distance relative entre les deux partenaires qui crée la compatibilité."*

---

### Slide 3 : Feature Engineering Avancé & Rigueur Méthodologique
- **Titre :** *De 30 Variables Brutes à 73 Caractéristiques Prédictives*
- **Innovations de Feature Engineering :**
  1. *Écarts différentiels absolus* : $\Delta \text{Trait} = |A_{\text{trait}} - B_{\text{trait}}|$.
  2. *Distance Euclidienne Multidimensionnelle* : Proximité spatiale dans l'espace des 8 traits psychologiques.
  3. *Score d'Harmonie Big 5* : $\frac{1}{1 + \text{mean}(\Delta \text{Big5})}$.
  4. *Alignement des Valeurs Partagées* : Localisation commune, même langage d'amour, même domaine de carrière.
- **Garantie contre le Data Leakage :**
  - Utilisation d'un `ColumnTransformer` scikit-learn (`OneHotEncoder` sans inconnues, `StandardScaler`).
  - La normalisation et l'encodage sont appris **exclusivement sur les 80 000 lignes du train set**, puis appliqués sans ré-apprentissage sur les 20 000 lignes du test set.
- **Script oral :**
  > *"Pour donner du pouvoir prédictif à nos modèles, nous avons créé 73 caractéristiques spécialisées. Afin de garantir une intégrité scientifique absolue, nous avons encapsulé ce préprocesseur dans une classe custom empêchant toute fuite d'information entre l'apprentissage et le test."*

---

### Slide 4 : Benchmark des Modèles Supervisés & Optimisation
- **Titre :** *Comparaison des Algorithmes & Modèle Champion*
- **Tableau de benchmark officiel :**

| Modèle | Accuracy | Précision | Rappel | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Régression Logistique** | 71.85% | 72.82% | 69.67% | 0.7121 | 0.7934 |
| **Arbre de Décision (Pruned)** | 68.69% | 68.04% | 70.52% | 0.6926 | 0.7255 |
| **Random Forest (100 arbres)** | 71.66% | 72.33% | 70.16% | 0.7123 | 0.7882 |
| **XGBoost (Baseline)** | 72.41% | 72.76% | 71.61% | 0.7218 | 0.7960 |
| **Tuned XGBoost (Champion)** | **72.65%** | **73.12%** | **71.74%** | **0.7242** | **0.7969** |

- **Hyperparamètres optimaux retenus :** `n_estimators=150`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Script oral :**
  > *"Après optimisation par validation croisée stratifiée sur 3 plis avec RandomizedSearchCV, le modèle XGBoost optimisé domine l'ensemble des métriques avec un ROC-AUC de 0.7969 et une précision de 73.12%, surpassant significativement les modèles linéaires et les arbres isolés."*

---

### Slide 5 : Étude d'Ablation & Courbes d'Apprentissage
- **Titre :** *Validation Empirique de la Valeur Ajoutée*
- **Graphiques projetés :** `results/figures/ablation_study.png` et `learning_curve_xgboost.png`.
- **Résultats clés de l'ablation :**
  - Modèle avec attributs bruts seuls : ROC-AUC = **0.7059**
  - Ajout des différences de traits : ROC-AUC = **0.7938** (+8.8 points !)
  - Retrait de l'alignement du langage de l'amour : Chute du ROC-AUC à **0.7275** (-6.8 points !)
- **Courbes d'apprentissage :**
  - Le modèle converge dès 30 000 paires et se stabilise à 80 000 paires avec un faible écart train/validation, confirmant l'absence de surapprentissage critique.

---

### Slide 6 : Robustesse, Calibration & Tests Statistiques
- **Titre :** *Stress-Tests & Rigueur Statistique*
- **Éléments scientifiques :**
  - **Test de Wilcoxon Signed-Rank** (10-fold CV) : Prouve que la supériorité de XGBoost face au Random Forest et à l'Arbre de Décision est statistiquement significative ($p < 0.001$).
  - **Calibration des probabilités** : Brier Score Loss de **0.1808** (courbe de calibration très proche de la diagonale idéale).
  - **Test de robustesse au bruit gaussien** : Même avec un bruit $\sigma=0.35$ sur les entrées, le modèle conserve un ROC-AUC supérieur à 0.746, démontrant sa résilience en conditions réelles.

---

### Slide 7 : Clustering Non Supervisé : Les 4 Archétypes
- **Titre :** *Découverte Non Supervisée des Tempéraments Psychologiques*
- **Méthodes comparées :** K-Means ($k \in [2, 8]$), CAH (Ward dendrogramme), DBSCAN ($\varepsilon=1.85, \text{min\_samples}=25$).
- **Graphiques projetés :** `results/figures/archetype_radar_chart.png` et `clustering_pca_comparison.png`.
- **Les 4 Archétypes identifiés :**
  1. **The Grounded Stabilizers (24.8%)** : Agréabilité et Conscienciosité très élevées, profil empathique et sécurisant.
  2. **The Dynamic Visionaries (25.1%)** : Forte ouverture d'esprit et ambition professionnelle élevée.
  3. **The Spontaneous Socializers (25.3%)** : Extraversion et spontanéité maximales, expressivité affective vive.
  4. **The Independent Thinkers (24.8%)** : Calme, introspection, grande autonomie de décision.

---

### Slide 8 : Matrice d'Affinité Croisée & Longévité Estimée
- **Titre :** *Passerelle Non Supervisé $\rightarrow$ Supervisé*
- **Graphique projeté :** `results/figures/clustering_archetype_affinity_heatmap.png`.
- **Enseignements majeurs :**
  - **Synergies les plus fortes :** Les couples *Dynamic Visionaries $\times$ Dynamic Visionaries* (49.7% de réussite) et *Independent Thinkers $\times$ Independent Thinkers* (51.3%).
  - **Frictions identifiées :** Les profils *Independent Thinkers $\times$ Grounded Stabilizers* présentent le taux de compatibilité le plus faible (37.4%), nécessitant un compromis accru sur les besoins sociaux.

---

### Slide 9 : Explicabilité en Direct (XAI via SHAP)
- **Titre :** *Ouvrir la Boîte Noire : Intelligence Artificielle Explicable*
- **Graphiques projetés :** `results/figures/shap_summary.png` (Global) et `shap_waterfall.png` (Local).
- **Les 3 facteurs les plus déterminants à l'échelle globale :**
  1. `same_love_language` : Le facteur n°1 influençant la longévité affective.
  2. `personality_euclidean_dist` : Plus la distance psychologique globale est faible, plus la compatibilité est élevée.
  3. `diff_career_ambition` : Un fort décalage d'ambition de carrière constitue le principal facteur de rupture.
- **Explication locale en temps réel :** Le diagramme en cascade (*Waterfall*) décompose pour un couple précis exactement quel trait apporte $+18\%$ ou $-15\%$ à la décision.

---

### Slide 10 : Démonstration en Direct de l'Application Web
- **Titre :** *L'Écosystème Web Cupid AI (Flask & Glassmorphism UI)*
- **Points de démonstration :**
  - **Page d'accueil (`/`)** : Formulaire interactif avec presets en 1 clic (*Perfect Match, Balanced, Opposites Attract, Clash*).
  - **Jauge animée & Explication XAI** : Affichage en direct du score (%), des forces motrices et des frictions détectées.
  - **Dashboard de performance (`/dashboard`)** : Métriques comparatives, courbes ROC et matrice de confusion interactives.
  - **Explorateur d'archétypes (`/archetypes`)** : Radar chart interactif en Chart.js et quiz d'auto-positionnement.

---

### Slide 11 : Architecture Logicielle & Rigueur Industrielle
- **Titre :** *Normes de Développement & Intégrité CI/CD*
- **Points techniques forts :**
  - Architecture modulaire claire : `src/data`, `src/models`, `src/evaluation`, `src/api`, `experiments/`.
  - **76 tests unitaires et d'intégration automatisés (100% passing)** sous `pytest`.
  - Dépôt Git versionné proprement, aucun chemin codé en dur, documentation complète de reproductibilité (`REPRODUCIBILITY_GUIDE.md`).

---

### Slide 12 : Conclusion & Perspectives
- **Titre :** *Bilan du Projet & Évolutions Futures*
- **Bilan :** Objectif atteint à 100% : un pipeline ML complet, benchmarké, interprétable et déployé sous forme de produit applicatif.
- **Perspectives :**
  - Intégration de modèles NLP sur des descriptions textuelles libres (biographies de profils).
  - Algorithmes de matching bilatéral de Gale-Shapley pour optimiser l'ensemble d'une communauté simultanément.
  - Déploiement cloud containerisé sous Docker.

---

## 🎯 Anticipation des Questions Pièges du Jury (Q&A)

### Q1 : *"Pourquoi avoir choisi XGBoost plutôt qu'un réseau de neurones profond (Deep Learning) ?"*
> **Réponse modèle :** *"Sur des données tabulaires hétérogènes de 100 000 lignes combinant variables continues et catégorielles, les arbres de décision boostés par gradient (GBDT / XGBoost) surpassent systématiquement les réseaux de neurones denses, comme le démontrent les benchmarks académiques (ex. Grinsztajn et al., 2022). De plus, XGBoost offre une convergence plus rapide, une insensibilité naturelle aux distributions non gaussiennes et permet un calcul direct des valeurs de SHAP exactes via TreeSHAP en temps réel, ce qui est crucial pour notre exigence d'explicabilité."*

### Q2 : *"Comment vous êtes-vous assurés qu'il n'y a pas eu de data leakage lors de la normalisation ?"*
> **Réponse modèle :** *"Nous avons conçu la classe `CupidDataPreprocessor` de manière strictement découplée : la division en Train (80%) et Test (20%) est opérée avant tout calcul de moyenne ou d'écart-type. Le `StandardScaler` et le `OneHotEncoder` appellent `fit_transform()` uniquement sur le Train Set. Le Test Set n'est transformé qu'à l'aide de la méthode `transform()`. Nous avons même écrit un test unitaire dédié (`test_prepare_train_test_fits_scaler_only_on_training_rows`) qui vérifie mathématiquement que la moyenne apprise est exactement celle du fold d'entraînement."*

### Q3 : *"Dans le clustering, pourquoi avoir choisi k=4 plutôt que k=3 ou k=5 ?"*
> **Réponse modèle :** *"Nous avons combiné la méthode du coude sur la somme des carrés intra-cluster (Inertie) et l'analyse du coefficient de Silhouette. Même si mathématiquement k=2 a un score légèrement supérieur (ce qui est classique en silhouette), k=4 représente le point d'inflexion optimal où la variance intra-cluster décroît significativement tout en conservant une interprétabilité psychologique forte, alignée avec les grands contrastes de notre analyse factorielle en 2D (PCA)."*

### Q4 : *"Que se passe-t-il si un utilisateur entre des traits sur une échelle de 1 à 10 alors que votre modèle a été entraîné sur 0 à 1 ?"*
> **Réponse modèle :** *"Nous avons implémenté une normalisation dynamique dans l'API Flask (`app.py`). Dès qu'un trait dépasse la valeur 1.0, l'API le ramène automatiquement sur l'intervalle [0, 1] en le divisant par 10. Ainsi, que l'utilisateur manipule les sliders 1–10 du formulaire web ou qu'un système externe interroge l'API avec des décimales 0–1, le pipeline fonctionne de manière transparente et sans distorsion."*
