# 💘 Cupid Matchmaking — Machine Learning Compatibility & Profiling

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-green)
![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)
![Flask](https://img.shields.io/badge/Flask-web--app-lightgrey?logo=flask)
![Tests](https://img.shields.io/badge/tests-36%2F36%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

> **Can machine learning predict romantic compatibility and segment human temperaments?**  
> Cupid Matchmaking is an end-to-end data science and machine learning system that predicts compatibility between partner profile pairs using demographic data, Big 5 personality traits, and lifestyle metrics. It combines **supervised predictive modeling**, **unsupervised personality clustering**, **SHAP explainability**, and a **real-time Flask web application**.

---

## 📋 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Project Structure](#-project-structure)
- [Data Description](#-data-description)
- [Feature Engineering Pipeline](#-feature-engineering-pipeline)
- [Supervised Learning Models](#-supervised-learning-models)
- [Results & Benchmark](#-results--benchmark)
- [Model Explainability (SHAP XAI)](#-model-explainability-shap-xai)
- [Unsupervised Learning & Clustering](#-unsupervised-learning--clustering)
- [Cross-Archetype Compatibility Affinity](#-cross-archetype-compatibility-affinity)
- [Web Application (Flask API)](#-web-application-flask-api)
- [Installation & Quickstart](#-installation--quickstart)
- [Running Tests](#-running-tests)
- [Guided Notebooks (01 to 08)](#-guided-notebooks-01-to-08)
- [Soutenance & Presentation](#-soutenance--presentation)

---

## 🌟 Overview & Architecture

Cupid AI provides a dual-engine architecture:
1. **Supervised Prediction Engine**: A tuned gradient-boosted decision tree pipeline (XGBoost) achieving an **ROC-AUC of 0.7969** and **72.65% Accuracy** on 100,000 profile pairs.
2. **Unsupervised Profiling Engine**: Multi-algorithmic segmentation (K-Means, Ward Hierarchical, DBSCAN) discovering **4 distinct personality archetypes** across multidimensional psychological traits.
3. **Explainable AI Layer (SHAP)**: Global feature attribution and local waterfall explanations explaining *why* a couple is compatible or incompatible.
4. **Production Web Application**: Interactive Glassmorphism UI built on Flask with real-time prediction, health checks, and metadata endpoints.

---

## 🗂️ Project Structure

```
DM_PROJECT/
├── data/
│   ├── raw/
│   │   └── cupid_algorithm_dataset.csv      # 100,000 profile pairs (raw)
│   └── processed/
│       ├── cupid_cleaned.csv                # Cleaned dataset
│       └── cupid_cleaned_eda.csv            # EDA version
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb               # Cleaning, encoding & scaling
│   ├── 03_logistic_regression.ipynb         # Logistic Regression baseline
│   ├── 04_decision_tree.ipynb               # Decision Tree classifier
│   ├── 05_random_forest.ipynb               # Random Forest ensemble
│   ├── 06_xgboost.ipynb                     # XGBoost gradient boosting
│   ├── 07_evaluation_tuning.ipynb           # RandomizedSearchCV & CV
│   └── 08_clustering.ipynb                  # Unsupervised learning & K-Means/DBSCAN
├── src/
│   ├── data/
│   │   └── preprocessing.py                 # CupidDataPreprocessor pipeline
│   ├── models/
│   │   ├── base_model.py                    # OOP Abstract Base Classifier
│   │   ├── logistic_regression.py          # CupidLogisticRegression
│   │   ├── decision_tree.py                # CupidDecisionTree
│   │   ├── random_forest.py                # CupidRandomForest
│   │   ├── xgboost_model.py                # CupidXGBoost
│   │   └── clustering.py                   # KMeans, Hierarchical & DBSCAN
│   ├── evluation/                           # Metrics, Cross-Validation & Tuning
│   │   ├── metrics.py
│   │   ├── cross_validation.py
│   │   └── tuning.py
│   ├── evaluation/
│   │   └── shap_explainer.py                # Global & Local SHAP Explainability
│   ├── train_and_evaluate.py                # Full automated training pipeline
│   ├── run_clustering.py                    # Clustering pipeline & benchmarking
│   ├── run_archetype_affinity.py            # Cross-archetype affinity matrix
│   └── run_shap_analysis.py                 # SHAP figures generation script
├── results/
│   ├── best_model.pkl                       # Serialized Tuned XGBoost
│   ├── preprocessor.pkl                     # Serialized preprocessor
│   ├── summary.json                         # Best params and evaluation metrics
│   ├── figures/                             # Visualizations (ROC, SHAP, Heatmaps, PCA...)
│   └── tables/                              # Evaluation CSVs (Comparison, Affinity...)
├── static/ & templates/                     # Flask Glassmorphism Frontend (HTML, CSS, JS)
├── tests/                                   # Pytest Unit & Integration Suite (36 tests)
├── slides_soutenance.md                     # Oral presentation slides & defense guide
├── rapport_final.md                         # Full academic report (12 chapters)
└── app.py                                   # Flask web application server
```

---

## 📊 Data Description

The dataset comprises **100,000 synthetic dating profile pairs** modeled after OkCupid-style questionnaires. Each record contains pairwise attributes for **Partner A** and **Partner B**:

| Feature Category | Variable Names | Description |
|---|---|---|
| **Demographics** | `a_age`, `b_age`, `a_education`, `b_education`, `a_location`, `b_location` | Age, education degree level (1-4), city |
| **Big 5 Traits** | `openness`, `extraversion`, `agreeableness`, `conscientiousness` | Core psychological dimensions on standardized scales |
| **Lifestyle & Habits** | `chronotype`, `spontaneity`, `emotional_expressiveness`, `career_ambition`, `career_field`, `love_language` | Circadian rhythm, planning style, communication style, industry |
| **Targets** | `compatible` (binary 0/1), `compatibility_score` (0-100), `relationship_longevity_months` | Primary classification target and relational longevity |

---

## ⚙️ Feature Engineering Pipeline

To capture inter-partner chemistry, `CupidDataPreprocessor` engineers **16 interaction and distance features** prior to model fitting:

| Feature Name | Calculation / Concept | Meaning |
|---|---|---|
| `age_diff` | $\|a\_age - b\_age\|$ | Age gap |
| `education_diff` | $\|a\_education - b\_education\|$ | Educational gap |
| `same_location` | $\mathbb{I}(a\_location == b\_location)$ | Geographical co-location |
| `same_career_field` | $\mathbb{I}(a\_field == b\_field)$ | Shared professional background |
| `same_love_language` | $\mathbb{I}(a\_love == b\_love)$ | Shared love language |
| `diff_*` (8 traits) | $\|a\_trait - b\_trait\|$ | Absolute divergence per psychological trait |
| `personality_euclidean_dist` | $\sqrt{\sum (a\_trait - b\_trait)^2}$ | Multidimensional Euclidean psychological distance |
| `personality_similarity_index` | $1 / (1 + \text{euclidean\_dist})$ | Inverse distance normalized similarity index |
| `avg_trait_diff` | $\frac{1}{8}\sum \|a\_trait - b\_trait\|$ | Average discrepancy across all personality traits |
| `big5_harmony_score` | Composite Big 5 harmony score | Alignment on fundamental stability traits |

---

## 🤖 Supervised Learning Models

Five configurations were benchmarked on a stratified 80% train / 20% test split:

1. **Logistic Regression**: Linear baseline with $L_2$ regularization.
2. **Decision Tree**: Non-linear tree classifier (`max_depth=12`, `min_samples_split=10`).
3. **Random Forest**: Ensemble bagging classifier with 100 trees (`max_depth=14`).
4. **XGBoost (Default)**: Gradient boosted trees with default hyperparameters.
5. **Tuned XGBoost**: Optimized via `RandomizedSearchCV` across a 5D parameter grid with 3-fold Stratified CV.

### Best Tuned Hyperparameters:
```json
{
  "n_estimators": 200,
  "max_depth": 6,
  "learning_rate": 0.12,
  "subsample": 0.9,
  "colsample_bytree": 0.8
}
```

---

## 📈 Results & Benchmark

Evaluation on the hold-out test set ($N=20,000$):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.6821 | 0.6453 | 0.5912 | 0.6170 | 0.7312 |
| Decision Tree | 0.5866 | 0.5172 | 0.5391 | 0.5279 | 0.6032 |
| Random Forest | 0.7045 | 0.6821 | 0.6201 | 0.6495 | 0.7721 |
| XGBoost | 0.7138 | 0.6934 | 0.6388 | 0.6650 | 0.7891 |
| **Tuned XGBoost** 🏆 | **0.7265** | **0.7042** | **0.6242** | **0.6618** | **0.7969** |

---

## 🔍 Model Explainability (SHAP XAI)

To satisfy modern interpretability standards, Cupid integrates **SHAP (SHapley Additive exPlanations)**:

- **Global Attribution** (`results/figures/shap_summary.png`): Identifies that `diff_career_ambition`, `personality_euclidean_dist`, `same_love_language`, and `same_location` are the dominant global drivers of compatibility.
- **Local Waterfall Attribution** (`results/figures/shap_waterfall.png`): Computes individual Shapley values for any specific profile pair, pinpointing the exact factors contributing to compatibility or creating friction.

Run SHAP generation independently:
```bash
python src/run_shap_analysis.py
```

---

## 🔮 Unsupervised Learning & Clustering

User profile segmentation across 8 personality and lifestyle dimensions:

1. **K-Means**: Selection of optimal $k=4$ via the **Elbow Method (WCSS)** and **Silhouette Analysis** across $k \in [2, 8]$.
2. **Hierarchical Agglomerative Clustering**: Bottom-up merging with **Ward linkage** and dendrogram cut-off visualization.
3. **DBSCAN**: Density-based spatial clustering with $k$-distance knee analysis ($\epsilon=1.85, \text{min\_samples}=25$), identifying dense clusters while isolating genuine outlier/noise profiles ($0.15\%$).

### 👥 The 4 Personality Archetypes:
- **Arch. 0 — The Grounded Stabilizers (23.3%)**: High Agreeableness & Conscientiousness. Reliable, structured.
- **Arch. 1 — The Dynamic Visionaries (25.7%)**: High Openness, Extraversion & Career Ambition.
- **Arch. 2 — The Spontaneous Socializers (25.9%)**: High Spontaneity & Extraversion. Fun-loving and expressive.
- **Arch. 3 — The Independent Thinkers (25.1%)**: Analytical, quiet, high focus and introverted depth.

Run clustering:
```bash
python src/run_clustering.py
```

---

## 🧩 Cross-Archetype Compatibility Affinity

Cupid bridges unsupervised clustering and supervised matchmaking by evaluating the **cross-cluster compatibility rate**:

| Partner A \ Partner B | 0. Grounded Stabilizers | 1. Dynamic Visionaries | 2. Spontaneous Socializers | 3. Independent Thinkers |
|---|:---:|:---:|:---:|:---:|
| **0. Grounded Stabilizers** | **49.0%** | 40.5% | 42.7% | 37.4% |
| **1. Dynamic Visionaries** | 40.5% | 41.2% | 43.1% | 40.1% |
| **2. Spontaneous Socializers** | 42.7% | 43.1% | **46.8%** | 39.4% |
| **3. Independent Thinkers** | 37.4% | 40.1% | 39.4% | **51.3%** 🏆 |

Heatmaps are available in `results/figures/clustering_archetype_affinity_heatmap.png`.

---

## 🌐 Web Application (Flask API)

The Flask web app exposes an interactive Glassmorphism UI and a REST API:

- `GET /` : Interactive prediction interface with presets (Perfect Harmony, Balanced Duo, Opposites Attract, High Friction).
- `POST /api/predict` : Predicts compatibility score (0-100), predicted class (0/1), estimated relationship longevity, Big 5 harmony score, and trait distances.
- `GET /api/health` : Health check confirming model and preprocessor status.
- `GET /api/model-info` : Returns model parameters, test size, and benchmark metrics from `summary.json`.

---

## 🛠️ Installation & Quickstart

```bash
# 1. Clone repository
git clone https://github.com/khaldamin/DM_PROJECT.git
cd DM_PROJECT

# 2. Install requirements
pip install -r requirement.txt

# 3. Launch Flask Web Application
python app.py
# Open http://localhost:5000 in your browser!
```

To retrain models from scratch:
```bash
python src/train_and_evaluate.py
```

---

## 🧪 Running Tests

The test suite contains **36 unit and integration tests** verifying:
- Prevention of data leakage during scaling and encoding
- Stratified cross-validation fold preprocessor isolation
- Flask API contract, error handling, and health endpoints
- K-Means, Hierarchical, and DBSCAN clustering logic
- SHAP global and local attribution mechanics

```bash
python -m pytest tests/ -v
```

Output:
```text
======================= 36 passed, 13 warnings in 7.94s =======================
```

---

## 📓 Guided Notebooks (01 to 08)

| Notebook | Topic |
|---|---|
| `01_data_understanding_eda.ipynb` | Dataset profiling, distributions, correlation analysis |
| `02_preprocessing.ipynb` | Handling missing values, OHE, StandardScaler, leakage prevention |
| `03_logistic_regression.ipynb` | Linear baseline, odds ratios, decision boundaries |
| `04_decision_tree.ipynb` | Decision tree depth tuning, non-linear classification |
| `05_random_forest.ipynb` | Bagging ensemble, out-of-bag validation, feature importance |
| `06_xgboost.ipynb` | Gradient boosted decision trees |
| `07_evaluation_tuning.ipynb` | RandomizedSearchCV, Stratified K-Fold, ROC/PR curves |
| `08_clustering.ipynb` | Unsupervised profiling: K-Means (Elbow/Silhouette), Dendrogram, DBSCAN |

---

## 🎤 Soutenance & Presentation

A complete defense slide deck is provided in [`slides_soutenance.md`](file:///c:/Users/ENPEI/Desktop/Nouveau%20dossier/DM_PROJECT/slides_soutenance.md), covering:
1. Problem statement & objectives
2. Data exploration & preprocessing
3. Model comparison & tuning results
4. Unsupervised clustering & archetypes
5. SHAP explainability demo
6. Web app architecture & live demonstration
7. Limitations, ethical considerations & conclusion
