# 💘 Cupid Matchmaking — ML Compatibility Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-green)
![Flask](https://img.shields.io/badge/Flask-web--app-lightgrey?logo=flask)
![Tests](https://img.shields.io/badge/tests-34%2F34%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

> **Can a machine predict love?**  
> Cupid Matchmaking is a machine-learning project that predicts romantic compatibility between two OkCupid-style profiles using personality traits, demographic features, and engineered interaction variables.

---

## 📋 Table of Contents

- [Description](#description)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Tests](#tests)
- [Notebooks](#notebooks)

---

## 📖 Description

This project builds a binary classification pipeline to determine whether two people are **compatible** (`1`) or not (`0`). It compares four machine learning algorithms — Logistic Regression, Decision Tree, Random Forest, and XGBoost — with the best model (Tuned XGBoost) achieving a **ROC-AUC of 0.7969** and **Accuracy of 72.65%**.

A **Flask web application** wraps the best model, allowing users to input two profiles and receive a real-time compatibility prediction.

---

## 🗂️ Project Structure

```
DM_PROJECT/
├── data/
│   ├── raw/
│   │   └── cupid_algorithm_dataset.csv      # Original dataset
│   └── processed/
│       ├── cupid_cleaned.csv                # Cleaned dataset
│       ├── cupid_cleaned_eda.csv            # EDA-ready version
│       └── cupid_algorithm_dataset.csv      # Feature-engineered dataset
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb               # Cleaning & preprocessing
│   ├── 03_logistic_regression.ipynb         # Logistic Regression model
│   ├── 04_decision_tree.ipynb               # Decision Tree model
│   ├── 05_random_forest.ipynb               # Random Forest model
│   ├── 06_xgboost.ipynb                     # XGBoost model
│   └── 07_evaluation_tuning.ipynb           # Tuning & final evaluation
├── src/
│   └── train_and_evaluate.py                # Full training pipeline script
├── results/
│   ├── best_model.pkl                       # Serialized best model
│   ├── preprocessor.pkl                     # Fitted preprocessor
│   ├── model_comparison.csv                 # Metrics for all models
│   ├── summary.json                         # Best model summary
│   ├── figures/                             # Plots (ROC, confusion matrix…)
│   └── tables/                              # CSV exports of metrics
├── templates/
│   └── index.html                           # Flask HTML template
├── static/                                  # CSS / JS assets
├── tests/
│   └── test_pipeline.py                     # Unit tests (pytest)
├── app.py                                   # Flask web application
├── requirement.txt                          # Python dependencies
├── pytest.ini                               # Pytest configuration
└── README.md
```

---

## 📊 Data Description

| Property | Value |
|---|---|
| **Source** | OkCupid-style synthetic profile pairs |
| **Rows** | 100 000 pairs |
| **Target** | `compatible` — binary (0 = not compatible, 1 = compatible) |
| **Train / Test split** | 80 % / 20 % (stratified) |
| **Features after preprocessing** | 51 (OHE + StandardScaler) |

### Raw Variables

Each row contains two profiles, prefixed `a_` and `b_`:

| Variable | Type | Description |
|---|---|---|
| `a_age` / `b_age` | Numeric | Age in years |
| `a_education` / `b_education` | Numeric | Education level (encoded) |
| `a_career_ambition` / `b_career_ambition` | Numeric | Career ambition score |
| `a_openness` / `b_openness` | Numeric | Big 5 — Openness |
| `a_extraversion` / `b_extraversion` | Numeric | Big 5 — Extraversion |
| `a_agreeableness` / `b_agreeableness` | Numeric | Big 5 — Agreeableness |
| `a_conscientiousness` / `b_conscientiousness` | Numeric | Big 5 — Conscientiousness |
| `a_chronotype` / `b_chronotype` | Numeric | Morning / evening preference |
| `a_spontaneity` / `b_spontaneity` | Numeric | Spontaneity score |
| `a_emotional_expressiveness` / `b_emotional_expressiveness` | Numeric | Emotional expressiveness |
| `a_location` / `b_location` | Categorical | City / region |
| `a_career_field` / `b_career_field` | Categorical | Occupation domain |
| `a_love_language` / `b_love_language` | Categorical | Primary love language |
| `compatible` | Binary | **Target** (0 / 1) |

---

## ⚙️ Feature Engineering

Beyond the raw variables, the following interaction and similarity features are computed:

| Feature | Description |
|---|---|
| `age_diff` | Absolute age difference |
| `education_diff` | Absolute education level difference |
| `same_location` | 1 if both profiles share the same location |
| `same_career_field` | 1 if both profiles share the same career field |
| `same_love_language` | 1 if both profiles share the same love language |
| `diff_openness` | \|a_openness − b_openness\| |
| `diff_extraversion` | \|a_extraversion − b_extraversion\| |
| `diff_agreeableness` | \|a_agreeableness − b_agreeableness\| |
| `diff_conscientiousness` | \|a_conscientiousness − b_conscientiousness\| |
| `diff_chronotype` | \|a_chronotype − b_chronotype\| |
| `diff_spontaneity` | \|a_spontaneity − b_spontaneity\| |
| `diff_emotional_expressiveness` | \|a_emotional_expressiveness − b_emotional_expressiveness\| |
| `personality_euclidean_dist` | Euclidean distance across all Big 5 traits |
| `personality_similarity_index` | 1 / (1 + euclidean_dist) |
| `avg_trait_diff` | Mean of all pairwise trait differences |
| `big5_harmony_score` | Composite harmony score across Big 5 dimensions |

---

## 🤖 Models

Four classifiers were trained and evaluated:

1. **Logistic Regression** — baseline linear model with regularization
2. **Decision Tree** — interpretable tree-based model
3. **Random Forest** — ensemble of decision trees (bagging)
4. **XGBoost** — gradient boosting (default parameters)
5. **XGBoost Tuned** — XGBoost optimized with `RandomizedSearchCV` (6 configurations, 3-fold Stratified CV)

Best hyperparameters found:

```python
{
    "n_estimators":    200,
    "max_depth":       6,
    "learning_rate":   0.12,
    "subsample":       0.9,
    "colsample_bytree": 0.8
}
```

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.6821 | 0.6453 | 0.5912 | 0.6170 | 0.7312 |
| Decision Tree | 0.5866 | 0.5172 | 0.5391 | 0.5279 | 0.6032 |
| Random Forest | 0.7045 | 0.6821 | 0.6201 | 0.6495 | 0.7721 |
| XGBoost | 0.7138 | 0.6934 | 0.6388 | 0.6650 | 0.7891 |
| **XGBoost Tuned** ✅ | **0.7265** | **0.7042** | **0.6242** | **0.6618** | **0.7969** |

> The **Tuned XGBoost** is the best model across all metrics and is saved to `results/best_model.pkl`.

---

## 🔮 Unsupervised Learning & Clustering (Profile Segmentation)

In addition to supervised classification, the project implements unsupervised personality profiling:

1. **K-Means Clustering**:
   - **Elbow Method (WCSS / Inertia)** & **Silhouette Analysis** across $k \\in [2, 8]$.
   - Optimal $k=4$ segments users into 4 balanced personality archetypes:
     - *The Grounded Stabilizers* (High Agreeableness & Conscientiousness)
     - *The Dynamic Visionaries* (High Openness & Ambition)
     - *The Spontaneous Socializers* (High Extraversion & Spontaneity)
     - *The Independent Thinkers* (Low Extraversion, High Conscientiousness)
2. **Hierarchical Agglomerative Clustering**:
   - Bottom-up clustering with **Ward linkage**.
   - Dendrogram visualization showing natural separation at distance threshold cut-off.
3. **DBSCAN (Density-Based Clustering)**:
   - $k$-distance elbow curve to select optimal $\\epsilon = 1.85$.
   - Successfully detects dense clusters while isolating genuine outlier/noise profiles ($<0.2\\%$).
4. **Intrinsic Evaluation**:
   - Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score benchmarking across algorithms.
   - PCA 2D projections saved in `results/figures/clustering_pca_comparison.png`.

To run clustering independently:
```bash
python src/run_clustering.py
```

---

## 🛠️ Installation

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cupid-matchmaking.git
cd cupid-matchmaking

# 2. (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS

# 3. Install dependencies
pip install -r requirement.txt
```

---

## 🚀 Usage

### Train & Evaluate All Models

```bash
python src/train_and_evaluate.py
```

This script will:
- Load and preprocess the data
- Engineer all features
- Train all five models (including tuned XGBoost)
- Save metrics to `results/model_comparison.csv`
- Save the best model to `results/best_model.pkl`

### Launch the Flask Web Application

```bash
python app.py
```

Then open your browser at **http://localhost:5000**  
Enter two profiles and get an instant compatibility prediction.

---

## 🧪 Tests

```bash
pytest
```

The test suite covers 10 unit tests including:
- Data loading and schema validation
- Feature engineering correctness
- Preprocessing pipeline (OHE + StandardScaler)
- Model training and prediction shapes
- Flask routes and response codes

All **10/10 tests pass** ✅

---

## 📓 Notebooks

Work through the notebooks in order for a full guided walkthrough:

| Notebook | Description |
|---|---|
| `01_data_understanding_eda.ipynb` | Dataset overview, distributions, missing values, correlation analysis |
| `02_preprocessing.ipynb` | Cleaning, encoding, feature engineering, train/test split |
| `03_logistic_regression.ipynb` | Logistic Regression training, coefficients, ROC curve |
| `04_decision_tree.ipynb` | Decision Tree training, depth tuning, visualization |
| `05_random_forest.ipynb` | Random Forest training, feature importance |
| `06_xgboost.ipynb` | XGBoost training, default parameters |
| `07_evaluation_tuning.ipynb` | RandomizedSearchCV tuning, cross-validation, final comparison |
| `08_clustering.ipynb` | Unsupervised profile segmentation: K-Means (Elbow/Silhouette), Hierarchical (Dendrogram), DBSCAN (Noise detection) |

```bash
# Launch Jupyter
jupyter notebook notebooks/
```

---

## 📄 License

This project is released under the **MIT License**.

---

*Made with ❤️ and data science — because love deserves an algorithm.*
