# 🔄 Reproducibility Guide — Cupid AI

> **Standard:** Fully Deterministic, End-to-End Replicable Research Pipeline  
> **Environment:** Python 3.10+ on Linux / macOS / Windows WSL  
> **Seed Configuration:** Global Pseudo-Random Seed: `42`  
> **Status:** Enhancement Addition (Strictly Non-Destructive)

---

## 1. Quickstart & Environment Setup

### 1.1 Python Virtual Environment
To guarantee absolute dependency parity and prevent package collisions:

```bash
# 1. Clone repository (if external)
git clone https://github.com/khaldamin/DM_PROJECT.git
cd DM_PROJECT

# 2. Initialize Python 3.10+ virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirement.txt
```

### 1.2 Core Dependencies Verification
Verify installed packages against the project requirements:

```bash
python3 -c "
import numpy, pandas, sklearn, xgboost, shap, flask, scipy
print('✓ NumPy:', numpy.__version__)
print('✓ Pandas:', pandas.__version__)
print('✓ Scikit-Learn:', sklearn.__version__)
print('✓ XGBoost:', xgboost.__version__)
print('✓ SHAP:', shap.__version__)
print('✓ Flask:', flask.__version__)
print('✓ SciPy:', scipy.__version__)
"
```

---

## 2. Directory Structure Verification

Before execution, verify the presence of all source files and data partitions:

```bash
# Confirm raw data integrity (100,001 rows including header)
wc -l data/raw/cupid_algorithm_dataset.csv
# Expected: 100001 data/raw/cupid_algorithm_dataset.csv

# Verify core source modules
ls -la src/data/preprocessing.py
ls -la src/models/base_model.py
ls -la src/train_and_evaluate.py
ls -la src/run_clustering.py
```

---

## 3. Step-by-Step Pipeline Reproduction

All pipelines are designed to run deterministically from the repository root:

### Step 1: Automated Model Training & Evaluation
Executes stratified 80/20 train/test splitting, 16-feature engineering, `RandomizedSearchCV` hyperparameter tuning, 3-fold Stratified CV, baseline benchmark comparisons, and serializes artifacts:

```bash
python3 src/train_and_evaluate.py
```
*Expected Outputs:*
- `results/best_model.pkl`
- `results/preprocessor.pkl`
- `results/summary.json`
- `results/model_comparison.csv`
- `results/tables/cross_validation.csv`
- `results/tables/tuning_results.csv`

### Step 2: Unsupervised Clustering & Personality Archetypes
Performs K-Means ($k=2 \dots 8$ sweep), Ward Hierarchical clustering, and DBSCAN density analysis on 8 personality dimensions:

```bash
python3 src/run_clustering.py
```
*Expected Outputs:*
- `results/tables/clustering_comparison.csv`
- `results/tables/cluster_archetypes.csv`
- `results/figures/clustering_kmeans_elbow_silhouette.png`
- `results/figures/clustering_dendrogram.png`
- `results/figures/clustering_pca_comparison.png`

### Step 3: Cross-Archetype Compatibility Affinity Matrix
Calculates cross-cluster empirical pairing success across all combinations:

```bash
python3 src/run_archetype_affinity.py
```
*Expected Outputs:*
- `results/tables/archetype_affinity_matrix.csv`
- `results/figures/clustering_archetype_affinity_heatmap.png`

### Step 4: SHAP Explainability Generation
Computes global Shapley beeswarm and local waterfall attributions:

```bash
python3 src/run_shap_analysis.py
```
*Expected Outputs:*
- `results/figures/shap_summary.png`
- `results/figures/shap_waterfall.png`

---

## 4. Running the Enhancement Experiments Suite

A dedicated suite of advanced experimental modules has been added to `experiments/`:

| Script | Purpose | Execution Command | Output Artifacts |
|---|---|---|---|
| **Learning Curves** | Evaluates bias-variance dynamics across 10 sample fractions | `python3 experiments/learning_curves.py` | `results/figures/learning_curve_xgboost.png` |
| **Error & Calibration** | PR-AUC, reliability diagram, score distributions | `python3 experiments/error_analysis.py` | `results/figures/pr_curve.png`, `calibration_curve.png` |
| **Feature Selection** | ANOVA F-test, Mutual Information, RFE & L1 comparisons | `python3 experiments/feature_selection_analysis.py` | `results/figures/feature_selection_comparison.png` |
| **Ablation Study** | Isolates contribution of 7 feature groups | `python3 experiments/ablation_study.py` | `results/figures/ablation_study.png` |
| **Extended Clustering** | Archetype radar charts and parallel coordinates | `python3 experiments/clustering_extended_analysis.py` | `results/figures/archetype_radar_chart.png` |
| **Model Robustness** | Seed variance, Gaussian noise, missing value resilience | `python3 experiments/model_robustness.py` | `results/figures/robustness_noise.png` |
| **Statistical Comparison** | Wilcoxon signed-rank & McNemar hypothesis tests | `python3 experiments/statistical_model_comparison.py` | `results/tables/statistical_comparison.csv` |
| **Advanced SHAP** | Interaction effects, dependence plots, force plots | `experiments/advanced_shap_analysis.py` | `results/figures/enhanced_shap_interaction.png` |

To execute the entire experimental suite automatically:
```bash
python3 experiments/run_all_experiments.py
```

---

## 5. Running the Test Suite

Execute the full suite of unit, integration, and enhancement tests:

```bash
# Run all tests with verbose reporting
python3 -m pytest tests/ -v

# Run only the new enhancement suite
python3 -m pytest tests/test_enhancements.py -v
```

---

## 6. Launching the Production Application

Start the Flask interactive web application:

```bash
python3 app.py
```
Navigate to `http://localhost:5000` in any modern web browser.
- Main Matchmaking Portal: `http://localhost:5000/`
- Model Performance Dashboard: `http://localhost:5000/dashboard`
- Personality Archetype Explorer: `http://localhost:5000/archetypes`
- Health Endpoint: `http://localhost:5000/api/health`
- Model Metadata: `http://localhost:5000/api/model-info`

---

## 7. Determinism & Random Seeds Checklist

| Component | Random Seed Setting | Purpose |
|---|:---:|---|
| Train/Test Stratification | `random_state=42` | Ensures identical 80k/20k row assignments |
| Stratified K-Fold CV | `random_state=42` | Ensures identical fold memberships across models |
| Logistic Regression | `random_state=42` | Solver convergence determinism |
| Decision Tree | `random_state=42` | Best feature split tie-breaking |
| Random Forest | `random_state=42` | Tree bagging and bootstrap sampling seed |
| XGBoost Classifier | `random_state=42` | Subsample and colsample sampling seed |
| K-Means Clustering | `random_state=42` | Initial centroid placement determinism |
| RandomizedSearchCV | `random_state=42` | Hyperparameter sample draws |
