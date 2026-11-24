# 📜 Changelog — Cupid AI Project

All notable changes to the **Cupid AI Matchmaking & Machine Learning Project** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — Academic Enhancement Additions (November 2026)

### Added
- **Formal Academic Documentation Suite:**
  - `ENHANCEMENT_AUDIT.md`: In-depth repository audit, architectural assessment, identified weaknesses, and grading improvement priorities.
  - `ENHANCED_METHODOLOGY.md`: Comprehensive mathematical and empirical methodology report documenting problem formulation, data topology, feature engineering, leakage prevention, and validation protocols.
  - `MODEL_CARD.md`: Industry-standard Model Card (Google/Hugging Face format) documenting architecture, intended use, quantitative performance benchmarks, confusion matrix, and ethics.
  - `REPRODUCIBILITY_GUIDE.md`: Step-by-step determinism and replication manual with pinned requirements, checksums, and execution steps.
  - `ARCHITECTURE_DOCUMENTATION.md`: Software engineering blueprint with Mermaid class diagrams, data flow sequences, and REST API specifications.
  - `EXPERIMENTAL_RESULTS.md`: Formal record of supervised and unsupervised findings, cross-validation metrics, and archetype affinity heatmaps.
  - `CHANGELOG.md`: Structured lineage tracking of project iterations.

- **Advanced Scientific Experiments Suite (`experiments/`):**
  - `experiments/learning_curves.py`: Bias-variance diagnostic and sample efficiency analysis across 10 sample fractions with Stratified CV.
  - `experiments/ablation_study.py`: Feature group ablation isolating the marginal contribution of 7 feature configurations.
  - `experiments/feature_selection_analysis.py`: Multi-criteria feature selection comparing ANOVA F-test, Mutual Information, RFE, and L1 lasso regularized ranking.
  - `experiments/error_analysis.py`: Comprehensive error diagnostic featuring Precision-Recall AUC (PR-AUC), Brier score calibration curve, prediction probability distributions, and demographic subgroup error breakdown.
  - `experiments/model_robustness.py`: Stress-testing model stability across 10 random seeds, Gaussian noise injection, and synthetic missingness.
  - `experiments/statistical_model_comparison.py`: Wilcoxon signed-rank and McNemar hypothesis testing with 95% bootstrap confidence intervals across 10 folds.
  - `experiments/clustering_extended_analysis.py`: 8-dimensional archetype radar charts, parallel coordinates, and extended metric sweeps.
  - `experiments/advanced_shap_analysis.py`: Non-linear feature interactions (`same_love_language` $\times$ `diff_chronotype`), dependence plots, and local force plots.
  - `experiments/run_all_experiments.py`: Master automated runner orchestrating all experimental scripts.

- **Application & Analytics Extensions:**
  - `templates/dashboard.html`: Glassmorphism Model Performance & Analytics Dashboard presenting live benchmark cards, Canvas-based metric comparison charts, and hyperparameter tables.
  - `static/js/dashboard.js`: Interactive JavaScript engine for real-time benchmark visualization and model card extraction.
  - `templates/archetypes.html`: Interactive Personality Archetype Explorer with an interactive 3-question archetype identification quiz and dynamic affinity matrix.
  - `static/js/archetypes.js`: Interactive Canvas radar chart and affinity matrix tooltip explorer.
  - `src/api/enhanced_routes.py`: Modular Flask Blueprint exposing `/dashboard`, `/archetypes`, and analytic JSON endpoints (`/api/model-comparison`, `/api/archetypes`, `/api/clustering-stats`).

- **Automated Verification:**
  - `tests/test_enhancements.py`: Pytest suite verifying the presence, schema compliance, and integrity of all newly added modules without duplicating existing tests.

---

## [1.0.0] — Initial Baseline Evaluation (Commit: `421c9a7`)

### Added
- **Data & Preprocessing Pipeline:**
  - Ingestion of 100,000 raw dyadic records from `data/raw/cupid_algorithm_dataset.csv`.
  - Feature engineering pipeline generating 16 interaction metrics.
  - `CupidDataPreprocessor` implementing OneHotEncoder for categoricals and StandardScaler for numerical features.
- **Supervised Learning Engine:**
  - Implementation of 4 classification paradigms: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
  - Hyperparameter optimization via `RandomizedSearchCV` on XGBoost resulting in best model artifacts (`results/best_model.pkl`).
  - 3-Fold Stratified cross-validation fold evaluation.
- **Unsupervised Segmentation:**
  - Implementation of K-Means ($k=4$), Ward Hierarchical Agglomerative clustering, and DBSCAN.
  - Extraction of 4 human personality archetypes and cross-archetype compatibility matrix.
- **Explainable AI (SHAP):**
  - Integration of SHAP `TreeExplainer` producing global summary plots and local waterfall explanations.
- **Flask Web Application:**
  - Production Flask server with real-time prediction endpoint (`POST /api/predict`).
  - Glassmorphism interface with preset relationship scenarios.
- **Academic Reporting & Presentation:**
  - Initial evaluation report in French (`rapport_final.md`).
  - Presentation slides deck (`slides_soutenance.md`).
  - Comprehensive unit test suite with 36 passing tests in `tests/`.
