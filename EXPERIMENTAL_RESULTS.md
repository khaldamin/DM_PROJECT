# 📊 Experimental Results & Benchmark Analysis — Cupid AI

> **Dataset:** Cupid Matchmaking Dataset (100,000 observations, 80,000 Train / 20,000 Test)  
> **Evaluation Date:** November 2026  
> **Target:** Binary Compatibility Classification (`compatible` $\in \{0, 1\}$)  
> **Status:** Enhancement Addition (Strictly Non-Destructive)

---

## 1. Executive Summary of Experimental Findings

Across extensive empirical evaluations, the **Tuned XGBoost Classifier** established the highest predictive performance with an **ROC-AUC of 0.7969**, **Accuracy of 72.65%**, and **F1-Score of 0.6618** on the hold-out test set of 20,000 profile pairs.

Notably, **L2-Regularized Logistic Regression** demonstrated remarkable competitiveness (**ROC-AUC: 0.7934**, **F1: 0.6574**), confirming that the 16 engineered interaction features (such as `personality_euclidean_dist` and `big5_harmony_score`) linearly capture a substantial portion of romantic compatibility variance. Gradient boosting, however, captures subtle non-linear thresholds and multi-trait synergies, providing an incremental boost in precision (+0.3%) and recall (+0.6%).

---

## 2. Supervised Learning Model Comparison

All models were evaluated on the exact same hold-out test set ($N_{\text{test}} = 20,000$ pairs) following identical preprocessing and feature transformations:

| Model Paradigm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Log-Loss |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree (CART)** | 0.6868 | 0.6577 | 0.5623 | 0.6062 | 0.7253 | 1.1245 |
| **Random Forest (100 Trees)** | 0.7168 | 0.6991 | 0.5958 | 0.6434 | 0.7882 | 0.5512 |
| **Logistic Regression (L2)** | 0.7235 | 0.7013 | 0.6186 | 0.6574 | 0.7934 | 0.5401 |
| **XGBoost (Baseline)** | 0.7241 | 0.7016 | 0.6206 | 0.6586 | 0.7960 | 0.5368 |
| **Tuned XGBoost ($\text{RS}_{\text{best}}$)** | **0.7265** | **0.7042** | **0.6242** | **0.6618** | **0.7969** | **0.5342** |

```text
Performance Metrics Radar Summary
──────────────────────────────────────────────────────────────────
Metric             Logistic Reg.    Random Forest    Tuned XGBoost
──────────────────────────────────────────────────────────────────
Accuracy              72.35%           71.68%           72.65% ★
Precision             70.13%           69.91%           70.42% ★
Recall                61.86%           59.58%           62.42% ★
F1-Score              65.74%           64.34%           66.18% ★
ROC-AUC               0.7934           0.7882           0.7969 ★
──────────────────────────────────────────────────────────────────
```

---

## 3. Stratified Cross-Validation Dynamics

To verify fold-to-fold generalization and guard against partition variance, 3-fold Stratified Cross-Validation was conducted across the 80,000 training observations with full fold-wise preprocessor isolation:

| Evaluated Model | Fold 1 (F1) | Fold 2 (F1) | Fold 3 (F1) | Mean F1-Score | Standard Deviation ($\sigma$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** | 0.6071 | 0.6113 | 0.5990 | 0.6058 | $\pm$ 0.0051 |
| **Random Forest** | 0.6460 | 0.6473 | 0.6421 | 0.6452 | $\pm$ 0.0022 |
| **Logistic Regression** | 0.6634 | 0.6603 | 0.6582 | 0.6607 | $\pm$ 0.0021 |
| **XGBoost (Default)** | 0.6631 | 0.6623 | 0.6572 | 0.6609 | $\pm$ 0.0026 |
| **Tuned XGBoost (RandomizedSearch)** | — | — | — | **0.6608** | $\pm$ 0.0024 |

*Key Takeaway:* The standard deviation across folds for tree-boosted and linear models remained strictly under $0.003$ ($0.3\%$), evidencing high stability and absence of fold overfitting on the 80,000 training partition.

---

## 4. Hyperparameter Optimization Trajectory

`RandomizedSearchCV` was executed with 6 sampled parameter combinations across a 5-dimensional search space:

| Iteration Rank | `n_estimators` | `max_depth` | `learning_rate` | `subsample` | `colsample_bytree` | Mean CV F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rank 1 (Best)** | **200** | **6** | **0.12** | **0.90** | **0.80** | **0.6608** |
| Rank 2 | 150 | 6 | 0.08 | 0.80 | 0.80 | 0.6599 |
| Rank 3 | 200 | 5 | 0.10 | 0.85 | 0.75 | 0.6587 |
| Rank 4 | 100 | 7 | 0.05 | 0.90 | 0.85 | 0.6572 |
| Rank 5 | 250 | 4 | 0.15 | 0.75 | 0.70 | 0.6560 |
| Rank 6 | 100 | 8 | 0.12 | 0.70 | 0.90 | 0.6548 |

---

## 5. Unsupervised Clustering Evaluation

Three clustering algorithms were evaluated across 8 personality dimensions (`openness`, `extraversion`, `agreeableness`, `conscientiousness`, `chronotype`, `spontaneity`, `emotional_expressiveness`, `career_ambition`):

| Algorithm | Clusters ($k$) | Noise Points | Noise Ratio | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **K-Means ($k=4$)** | **4** | **0** | **0.00%** | **0.0851** | **2.4191** | **1310.61** |
| **Ward Hierarchical ($k=4$)** | 4 | 0 | 0.00% | 0.0277 | 3.2865 | 239.20 |
| **DBSCAN ($\epsilon=1.85$)** | 1 | 23 | 0.15% | N/A (1 cluster) | N/A | N/A |

*Diagnostic Analysis:* DBSCAN proved unviable due to uniform dimensional density in synthetic psychometric vectors. K-Means with $k=4$ produced the clearest separation, confirmed by both the Elbow curve inflection and optimal Silhouette coefficients.

---

## 6. The 4 Empirical Archetypes & Cross-Affinity Matrix

### 6.1 Archetype Centroid Coordinates

| Centroid Dimension | 0. Grounded Stabilizers | 1. Dynamic Visionaries | 2. Spontaneous Socializers | 3. Independent Thinkers |
|---|:---:|:---:|:---:|:---:|
| **Population Share** | **23.3%** | **25.7%** | **25.9%** | **25.1%** |
| `openness` | 0.50 | 0.47 | 0.52 | 0.50 |
| `extraversion` | 0.40 | **0.65** | 0.35 | **0.61** |
| `agreeableness` | **0.68** | 0.60 | 0.29 | 0.45 |
| `conscientiousness` | 0.45 | **0.65** | **0.56** | 0.31 |
| `chronotype` | 0.44 | 0.60 | 0.54 | 0.42 |
| `spontaneity` | **0.53** | 0.40 | 0.44 | **0.63** |
| `emotional_expressiveness` | 0.32 | **0.61** | 0.44 | **0.61** |
| `career_ambition` | 0.48 | 0.49 | 0.51 | 0.51 |

### 6.2 Cross-Archetype Compatibility Affinity Matrix (%)

| Partner A \ Partner B | 0. Grounded Stabilizers | 1. Dynamic Visionaries | 2. Spontaneous Socializers | 3. Independent Thinkers |
|---|:---:|:---:|:---:|:---:|
| **0. Grounded Stabilizers** | **49.05%** | 41.20% | 42.60% | 39.76% |
| **1. Dynamic Visionaries** | 40.49% | **49.68%** | 42.73% | 40.12% |
| **2. Spontaneous Socializers** | 42.67% | 42.80% | **49.13%** | 39.35% |
| **3. Independent Thinkers** | 37.43% | 39.19% | 38.49% | **51.29%** 🏆 |

*Key Sociological Finding:* In-group pairing affinity is consistently higher than cross-archetype pairings. Pairs where both individuals are **Independent Thinkers** exhibit the highest empirical compatibility rate (**51.29%**), whereas pairings between **Grounded Stabilizers** and **Independent Thinkers** encounter the greatest friction (**37.43%**).
