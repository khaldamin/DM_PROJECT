# 📐 Enhanced Scientific Methodology — Cupid AI

> **Document Type:** Formal Research & Technical Methodology Report  
> **Domain:** Interpersonal Relationship Compatibility & Personality Segmentation via Machine Learning  
> **Dataset Scale:** 100,000 Dyadic Profile Pairs (26 Raw Features + 16 Interaction Dimensions)  
> **Status:** Enhancement Addition (Strictly Non-Destructive)

---

## 1. Executive Summary & Problem Formulation

Predicting human romantic and interpersonal compatibility is fundamentally challenging due to the high-dimensional, subjective, and non-linear nature of human relationships. Commercial dating platforms commonly rely on proprietary heuristics or simple cosine similarity across profile vectors without public empirical validation.

**Cupid AI** formalizes dyadic matchmaking as a binary classification problem alongside multidimensional personality clustering. Given a pair of individuals $(A, B)$ represented by profile feature vectors $\mathbf{x}_A \in \mathbb{R}^{d}$ and $\mathbf{x}_B \in \mathbb{R}^{d}$, the primary objective is to learn a mapping:

$$f: (\mathbf{x}_A, \mathbf{x}_B) \mapsto \hat{y} \in \{0, 1\}$$

where $\hat{y} = 1$ denotes a compatible pairing (high probability of sustained romantic affinity) and $\hat{y} = 0$ denotes incompatibility. Concurrently, an unsupervised operator $g: \mathbf{x} \mapsto c \in \{0, 1, 2, 3\}$ maps individual temperaments into distinct psychometric archetypes.

---

## 2. Dataset Topology & Statistical Profile

### 2.1 Data Ingestion & Partitions
The dataset comprises $N = 100,000$ labeled profile pairs. Data splitting is conducted prior to any normalization or transformation to guarantee zero data leakage:

| Partition | Sample Count ($N$) | Proportion | Target Distribution ($y=1$ / $y=0$) |
|---|:---:|:---:|:---:|
| **Training Set** | 80,000 | 80.0% | 49.98% / 50.02% (Balanced) |
| **Hold-out Test Set** | 20,000 | 20.0% | 49.97% / 50.03% (Balanced) |
| **Total** | 100,000 | 100.0% | 49.98% / 50.02% |

### 2.2 Feature Space Definition
Each dyad consists of 13 attributes per individual (prefix `a_` and `b_`):
- **Demographics:** `age` ($\in [18, 80]$), `education` ($\in \{1, 2, 3, 4\}$ ordinal), `location` ($\in \{\text{Urban, Suburban, Rural}\}$).
- **Career & Lifestyle:** `career_field` (10 modalities), `career_ambition` ($\in [0.1, 1.0]$), `chronotype` ($\in [0.0, 1.0]$ representing early bird vs. night owl).
- **Big Five Personality Dimensions:** `openness`, `extraversion`, `agreeableness`, `conscientiousness` ($\in [0.0, 1.0]$).
- **Relational Expressiveness:** `spontaneity`, `emotional_expressiveness` ($\in [0.0, 1.0]$), and `love_language` (5 categories).

---

## 3. Mathematical Feature Engineering Pipeline

Standard concatenated features $[\mathbf{x}_A, \mathbf{x}_B]$ fail to expose relational dynamics. Cupid AI derives **16 domain-specific interaction features** prior to model ingestion:

### 3.1 Categorical Concordance Indicators
Binary indicators capture shared environmental and emotional alignments:
$$\text{same\_location} = \mathbb{I}(a_{\text{location}} = b_{\text{location}})$$
$$\text{same\_career\_field} = \mathbb{I}(a_{\text{career\_field}} = b_{\text{career\_field}})$$
$$\text{same\_love\_language} = \mathbb{I}(a_{\text{love\_language}} = b_{\text{love\_language}})$$

### 3.2 Dyadic Differential Metrics
Absolute divergence metrics capture friction across life stages and personality traits:
$$\Delta_{\text{age}} = |a_{\text{age}} - b_{\text{age}}|$$
$$\Delta_{\text{education}} = |a_{\text{education}} - b_{\text{education}}|$$
$$\Delta_{\text{trait}, k} = |a_k - b_k|, \quad \forall k \in \{\text{openness}, \text{extraversion}, \text{agreeableness}, \text{conscientiousness}, \text{chronotype}, \text{spontaneity}, \text{emotional\_expressiveness}, \text{career\_ambition}\}$$

### 3.3 Composite Psychological Metrics
1. **Psychological Euclidean Distance ($D_{\text{psych}}$):**  
   $$D_{\text{psych}} = \sqrt{\sum_{k \in \mathcal{K}} (a_k - b_k)^2}, \quad \text{where } |\mathcal{K}| = 8$$

2. **Personality Similarity Index ($S_{\text{psych}}$):**  
   $$S_{\text{psych}} = \frac{1}{1 + D_{\text{psych}}} \in (0, 1]$$

3. **Average Absolute Trait Divergence ($\bar{\Delta}_{\text{trait}}$):**  
   $$\bar{\Delta}_{\text{trait}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} |a_k - b_k|$$

4. **Big 5 Harmony Score ($H_{\text{Big5}}$):**  
   Evaluates fundamental temperament alignment across core psychometric traits $\mathcal{B} = \{\text{openness}, \text{extraversion}, \text{agreeableness}, \text{conscientiousness}\}$:
   $$H_{\text{Big5}} = \frac{1}{1 + \frac{1}{4} \sum_{k \in \mathcal{B}} |a_k - b_k|}$$

---

## 4. Preprocessing & Leakage Prevention Protocol

Data leakage occurs when information from outside the training dataset is used to build the model. Cupid AI enforces strict partitioning via `CupidDataPreprocessor`:

```text
[Raw Dataset (100k)]
         │
         ▼  StratifiedSplit (seed=42, test_size=0.20)
    ┌────┴─────────────────────────────┐
    ▼                                  ▼
[Train Partition (80k)]        [Test Partition (20k)]
    │                                  │
    ├─ fit_transform()                 │
    │  • OneHotEncoder (learns cats)   │
    │  • StandardScaler (learns µ, σ)  │
    ▼                                  │
[Transformed X_train]                  │
                                       ▼ transform() (uses learned µ, σ, cats)
                               [Transformed X_test]
```

1. **One-Hot Encoding:** Applied to 6 categorical features (`a_location`, `b_location`, `a_career_field`, `b_career_field`, `a_love_language`, `b_love_language`) with `handle_unknown='ignore'`.
2. **Feature Standardization:** Numerical features $\mathbf{z}$ are standardized using parameters computed strictly on training data:
   $$z_i = \frac{x_i - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
3. **Cross-Validation Fold Isolation:** During Stratified $K$-Fold cross-validation, a dedicated preprocessor instance is re-fitted on each inner training fold, preserving absolute statistical independence.

---

## 5. Supervised Learning Methodology

Four distinct algorithmic families were evaluated to balance interpretability, linear separability, non-linear interaction capacity, and ensemble resilience:

1. **L2-Regularized Logistic Regression (Linear Baseline):**
   $$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
   *Objective:* Prove whether romantic compatibility can be captured by monotonic linear combinations of trait discrepancies.

2. **CART Decision Tree (Non-Linear Single Tree):**
   Splitting criterion based on Gini Impurity:
   $$I_G(t) = 1 - \sum_{i=1}^{C} p(i|t)^2$$
   *Hyperparameters:* `max_depth=12`, `min_samples_split=10`.

3. **Random Forest Classifier (Bagging Ensemble):**
   Aggregates $B=100$ decorrelated decision trees with bootstrap aggregation and feature sub-sampling:
   $$\hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} f_b(\mathbf{x})$$

4. **Extreme Gradient Boosting — XGBoost (Boosting Ensemble):**
   Minimizes regularized objective through second-order Taylor expansion:
   $$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$
   where $g_i = \partial_{\hat{y}^{(t-1)}} \ell(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial_{\hat{y}^{(t-1)}}^2 \ell(y_i, \hat{y}^{(t-1)})$.

### 5.1 Hyperparameter Tuning Protocol
Optimization was performed via `RandomizedSearchCV` across a continuous-discrete parameter grid evaluated with 3-fold Stratified CV optimizing the F1-Score:
- `n_estimators`: $\{100, 150, 200, 250\}$ $\to$ **Selected: 200**
- `max_depth`: $\{4, 5, 6, 8\}$ $\to$ **Selected: 6**
- `learning_rate`: $\{0.05, 0.08, 0.10, 0.12\}$ $\to$ **Selected: 0.12**
- `subsample`: $\{0.7, 0.8, 0.9, 1.0\}$ $\to$ **Selected: 0.9**
- `colsample_bytree`: $\{0.7, 0.8, 0.9, 1.0\}$ $\to$ **Selected: 0.8**

---

## 6. Unsupervised Clustering & Archetype Profiling

Individual personality vectors $\mathbf{p} \in [0, 1]^8$ were clustered using three distinct paradigms to reveal underlying psychological phenotypes:

### 6.1 Cluster Validation Metrics
- **Silhouette Coefficient ($s_i$):**
  $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}, \quad S = \frac{1}{N} \sum_{i=1}^{N} s_i$$
- **Davies-Bouldin Index ($DB$):**
  $$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\bar{d}_i + \bar{d}_j}{d(c_i, c_j)} \right)$$
- **Calinski-Harabasz Index ($CH$):**
  $$CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{N - k}{k - 1}$$

### 6.2 The Four Empirical Archetypes ($k=4$)
Clustering revealed four stable centroids across 100,000 observations:
1. **Archetype 0 — The Grounded Stabilizers (23.3%):** Prominent Agreeableness ($0.68$) and Conscientiousness ($0.45$). Characterized by reliability, empathy, and preference for relationship security.
2. **Archetype 1 — The Dynamic Visionaries (25.7%):** High Openness ($0.47$), Extraversion ($0.65$), and Career Ambition ($0.49$). Forward-looking and goal-oriented.
3. **Archetype 2 — The Spontaneous Socializers (25.9%):** High Spontaneity ($0.44$) and Extraversion ($0.35$ relative baseline). Expressive, present-focused, and socially vibrant.
4. **Archetype 3 — The Independent Thinkers (25.1%):** Introverted depth ($0.61$), analytical focus, and deliberate pacing.

---

## 7. Explainable AI (SHAP) Attribution Protocol

To prevent opacity in matchmaking decisions, Cupid AI incorporates Shapley Additive Explanations grounded in cooperative game theory:

$$g(z') = \phi_0 + \sum_{j=1}^{M} \phi_j z'_j$$

where the Shapley value $\phi_j$ represents the marginal contribution of feature $j$ across all possible feature coalitions $\mathcal{S} \subseteq \mathcal{F} \setminus \{j\}$:

$$\phi_j = \sum_{\mathcal{S} \subseteq \mathcal{F} \setminus \{j\}} \frac{|\mathcal{S}|! (|\mathcal{F}| - |\mathcal{S}| - 1)!}{|\mathcal{F}|!} \left[ f_x(\mathcal{S} \cup \{j\}) - f_x(\mathcal{S}) \right]$$

- **Global Attribution:** Evaluated over representative test samples via TreeExplainer, revealing that `same_love_language`, `diff_career_ambition`, `diff_chronotype`, and `personality_euclidean_dist` dominate decision boundaries.
- **Local Attribution:** Visualized in real time for individual user pairings through waterfall diagrams, isolating positive compatibility drivers ($\phi_j > 0$) from relational friction points ($\phi_j < 0$).

---

## 8. Summary of Scientific Standards & Rigor

1. **Replicability:** Fixed pseudo-random seeds (`random_state=42`) applied across all stochastic steps.
2. **Leakage Immunity:** Strict isolation between parameter fitting and evaluation spaces.
3. **Multi-Faceted Evaluation:** Metrics balance threshold-dependent criteria (Accuracy, Precision, Recall, F1) and rank-order discrimination (ROC-AUC, PR-AUC).
4. **Algorithmic Transparency:** Direct translation of tree decisions into human-interpretable psychometric insights via SHAP values.
