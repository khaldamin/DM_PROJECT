# 🏛️ Architecture Documentation — Cupid AI System

> **Document Type:** Technical System & Software Architecture Blueprint  
> **System Name:** Cupid AI Matchmaking & Personality Analytics Engine  
> **Design Pattern:** Object-Oriented Domain Layer with Scikit-Learn Transformers & Flask Microservice  
> **Status:** Enhancement Addition (Strictly Non-Destructive)

---

## 1. High-Level System Architecture

Cupid AI is designed as a decoupled, multi-tiered architecture that separates **raw data ingestion**, **feature engineering**, **supervised predictive inference**, **unsupervised clustering**, **interpretability (SHAP)**, and **client-facing presentation layers**.

```mermaid
graph TD
    subgraph Presentation & Client Layer
        CLIENT["Web Browser (User / Evaluator)"]
        CLIENT -->|HTTP GET / POST| FLASK["Flask Web Server (app.py)"]
        UI_MAIN["Main Matching Portal (templates/index.html)"]
        UI_DASH["Analytics Dashboard (templates/dashboard.html)"]
        UI_ARCH["Archetype Explorer (templates/archetypes.html)"]
        FLASK --> UI_MAIN
        FLASK --> UI_DASH
        FLASK --> UI_ARCH
    end

    subgraph API Routing Layer
        FLASK --> PREDICT_API["POST /api/predict"]
        FLASK --> HEALTH_API["GET /api/health"]
        FLASK --> INFO_API["GET /api/model-info"]
        FLASK --> ENHANCED_BP["Enhanced Blueprint (src/api/enhanced_routes.py)"]
    end

    subgraph Inference & Intelligence Engines
        PREDICT_API --> PIPELINE["CupidDataPreprocessor (results/preprocessor.pkl)"]
        PIPELINE -->|Transformed X| MODEL["CupidXGBoost (results/best_model.pkl)"]
        MODEL -->|Predict Proba| RESPONSE["Compatibility Score & Longevity"]
        PREDICT_API --> SHAP["CupidExplainer / TreeExplainer"]
        PREDICT_API --> ARCH_ENGINE["Archetype Matcher (src/models/archetypes.py)"]
        SHAP -->|Positive & Negative Drivers| RESPONSE
        ARCH_ENGINE -->|Centroid Distance & Affinity| RESPONSE
    end

    subgraph Offline Training & Evaluation Pipeline
        RAW_DATA["data/raw/cupid_algorithm_dataset.csv (100k)"] --> TRAIN_SCRIPT["src/train_and_evaluate.py"]
        TRAIN_SCRIPT --> PREP_CLASS["CupidDataPreprocessor.fit_transform()"]
        TRAIN_SCRIPT --> CV_ENGINE["StratifiedKFold + RandomizedSearchCV"]
        CV_ENGINE --> SERIALIZE["Artifact Serializer (results/*.pkl)"]
        RAW_DATA --> CLUST_SCRIPT["src/run_clustering.py"]
        CLUST_SCRIPT --> CLUST_RESULTS["Cluster Metrics & Archetypes"]
    end
```

---

## 2. Object-Oriented Class Hierarchy

The modeling core utilizes an Object-Oriented Programming (OOP) hierarchy adhering to the **Open/Closed Principle** and **Interface Segregation**:

```mermaid
classDiagram
    class BaseModel {
        <<abstract>>
        +str model_name
        +Any model
        +fit(X_train, y_train)*
        +predict(X) ndarray
        +predict_proba(X) ndarray
        +save(filepath)
        +load(filepath)$ BaseModel
    }

    class CupidLogisticRegression {
        +int max_iter
        +float C
        +fit(X_train, y_train)
        +get_coefficients() ndarray
    }

    class CupidDecisionTree {
        +int max_depth
        +int min_samples_split
        +fit(X_train, y_train)
        +get_feature_importances() ndarray
    }

    class CupidRandomForest {
        +int n_estimators
        +int max_depth
        +fit(X_train, y_train)
        +get_feature_importances() ndarray
    }

    class CupidXGBoost {
        +int n_estimators
        +int max_depth
        +float learning_rate
        +float subsample
        +float colsample_bytree
        +fit(X_train, y_train)
        +get_feature_importances() ndarray
    }

    BaseModel <|-- CupidLogisticRegression
    BaseModel <|-- CupidDecisionTree
    BaseModel <|-- CupidRandomForest
    BaseModel <|-- CupidXGBoost
```

### 2.1 Model Base Contract (`src/models/base_model.py`)
All classification algorithms inherit from `BaseModel`. This ensures that downstream serving scripts, the Flask API, cross-validation runners, and SHAP explainers interact with a consistent interface regardless of the underlying algorithm (`predict()`, `predict_proba()`, `save()`, `load()`).

---

## 3. Preprocessing Pipeline Architecture

The preprocessing architecture encapsulates domain-specific logic, feature engineering, and statistical normalization inside `CupidDataPreprocessor`:

```mermaid
sequenceDiagram
    participant Caller as Ingestion / Inference
    participant Prep as CupidDataPreprocessor
    participant FE as Feature Engineering Engine
    participant CT as ColumnTransformer
    participant Scaler as StandardScaler
    participant OHE as OneHotEncoder

    Caller->>Prep: transform(df_raw)
    Prep->>FE: create_features(df_raw)
    Note over FE: Generates 16 interaction features<br/>(diff_*, same_*, euclidean_dist, big5_harmony)
    FE-->>Prep: df_engineered
    Prep->>CT: transform(df_engineered)
    CT->>OHE: transform(categorical_cols)
    CT->>Scaler: transform(numerical_cols)
    OHE-->>CT: encoded_cats
    Scaler-->>CT: scaled_numerics
    CT-->>Prep: X_transformed (shape: [N, 56])
    Prep-->>Caller: X_transformed
```

---

## 4. API Endpoints Specification

| Method | Endpoint | Description | Request Body / Parameters | Response Structure |
|---|---|---|---|---|
| `GET` | `/` | Serves main Glassmorphism matchmaking web application | None | HTML document (`templates/index.html`) |
| `GET` | `/dashboard` | Serves Model Performance & Benchmark Dashboard | None | HTML document (`templates/dashboard.html`) |
| `GET` | `/archetypes` | Serves Personality Archetype Explorer & Quiz | None | HTML document (`templates/archetypes.html`) |
| `POST` | `/api/predict` | Computes compatibility prediction, longevity, and SHAP drivers | JSON payload with 26 dyadic profile attributes | `{"success": true, "compatibility_score": 88.5, "compatible": 1, "estimated_longevity_months": 84, "archetype_a": {...}, "xai_breakdown": {...}}` |
| `GET` | `/api/health` | Service health status and artifact verification | None | `{"status": "healthy", "model_loaded": true, "preprocessor_loaded": true}` |
| `GET` | `/api/model-info` | Returns model hyperparameters and benchmark test metrics | None | `{"model_name": "Tuned XGBoost", "metrics": {...}, "best_params": {...}}` |
| `GET` | `/api/model-comparison` | Comparative metrics across all 5 evaluated algorithms | None | `{"models": [{"Model": "XGBoost", "Accuracy": 0.7265, ...}]}` |
| `GET` | `/api/archetypes` | Centroids, names, and trait distributions for all 4 archetypes | None | `{"archetypes": [{"id": 0, "name": "...", "traits": {...}}]}` |

---

## 5. Security & Robustness Considerations

1. **Input Validation:** Required numerical and categorical fields are validated on `/api/predict`. Missing or out-of-range keys trigger explicit `400 Bad Request` messages without crashing the server.
2. **Graceful Fallback:** If serialized pickle artifacts are unavailable, `app.py` falls back to a deterministic heuristic compatibility model based on normalized Euclidean trait distance.
3. **In-Memory Caching:** Heavy machine learning models (`best_model.pkl`, `preprocessor.pkl`, `TreeExplainer`) are loaded once during Flask bootstrap into global state, preventing per-request disk I/O bottlenecks.
