# 📚 Documentation complète du projet *Cupid Matchmaking*

## 🎤 Présentation (extraits du slide de soutenance)

```
---
marp: true
theme: default
paginate: true
header: "Cupid AI — Data Mining & Matchmaking Prediction"
footer: "DM_PROJECT — Soutenance de Projet Machine Learning"
---

# 💘 Cupid Matchmaking
### Prédiction de la Compatibilité Relationnelle & Segmentation des Personnalités

*Auteurs : Équipe DM_PROJECT*
*Date : Novembre 2026*
*Dépôt GitHub : https://github.com/khaldamin/DM_PROJECT*
```

*(Le reste du slide détaille le contexte, les données, le pré‑traitement, les modèles, les résultats, l'explicabilité SHAP, le clustering, le déploiement Flask, les limites et la conclusion.)*

---

## 📄 README (extraits principaux)

```
# 💘 Cupid Matchmaking — Machine Learning Compatibility & Profiling

> Can machine learning predict romantic compatibility and segment human temperaments?

## 📋 Table of Contents
- Overview & Architecture
- Project Structure
- Data Description
- Feature Engineering Pipeline
- Supervised Learning Models
- Results & Benchmark
- Model Explainability (SHAP XAI)
- Unsupervised Learning & Clustering
- Web Application (Flask API)
- Installation & Quickstart
- Running Tests
- Guided Notebooks (01 to 08)
- Soutenance & Presentation

## ⚙️ Installation & Quickstart
```bash
pip install -r requirement.txt
python app.py            # launch Flask UI
python src/run_batch_predict.py <path/to/csv>   # batch inference
```

## ✅ Tests
```bash
python -m pytest tests/ -v
```

## 📈 Results & Benchmark
| Model | Accuracy | Precision | Recall | F1-Score | ROC‑AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6821 | 0.6453 | 0.5912 | 0.6170 | 0.7312 |
| Decision Tree | 0.5866 | 0.5172 | 0.5391 | 0.5279 | 0.6032 |
| Random Forest | 0.7045 | 0.6821 | 0.6201 | 0.6495 | 0.7721 |
| XGBoost | 0.7138 | 0.6934 | 0.6388 | 0.6650 | 0.7891 |
| **Tuned XGBoost** 🏆 | **0.7265** | **0.7042** | **0.6242** | **0.6618** | **0.7969** |
```

*Le README complet est disponible dans le dépôt.*
```
```
---

## 📦 Requirements (requirement.txt)

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
flask
scipy
joblib
pytest
jupyter
```

---

*Cette documentation regroupe la présentation, le README et les dépendances du projet pour faciliter la remise et la revue.*
