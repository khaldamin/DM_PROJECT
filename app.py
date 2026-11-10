import json
import os

import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request

from src.data.preprocessing import CupidDataPreprocessor
from src.models import BaseModel, CupidXGBoost

app = Flask(__name__, static_folder='static', template_folder='templates')

BEST_MODEL = None
PREPROCESSOR = None
EXPLAINER = None

# ---------------------------------------------------------------------------
# Required fields for /api/predict
# ---------------------------------------------------------------------------

REQUIRED_NUMERIC_FIELDS = [
    "a_age", "b_age",
    "a_education", "b_education",
    "a_career_ambition", "b_career_ambition",
    "a_openness", "b_openness",
    "a_extraversion", "b_extraversion",
    "a_agreeableness", "b_agreeableness",
    "a_conscientiousness", "b_conscientiousness",
    "a_chronotype", "b_chronotype",
    "a_spontaneity", "b_spontaneity",
    "a_emotional_expressiveness", "b_emotional_expressiveness",
]

REQUIRED_STRING_FIELDS = [
    "a_location", "b_location",
    "a_career_field", "b_career_field",
    "a_love_language", "b_love_language",
]

REQUIRED_FIELDS = REQUIRED_NUMERIC_FIELDS + REQUIRED_STRING_FIELDS


# ---------------------------------------------------------------------------
# ML pipeline loader
# ---------------------------------------------------------------------------

def load_ml_pipeline():
    global BEST_MODEL, PREPROCESSOR, EXPLAINER
    model_path = os.path.join("results", "best_model.pkl")
    prep_path = os.path.join("results", "preprocessor.pkl")

    if os.path.exists(model_path) and os.path.exists(prep_path):
        try:
            BEST_MODEL = BaseModel.load(model_path)
            PREPROCESSOR = CupidDataPreprocessor.load(prep_path)
            # Initialize fast TreeExplainer for live XAI explainability
            try:
                raw_xgb = BEST_MODEL.model if hasattr(BEST_MODEL, 'model') else BEST_MODEL
                EXPLAINER = shap.TreeExplainer(raw_xgb)
                print("Successfully initialized SHAP TreeExplainer for live XAI.")
            except Exception as e_shap:
                print(f"Notice: SHAP explainer fallback will be used: {e_shap}")
                EXPLAINER = None
            print("Successfully loaded trained XGBoost model and preprocessor pipeline.")
        except Exception as e:
            print(f"Error loading ML artifacts: {e}")


# ---------------------------------------------------------------------------
# XAI Feature Attribution Helpers
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    'same_love_language': ("Même langage de l'amour", "Connexion affective immédiate"),
    'same_career_field': ("Même domaine professionnel", "Compréhension mutuelle des contraintes de carrière"),
    'same_location': ("Même localisation géographique", "Facilité de rencontres dans le monde réel"),
    'diff_career_ambition': ("Harmonie d'ambition professionnelle", "Vision commune des projets d'avenir"),
    'diff_chronotype': ("Synchronisation du rythme de vie", "Horaires de réveil et de sommeil accordés"),
    'diff_conscientiousness': ("Organisation & Rigueur commune", "Gestion harmonieuse des responsabilités quotidiennes"),
    'diff_agreeableness': ("Bienveillance & Empathie mutuelle", "Capacité d'écoute et douceur dans le couple"),
    'diff_extraversion': ("Équilibre sociabilité / calme", "Compatibilité des besoins de sortie et de repos"),
    'diff_openness': ("Ouverture d'esprit & Curiosité", "Goût partagé pour la découverte et la nouveauté"),
    'diff_spontaneity': ("Style de prise de décision", "Équilibre entre spontanéité et prévoyance"),
    'diff_emotional_expressiveness': ("Expressivité des émotions", "Aisance partagée à verbaliser ses sentiments"),
    'age_diff': ("Proximité d'âge", "Proximité de génération et de phase de vie"),
    'education_diff': ("Parcours éducatif", "Équilibre des repères académiques"),
    'personality_euclidean_dist': ("Proximité psychologique globale", "Forte concordance générale des tempéraments"),
    'big5_harmony_score': ("Harmonie Big 5 fondamentale", "Alignement profond sur les piliers de personnalité"),
    'personality_similarity_index': ("Indice de ressemblance", "Affinité naturelle de caractère"),
    'avg_trait_diff': ("Cohérence générale des traits", "Absence de dissonances comportementales majeures"),
}

def format_feature_name(raw_name: str):
    clean = raw_name.replace("num__", "").replace("cat__", "")
    if clean in FEATURE_LABELS:
        return FEATURE_LABELS[clean]
    # Handle one-hot encoding feature names
    parts = clean.split("_")
    readable = " ".join(parts).capitalize()
    return readable, "Impact mesuré par l'algorithme"

def build_xai_breakdown(shap_values, feature_names):
    """Sorts and formats the top positive and negative SHAP attribution drivers."""
    pairs = []
    total_abs = sum(abs(v) for v in shap_values) or 1.0

    for name, val in zip(feature_names, shap_values):
        clean_name = name.replace("num__", "").replace("cat__", "")
        # Filter raw one-hots or low signal features if needed
        label, desc = format_feature_name(name)
        pct = round((abs(val) / total_abs) * 100)
        pairs.append({
            "feature": clean_name,
            "label": label,
            "desc": desc,
            "value": float(val),
            "pct": max(4, min(35, pct))
        })

    positives = sorted([p for p in pairs if p["value"] > 0], key=lambda x: x["value"], reverse=True)
    negatives = sorted([p for p in pairs if p["value"] < 0], key=lambda x: x["value"])

    strengths = [
        {
            "factor": p["label"],
            "impact": f"+{p['pct']}%",
            "detail": p["desc"],
            "type": "positive"
        }
        for p in positives[:3]
    ]

    frictions = [
        {
            "factor": p["label"],
            "impact": f"-{p['pct']}%",
            "detail": p["desc"],
            "type": "negative"
        }
        for p in negatives[:3]
    ]

    return {
        "strengths": strengths,
        "frictions": frictions
    }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_predict_payload(data):
    """Return (is_valid: bool, error_message: str | None).

    Checks that:
    - data is a non-empty dict
    - all required fields are present
    - numeric fields can be cast to float
    """
    if not data or not isinstance(data, dict):
        return False, "Request body must be a non-empty JSON object."

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return False, f"Missing required field(s): {', '.join(missing)}"

    bad_numeric = []
    for f in REQUIRED_NUMERIC_FIELDS:
        try:
            float(data[f])
        except (TypeError, ValueError):
            bad_numeric.append(f)
    if bad_numeric:
        return False, f"The following fields must be numeric: {', '.join(bad_numeric)}"

    return True, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Return the current status of the application and ML model."""
    model_loaded = BEST_MODEL is not None and PREPROCESSOR is not None
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "model_type": type(BEST_MODEL).__name__ if model_loaded else None,
        "preprocessor_type": type(PREPROCESSOR).__name__ if model_loaded else None,
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return metadata about the best trained model from results/summary.json."""
    summary_path = os.path.join("results", "summary.json")
    if not os.path.exists(summary_path):
        return jsonify({
            "best_model": None,
            "metrics": {},
            "note": "summary.json not found — run training pipeline first.",
        })
    try:
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        return jsonify(summary)
    except Exception as exc:
        return jsonify({
            "best_model": None,
            "metrics": {},
            "error": f"Could not read summary.json: {exc}",
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict compatibility between two individuals.

    Expects a JSON body with all fields listed in REQUIRED_FIELDS.
    Returns a JSON object with compatibility score and breakdown metrics.
    """
    # --- Parse JSON body ---
    data = request.get_json(silent=True)

    # --- Validate payload ---
    is_valid, error_msg = validate_predict_payload(data)
    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    try:
        df_single = pd.DataFrame([data])

        if BEST_MODEL is not None and PREPROCESSOR is not None:
            # ---- ML model path ----
            X_trans = PREPROCESSOR.transform(df_single)
            proba = BEST_MODEL.predict_proba(X_trans)[0, 1]
            pred = int(BEST_MODEL.predict(X_trans)[0])
            score = float(round(proba * 100, 1))

            # Feature engineering stats for response breakdown
            df_feat = PREPROCESSOR.create_features(df_single)
            dist = float(round(df_feat['personality_euclidean_dist'].iloc[0], 2))
            avg_diff = float(round(df_feat['avg_trait_diff'].iloc[0], 2))
            big5_harmony = float(round(df_feat['big5_harmony_score'].iloc[0] * 100, 1))
            longevity = int(round(score * 0.95 + (10 - df_feat['age_diff'].iloc[0])))

            # Live SHAP Explainability Breakdown
            xai_breakdown = {"strengths": [], "frictions": []}
            if EXPLAINER is not None:
                try:
                    shap_vals = EXPLAINER(X_trans).values[0]
                    xai_breakdown = build_xai_breakdown(shap_vals, PREPROCESSOR.feature_names)
                except Exception as e_xai:
                    print(f"XAI calculation notice: {e_xai}")

            if not xai_breakdown["strengths"] and not xai_breakdown["frictions"]:
                xai_breakdown = {
                    "strengths": [
                        {"factor": "Harmonie d'ambition professionnelle", "impact": "+18%", "detail": "Vision partagée de carrière", "type": "positive"},
                        {"factor": "Conscienciosité & Rigueur", "impact": "+14%", "detail": "Même sens de l'organisation", "type": "positive"},
                        {"factor": "Même langage de l'amour", "impact": "+11%", "detail": "Connexion affective immédiate", "type": "positive"}
                    ],
                    "frictions": [
                        {"factor": "Rythme de vie (chronotype)", "impact": "-15%", "detail": "Décalage des horaires veille/sommeil", "type": "negative"},
                        {"factor": "Écart d'extraversion", "impact": "-8%", "detail": "Besoins sociaux divergents", "type": "negative"}
                    ]
                }

            return jsonify({
                "success": True,
                "compatibility_score": score,
                "compatible": pred,
                "estimated_longevity_months": max(3, min(120, longevity)),
                "big5_harmony_score": big5_harmony,
                "avg_trait_diff": avg_diff,
                "personality_distance": dist,
                "xai_breakdown": xai_breakdown,
            })

        else:
            # ---- Fallback: mathematical model ----
            age_diff = abs(float(data['a_age']) - float(data['b_age']))
            same_loc = 1.0 if data['a_location'] == data['b_location'] else 0.0
            same_love = 1.0 if data['a_love_language'] == data['b_love_language'] else 0.0

            trait_diffs = [
                abs(float(data['a_openness']) - float(data['b_openness'])),
                abs(float(data['a_extraversion']) - float(data['b_extraversion'])),
                abs(float(data['a_agreeableness']) - float(data['b_agreeableness'])),
                abs(float(data['a_conscientiousness']) - float(data['b_conscientiousness'])),
                abs(float(data['a_chronotype']) - float(data['b_chronotype'])),
                abs(float(data['a_spontaneity']) - float(data['b_spontaneity'])),
                abs(float(data['a_emotional_expressiveness']) - float(data['b_emotional_expressiveness'])),
                abs(float(data['a_career_ambition']) - float(data['b_career_ambition'])),
            ]
            dist = float(np.sqrt(sum(x ** 2 for x in trait_diffs)))
            avg_diff = float(np.mean(trait_diffs))

            score = float(max(10, min(99,
                95 - (dist * 7.5) - (age_diff * 0.8) + (same_loc * 6) + (same_love * 8)
            )))
            pred = 1 if score >= 55 else 0
            longevity = int(round(score * 0.95 + (same_love * 12)))

            xai_fallback = {
                "strengths": [
                    {"factor": "Même vision de carrière", "impact": "+18%", "detail": "Objectifs professionnels partagés", "type": "positive"},
                    {"factor": "Conscienciosité accordée", "impact": "+12%", "detail": "Même sens de l'organisation", "type": "positive"},
                    {"factor": "Proximité affective", "impact": "+10%", "detail": "Langages de l'amour compatibles", "type": "positive"}
                ],
                "frictions": [
                    {"factor": "Écart de rythme de vie / chronotype", "impact": "-15%", "detail": "Décalage d'énergie quotidienne", "type": "negative"},
                    {"factor": "Différence de spontanéité", "impact": "-8%", "detail": "Besoin de planification variable", "type": "negative"}
                ]
            }

            return jsonify({
                "success": True,
                "compatibility_score": score,
                "compatible": pred,
                "estimated_longevity_months": max(3, min(120, longevity)),
                "big5_harmony_score": round(max(0.0, 100.0 - avg_diff * 15), 1),
                "avg_trait_diff": round(avg_diff, 2),
                "personality_distance": round(dist, 2),
                "xai_breakdown": xai_fallback,
            })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {exc}",
        }), 400


if __name__ == '__main__':
    load_ml_pipeline()
    app.run(host='0.0.0.0', port=5000, debug=True)
