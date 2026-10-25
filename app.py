import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from src.data.preprocessing import CupidDataPreprocessor
from src.models import CupidXGBoost, BaseModel

app = Flask(__name__, static_folder='static', template_folder='templates')

BEST_MODEL = None
PREPROCESSOR = None

def load_ml_pipeline():
    global BEST_MODEL, PREPROCESSOR
    model_path = os.path.join("results", "best_model.pkl")
    prep_path = os.path.join("results", "preprocessor.pkl")

    if os.path.exists(model_path) and os.path.exists(prep_path):
        try:
            BEST_MODEL = BaseModel.load(model_path)
            PREPROCESSOR = CupidDataPreprocessor.load(prep_path)
            print("Successfully loaded trained XGBoost model and preprocessor pipeline.")
        except Exception as e:
            print(f"Error loading ML artifacts: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df_single = pd.DataFrame([data])

        if BEST_MODEL is not None and PREPROCESSOR is not None:
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

            return jsonify({
                "success": True,
                "compatibility_score": score,
                "compatible": pred,
                "estimated_longevity_months": max(3, min(120, longevity)),
                "big5_harmony_score": big5_harmony,
                "avg_trait_diff": avg_diff,
                "personality_distance": dist
            })
        else:
            # Mathematical model calculation if artifacts not loaded yet
            age_diff = abs(data['a_age'] - data['b_age'])
            same_loc = 1.0 if data['a_location'] == data['b_location'] else 0.0
            same_love = 1.0 if data['a_love_language'] == data['b_love_language'] else 0.0

            trait_diffs = [
                abs(data['a_openness'] - data['b_openness']),
                abs(data['a_extraversion'] - data['b_extraversion']),
                abs(data['a_agreeableness'] - data['b_agreeableness']),
                abs(data['a_conscientiousness'] - data['b_conscientiousness']),
                abs(data['a_chronotype'] - data['b_chronotype']),
                abs(data['a_spontaneity'] - data['b_spontaneity']),
                abs(data['a_emotional_expressiveness'] - data['b_emotional_expressiveness']),
                abs(data['a_career_ambition'] - data['b_career_ambition'])
            ]
            dist = float(np.sqrt(sum(x**2 for x in trait_diffs)))
            avg_diff = float(np.mean(trait_diffs))

            score = float(max(10, min(99, 95 - (dist * 7.5) - (age_diff * 0.8) + (same_loc * 6) + (same_love * 8))))
            pred = 1 if score >= 55 else 0
            longevity = int(round(score * 0.95 + (same_love * 12)))

            return jsonify({
                "success": True,
                "compatibility_score": score,
                "compatible": pred,
                "estimated_longevity_months": max(3, min(120, longevity)),
                "big5_harmony_score": round(max(0, 100 - avg_diff * 15), 1),
                "avg_trait_diff": round(avg_diff, 2),
                "personality_distance": round(dist, 2)
            })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    load_ml_pipeline()
    app.run(host='0.0.0.0', port=5000, debug=True)
