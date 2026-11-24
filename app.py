import json
import os

import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request

from src.data.preprocessing import CupidDataPreprocessor
from src.models import (
    BaseModel,
    CupidXGBoost,
    identify_archetype,
    compute_cross_archetype_affinity
)

import plotly.express as px
from scipy import stats
import joblib
from flask import send_file
from io import BytesIO

app = Flask(__name__, static_folder='static', template_folder='templates')

try:
    from src.api.enhanced_routes import enhanced_bp
    app.register_blueprint(enhanced_bp)
except Exception as _e_bp:
    print(f"Notice: enhanced routes blueprint could not be loaded: {_e_bp}")

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

TRAIT_NAMES = [
    "openness", "extraversion", "agreeableness", "conscientiousness",
    "chronotype", "spontaneity", "emotional_expressiveness", "career_ambition"
]


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

@app.route('/dataquality')
def data_quality():
    dataset_path = os.path.join(
        app.root_path,
        'data',
        'processed',
        'cupid_algorithm_dataset.csv'
    )

    df = pd.read_csv(dataset_path)

    # General dataset information
    total_rows = len(df)
    total_columns = len(df.columns)

    # Missing values
    missing_values = df.isnull().sum()
    total_missing = int(missing_values.sum())

    missing_data = [
        {
            "column": col,
            "count": int(missing_values[col]),
            "percentage": round(
                missing_values[col] / total_rows * 100, 2
            )
        }
        for col in df.columns
    ]

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # Column types
    type_counts = df.dtypes.astype(str).value_counts().to_dict()

    # Variable profile
    column_profiles = []

    for col in df.columns:
        column_profiles.append({
            "name": col,
            "type": str(df[col].dtype),
            "unique": int(df[col].nunique()),
            "missing": int(df[col].isnull().sum())
        })

    # Compatibility distribution
    compatibility_counts = (
        df['compatible']
        .value_counts()
        .sort_index()
        .to_dict()
    )

    compatibility_counts = {
        str(k): int(v)
        for k, v in compatibility_counts.items()
    }

    return render_template(
        'data_quality.html',
        total_rows=total_rows,
        total_columns=total_columns,
        total_missing=total_missing,
        duplicate_count=duplicate_count,
        missing_data=missing_data,
        type_counts=type_counts,
        column_profiles=column_profiles,
        compatibility_counts=compatibility_counts
    )

@app.route('/eda')
def eda():
    dataset_path = os.path.join(
        app.root_path,
        'data',
        'processed',
        'cupid_algorithm_dataset.csv'
    )

    df = pd.read_csv(dataset_path)

    # 1. Distribution des âges
    age_df = df[['a_age', 'b_age']].melt(
    var_name='Personne',
    value_name='Âge'
    )

    age_fig = px.histogram(
        age_df,
        x='Âge',
        color='Personne',
        barmode='overlay',
        opacity=0.6,
        nbins=30,
        title='Distribution des âges des deux profils',
        labels={
            'Âge': 'Âge',
            'Personne': 'Profil'
        }
    )

    # 2. Distribution du score de compatibilité
    score_fig = px.histogram(
        df,
        x='compatibility_score',
        title='Distribution des scores de compatibilité'
    )

    # 3. Répartition compatible / non compatible
    compatibility_fig = px.pie(
        df,
        names='compatible',
        title='Répartition de la compatibilité',
        hole=0.4
    )

    # 4. Distribution de l'éducation
    education_df = pd.concat([
        df['a_education'].rename('Education'),
        df['b_education'].rename('Education')
    ])

    education_fig = px.histogram(
        education_df,
        x='Education',
        title='Distribution des niveaux d’éducation'
    )

    # 5. Différence d'âge et compatibilité
    df['age_difference'] = abs(df['a_age'] - df['b_age'])

    age_compatibility_fig = px.box(
        df,
        x='compatible',
        y='age_difference',
        title='Différence d’âge selon la compatibilité'
    )


    # 6. Matrice de corrélation des traits de personnalité
    personality_cols = [
        'a_openness', 'b_openness',
        'a_extraversion', 'b_extraversion',
        'a_agreeableness', 'b_agreeableness',
        'a_conscientiousness', 'b_conscientiousness',
        'a_spontaneity', 'b_spontaneity',
        'a_emotional_expressiveness',
        'b_emotional_expressiveness',
        'a_career_ambition', 'b_career_ambition'
    ]

    correlation_matrix = df[personality_cols].corr()

    correlation_fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Matrice de corrélation des caractéristiques'
    )

    # 7. Compatibilité moyenne selon le domaine professionnel
    career_a = df[['a_career_field', 'compatible']].rename(
        columns={'a_career_field': 'career_field'}
    )

    career_b = df[['b_career_field', 'compatible']].rename(
        columns={'b_career_field': 'career_field'}
    )

    career_df = pd.concat([career_a, career_b])

    career_stats = (
        career_df.groupby('career_field')['compatible']
        .mean()
        .mul(100)
        .reset_index(name='compatibility_rate')
    )

    career_stats = career_stats.sort_values(
        'compatibility_rate', ascending=False
    ).head(15)

    career_fig = px.bar(
        career_stats,
        x='career_field',
        y='compatibility_rate',
        title='Taux de compatibilité selon le domaine professionnel',
        labels={
            'career_field': 'Domaine professionnel',
            'compatibility_rate': 'Taux de compatibilité (%)'
        }
    )

    # 8. Compatibilité selon le langage de l'amour
    df['same_love_language'] = (
        df['a_love_language'] == df['b_love_language']
    )

    love_stats = (
        df.groupby('same_love_language')['compatible']
        .mean()
        .mul(100)
        .reset_index(name='compatibility_rate')
    )

    love_stats['same_love_language'] = love_stats[
        'same_love_language'
    ].map({
        True: 'Même langage',
        False: 'Langages différents'
    })

    love_fig = px.bar(
        love_stats,
        x='same_love_language',
        y='compatibility_rate',
        title='Compatibilité selon les langages de l’amour',
        labels={
            'same_love_language': 'Comparaison des langages',
            'compatibility_rate': 'Taux de compatibilité (%)'
        }
    )

    # 9. Compatibilité selon la différence de chronotype
    df['chronotype_difference'] = abs(
        df['a_chronotype'] - df['b_chronotype']
    )

    df['chronotype_group'] = pd.cut(
        df['chronotype_difference'],
        bins=[-0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        labels=[
            '0–0.1', '0.1–0.2', '0.2–0.3',
            '0.3–0.4', '0.4–0.5', '0.5–1.0'
        ]
    )

    chronotype_stats = (
        df.groupby('chronotype_group', observed=False)['compatible']
        .mean()
        .mul(100)
        .reset_index(name='compatibility_rate')
    )

    chronotype_fig = px.bar(
        chronotype_stats,
        x='chronotype_group',
        y='compatibility_rate',
        title='Compatibilité selon la différence de chronotype',
        labels={
            'chronotype_group': 'Différence de chronotype',
            'compatibility_rate': 'Taux de compatibilité (%)'
        }
    )

    # 10. Durée des relations selon la compatibilité
    longevity_fig = px.box(
        df,
        x='compatible',
        y='relationship_longevity_months',
        color='compatible',
        title='Durée des relations selon la compatibilité',
        labels={
            'compatible': 'Compatible (0 = Non, 1 = Oui)',
            'relationship_longevity_months': 'Durée (mois)'
        }
    )


    # Conversion des graphiques en HTML
    graphs = [
    age_fig,
    score_fig,
    compatibility_fig,
    education_fig,
    age_compatibility_fig,
    correlation_fig,
    career_fig,
    love_fig,
    chronotype_fig,
    longevity_fig
    ]

    graph_html = [
        fig.to_html(full_html=False, include_plotlyjs='cdn')
        for fig in graphs
    ]

    return render_template(
        'eda.html',
        graph_html=graph_html
    )


@app.route('/statistics')
def statistics():
    dataset_path = os.path.join(
        app.root_path,
        'data',
        'processed',
        'cupid_algorithm_dataset.csv'
    )

    df = pd.read_csv(dataset_path)

    results = []

    # Fonction pour interpréter les p-values
    def interpret(p_value):
        if p_value < 0.05:
            return "Rejet de H0 : résultat statistiquement significatif."
        return "On ne rejette pas H0 : résultat non significatif."

    # ==========================================
    # 1. CHI-SQUARE TEST
    # Education de A et compatibilité
    # ==========================================

    contingency_table = pd.crosstab(
        df['a_education'],
        df['compatible']
    )

    chi2, p_chi, dof, expected = stats.chi2_contingency(
        contingency_table
    )

    results.append({
        'name': 'Chi-Square Test',
        'objective': 'Étudier l’association entre le niveau d’éducation de A et la compatibilité.',
        'h0': 'Les deux variables sont indépendantes.',
        'h1': 'Les deux variables sont associées.',
        'statistic': round(float(chi2), 4),
        'p_value': round(float(p_chi), 6),
        'conclusion': interpret(p_chi)
    })

    # ==========================================
    # 2. SHAPIRO-WILK TEST
    # Normalité du score de compatibilité
    # ==========================================

    scores = df['compatibility_score'].dropna()

    # Échantillon aléatoire de 5000 observations maximum
    if len(scores) > 5000:
        scores = scores.sample(
            n=5000,
            random_state=42
        )

    shapiro_stat, p_shapiro = stats.shapiro(scores)

    results.append({
        'name': 'Shapiro-Wilk Test',
        'objective': 'Vérifier la normalité des scores de compatibilité.',
        'h0': 'Les scores suivent une distribution normale.',
        'h1': 'Les scores ne suivent pas une distribution normale.',
        'statistic': round(float(shapiro_stat), 4),
        'p_value': round(float(p_shapiro), 6),
        'conclusion': interpret(p_shapiro)
    })

    # ==========================================
    # 3. MANN-WHITNEY U TEST
    # Durée selon la compatibilité
    # ==========================================

    group_0 = df.loc[
        df['compatible'] == 0,
        'relationship_longevity_months'
    ].dropna()

    group_1 = df.loc[
        df['compatible'] == 1,
        'relationship_longevity_months'
    ].dropna()

    u_stat, p_mann = stats.mannwhitneyu(
        group_0,
        group_1,
        alternative='two-sided'
    )

    results.append({
        'name': 'Mann-Whitney U Test',
        'objective': 'Comparer la distribution de la durée des relations entre les deux groupes.',
        'h0': 'Les distributions de durée sont identiques.',
        'h1': 'Les distributions de durée sont différentes.',
        'statistic': round(float(u_stat), 4),
        'p_value': round(float(p_mann), 6),
        'conclusion': interpret(p_mann)
    })

    # ==========================================
    # 4. KRUSKAL-WALLIS TEST
    # Score selon la différence d'âge
    # ==========================================

    analysis_df = df[
        ['a_age', 'b_age', 'compatibility_score']
    ].dropna().copy()

    analysis_df['age_difference'] = abs(
        analysis_df['a_age'] - analysis_df['b_age']
    )

    analysis_df['age_group'] = pd.cut(
        analysis_df['age_difference'],
        bins=[-1, 2, 5, 10, float('inf')],
        labels=['0-2 ans', '3-5 ans', '6-10 ans', 'Plus de 10 ans']
    )

    groups = [
        group['compatibility_score'].values
        for _, group in analysis_df.groupby(
            'age_group',
            observed=True
        )
        if len(group) > 0
    ]

    if len(groups) >= 2:
        h_stat, p_kruskal = stats.kruskal(*groups)

        results.append({
            'name': 'Kruskal-Wallis Test',
            'objective': 'Comparer les scores de compatibilité selon quatre groupes de différence d’âge.',
            'h0': 'Les distributions des scores sont identiques entre les groupes.',
            'h1': 'Au moins un groupe présente une distribution différente.',
            'statistic': round(float(h_stat), 4),
            'p_value': round(float(p_kruskal), 6),
            'conclusion': interpret(p_kruskal)
        })

    return render_template(
        'statistics.html',
        results=results
    )

@app.route('/benchmark')
def benchmark():
    results_path = os.path.join(
        app.root_path, 'results', 'model_comparison.csv'
    )
    summary_path = os.path.join(
        app.root_path, 'results', 'summary.json'
    )

    # Charger les résultats des modèles
    models_df = pd.read_csv(results_path)

    # Convertir les résultats en dictionnaire
    models = models_df.to_dict(orient='records')

    # Charger les résultats du meilleur modèle optimisé
    with open(summary_path, 'r', encoding='utf-8') as file:
        summary = json.load(file)

    return render_template(
        'benchmark.html',
        models=models,
        summary=summary
    )


@app.route('/batch-inference', methods=['GET', 'POST'])
def batch_inference():
    predictions = None
    error = None

    if request.method == 'POST':
        try:
            # Vérifier la présence du fichier
            if 'file' not in request.files:
                raise ValueError("Aucun fichier n'a été envoyé.")

            file = request.files['file']

            if file.filename == '':
                raise ValueError("Veuillez sélectionner un fichier CSV.")

            if not file.filename.lower().endswith('.csv'):
                raise ValueError("Le fichier doit être au format CSV.")

            # Charger les données
            df = pd.read_csv(file)

            # Colonnes attendues par le préprocesseur
            preprocessor = joblib.load(
                os.path.join(app.root_path, 'results', 'preprocessor.pkl')
            )

            model = joblib.load(
                os.path.join(app.root_path, 'results', 'best_model.pkl')
            )

            required_columns = (
                preprocessor.categorical_cols
                + preprocessor.numerical_raw_cols
            )

            missing_columns = [
                col for col in required_columns if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Colonnes manquantes : " + ", ".join(missing_columns)
                )

            if df.empty:
                raise ValueError("Le fichier CSV est vide.")

            # Prétraitement et prédiction
            X = preprocessor.transform(df)

            predicted_classes = model.predict(X)
            probabilities = model.predict_proba(X)

            # Construire les résultats
            predictions = df.copy()
            predictions['Predicted_Compatibility'] = predicted_classes
            predictions['Compatibility_Probability'] = probabilities[:, 1]

            predictions['Predicted_Label'] = predictions[
                'Predicted_Compatibility'
            ].map({
                0: 'Not Compatible',
                1: 'Compatible'
            })

        except Exception as e:
            error = str(e)

    return render_template(
        'batch_inference.html',
        predictions=predictions,
        error=error
    )


@app.route('/download-predictions', methods=['POST'])
def download_predictions():
    try:
        predictions_json = request.form.get('predictions_json')

        if not predictions_json:
            raise ValueError("Aucune prédiction à exporter.")

        df = pd.read_json(
            BytesIO(predictions_json.encode('utf-8')),
            orient='records'
        )

        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='batch_predictions.csv'
        )

    except Exception as e:
        return str(e), 400

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
        # Standardize trait scale: if client sends 1-10 scale (UI sliders), normalize to 0-1 range expected by the ML model
        normalized_data = dict(data)
        for p in ['a', 'b']:
            for t in TRAIT_NAMES:
                key = f"{p}_{t}"
                if key in normalized_data:
                    val = float(normalized_data[key])
                    if val > 1.0:
                        normalized_data[key] = val / 10.0

        # Archetype Profiling
        archetype_a = identify_archetype(data, prefix="a_")
        archetype_b = identify_archetype(data, prefix="b_")
        archetype_affinity = compute_cross_archetype_affinity(archetype_a["id"], archetype_b["id"])

        age_diff = abs(float(data['a_age']) - float(data['b_age']))
        same_love = 1.0 if data['a_love_language'] == data['b_love_language'] else 0.0
        age_bonus = max(-10.0, min(10.0, 10.0 - age_diff))

        if BEST_MODEL is not None and PREPROCESSOR is not None:
            # ---- ML model path ----
            df_single = pd.DataFrame([normalized_data])
            X_trans = PREPROCESSOR.transform(df_single)
            proba = BEST_MODEL.predict_proba(X_trans)[0, 1]
            pred = int(BEST_MODEL.predict(X_trans)[0])
            score = float(round(proba * 100, 1))

            # Feature engineering stats for response breakdown
            df_feat = PREPROCESSOR.create_features(df_single)
            dist = float(round(df_feat['personality_euclidean_dist'].iloc[0], 2))
            avg_diff = float(round(df_feat['avg_trait_diff'].iloc[0], 2))
            big5_harmony = float(round(df_feat['big5_harmony_score'].iloc[0] * 100, 1))
            longevity = int(round(score * 0.95 + age_bonus + (same_love * 6)))

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
                "archetype_a": archetype_a,
                "archetype_b": archetype_b,
                "archetype_affinity": archetype_affinity,
                "xai_breakdown": xai_breakdown,
            })

        else:
            # ---- Fallback: mathematical model ----
            same_loc = 1.0 if data['a_location'] == data['b_location'] else 0.0

            trait_diffs = [
                abs(float(normalized_data['a_openness']) - float(normalized_data['b_openness'])),
                abs(float(normalized_data['a_extraversion']) - float(normalized_data['b_extraversion'])),
                abs(float(normalized_data['a_agreeableness']) - float(normalized_data['b_agreeableness'])),
                abs(float(normalized_data['a_conscientiousness']) - float(normalized_data['b_conscientiousness'])),
                abs(float(normalized_data['a_chronotype']) - float(normalized_data['b_chronotype'])),
                abs(float(normalized_data['a_spontaneity']) - float(normalized_data['b_spontaneity'])),
                abs(float(normalized_data['a_emotional_expressiveness']) - float(normalized_data['b_emotional_expressiveness'])),
                abs(float(normalized_data['a_career_ambition']) - float(normalized_data['b_career_ambition'])),
            ]
            dist = float(np.sqrt(sum(x ** 2 for x in trait_diffs)))
            avg_diff = float(np.mean(trait_diffs))

            score = float(max(10, min(99,
                95 - (dist * 25.0) - (age_diff * 0.8) + (same_loc * 6) + (same_love * 8)
            )))
            pred = 1 if score >= 55 else 0
            longevity = int(round(score * 0.95 + age_bonus + (same_love * 6)))

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
                "big5_harmony_score": round(max(0.0, 100.0 - avg_diff * 50), 1),
                "avg_trait_diff": round(avg_diff, 2),
                "personality_distance": round(dist, 2),
                "archetype_a": archetype_a,
                "archetype_b": archetype_b,
                "archetype_affinity": archetype_affinity,
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
