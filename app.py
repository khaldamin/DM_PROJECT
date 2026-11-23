import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from src.data.preprocessing import CupidDataPreprocessor
from src.models import CupidXGBoost, BaseModel
import plotly.express as px
from scipy import stats
import json
import joblib
from flask import request, send_file
from io import BytesIO

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
