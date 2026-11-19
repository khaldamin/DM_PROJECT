"""
Feature Engineering Ablation Study for Cupid AI.
Evaluates model performance degradation when systematically omitting
domain-specific interaction feature categories to quantify their marginal utility.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath('.'))

from src.models.xgboost_model import CupidXGBoost


def run_ablation_study():
    print("==================================================")
    print(" Cupid AI: Feature Engineering Ablation Study")
    print("==================================================")

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    fig_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df_raw = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df_raw, test_size=0.2, random_state=42, stratify=df_raw['compatible']
    )
    y_train = train_df['compatible'].to_numpy()
    y_test = test_df['compatible'].to_numpy()

    # Define Ablation Configurations
    ablation_configs = [
        ("Raw Profile Features Only", "raw_only"),
        ("+ Demographic & Categorical Match", "raw_plus_demographics"),
        ("+ Personality Trait Differences", "raw_plus_trait_diffs"),
        ("+ Psychological Distance & Big5 Harmony", "raw_plus_psychological"),
        ("Full Engineered Pipeline (Baseline)", "full_pipeline"),
        ("Without Big 5 Harmony Score", "without_big5"),
        ("Without Love Language Alignment", "without_love_language")
    ]

    categorical_cols = [
        'a_location', 'b_location',
        'a_career_field', 'b_career_field',
        'a_love_language', 'b_love_language'
    ]
    numerical_raw_cols = [
        'a_age', 'a_education', 'a_career_ambition',
        'a_openness', 'a_extraversion', 'a_agreeableness', 'a_conscientiousness',
        'a_chronotype', 'a_spontaneity', 'a_emotional_expressiveness',
        'b_age', 'b_education', 'b_career_ambition',
        'b_openness', 'b_extraversion', 'b_agreeableness', 'b_conscientiousness',
        'b_chronotype', 'b_spontaneity', 'b_emotional_expressiveness'
    ]

    def create_features_custom(df, mode):
        df_f = df.copy()
        num_cols = list(numerical_raw_cols)

        if mode == "raw_only":
            return df_f[categorical_cols + num_cols]

        # Demographic features
        if mode in ["raw_plus_demographics", "raw_plus_trait_diffs", "raw_plus_psychological", "full_pipeline", "without_big5"]:
            df_f['age_diff'] = (df_f['a_age'] - df_f['b_age']).abs()
            df_f['education_diff'] = (df_f['a_education'] - df_f['b_education']).abs()
            df_f['same_location'] = (df_f['a_location'] == df_f['b_location']).astype(int)
            df_f['same_career_field'] = (df_f['a_career_field'] == df_f['b_career_field']).astype(int)
            num_cols.extend(['age_diff', 'education_diff', 'same_location', 'same_career_field'])

        if mode != "without_love_language":
            df_f['same_love_language'] = (df_f['a_love_language'] == df_f['b_love_language']).astype(int)
            num_cols.append('same_love_language')

        # Trait diffs
        if mode in ["raw_plus_trait_diffs", "raw_plus_psychological", "full_pipeline", "without_big5", "without_love_language"]:
            trait_diff_cols = [
                'diff_openness', 'diff_extraversion', 'diff_agreeableness', 'diff_conscientiousness',
                'diff_chronotype', 'diff_spontaneity', 'diff_emotional_expressiveness', 'diff_career_ambition'
            ]
            for t in ['openness', 'extraversion', 'agreeableness', 'conscientiousness',
                      'chronotype', 'spontaneity', 'emotional_expressiveness', 'career_ambition']:
                df_f[f'diff_{t}'] = (df_f[f'a_{t}'] - df_f[f'b_{t}']).abs()
            num_cols.extend(trait_diff_cols)

        # Psychological composites
        if mode in ["raw_plus_psychological", "full_pipeline", "without_love_language"]:
            trait_diff_cols = [
                'diff_openness', 'diff_extraversion', 'diff_agreeableness', 'diff_conscientiousness',
                'diff_chronotype', 'diff_spontaneity', 'diff_emotional_expressiveness', 'diff_career_ambition'
            ]
            df_f['personality_euclidean_dist'] = np.sqrt((df_f[trait_diff_cols] ** 2).sum(axis=1))
            df_f['personality_similarity_index'] = 1.0 / (1.0 + df_f['personality_euclidean_dist'])
            df_f['avg_trait_diff'] = df_f[trait_diff_cols].mean(axis=1)
            num_cols.extend(['personality_euclidean_dist', 'personality_similarity_index', 'avg_trait_diff'])

        if mode in ["full_pipeline", "without_love_language"]:
            big5_diff_cols = ['diff_openness', 'diff_extraversion', 'diff_agreeableness', 'diff_conscientiousness']
            df_f['big5_harmony_score'] = 1.0 / (1.0 + df_f[big5_diff_cols].mean(axis=1))
            num_cols.append('big5_harmony_score')

        return df_f[categorical_cols + list(set(num_cols))]

    results = []
    print("\nEvaluating ablation configurations...")

    for label, mode in ablation_configs:
        # Preprocess with fold isolation
        X_tr_df = create_features_custom(train_df, mode)
        X_te_df = create_features_custom(test_df, mode)

        num_features = [c for c in X_tr_df.columns if c not in categorical_cols]

        transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), num_features)
            ]
        )

        X_train_trans = transformer.fit_transform(X_tr_df)
        X_test_trans = transformer.transform(X_te_df)

        model = CupidXGBoost(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
        )
        model.fit(X_train_trans, y_train)

        preds = model.predict(X_test_trans)
        probas = model.predict_proba(X_test_trans)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)

        results.append({
            "Configuration": label,
            "Feature_Count": X_train_trans.shape[1],
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        print(f"   -> {label:42s} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    df_ablation = pd.DataFrame(results)
    csv_path = os.path.join(tables_dir, "ablation_study_results.csv")
    df_ablation.to_csv(csv_path, index=False)
    print(f"\nSaved table to {csv_path}")

    # Plot Ablation Bar Chart
    fig_path = os.path.join(fig_dir, "ablation_study.png")
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(df_ablation))
    plt.barh(y_pos, df_ablation["ROC-AUC"], color='#34495e', alpha=0.85, height=0.55)
    plt.yticks(y_pos, df_ablation["Configuration"], fontsize=10)
    plt.xlim(0.70, 0.81)
    plt.axvline(x=df_ablation.loc[df_ablation['Configuration'] == 'Full Engineered Pipeline (Baseline)', 'ROC-AUC'].values[0],
                color='#e74c3c', linestyle='--', label='Baseline Performance')
    plt.title("Feature Ablation Study: Marginal ROC-AUC Contribution", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Hold-out ROC-AUC", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")

    print("\nAblation study complete!")


if __name__ == '__main__':
    run_ablation_study()
