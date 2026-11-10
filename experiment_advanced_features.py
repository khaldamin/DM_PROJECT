"""
experiment_advanced_features.py
-------------------------------
Tests additional engineered features to push Accuracy and ROC-AUC higher.
Zero modification to existing codebase.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.abspath('.'))

def build_advanced_features(df):
    df_feat = df.copy()

    # 1. Base differences
    df_feat['age_diff'] = (df_feat['a_age'] - df_feat['b_age']).abs()
    df_feat['age_mean'] = (df_feat['a_age'] + df_feat['b_age']) / 2.0
    df_feat['education_diff'] = (df_feat['a_education'] - df_feat['b_education']).abs()
    df_feat['same_location'] = (df_feat['a_location'] == df_feat['b_location']).astype(int)
    df_feat['same_career_field'] = (df_feat['a_career_field'] == df_feat['b_career_field']).astype(int)
    df_feat['same_love_language'] = (df_feat['a_love_language'] == df_feat['b_love_language']).astype(int)

    # Cross categorical features (ordered pairs)
    df_feat['pair_love_language'] = df_feat.apply(
        lambda r: "_".join(sorted([str(r['a_love_language']), str(r['b_love_language'])])), axis=1
    )
    df_feat['pair_career_field'] = df_feat.apply(
        lambda r: "_".join(sorted([str(r['a_career_field']), str(r['b_career_field'])])), axis=1
    )

    traits = [
        'openness', 'extraversion', 'agreeableness', 'conscientiousness',
        'chronotype', 'spontaneity', 'emotional_expressiveness', 'career_ambition'
    ]

    for t in traits:
        df_feat[f'diff_{t}'] = (df_feat[f'a_{t}'] - df_feat[f'b_{t}']).abs()
        df_feat[f'prod_{t}'] = df_feat[f'a_{t}'] * df_feat[f'b_{t}']
        df_feat[f'mean_{t}'] = (df_feat[f'a_{t}'] + df_feat[f'b_{t}']) / 2.0

    trait_diff_cols = [f'diff_{t}' for t in traits]
    df_feat['personality_euclidean_dist'] = np.sqrt((df_feat[trait_diff_cols] ** 2).sum(axis=1))
    df_feat['personality_similarity_index'] = 1.0 / (1.0 + df_feat['personality_euclidean_dist'])
    df_feat['avg_trait_diff'] = df_feat[trait_diff_cols].mean(axis=1)

    big5_diffs = ['diff_openness', 'diff_extraversion', 'diff_agreeableness', 'diff_conscientiousness']
    df_feat['big5_harmony_score'] = 1.0 / (1.0 + df_feat[big5_diffs].mean(axis=1))

    return df_feat

def main():
    print("Loading data...")
    raw_df = pd.read_csv("data/raw/cupid_algorithm_dataset.csv")

    raw_train_df, raw_test_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df['compatible']
    )

    y_train = raw_train_df['compatible'].to_numpy()
    y_test = raw_test_df['compatible'].to_numpy()
    y_score_train = raw_train_df['compatibility_score'].to_numpy()

    print("Building advanced features...")
    feat_train = build_advanced_features(raw_train_df)
    feat_test = build_advanced_features(raw_test_df)

    cols_to_drop = ['pair_id', 'compatibility_score', 'relationship_longevity_months', 'compatible']
    cols_to_drop = [c for c in cols_to_drop if c in feat_train.columns]

    X_train_df = feat_train.drop(columns=cols_to_drop)
    X_test_df = feat_test.drop(columns=cols_to_drop)

    cat_cols = ['a_location', 'b_location', 'a_career_field', 'b_career_field',
                'a_love_language', 'b_love_language', 'pair_love_language', 'pair_career_field']
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    print("Preprocessing data matrix...")
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    print(f"Engineered feature shape: {X_train.shape}")

    # Train Tuned XGBoost with new features
    print("Training XGBoost Classifier with Advanced Features...")
    xgb_clf = XGBClassifier(
        n_estimators=220, max_depth=6, learning_rate=0.10,
        subsample=0.9, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train, y_train)
    probs = xgb_clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    print("\n" + "=" * 50)
    print(" ADVANCED FEATURE RESULTS (TEST SET)")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f} (Baseline: 0.7265)")
    print(f"Precision: {prec:.4f} (Baseline: 0.7042)")
    print(f"Recall   : {rec:.4f} (Baseline: 0.6242)")
    print(f"F1-Score : {f1:.4f} (Baseline: 0.6618)")
    print(f"ROC-AUC  : {auc:.4f} (Baseline: 0.7969)")

if __name__ == "__main__":
    main()

