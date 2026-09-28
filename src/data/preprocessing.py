import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

class CupidDataPreprocessor:
    """
    Data preprocessor and feature engineering pipeline for the Cupid Matchmaking Dataset.
    """
    def __init__(self, target_col='compatible', test_size=0.2, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.encoder = None
        self.scaler = None
        self.feature_names = None
        self.categorical_cols = [
            'a_location', 'b_location',
            'a_career_field', 'b_career_field',
            'a_love_language', 'b_love_language'
        ]
        self.numerical_raw_cols = [
            'a_age', 'a_education', 'a_career_ambition',
            'a_openness', 'a_extraversion', 'a_agreeableness', 'a_conscientiousness',
            'a_chronotype', 'a_spontaneity', 'a_emotional_expressiveness',
            'b_age', 'b_education', 'b_career_ambition',
            'b_openness', 'b_extraversion', 'b_agreeableness', 'b_conscientiousness',
            'b_chronotype', 'b_spontaneity', 'b_emotional_expressiveness'
        ]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific matchmaking and compatibility features.
        """
        df_feat = df.copy()

        # 1. Demographic & Categorical Match Features
        df_feat['age_diff'] = (df_feat['a_age'] - df_feat['b_age']).abs()
        df_feat['education_diff'] = (df_feat['a_education'] - df_feat['b_education']).abs()
        df_feat['same_location'] = (df_feat['a_location'] == df_feat['b_location']).astype(int)
        df_feat['same_career_field'] = (df_feat['a_career_field'] == df_feat['b_career_field']).astype(int)
        df_feat['same_love_language'] = (df_feat['a_love_language'] == df_feat['b_love_language']).astype(int)

        # 2. Personality Trait Differences & Ambition
        df_feat['diff_openness'] = (df_feat['a_openness'] - df_feat['b_openness']).abs()
        df_feat['diff_extraversion'] = (df_feat['a_extraversion'] - df_feat['b_extraversion']).abs()
        df_feat['diff_agreeableness'] = (df_feat['a_agreeableness'] - df_feat['b_agreeableness']).abs()
        df_feat['diff_conscientiousness'] = (df_feat['a_conscientiousness'] - df_feat['b_conscientiousness']).abs()
        df_feat['diff_chronotype'] = (df_feat['a_chronotype'] - df_feat['b_chronotype']).abs()
        df_feat['diff_spontaneity'] = (df_feat['a_spontaneity'] - df_feat['b_spontaneity']).abs()
        df_feat['diff_emotional_expressiveness'] = (df_feat['a_emotional_expressiveness'] - df_feat['b_emotional_expressiveness']).abs()
        df_feat['diff_career_ambition'] = (df_feat['a_career_ambition'] - df_feat['b_career_ambition']).abs()

        # 3. Aggregate Personality Metrics
        trait_diff_cols = [
            'diff_openness', 'diff_extraversion', 'diff_agreeableness', 'diff_conscientiousness',
            'diff_chronotype', 'diff_spontaneity', 'diff_emotional_expressiveness', 'diff_career_ambition'
        ]
        
        # Euclidean distance across personality dimensions
        df_feat['personality_euclidean_dist'] = np.sqrt((df_feat[trait_diff_cols] ** 2).sum(axis=1))
        # Personality similarity index (inverse distance)
        df_feat['personality_similarity_index'] = 1.0 / (1.0 + df_feat['personality_euclidean_dist'])
        # Mean absolute trait discrepancy
        df_feat['avg_trait_diff'] = df_feat[trait_diff_cols].mean(axis=1)

        # 4. Big 5 Compatibility Index (Openness, Extraversion, Agreeableness, Conscientiousness)
        big5_diff_cols = ['diff_openness', 'diff_extraversion', 'diff_agreeableness', 'diff_conscientiousness']
        df_feat['big5_harmony_score'] = 1.0 / (1.0 + df_feat[big5_diff_cols].mean(axis=1))

        return df_feat

    def fit_transform(self, df: pd.DataFrame):
        """
        Fits the preprocessor on df and transforms features and target.
        """
        # Feature Engineering
        df_engineered = self.create_features(df)

        # Drop non-feature ID columns and extra targets if present
        cols_to_drop = ['pair_id', 'compatibility_score', 'relationship_longevity_months', self.target_col]
        cols_to_drop = [c for c in cols_to_drop if c in df_engineered.columns]

        X_raw = df_engineered.drop(columns=cols_to_drop)
        y = df_engineered[self.target_col].values if self.target_col in df_engineered.columns else None

        # Identify numeric columns including newly engineered ones
        engineered_numeric_cols = [
            'age_diff', 'education_diff', 'same_location', 'same_career_field', 'same_love_language',
            'diff_openness', 'diff_extraversion', 'diff_agreeableness', 'diff_conscientiousness',
            'diff_chronotype', 'diff_spontaneity', 'diff_emotional_expressiveness', 'diff_career_ambition',
            'personality_euclidean_dist', 'personality_similarity_index', 'avg_trait_diff', 'big5_harmony_score'
        ]
        numeric_cols = self.numerical_raw_cols + engineered_numeric_cols

        # Initialize ColumnTransformer
        self.encoder = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ]
        )

        X_transformed = self.encoder.fit_transform(X_raw)

        # Retrieve output feature names
        cat_encoder = self.encoder.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names = cat_feature_names + numeric_cols

        return X_transformed, y

    def transform(self, df: pd.DataFrame):
        """
        Transforms new input data using the fitted encoder & scaler.
        """
        df_engineered = self.create_features(df)
        cols_to_drop = ['pair_id', 'compatibility_score', 'relationship_longevity_months', self.target_col]
        cols_to_drop = [c for c in cols_to_drop if c in df_engineered.columns]
        X_raw = df_engineered.drop(columns=cols_to_drop)
        return self.encoder.transform(X_raw)

    def prepare_train_test(self, csv_filepath: str):
        """
        Loads CSV dataset, fits/transforms data, and splits into train and test sets.
        """
        df = pd.read_csv(csv_filepath)
        X, y = self.fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def save(self, filepath: str):
        """
        Save preprocessor object to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """
        Load preprocessor object from disk.
        """
        return joblib.load(filepath)
