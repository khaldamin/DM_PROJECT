import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }

notebooks_dir = "notebooks"
os.makedirs(notebooks_dir, exist_ok=True)

# 01_data_understanding_eda.ipynb
nb1_cells = [
    make_markdown_cell("# 01. Data Understanding & Exploratory Data Analysis (EDA)\n\nThis notebook analyzes the Cupid Matchmaking Dataset (100,000 profile pairs), checking target distributions, feature correlations, and demographic patterns."),
    make_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n%matplotlib inline\nplt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')"),
    make_code_cell("df = pd.read_csv('../data/raw/cupid_algorithm_dataset.csv')\nprint(f'Dataset Shape: {df.shape}')\ndf.head()"),
    make_code_cell("df.info()"),
    make_code_cell("df.describe().T"),
    make_markdown_cell("## Target Variable Distribution"),
    make_code_cell("plt.figure(figsize=(6, 4))\nsns.countplot(data=df, x='compatible', palette='Set2')\nplt.title('Target Variable Distribution (compatible)')\nplt.xlabel('Compatible (0 = No, 1 = Yes)')\nplt.ylabel('Count')\nplt.show()\n\nprint(df['compatible'].value_counts(normalize=True))"),
    make_markdown_cell("## Relationship Longevity vs Compatibility"),
    make_code_cell("plt.figure(figsize=(8, 5))\nsns.boxplot(data=df, x='compatible', y='relationship_longevity_months', palette='Set2')\nplt.title('Relationship Longevity (Months) by Compatibility')\nplt.show()"),
    make_markdown_cell("## Feature Correlations with Target"),
    make_code_cell("numeric_df = df.select_dtypes(include=[np.number])\ncorrs = numeric_df.corr()['compatible'].sort_values(ascending=False)\nprint('Top Correlations with Target (compatible):')\nprint(corrs)")
]

with open(os.path.join(notebooks_dir, "01_data_understanding_eda.ipynb"), "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# 02_preprocessing.ipynb
nb2_cells = [
    make_markdown_cell("# 02. Preprocessing & Feature Engineering Pipeline\n\nThis notebook demonstrates feature engineering (demographic matches, personality trait distances, Big 5 harmony) and scaling."),
    make_code_cell("import pandas as pd\nimport numpy as np\nimport sys\nsys.path.append('..')\nfrom src.data.preprocessing import CupidDataPreprocessor"),
    make_code_cell("df = pd.read_csv('../data/raw/cupid_algorithm_dataset.csv')\npreprocessor = CupidDataPreprocessor(test_size=0.2, random_state=42)\nX_train, X_test, y_train, y_test = preprocessor.prepare_train_test('../data/raw/cupid_algorithm_dataset.csv')\nprint(f'Engineered Feature Count: {X_train.shape[1]}')\nprint(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')")
]

with open(os.path.join(notebooks_dir, "02_preprocessing.ipynb"), "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# 03_logistic_regression.ipynb
nb3_cells = [
    make_markdown_cell("# 03. Logistic Regression Classifier\n\nBaseline linear classification model for relationship compatibility."),
    make_code_cell("import sys\nsys.path.append('..')\nfrom src.data.preprocessing import CupidDataPreprocessor\nfrom src.models.logistic_regression import CupidLogisticRegression\nfrom src.evluation.metrics import evaluate_classification_model, print_evaluation_summary"),
    make_code_cell("preprocessor = CupidDataPreprocessor()\nX_train, X_test, y_train, y_test = preprocessor.prepare_train_test('../data/raw/cupid_algorithm_dataset.csv')"),
    make_code_cell("model = CupidLogisticRegression(max_iter=1000, random_state=42)\nmodel.fit(X_train, y_train)\nmetrics = evaluate_classification_model(model, X_test, y_test, 'Logistic Regression')\nprint_evaluation_summary(metrics)")
]

with open(os.path.join(notebooks_dir, "03_logistic_regression.ipynb"), "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

# 04_decision_tree.ipynb
nb4_cells = [
    make_markdown_cell("# 04. Decision Tree Classifier\n\nNon-linear decision tree classifier for matchmaking."),
    make_code_cell("import sys\nsys.path.append('..')\nfrom src.data.preprocessing import CupidDataPreprocessor\nfrom src.models.decision_tree import CupidDecisionTree\nfrom src.evluation.metrics import evaluate_classification_model, print_evaluation_summary"),
    make_code_cell("preprocessor = CupidDataPreprocessor()\nX_train, X_test, y_train, y_test = preprocessor.prepare_train_test('../data/raw/cupid_algorithm_dataset.csv')"),
    make_code_cell("model = CupidDecisionTree(max_depth=10, random_state=42)\nmodel.fit(X_train, y_train)\nmetrics = evaluate_classification_model(model, X_test, y_test, 'Decision Tree')\nprint_evaluation_summary(metrics)")
]

with open(os.path.join(notebooks_dir, "04_decision_tree.ipynb"), "w") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

# 05_random_forest.ipynb
nb5_cells = [
    make_markdown_cell("# 05. Random Forest Ensemble Classifier\n\nEnsemble bagger tree classifier for relationship compatibility."),
    make_code_cell("import sys\nsys.path.append('..')\nfrom src.data.preprocessing import CupidDataPreprocessor\nfrom src.models.random_forest import CupidRandomForest\nfrom src.evluation.metrics import evaluate_classification_model, print_evaluation_summary"),
    make_code_cell("preprocessor = CupidDataPreprocessor()\nX_train, X_test, y_train, y_test = preprocessor.prepare_train_test('../data/raw/cupid_algorithm_dataset.csv')"),
    make_code_cell("model = CupidRandomForest(n_estimators=100, max_depth=12, random_state=42)\nmodel.fit(X_train, y_train)\nmetrics = evaluate_classification_model(model, X_test, y_test, 'Random Forest')\nprint_evaluation_summary(metrics)")
]

with open(os.path.join(notebooks_dir, "05_random_forest.ipynb"), "w") as f:
    json.dump(make_notebook(nb5_cells), f, indent=2)

# 06_xgboost.ipynb
nb6_cells = [
    make_markdown_cell("# 06. XGBoost Gradient Boosting Classifier\n\nState-of-the-art gradient boosted decision tree classifier."),
    make_code_cell("import sys\nsys.path.append('..')\nfrom src.data.preprocessing import CupidDataPreprocessor\nfrom src.models.xgboost_model import CupidXGBoost\nfrom src.evluation.metrics import evaluate_classification_model, print_evaluation_summary"),
    make_code_cell("preprocessor = CupidDataPreprocessor()\nX_train, X_test, y_train, y_test = preprocessor.prepare_train_test('../data/raw/cupid_algorithm_dataset.csv')"),
    make_code_cell("model = CupidXGBoost(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42)\nmodel.fit(X_train, y_train)\nmetrics = evaluate_classification_model(model, X_test, y_test, 'XGBoost')\nprint_evaluation_summary(metrics)")
]

with open(os.path.join(notebooks_dir, "06_xgboost.ipynb"), "w") as f:
    json.dump(make_notebook(nb6_cells), f, indent=2)

# 07_evaluation_tuning.ipynb
nb7_cells = [
    make_markdown_cell("# 07. Model Comparison & Hyperparameter Tuning\n\nCompares performance metrics across all models and saves the best model."),
    make_code_cell("import pandas as pd\nimport sys\nsys.path.append('..')\nfrom src.train_and_evaluate import main\nmain()")
]

with open(os.path.join(notebooks_dir, "07_evaluation_tuning.ipynb"), "w") as f:
    json.dump(make_notebook(nb7_cells), f, indent=2)

print("Successfully generated all 7 Jupyter notebooks in 'notebooks/'!")
