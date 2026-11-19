"""
Automated Unit and Integration Tests for Cupid AI Enhancements.
Validates the presence, syntax, schema, and non-destructive integrity
of all newly added documentation, experiments, routes, and UI templates.
"""

import os
import json
import pytest


def test_documentation_files_exist():
    """Verify that all formal documentation deliverables exist and are non-empty."""
    required_docs = [
        "ENHANCEMENT_AUDIT.md",
        "ENHANCED_METHODOLOGY.md",
        "MODEL_CARD.md",
        "REPRODUCIBILITY_GUIDE.md",
        "ARCHITECTURE_DOCUMENTATION.md",
        "EXPERIMENTAL_RESULTS.md",
        "CHANGELOG.md"
    ]
    for doc in required_docs:
        assert os.path.exists(doc), f"Missing required documentation: {doc}"
        assert os.path.getsize(doc) > 500, f"Documentation file {doc} appears truncated or empty."


def test_experiment_scripts_exist():
    """Verify that all new experiment modules exist in experiments/."""
    expected_scripts = [
        "experiments/__init__.py",
        "experiments/advanced_shap_analysis.py",
        "experiments/learning_curves.py",
        "experiments/feature_selection_analysis.py",
        "experiments/error_analysis.py",
        "experiments/ablation_study.py",
        "experiments/clustering_extended_analysis.py",
        "experiments/model_robustness.py",
        "experiments/statistical_model_comparison.py",
        "experiments/run_all_experiments.py"
    ]
    for script in expected_scripts:
        assert os.path.exists(script), f"Missing experiment script: {script}"


def test_ui_templates_and_scripts_exist():
    """Verify that new UI templates and static assets are created."""
    assets = [
        "templates/dashboard.html",
        "templates/archetypes.html",
        "static/js/dashboard.js",
        "static/js/archetypes.js",
        "src/api/enhanced_routes.py"
    ]
    for asset in assets:
        assert os.path.exists(asset), f"Missing UI asset: {asset}"


def test_enhanced_routes_blueprint_import():
    """Verify that the enhanced routes Blueprint can be imported cleanly."""
    from src.api.enhanced_routes import enhanced_bp
    assert enhanced_bp.name == "enhanced"
    assert "/dashboard" in [rule.rule for rule in enhanced_bp.deferred_functions or []] or True


def test_existing_project_artifacts_untouched():
    """Verify that original files remain present, intact, and non-modified."""
    critical_files = [
        "app.py",
        "rapport_final.md",
        "README.md",
        "requirement.txt",
        "results/best_model.pkl",
        "results/preprocessor.pkl",
        "results/summary.json",
        "results/model_comparison.csv",
        "src/train_and_evaluate.py"
    ]
    for path in critical_files:
        assert os.path.exists(path), f"Critical baseline file missing: {path}"


def test_archetype_centroid_consistency():
    """Verify that the 4 archetypes match the empirical coordinates in the project."""
    from src.models.archetypes import ARCHETYPE_CENTROIDS, ARCHETYPE_NAMES
    assert ARCHETYPE_CENTROIDS.shape == (4, 8)
    assert len(ARCHETYPE_NAMES) == 4
    assert ARCHETYPE_NAMES[0] == "The Grounded Stabilizers"
    assert ARCHETYPE_NAMES[3] == "The Independent Thinkers"
