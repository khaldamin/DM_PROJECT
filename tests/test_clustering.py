import numpy as np
import pytest
from src.models.clustering import (
    CupidKMeansClusterer,
    CupidHierarchicalClusterer,
    CupidDBSCANClusterer,
    evaluate_clustering,
    compute_k_distance
)

@pytest.fixture
def cluster_data():
    np.random.seed(42)
    # Generate synthetic 2-cluster data
    c1 = np.random.randn(50, 3) + 3.0
    c2 = np.random.randn(50, 3) - 3.0
    return np.vstack([c1, c2])

def test_kmeans_fit_predict_clusters(cluster_data):
    km = CupidKMeansClusterer(n_clusters=2, random_state=42)
    labels = km.fit_predict(cluster_data)
    assert len(labels) == len(cluster_data)
    assert set(np.unique(labels)) == {0, 1}
    assert km.inertia_ > 0

def test_kmeans_inertia_decreases_with_k(cluster_data):
    km2 = CupidKMeansClusterer(n_clusters=2, random_state=42).fit(cluster_data)
    km4 = CupidKMeansClusterer(n_clusters=4, random_state=42).fit(cluster_data)
    assert km4.inertia_ < km2.inertia_

def test_hierarchical_fit_predict(cluster_data):
    hc = CupidHierarchicalClusterer(n_clusters=2, linkage_method='ward')
    labels = hc.fit_predict(cluster_data)
    assert len(labels) == len(cluster_data)
    assert set(np.unique(labels)) == {0, 1}

def test_dbscan_fit_predict_detects_clusters(cluster_data):
    db = CupidDBSCANClusterer(eps=1.5, min_samples=5)
    labels = db.fit_predict(cluster_data)
    assert len(labels) == len(cluster_data)
    # Non-noise clusters should exist
    assert len(set(labels) - {-1}) >= 1

def test_evaluate_clustering_metrics(cluster_data):
    km = CupidKMeansClusterer(n_clusters=2, random_state=42)
    labels = km.fit_predict(cluster_data)
    metrics = evaluate_clustering(cluster_data, labels, algorithm_name="KMeans_Test")
    assert "Silhouette_Score" in metrics
    assert "Davies_Bouldin_Index" in metrics
    assert "Calinski_Harabasz_Score" in metrics
    assert -1.0 <= metrics["Silhouette_Score"] <= 1.0

def test_compute_k_distance(cluster_data):
    distances = compute_k_distance(cluster_data, k=4)
    assert len(distances) == len(cluster_data)
    # Distances should be sorted in ascending order
    assert np.all(np.diff(distances) >= 0)
