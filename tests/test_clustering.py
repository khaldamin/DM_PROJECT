import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src.models.clustering import (
    CupidKMeansClusterer,
    CupidHierarchicalClusterer,
    CupidDBSCANClusterer,
    evaluate_clustering,
    compute_k_distance
)

@pytest.fixture
def synthetic_blobs():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)
    return X

def test_kmeans_clusterer_fit_predict(synthetic_blobs):
    km = CupidKMeansClusterer(n_clusters=3, random_state=42)
    labels = km.fit_predict(synthetic_blobs)
    
    assert len(labels) == len(synthetic_blobs)
    assert len(set(labels)) == 3
    assert km.inertia_ > 0
    assert km.cluster_centers_.shape == (3, 4)

def test_hierarchical_clusterer(synthetic_blobs):
    hier = CupidHierarchicalClusterer(n_clusters=3, linkage_method='ward')
    labels = hier.fit_predict(synthetic_blobs)
    
    assert len(labels) == len(synthetic_blobs)
    assert len(set(labels)) == 3

def test_hierarchical_linkage_matrix(synthetic_blobs):
    linkage_matrix = CupidHierarchicalClusterer.compute_linkage_matrix(synthetic_blobs[:50], method='ward')
    assert linkage_matrix.shape == (49, 4)

def test_dbscan_clusterer(synthetic_blobs):
    dbscan = CupidDBSCANClusterer(eps=1.5, min_samples=5)
    labels = dbscan.fit_predict(synthetic_blobs)
    
    assert len(labels) == len(synthetic_blobs)
    assert dbscan.n_clusters_ >= 1
    assert dbscan.n_noise_ >= 0

def test_compute_k_distance(synthetic_blobs):
    k_dist = compute_k_distance(synthetic_blobs, k=5)
    assert len(k_dist) == len(synthetic_blobs)
    # Check monotonically increasing
    assert np.all(np.diff(k_dist) >= 0)

def test_evaluate_clustering_metrics(synthetic_blobs):
    km = CupidKMeansClusterer(n_clusters=3, random_state=42)
    labels = km.fit_predict(synthetic_blobs)
    
    metrics = evaluate_clustering(synthetic_blobs, labels, "Test-KMeans")
    assert "Silhouette_Score" in metrics
    assert "Davies_Bouldin_Index" in metrics
    assert "Calinski_Harabasz_Score" in metrics
    assert -1.0 <= metrics["Silhouette_Score"] <= 1.0
    assert metrics["Davies_Bouldin_Index"] >= 0.0
    assert metrics["Calinski_Harabasz_Score"] > 0.0

