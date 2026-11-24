"""
Clustering and Unsupervised Learning module for Cupid Matchmaking.
Implements K-Means, Hierarchical Agglomerative Clustering, and DBSCAN
along with intrinsic evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import os


class CupidKMeansClusterer:
    """
    Partition-based K-Means clustering for profile personality segmentation.
    """
    def __init__(self, n_clusters: int = 4, init: str = 'k-means++', n_init: int = 10, max_iter: int = 300, random_state: int = 42):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray) -> 'CupidKMeansClusterer':
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


class CupidHierarchicalClusterer:
    """
    Agglomerative Hierarchical clustering for dendrogram tree exploration.
    """
    def __init__(self, n_clusters: int = 4, metric: str = 'euclidean', linkage_method: str = 'ward'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage_method = linkage_method
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.metric,
            linkage=self.linkage_method
        )
        self.labels_ = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    @staticmethod
    def compute_linkage_matrix(X: np.ndarray, method: str = 'ward', metric: str = 'euclidean') -> np.ndarray:
        """Computes linkage matrix for scipy dendrogram."""
        return linkage(X, method=method, metric=metric)


class CupidDBSCANClusterer:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Detects arbitrary shaped clusters and flags outliers / noise (-1).
    """
    def __init__(self, eps: float = 1.2, min_samples: int = 15, metric: str = 'euclidean', n_jobs: int = -1):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=self.n_jobs
        )
        self.labels_ = None
        self.n_clusters_ = 0
        self.n_noise_ = 0

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.labels_ = self.model.fit_predict(X)
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels - {-1})
        self.n_noise_ = int(np.sum(self.labels_ == -1))
        return self.labels_


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, algorithm_name: str = "Clustering", sample_size: int = 10000, random_state: int = 42) -> Dict[str, Any]:
    """
    Computes intrinsic clustering quality metrics:
    - Silhouette Score (in [-1, 1], higher is better)
    - Davies-Bouldin Index (>= 0, lower is better)
    - Calinski-Harabasz Score (higher is better)
    Handles noise labels (-1) properly by evaluating non-noise points.
    """
    non_noise_mask = (labels != -1)
    unique_clusters = set(labels[non_noise_mask])
    
    n_total = len(labels)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = float(n_noise / n_total) if n_total > 0 else 0.0

    if len(unique_clusters) < 2:
        return {
            "Algorithm": algorithm_name,
            "N_Clusters": len(unique_clusters),
            "N_Noise": n_noise,
            "Noise_Ratio": round(noise_ratio, 4),
            "Silhouette_Score": np.nan,
            "Davies_Bouldin_Index": np.nan,
            "Calinski_Harabasz_Score": np.nan
        }

    X_eval = X[non_noise_mask]
    labels_eval = labels[non_noise_mask]

    # Subsample if dataset is large for computational speed in silhouette calculation
    if len(X_eval) > sample_size:
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(X_eval), size=sample_size, replace=False)
        sil_score = float(silhouette_score(X_eval[sample_indices], labels_eval[sample_indices]))
    else:
        sil_score = float(silhouette_score(X_eval, labels_eval))

    db_score = float(davies_bouldin_score(X_eval, labels_eval))
    ch_score = float(calinski_harabasz_score(X_eval, labels_eval))

    return {
        "Algorithm": algorithm_name,
        "N_Clusters": len(unique_clusters),
        "N_Noise": n_noise,
        "Noise_Ratio": round(noise_ratio, 4),
        "Silhouette_Score": round(sil_score, 4),
        "Davies_Bouldin_Index": round(db_score, 4),
        "Calinski_Harabasz_Score": round(ch_score, 2)
    }


def compute_k_distance(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Computes sorted k-distance array to help determine optimal eps for DBSCAN via the elbow/knee.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_distances = np.sort(distances[:, k - 1])
    return k_distances

