"""
Execution script for Clustering & Unsupervised Learning on Cupid Matchmaking.
Implements:
1. K-Means with Elbow Method and Silhouette Analysis.
2. Hierarchical Agglomerative Clustering with Dendrogram visualization.
3. DBSCAN density-based clustering with noise/outlier detection.
4. Intrinsic evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).
5. PCA 2D visualizations and personality archetype profiling.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Ensure local imports work
sys.path.insert(0, os.path.abspath('.'))
from src.models.clustering import (
    CupidKMeansClusterer,
    CupidHierarchicalClusterer,
    CupidDBSCANClusterer,
    evaluate_clustering,
    compute_k_distance
)
from src.models.archetypes import ARCHETYPE_FULL_LABELS

def run_clustering_pipeline():
    print("=" * 65)
    print("  CUPID MATCHMAKING: CLUSTERING & UNSUPERVISED LEARNING PIPELINE")
    print("=" * 65)

    figures_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Data Loading & Feature Selection for Segmentation
    # -------------------------------------------------------------
    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    print(f"\n[1/6] Loading Cupid dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} columns.")

    # We extract individual personality and behavioral traits for profile segmentation
    # Matching the Customer/User Segmentation objective
    profile_features = [
        'openness', 'extraversion', 'agreeableness', 'conscientiousness',
        'chronotype', 'spontaneity', 'emotional_expressiveness', 'career_ambition'
    ]
    
    # We aggregate Partner A profiles (sample of 15,000 for high precision & fast clustering)
    a_cols = [f'a_{feat}' for feat in profile_features]
    df_sample = df[a_cols].sample(n=15000, random_state=42).copy()
    df_sample.columns = profile_features

    # Standardize traits for distance-based clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample)
    print(f"Feature matrix prepared: {X_scaled.shape[0]:,} profiles across {len(profile_features)} traits.")

    # -------------------------------------------------------------
    # 2. K-Means: Elbow Method & Silhouette Analysis
    # -------------------------------------------------------------
    print("\n[2/6] Running K-Means: Elbow Method (Inertia) & Silhouette Analysis...")
    k_range = list(range(2, 9))
    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = CupidKMeansClusterer(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score on a representative subsample for speed
        sil = evaluate_clustering(X_scaled, kmeans.labels_, sample_size=4000, random_state=42)["Silhouette_Score"]
        silhouette_scores.append(sil)
        print(f"  k={k} -> Inertia (WCSS): {kmeans.inertia_:10.2f} | Silhouette Score: {sil:.4f}")

    # Plot Elbow and Silhouette curves side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, marker='o', color='#3498db', linewidth=2.5, markersize=8)
    ax1.set_title("Elbow Method for Optimal k (WCSS / Inertia)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=11)
    ax1.set_xticks(k_range)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(k_range, silhouette_scores, marker='s', color='#e74c3c', linewidth=2.5, markersize=8)
    ax2.set_title("Silhouette Score vs Number of Clusters", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Average Silhouette Coefficient", fontsize=11)
    ax2.set_xticks(k_range)
    ax2.grid(True, linestyle='--', alpha=0.6)

    elbow_fig_path = os.path.join(figures_dir, "clustering_kmeans_elbow_silhouette.png")
    plt.tight_layout()
    plt.savefig(elbow_fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved Elbow & Silhouette plot to {elbow_fig_path}")

    # Optimal K selection (k=4 represents 4 distinct personality archetypes)
    optimal_k = 4
    best_kmeans = CupidKMeansClusterer(n_clusters=optimal_k, random_state=42)
    kmeans_labels = best_kmeans.fit_predict(X_scaled)
    kmeans_metrics = evaluate_clustering(X_scaled, kmeans_labels, f"K-Means (k={optimal_k})", sample_size=8000)

    # -------------------------------------------------------------
    # 3. Hierarchical Agglomerative Clustering & Dendrogram
    # -------------------------------------------------------------
    print("\n[3/6] Running Hierarchical Clustering & Computing Dendrogram...")
    # Generate dendrogram on a subsample of 150 points for visual clarity
    dendro_sample = X_scaled[:150]
    linkage_matrix = CupidHierarchicalClusterer.compute_linkage_matrix(dendro_sample, method='ward')

    plt.figure(figsize=(12, 6))
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=14, fontweight='bold')
    plt.xlabel("Sample Index / Cluster Size", fontsize=11)
    plt.ylabel("Euclidean Distance (Ward Dissimilarity)", fontsize=11)
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=25,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True
    )
    plt.axhline(y=14.0, color='r', linestyle='--', linewidth=2, label='Optimal Cut Threshold (k=4)')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    dendro_fig_path = os.path.join(figures_dir, "clustering_dendrogram.png")
    plt.tight_layout()
    plt.savefig(dendro_fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved Dendrogram plot to {dendro_fig_path}")

    # Fit Hierarchical on the dataset sample (sample 5000 for O(N^2) memory tractability)
    hierarchical_sample_size = 5000
    X_hier = X_scaled[:hierarchical_sample_size]
    hierarchical = CupidHierarchicalClusterer(n_clusters=optimal_k, linkage_method='ward')
    hier_labels = hierarchical.fit_predict(X_hier)
    hier_metrics = evaluate_clustering(X_hier, hier_labels, f"Hierarchical Ward (k={optimal_k})")

    # -------------------------------------------------------------
    # 4. DBSCAN: Density-Based Clustering & Noise Detection
    # -------------------------------------------------------------
    print("\n[4/6] Running DBSCAN: k-distance Knee Analysis & Noise Detection...")
    # Compute k-distance for knee identification
    k_dist = compute_k_distance(X_scaled[:5000], k=5)

    plt.figure(figsize=(8, 5))
    plt.plot(k_dist, color='#2c3e50', linewidth=2)
    plt.axhline(y=1.85, color='#e67e22', linestyle='--', linewidth=2, label='Selected Epsilon (eps=1.85)')
    plt.title("k-Distance Graph for DBSCAN Epsilon (eps) Selection", fontsize=13, fontweight='bold')
    plt.xlabel("Points Sorted by Distance to 5th Nearest Neighbor", fontsize=11)
    plt.ylabel("5-NN Distance", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    kdist_fig_path = os.path.join(figures_dir, "clustering_dbscan_kdistance.png")
    plt.tight_layout()
    plt.savefig(kdist_fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved k-distance plot to {kdist_fig_path}")

    # Fit DBSCAN
    dbscan = CupidDBSCANClusterer(eps=1.85, min_samples=25)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_metrics = evaluate_clustering(X_scaled, dbscan_labels, f"DBSCAN (eps={dbscan.eps})", sample_size=8000)

    print(f"  DBSCAN Results: {dbscan.n_clusters_} dense clusters found, {dbscan.n_noise_:,} noise/outlier points detected ({dbscan.n_noise_/len(X_scaled)*100:.1f}%).")

    # -------------------------------------------------------------
    # 5. PCA 2D Visualization & Comparison
    # -------------------------------------------------------------
    print("\n[5/6] Generating PCA 2D Projections for Cluster Visualizations...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    print(f"  PCA Explained Variance: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}% (Total={sum(var_exp)*100:.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # K-Means PCA
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5, s=15)
    # Centers
    centers_pca = pca.transform(best_kmeans.cluster_centers_)
    axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidth=1.5, label='Centroids')
    axes[0].set_title(f"K-Means (k={optimal_k})\nSilhouette: {kmeans_metrics['Silhouette_Score']}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # Hierarchical PCA (on its sample)
    X_hier_pca = X_pca[:hierarchical_sample_size]
    scatter2 = axes[1].scatter(X_hier_pca[:, 0], X_hier_pca[:, 1], c=hier_labels, cmap='tab10', alpha=0.5, s=15)
    axes[1].set_title(f"Hierarchical Ward (k={optimal_k})\nSilhouette: {hier_metrics['Silhouette_Score']}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)")
    axes[1].grid(True, linestyle='--', alpha=0.4)

    # DBSCAN PCA (with noise points in grey)
    import matplotlib.colors as mcolors
    colors = ['#95a5a6' if label == -1 else mcolors.to_hex(plt.cm.tab10(label % 10)) for label in dbscan_labels]
    axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=15)
    axes[2].set_title(f"DBSCAN (eps={dbscan.eps})\nClusters: {dbscan.n_clusters_} | Outliers: {dbscan.n_noise_}", fontsize=12, fontweight='bold')
    axes[2].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)")
    axes[2].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)")
    axes[2].grid(True, linestyle='--', alpha=0.4)

    pca_fig_path = os.path.join(figures_dir, "clustering_pca_comparison.png")
    plt.tight_layout()
    plt.savefig(pca_fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved 2D PCA cluster comparison to {pca_fig_path}")

    # -------------------------------------------------------------
    # 6. Profiling Personality Archetypes & Results Export
    # -------------------------------------------------------------
    print("\n[6/6] Profiling Personality Archetypes & Exporting Results...")
    df_sample['Cluster_KMeans'] = kmeans_labels
    cluster_profiles = df_sample.groupby('Cluster_KMeans')[profile_features].mean().round(2)
    cluster_sizes = df_sample['Cluster_KMeans'].value_counts().sort_index()
    cluster_profiles['Profile_Count'] = cluster_sizes
    cluster_profiles['Percentage'] = (cluster_sizes / len(df_sample) * 100).round(1)

    archetype_names = ARCHETYPE_FULL_LABELS
    cluster_profiles['Archetype_Name'] = [archetype_names.get(i, f"Archetype {i}") for i in cluster_profiles.index]

    # Reorder columns
    cols = ['Archetype_Name', 'Profile_Count', 'Percentage'] + profile_features
    cluster_profiles = cluster_profiles[cols]
    archetypes_csv_path = os.path.join(tables_dir, "cluster_archetypes.csv")
    cluster_profiles.to_csv(archetypes_csv_path, index=True)
    print(f"  -> Saved Cluster Archetypes table to {archetypes_csv_path}")

    # Comparison metrics table
    comparison_df = pd.DataFrame([kmeans_metrics, hier_metrics, dbscan_metrics])
    comparison_csv_path = os.path.join(tables_dir, "clustering_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"  -> Saved Clustering Comparison table to {comparison_csv_path}")

    print("\n" + "=" * 65)
    print("  CLUSTERING EVALUATION BENCHMARK:")
    print("=" * 65)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("  PERSONALITY ARCHETYPES SUMMARY:")
    print("=" * 65)
    print(cluster_profiles[['Archetype_Name', 'Profile_Count', 'Percentage']].to_string())

    print("\nClustering pipeline executed successfully! All figures & tables ready.")

if __name__ == "__main__":
    run_clustering_pipeline()
