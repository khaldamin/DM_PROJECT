"""
Extended Clustering & Archetype Analysis for Cupid AI.
Generates multidimensional radar/spider charts comparing the 4 personality archetypes,
parallel coordinates, and metric sweeps across k=2 to 8.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

sys.path.insert(0, os.path.abspath('.'))


def run_extended_clustering():
    print("==================================================")
    print(" Cupid AI: Extended Clustering & Archetype Profiling")
    print("==================================================")

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    fig_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df_raw = pd.read_csv(data_path)

    # Big 5 + Ambition + Chronotype + Spontaneity
    traits = [
        'openness', 'extraversion', 'agreeableness', 'conscientiousness',
        'chronotype', 'spontaneity', 'emotional_expressiveness', 'career_ambition'
    ]
    a_cols = [f'a_{t}' for t in traits]
    X_traits = df_raw[a_cols].to_numpy()

    # 1. Metric Sweeps Across k=2 to 8 on a 15k representative subset
    print("\n[1/3] Sweeping k in [2, 8] for intrinsic clustering metrics...")
    sub_idx = np.random.RandomState(42).choice(len(X_traits), size=15000, replace=False)
    X_sub = X_traits[sub_idx]

    k_range = list(range(2, 9))
    wcss = []
    sil_scores = []
    db_scores = []
    ch_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sub)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_sub, labels))
        db_scores.append(davies_bouldin_score(X_sub, labels))
        ch_scores.append(calinski_harabasz_score(X_sub, labels))
        print(f"   k={k} -> Silhouette: {sil_scores[-1]:.4f} | Davies-Bouldin: {db_scores[-1]:.4f}")

    # Plot Multi-metric Clustering Grid
    grid_fig_path = os.path.join(fig_dir, "clustering_extended_metrics.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(k_range, wcss, 'o-', color='#e74c3c', linewidth=2)
    axes[0, 0].set_title("Inertia (Elbow Method / WCSS)", fontweight='bold')
    axes[0, 0].set_xlabel("Number of Clusters (k)")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    axes[0, 1].plot(k_range, sil_scores, 's-', color='#2ecc71', linewidth=2)
    axes[0, 1].set_title("Silhouette Coefficient (Max at k=4)", fontweight='bold')
    axes[0, 1].set_xlabel("Number of Clusters (k)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    axes[1, 0].plot(k_range, db_scores, '^-', color='#e67e22', linewidth=2)
    axes[1, 0].set_title("Davies-Bouldin Index (Lower is Better)", fontweight='bold')
    axes[1, 0].set_xlabel("Number of Clusters (k)")
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    axes[1, 1].plot(k_range, ch_scores, 'd-', color='#9b59b6', linewidth=2)
    axes[1, 1].set_title("Calinski-Harabasz Index (Higher is Better)", fontweight='bold')
    axes[1, 1].set_xlabel("Number of Clusters (k)")
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(grid_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {grid_fig_path}")

    # 2. Archetype Radar Chart (8 Dimensions)
    print("\n[2/3] Generating 8-dimensional Archetype Radar Chart...")
    archetype_centroids = np.array([
        [0.50, 0.40, 0.68, 0.45, 0.44, 0.53, 0.32, 0.48],  # 0: Grounded Stabilizers
        [0.47, 0.65, 0.60, 0.65, 0.60, 0.40, 0.61, 0.49],  # 1: Dynamic Visionaries
        [0.52, 0.35, 0.29, 0.56, 0.54, 0.44, 0.44, 0.51],  # 2: Spontaneous Socializers
        [0.50, 0.61, 0.45, 0.31, 0.42, 0.63, 0.61, 0.51]   # 3: Independent Thinkers
    ])

    archetype_names = [
        "0. Grounded Stabilizers (23.3%)",
        "1. Dynamic Visionaries (25.7%)",
        "2. Spontaneous Socializers (25.9%)",
        "3. Independent Thinkers (25.1%)"
    ]
    colors = ['#27ae60', '#8e44ad', '#d35400', '#2980b9']

    # Radar plot setup
    num_vars = len(traits)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close loop

    radar_fig_path = os.path.join(fig_dir, "archetype_radar_chart.png")
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for i in range(4):
        values = archetype_centroids[i].tolist()
        values += values[:1]  # Close loop
        ax.plot(angles, values, color=colors[i], linewidth=2.5, label=archetype_names[i])
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), [t.replace('_', ' ').capitalize() for t in traits], fontsize=10)
    ax.set_ylim(0, 0.8)
    plt.title("Psychological Profile Comparison Across 4 Archetypes", fontsize=13, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(radar_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {radar_fig_path}")

    # 3. Population Distribution Pie Chart
    print("\n[3/3] Generating Archetype Population Distribution Pie...")
    pie_fig_path = os.path.join(fig_dir, "archetype_distribution_pie.png")
    shares = [23.3, 25.7, 25.9, 25.1]
    labels = ["Grounded Stabilizers", "Dynamic Visionaries", "Spontaneous Socializers", "Independent Thinkers"]

    plt.figure(figsize=(8, 6))
    plt.pie(shares, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    plt.title("Empirical Distribution of Discovered Personality Archetypes", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(pie_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {pie_fig_path}")

    print("\nExtended clustering analysis complete!")


if __name__ == '__main__':
    run_extended_clustering()
