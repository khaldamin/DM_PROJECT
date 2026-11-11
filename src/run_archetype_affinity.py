"""
Generates Cross-Cluster Compatibility Affinity Matrix.
Links Unsupervised Personality Archetypes to Supervised Matchmaking Outcomes.
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

sys.path.insert(0, os.path.abspath('.'))
from src.models.clustering import CupidKMeansClusterer
from src.models.archetypes import ARCHETYPE_NAMES

def compute_and_plot_cluster_affinity():
    print("=" * 65)
    print("  COMPUTING ARCHETYPE AFFINITY & COMPATIBILITY MATRIX")
    print("=" * 65)

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    df = pd.read_csv(data_path)

    profile_features = [
        'openness', 'extraversion', 'agreeableness', 'conscientiousness',
        'chronotype', 'spontaneity', 'emotional_expressiveness', 'career_ambition'
    ]

    a_cols = [f'a_{feat}' for feat in profile_features]
    b_cols = [f'b_{feat}' for feat in profile_features]

    # Fit clusterer on Partner A profiles (full dataset of 100,000 for high precision)
    scaler = StandardScaler()
    X_a_scaled = scaler.fit_transform(df[a_cols].to_numpy())
    X_b_scaled = scaler.transform(df[b_cols].to_numpy())

    # K-Means with optimal k=4
    kmeans = CupidKMeansClusterer(n_clusters=4, random_state=42)
    kmeans.fit(X_a_scaled)

    # Predict cluster for Partner A and Partner B
    df['cluster_A'] = kmeans.predict(X_a_scaled)
    df['cluster_B'] = kmeans.predict(X_b_scaled)

    archetype_names = {i: f"{i}. {name.replace('The ', '')}" for i, name in ARCHETYPE_NAMES.items()}

    df['archetype_A'] = df['cluster_A'].map(archetype_names)
    df['archetype_B'] = df['cluster_B'].map(archetype_names)

    # Compatibility Rate Matrix (Average 'compatible' per archetype pair)
    affinity_matrix = df.pivot_table(
        index='archetype_A',
        columns='archetype_B',
        values='compatible',
        aggfunc='mean'
    ) * 100

    # Relationship Longevity Matrix
    longevity_matrix = df.pivot_table(
        index='archetype_A',
        columns='archetype_B',
        values='relationship_longevity_months',
        aggfunc='mean'
    )

    # Save Tables
    tables_dir = os.path.join("results", "tables")
    figures_dir = os.path.join("results", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    affinity_csv = os.path.join(tables_dir, "archetype_affinity_matrix.csv")
    affinity_matrix.round(2).to_csv(affinity_csv)
    print(f"-> Saved Affinity Matrix to {affinity_csv}")

    # Plot Affinity Heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(affinity_matrix, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Compatibility Rate (%)'}, ax=ax1, linewidths=1)
    ax1.set_title("Cross-Archetype Compatibility Rate (% Success)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Partner B Archetype", fontsize=11)
    ax1.set_ylabel("Partner A Archetype", fontsize=11)

    sns.heatmap(longevity_matrix, annot=True, fmt=".1f", cmap="Purples", cbar_kws={'label': 'Average Longevity (Months)'}, ax=ax2, linewidths=1)
    ax2.set_title("Cross-Archetype Average Longevity (Months)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Partner B Archetype", fontsize=11)
    ax2.set_ylabel("Partner A Archetype", fontsize=11)

    plt.tight_layout()
    output_png = os.path.join(figures_dir, "clustering_archetype_affinity_heatmap.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"-> Saved Archetype Affinity Heatmap to {output_png}")

    print("\nCompatibility Rate Matrix (%):")
    print(affinity_matrix.round(1))

if __name__ == "__main__":
    compute_and_plot_cluster_affinity()
