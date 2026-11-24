"""
Personality Archetype Identification and Cross-Affinity Analysis Module.
Bridges Unsupervised Clustering Insights (K-Means) with Inference & Prediction.
"""

from typing import Dict, Any, Tuple
import numpy as np

ARCHETYPE_NAMES: Dict[int, str] = {
    0: "The Grounded Stabilizers",
    1: "The Dynamic Visionaries",
    2: "The Spontaneous Socializers",
    3: "The Independent Thinkers"
}

ARCHETYPE_FULL_LABELS: Dict[int, str] = {
    0: "The Grounded Stabilizers (High Agreeableness & Conscientiousness)",
    1: "The Dynamic Visionaries (High Openness & Ambition)",
    2: "The Spontaneous Socializers (High Extraversion & Spontaneity)",
    3: "The Independent Thinkers (Low Extraversion, High Conscientiousness)"
}

ARCHETYPE_DESCRIPTIONS: Dict[int, str] = {
    0: "Personnalité fiable, structurée et empathique (Agreeableness & Conscientiousness élevées).",
    1: "Profil ambitieux, curieux et leader (Openness & Ambition élevées).",
    2: "Caractère festif, expressif et dynamique (Extraversion & Spontanéité élevées).",
    3: "Esprit calme, analytique et réfléchi (Introspection et grande concentration)."
}

# Empirical normalized centroids from unsupervised K-Means clustering (100,000 dataset)
ARCHETYPE_CENTROIDS: np.ndarray = np.array([
    [0.50, 0.40, 0.68, 0.45, 0.44, 0.53, 0.32, 0.48],  # 0: Grounded Stabilizers
    [0.47, 0.65, 0.60, 0.65, 0.60, 0.40, 0.61, 0.49],  # 1: Dynamic Visionaries
    [0.52, 0.35, 0.29, 0.56, 0.54, 0.44, 0.44, 0.51],  # 2: Spontaneous Socializers
    [0.50, 0.61, 0.45, 0.31, 0.42, 0.63, 0.61, 0.51]   # 3: Independent Thinkers
], dtype=float)

TRAIT_ORDER = [
    'openness', 'extraversion', 'agreeableness', 'conscientiousness',
    'chronotype', 'spontaneity', 'emotional_expressiveness', 'career_ambition'
]

# Historical cross-archetype compatibility rate matrix (%)
AFFINITY_MATRIX: Dict[Tuple[int, int], float] = {
    (0, 0): 49.1, (0, 1): 41.2, (0, 2): 42.6, (0, 3): 39.8,
    (1, 0): 40.5, (1, 1): 49.7, (1, 2): 42.7, (1, 3): 40.1,
    (2, 0): 42.7, (2, 1): 42.8, (2, 2): 49.1, (2, 3): 39.4,
    (3, 0): 37.4, (3, 1): 39.2, (3, 2): 38.5, (3, 3): 51.3
}


def identify_archetype(traits: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Identifies the personality archetype of an individual by computing minimum
    Euclidean distance to the empirical cluster centroids.
    """
    values = []
    for key in TRAIT_ORDER:
        lookup_key = f"{prefix}{key}" if prefix else key
        raw_val = float(traits.get(lookup_key, 5.0))
        # Normalize 1-10 scale down to 0-1 range for distance computation
        norm_val = raw_val / 10.0 if raw_val > 1.0 else raw_val
        values.append(norm_val)

    vec = np.array(values, dtype=float)
    distances = np.linalg.norm(ARCHETYPE_CENTROIDS - vec, axis=1)
    best_cluster = int(np.argmin(distances))

    return {
        "id": best_cluster,
        "name": ARCHETYPE_NAMES[best_cluster],
        "description": ARCHETYPE_DESCRIPTIONS[best_cluster],
        "confidence_distance": round(float(distances[best_cluster]), 3)
    }


def compute_cross_archetype_affinity(arch_a_id: int, arch_b_id: int) -> Dict[str, Any]:
    """
    Looks up the historical matchmaking affinity rate between two personality archetypes.
    """
    affinity_pct = AFFINITY_MATRIX.get((arch_a_id, arch_b_id), 43.5)
    
    if affinity_pct >= 48.0:
        affinity_level = "Synergie Naturelle Forte"
    elif affinity_pct >= 41.0:
        affinity_level = "Affinité Équilibrée"
    else:
        affinity_level = "Complémentarité à Négocier"

    return {
        "affinity_pct": affinity_pct,
        "affinity_level": affinity_level,
        "archetype_a": ARCHETYPE_NAMES.get(arch_a_id, f"Archetype {arch_a_id}"),
        "archetype_b": ARCHETYPE_NAMES.get(arch_b_id, f"Archetype {arch_b_id}")
    }
