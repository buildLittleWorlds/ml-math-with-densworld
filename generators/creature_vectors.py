"""
Creature Feature Vectors Generator

Generates explicit numerical feature vectors for creatures from Yeller Quarry.
Used throughout the ML Math curriculum for teaching dot products, vector similarity,
and cosine similarity.

World Parameters (from yeller quarry ore.txt):
- Creatures have distinct hunting behaviors (ambush vs pursuit, solitary vs pack)
- Creatures inhabit different depths and environments
- Some creatures can become "yeller groups" (synchronized flocks of prime numbers)
- Danger ratings and metal content vary by species

Teaching Applications:
- Module 2: Vectors as data points in n-dimensional space
- Module 2: Dot product as similarity measure
- Module 2: Cosine similarity (normalized dot product)
- Module 2: Orthogonality (creatures with completely different behaviors)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# =============================================================================
# CREATURE DATABASE (from yeller-quarry-data-science)
# =============================================================================

# Core creature data with behavioral and habitat features
CREATURES = [
    {
        'creature_id': 'CR001',
        'common_name': 'Leatherback Burrower',
        'category': 'bird',
        # Behavioral features (0-1 scale)
        'aggression': 0.3,
        'sociality': 0.4,  # somewhat social
        'nocturnality': 0.7,  # mostly nocturnal
        'territoriality': 0.5,
        'hunting_strategy': 0.2,  # 0=ambush, 1=pursuit, 0.2=mostly ambush
        # Habitat features (0-1 scale)
        'depth_preference': 0.4,  # prefers medium depth
        'moisture_preference': 0.3,  # prefers drier
        'light_tolerance': 0.2,  # low light tolerance
        'cave_affinity': 0.8,  # strong cave preference
        'surface_affinity': 0.2
    },
    {
        'creature_id': 'CR002',
        'common_name': 'Stone Spine Lizard',
        'category': 'reptile',
        'aggression': 0.6,
        'sociality': 0.1,  # very solitary
        'nocturnality': 0.4,
        'territoriality': 0.9,  # highly territorial
        'hunting_strategy': 0.1,  # pure ambush
        'depth_preference': 0.3,
        'moisture_preference': 0.2,
        'light_tolerance': 0.5,
        'cave_affinity': 0.6,
        'surface_affinity': 0.4
    },
    {
        'creature_id': 'CR003',
        'common_name': 'Cave Bat',
        'category': 'mammal',
        'aggression': 0.1,
        'sociality': 0.95,  # highly social (roost in colonies)
        'nocturnality': 0.95,
        'territoriality': 0.2,
        'hunting_strategy': 0.8,  # pursuit (flying insects)
        'depth_preference': 0.5,
        'moisture_preference': 0.5,
        'light_tolerance': 0.1,
        'cave_affinity': 0.95,
        'surface_affinity': 0.3
    },
    {
        'creature_id': 'CR004',
        'common_name': 'Marsh Hornet',
        'category': 'insect',
        'aggression': 0.9,
        'sociality': 0.9,  # swarm behavior
        'nocturnality': 0.2,  # diurnal
        'territoriality': 0.95,
        'hunting_strategy': 0.9,  # pursuit attacks
        'depth_preference': 0.1,  # surface
        'moisture_preference': 0.8,  # needs moisture
        'light_tolerance': 0.9,
        'cave_affinity': 0.1,
        'surface_affinity': 0.95
    },
    {
        'creature_id': 'CR005',
        'common_name': 'Maw Beast',
        'category': 'mammal',
        'aggression': 0.95,  # extremely aggressive (killed Truck)
        'sociality': 0.1,  # solitary hunter
        'nocturnality': 0.6,
        'territoriality': 0.8,
        'hunting_strategy': 0.3,  # ambush with pursuit
        'depth_preference': 0.9,  # deep caves
        'moisture_preference': 0.4,
        'light_tolerance': 0.1,
        'cave_affinity': 0.95,
        'surface_affinity': 0.05
    },
    {
        'creature_id': 'CR006',
        'common_name': 'Yeller Frog',
        'category': 'amphibian',
        'aggression': 0.2,
        'sociality': 0.8,  # forms yeller groups
        'nocturnality': 0.5,
        'territoriality': 0.3,
        'hunting_strategy': 0.0,  # pure ambush (tongue strike)
        'depth_preference': 0.2,
        'moisture_preference': 0.95,
        'light_tolerance': 0.4,
        'cave_affinity': 0.3,
        'surface_affinity': 0.7
    },
    {
        'creature_id': 'CR007',
        'common_name': 'Coil Tube Serpent',
        'category': 'reptile',
        'aggression': 0.7,
        'sociality': 0.15,
        'nocturnality': 0.8,
        'territoriality': 0.6,
        'hunting_strategy': 0.05,  # constriction ambush
        'depth_preference': 0.5,
        'moisture_preference': 0.6,
        'light_tolerance': 0.2,
        'cave_affinity': 0.7,
        'surface_affinity': 0.3
    },
    {
        'creature_id': 'CR008',
        'common_name': 'Metal-Beaked Finch',
        'category': 'bird',
        'aggression': 0.2,
        'sociality': 0.7,  # flocks
        'nocturnality': 0.1,  # diurnal
        'territoriality': 0.4,
        'hunting_strategy': 0.6,  # foraging pursuit
        'depth_preference': 0.1,
        'moisture_preference': 0.3,
        'light_tolerance': 0.95,
        'cave_affinity': 0.2,
        'surface_affinity': 0.9
    },
    {
        'creature_id': 'CR009',
        'common_name': 'Wharver',
        'category': 'reptile',
        'aggression': 0.85,
        'sociality': 0.3,
        'nocturnality': 0.4,
        'territoriality': 0.7,
        'hunting_strategy': 0.4,  # mixed
        'depth_preference': 0.7,  # Grimslew Shore underwater
        'moisture_preference': 0.95,
        'light_tolerance': 0.3,
        'cave_affinity': 0.4,
        'surface_affinity': 0.3
    },
    {
        'creature_id': 'CR010',
        'common_name': 'Crawler',
        'category': 'insect',
        'aggression': 0.05,
        'sociality': 0.6,
        'nocturnality': 0.9,
        'territoriality': 0.1,
        'hunting_strategy': 0.2,  # scavenging
        'depth_preference': 0.6,
        'moisture_preference': 0.7,
        'light_tolerance': 0.1,
        'cave_affinity': 0.9,
        'surface_affinity': 0.1
    },
    {
        'creature_id': 'CR011',
        'common_name': 'Grimslew Fish',
        'category': 'fish',
        'aggression': 0.5,
        'sociality': 0.4,
        'nocturnality': 0.5,
        'territoriality': 0.3,
        'hunting_strategy': 0.7,
        'depth_preference': 0.95,
        'moisture_preference': 1.0,
        'light_tolerance': 0.05,
        'cave_affinity': 0.5,
        'surface_affinity': 0.0
    },
    {
        'creature_id': 'CR012',
        'common_name': 'Mud Worm',
        'category': 'worm',
        'aggression': 0.0,
        'sociality': 0.3,
        'nocturnality': 0.5,
        'territoriality': 0.0,
        'hunting_strategy': 0.0,
        'depth_preference': 0.4,
        'moisture_preference': 0.9,
        'light_tolerance': 0.0,
        'cave_affinity': 0.5,
        'surface_affinity': 0.5
    },
    {
        'creature_id': 'CR013',
        'common_name': 'Deep Borer',
        'category': 'worm',
        'aggression': 0.3,
        'sociality': 0.2,
        'nocturnality': 0.5,  # irrelevant at depth
        'territoriality': 0.5,
        'hunting_strategy': 0.1,
        'depth_preference': 0.95,
        'moisture_preference': 0.6,
        'light_tolerance': 0.0,
        'cave_affinity': 0.95,
        'surface_affinity': 0.0
    },
    {
        'creature_id': 'CR014',
        'common_name': 'Quarry Moth',
        'category': 'insect',
        'aggression': 0.0,
        'sociality': 0.5,
        'nocturnality': 0.95,
        'territoriality': 0.0,
        'hunting_strategy': 0.0,  # herbivore
        'depth_preference': 0.2,
        'moisture_preference': 0.4,
        'light_tolerance': 0.1,
        'cave_affinity': 0.6,
        'surface_affinity': 0.6
    },
    {
        'creature_id': 'CR015',
        'common_name': 'Rust Cat',
        'category': 'mammal',
        'aggression': 0.6,
        'sociality': 0.2,
        'nocturnality': 0.8,
        'territoriality': 0.7,
        'hunting_strategy': 0.15,  # stalk and pounce
        'depth_preference': 0.3,
        'moisture_preference': 0.3,
        'light_tolerance': 0.2,
        'cave_affinity': 0.5,
        'surface_affinity': 0.6
    },
    {
        'creature_id': 'CR016',
        'common_name': 'Stakdur',
        'category': 'reptile',
        'aggression': 0.75,
        'sociality': 0.05,  # extremely solitary
        'nocturnality': 0.6,
        'territoriality': 0.85,
        'hunting_strategy': 0.2,
        'depth_preference': 0.6,
        'moisture_preference': 0.4,
        'light_tolerance': 0.2,
        'cave_affinity': 0.8,
        'surface_affinity': 0.2
    },
    {
        'creature_id': 'CR017',
        'common_name': 'Yeller Cat',
        'category': 'mammal',
        'aggression': 0.5,
        'sociality': 0.85,  # forms yeller groups
        'nocturnality': 0.7,
        'territoriality': 0.4,
        'hunting_strategy': 0.5,  # pack hunting
        'depth_preference': 0.3,
        'moisture_preference': 0.4,
        'light_tolerance': 0.3,
        'cave_affinity': 0.4,
        'surface_affinity': 0.6
    },
    {
        'creature_id': 'CR018',
        'common_name': 'Yeller Bat',
        'category': 'mammal',
        'aggression': 0.2,
        'sociality': 0.95,  # forms yeller groups AND colonies
        'nocturnality': 0.95,
        'territoriality': 0.2,
        'hunting_strategy': 0.75,
        'depth_preference': 0.4,
        'moisture_preference': 0.5,
        'light_tolerance': 0.1,
        'cave_affinity': 0.9,
        'surface_affinity': 0.4
    },
    {
        'creature_id': 'CR019',
        'common_name': 'Yeller Bird Alpha',
        'category': 'bird',
        'aggression': 0.3,
        'sociality': 1.0,  # always in flock of 5
        'nocturnality': 0.3,
        'territoriality': 0.3,
        'hunting_strategy': 0.6,
        'depth_preference': 0.2,
        'moisture_preference': 0.4,
        'light_tolerance': 0.7,
        'cave_affinity': 0.3,
        'surface_affinity': 0.8
    },
    {
        'creature_id': 'CR020',
        'common_name': 'Scuttler',
        'category': 'insect',
        'aggression': 0.1,
        'sociality': 0.4,
        'nocturnality': 0.8,
        'territoriality': 0.1,
        'hunting_strategy': 0.3,
        'depth_preference': 0.3,
        'moisture_preference': 0.6,
        'light_tolerance': 0.2,
        'cave_affinity': 0.7,
        'surface_affinity': 0.4
    },
    {
        'creature_id': 'CR021',
        'common_name': 'Stone Leech',
        'category': 'worm',
        'aggression': 0.4,
        'sociality': 0.1,
        'nocturnality': 0.5,
        'territoriality': 0.2,
        'hunting_strategy': 0.0,  # parasitic attachment
        'depth_preference': 0.5,
        'moisture_preference': 0.8,
        'light_tolerance': 0.1,
        'cave_affinity': 0.7,
        'surface_affinity': 0.3
    },
    {
        'creature_id': 'CR022',
        'common_name': 'Witch Creature',
        'category': 'unknown',
        'aggression': 1.0,  # mythically dangerous
        'sociality': 0.0,  # unknown
        'nocturnality': 0.9,
        'territoriality': 1.0,
        'hunting_strategy': 0.5,
        'depth_preference': 0.8,
        'moisture_preference': 0.5,
        'light_tolerance': 0.0,
        'cave_affinity': 0.9,
        'surface_affinity': 0.1
    },
    {
        'creature_id': 'CR023',
        'common_name': 'Tunnel Newt',
        'category': 'amphibian',
        'aggression': 0.15,
        'sociality': 0.5,
        'nocturnality': 0.6,
        'territoriality': 0.3,
        'hunting_strategy': 0.2,
        'depth_preference': 0.5,
        'moisture_preference': 0.85,
        'light_tolerance': 0.2,
        'cave_affinity': 0.8,
        'surface_affinity': 0.2
    },
    {
        'creature_id': 'CR024',
        'common_name': 'Glint Beetle',
        'category': 'insect',
        'aggression': 0.05,
        'sociality': 0.3,
        'nocturnality': 0.7,
        'territoriality': 0.1,
        'hunting_strategy': 0.1,
        'depth_preference': 0.4,
        'moisture_preference': 0.5,
        'light_tolerance': 0.3,
        'cave_affinity': 0.6,
        'surface_affinity': 0.5
    },
    {
        'creature_id': 'CR025',
        'common_name': 'Marsh Eel',
        'category': 'fish',
        'aggression': 0.55,
        'sociality': 0.2,
        'nocturnality': 0.7,
        'territoriality': 0.5,
        'hunting_strategy': 0.6,
        'depth_preference': 0.6,
        'moisture_preference': 1.0,
        'light_tolerance': 0.1,
        'cave_affinity': 0.4,
        'surface_affinity': 0.2
    },
]

# Feature names for vector operations
BEHAVIORAL_FEATURES = ['aggression', 'sociality', 'nocturnality', 'territoriality', 'hunting_strategy']
HABITAT_FEATURES = ['depth_preference', 'moisture_preference', 'light_tolerance', 'cave_affinity', 'surface_affinity']
ALL_FEATURES = BEHAVIORAL_FEATURES + HABITAT_FEATURES


def compute_dot_product(creature_a: dict, creature_b: dict, features: list) -> float:
    """Compute dot product between two creature vectors."""
    vec_a = np.array([creature_a[f] for f in features])
    vec_b = np.array([creature_b[f] for f in features])
    return float(np.dot(vec_a, vec_b))


def compute_cosine_similarity(creature_a: dict, creature_b: dict, features: list) -> float:
    """Compute cosine similarity between two creature vectors."""
    vec_a = np.array([creature_a[f] for f in features])
    vec_b = np.array([creature_b[f] for f in features])
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_euclidean_distance(creature_a: dict, creature_b: dict, features: list) -> float:
    """Compute Euclidean distance between two creature vectors."""
    vec_a = np.array([creature_a[f] for f in features])
    vec_b = np.array([creature_b[f] for f in features])
    return float(np.linalg.norm(vec_a - vec_b))


def generate_creature_vectors_dataset() -> pd.DataFrame:
    """Generate the creature vectors dataset."""

    records = []

    for creature in CREATURES:
        record = {
            'creature_id': creature['creature_id'],
            'common_name': creature['common_name'],
            'category': creature['category'],
        }

        # Add behavioral features
        for feat in BEHAVIORAL_FEATURES:
            record[feat] = creature[feat]

        # Add habitat features
        for feat in HABITAT_FEATURES:
            record[feat] = creature[feat]

        # Compute vector norms
        behavioral_vec = np.array([creature[f] for f in BEHAVIORAL_FEATURES])
        habitat_vec = np.array([creature[f] for f in HABITAT_FEATURES])
        full_vec = np.array([creature[f] for f in ALL_FEATURES])

        record['behavioral_norm_l2'] = round(np.linalg.norm(behavioral_vec), 4)
        record['habitat_norm_l2'] = round(np.linalg.norm(habitat_vec), 4)
        record['full_norm_l2'] = round(np.linalg.norm(full_vec), 4)

        records.append(record)

    return pd.DataFrame(records)


def generate_similarity_matrix() -> pd.DataFrame:
    """Generate pairwise similarity matrix for all creatures."""

    n = len(CREATURES)
    similarities = []

    for i, creature_a in enumerate(CREATURES):
        for j, creature_b in enumerate(CREATURES):
            if i < j:  # Only upper triangle (symmetric)
                sim = {
                    'creature_a_id': creature_a['creature_id'],
                    'creature_a_name': creature_a['common_name'],
                    'creature_b_id': creature_b['creature_id'],
                    'creature_b_name': creature_b['common_name'],
                    'dot_product_behavioral': round(compute_dot_product(creature_a, creature_b, BEHAVIORAL_FEATURES), 4),
                    'dot_product_habitat': round(compute_dot_product(creature_a, creature_b, HABITAT_FEATURES), 4),
                    'dot_product_full': round(compute_dot_product(creature_a, creature_b, ALL_FEATURES), 4),
                    'cosine_sim_behavioral': round(compute_cosine_similarity(creature_a, creature_b, BEHAVIORAL_FEATURES), 4),
                    'cosine_sim_habitat': round(compute_cosine_similarity(creature_a, creature_b, HABITAT_FEATURES), 4),
                    'cosine_sim_full': round(compute_cosine_similarity(creature_a, creature_b, ALL_FEATURES), 4),
                    'euclidean_dist_full': round(compute_euclidean_distance(creature_a, creature_b, ALL_FEATURES), 4),
                }
                similarities.append(sim)

    return pd.DataFrame(similarities)


if __name__ == '__main__':
    print("Generating creature vectors dataset...")
    print("=" * 60)

    # Generate main dataset
    df_vectors = generate_creature_vectors_dataset()

    # Generate similarity matrix
    df_similarity = generate_similarity_matrix()

    # Save to data directory
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    vectors_path = output_dir / 'creature_vectors.csv'
    similarity_path = output_dir / 'creature_similarity.csv'

    df_vectors.to_csv(vectors_path, index=False)
    df_similarity.to_csv(similarity_path, index=False)

    print(f"Saved {len(df_vectors)} creature vectors to {vectors_path}")
    print(f"Saved {len(df_similarity)} pairwise similarities to {similarity_path}")

    # Print summary statistics
    print("\n--- Creature Vectors Summary ---")
    print(f"Number of creatures: {len(df_vectors)}")
    print(f"Behavioral features: {BEHAVIORAL_FEATURES}")
    print(f"Habitat features: {HABITAT_FEATURES}")

    print(f"\nCategory distribution:")
    print(df_vectors['category'].value_counts())

    print(f"\nVector norm statistics:")
    print(df_vectors[['behavioral_norm_l2', 'habitat_norm_l2', 'full_norm_l2']].describe())

    # Find most and least similar pairs
    print("\n--- Teaching Examples ---")

    most_similar = df_similarity.loc[df_similarity['cosine_sim_full'].idxmax()]
    print(f"\nMost similar creatures (cosine):")
    print(f"  {most_similar['creature_a_name']} & {most_similar['creature_b_name']}")
    print(f"  Cosine similarity: {most_similar['cosine_sim_full']:.4f}")

    least_similar = df_similarity.loc[df_similarity['cosine_sim_full'].idxmin()]
    print(f"\nLeast similar creatures (nearly orthogonal):")
    print(f"  {least_similar['creature_a_name']} & {least_similar['creature_b_name']}")
    print(f"  Cosine similarity: {least_similar['cosine_sim_full']:.4f}")

    # Find creatures with highest behavioral similarity but different habitats
    df_similarity['behavior_habitat_diff'] = abs(df_similarity['cosine_sim_behavioral'] - df_similarity['cosine_sim_habitat'])
    most_different = df_similarity.loc[df_similarity['behavior_habitat_diff'].idxmax()]
    print(f"\nSimilar behavior, different habitat:")
    print(f"  {most_different['creature_a_name']} & {most_different['creature_b_name']}")
    print(f"  Behavioral cosine: {most_different['cosine_sim_behavioral']:.4f}")
    print(f"  Habitat cosine: {most_different['cosine_sim_habitat']:.4f}")

    print("\n--- Teaching Applications ---")
    print("1. Dot product: Compute similarity between creature pairs")
    print("2. Cosine similarity: Normalize for magnitude, compare directions")
    print("3. Orthogonality: Find creatures with similarity ~0 (unrelated)")
    print("4. Vector norms: Compare total 'intensity' of creature behaviors")
