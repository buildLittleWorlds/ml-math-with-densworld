"""
Manuscript Features Simulator

Generates stylometric features for manuscript authenticity detection.
Demonstrates dot products, similarity, hypothesis testing, and regularization.

World Parameters (from ore files):
- Three philosophical schools: Stone, Water, Pebble
- Scholars have distinct writing styles
- Forgeries exist — Mink Pavar suspected of forging Grigsu Haldo
- Manuscripts have word frequency patterns tied to era and school
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# =============================================================================
# PHILOSOPHICAL SCHOOLS (canonical word vectors)
# =============================================================================

# Each school has characteristic word usage patterns
# These become the "target vectors" for similarity analysis

SCHOOL_VECTORS = {
    'stone_school': {
        'permanence': 0.9,
        'origin': 0.8,
        'foundation': 0.85,
        'residue': 0.2,
        'dissolution': 0.1,
        'flowing': 0.15,
        'lending': 0.2,
        'discovery': 0.7,
        'structure': 0.75,
        'ancient': 0.65
    },
    'water_school': {
        'permanence': 0.15,
        'origin': 0.3,
        'foundation': 0.2,
        'residue': 0.85,
        'dissolution': 0.9,
        'flowing': 0.8,
        'lending': 0.3,
        'discovery': 0.4,
        'structure': 0.25,
        'ancient': 0.4
    },
    'pebble_school': {
        'permanence': 0.5,
        'origin': 0.6,
        'foundation': 0.4,
        'residue': 0.5,
        'dissolution': 0.4,
        'flowing': 0.5,
        'lending': 0.8,
        'discovery': 0.55,
        'structure': 0.5,
        'ancient': 0.7
    }
}

# Key terms for normalization
TERMS = list(SCHOOL_VECTORS['stone_school'].keys())

# =============================================================================
# SCHOLARS DATABASE
# =============================================================================

SCHOLARS = {
    'SCH-001': {
        'name': 'Grigsu Haldo',
        'school': 'stone_school',
        'avg_sentence_length': 18.2,
        'vocabulary_richness': 0.72,
        'birth_year': 847,
        'death_year': 891,
        'active_start': 865,
        'active_end': 889
    },
    'SCH-002': {
        'name': 'Yasho Krent',
        'school': 'water_school',
        'avg_sentence_length': 24.5,
        'vocabulary_richness': 0.81,
        'birth_year': 842,
        'death_year': 912,
        'active_start': 860,
        'active_end': 910
    },
    'SCH-003': {
        'name': 'Vagabu Olt',
        'school': 'none',
        'avg_sentence_length': 15.3,
        'vocabulary_richness': 0.65,
        'birth_year': 851,
        'death_year': None,
        'active_start': 870,
        'active_end': 900
    },
    'SCH-004': {
        'name': 'Boffa Trent',
        'school': 'water_school',
        'avg_sentence_length': 32.1,
        'vocabulary_richness': 0.78,
        'birth_year': 849,
        'death_year': 903,
        'active_start': 868,
        'active_end': 901
    },
    'SCH-005': {
        'name': 'Mink Pavar',
        'school': 'water_school',
        'avg_sentence_length': 28.4,
        'vocabulary_richness': 0.84,
        'birth_year': 855,
        'death_year': 925,
        'active_start': 875,
        'active_end': 920
    },
    'SCH-006': {
        'name': 'Fibon Arrel',
        'school': 'pebble_school',
        'avg_sentence_length': 21.7,
        'vocabulary_richness': 0.69,
        'birth_year': 810,
        'death_year': 872,
        'active_start': 830,
        'active_end': 870
    },
    'SCH-007': {
        'name': 'Eulr Voss',
        'school': 'water_school',
        'avg_sentence_length': 26.8,
        'vocabulary_richness': 0.76,
        'birth_year': 790,
        'death_year': 845,
        'active_start': 810,
        'active_end': 843
    }
}


def compute_school_alignment(word_vector: dict, school: str) -> float:
    """
    Compute dot product between manuscript word vector and school canonical vector.
    This is the core similarity measure.
    """
    school_vec = SCHOOL_VECTORS[school]

    dot_product = 0
    for term in TERMS:
        dot_product += word_vector.get(term, 0) * school_vec[term]

    # Normalize by vector magnitudes
    mag1 = np.sqrt(sum(v**2 for v in word_vector.values()))
    mag2 = np.sqrt(sum(v**2 for v in school_vec.values()))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot_product / (mag1 * mag2)


def generate_authentic_manuscript(manuscript_id: int, scholar_id: str) -> dict:
    """Generate features for an authentic manuscript."""
    scholar = SCHOLARS[scholar_id]
    school = scholar['school']

    # Word count: log-normal distribution
    word_count = int(np.random.lognormal(mean=8, sigma=0.8))
    word_count = max(500, min(50000, word_count))

    # Average sentence length: scholar's style with variation
    avg_sentence_length = scholar['avg_sentence_length'] + np.random.normal(0, 2)
    avg_sentence_length = max(8, avg_sentence_length)

    # Generate word usage vector based on school
    word_vector = {}
    if school in SCHOOL_VECTORS:
        for term in TERMS:
            base = SCHOOL_VECTORS[school][term]
            # Add author-specific variation
            word_vector[term] = max(0, min(1, base + np.random.normal(0, 0.1)))
    else:
        # No school affiliation - mixed signal
        for term in TERMS:
            word_vector[term] = np.random.uniform(0.3, 0.7)

    # Vocabulary richness
    vocabulary_richness = scholar['vocabulary_richness'] + np.random.normal(0, 0.05)
    vocabulary_richness = max(0.3, min(0.95, vocabulary_richness))

    # Philosophical term density
    term_density = sum(word_vector.values()) / len(TERMS)

    # Stylometric variance (consistency within document)
    # Authentic documents have low variance
    stylometric_variance = np.random.exponential(0.05)

    # Era markers: authentic documents should match their era
    # This is 0 for authentic, > 0 indicates anachronisms
    era_marker_score = np.random.exponential(0.02)

    # Composition date within scholar's active period
    composition_year = np.random.randint(scholar['active_start'], scholar['active_end'] + 1)

    return {
        'manuscript_id': f'MS-{manuscript_id:04d}',
        'attributed_author': scholar['name'],
        'attributed_school': school,
        'is_forgery': False,
        'true_author': scholar['name'],
        'word_count': word_count,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'vocabulary_richness': round(vocabulary_richness, 3),
        'philosophical_term_density': round(term_density, 3),
        'school_alignment_stone': round(compute_school_alignment(word_vector, 'stone_school'), 3),
        'school_alignment_water': round(compute_school_alignment(word_vector, 'water_school'), 3),
        'school_alignment_pebble': round(compute_school_alignment(word_vector, 'pebble_school'), 3),
        'era_marker_score': round(era_marker_score, 4),
        'stylometric_variance': round(stylometric_variance, 4),
        'composition_year': composition_year
    }


def generate_forged_manuscript(manuscript_id: int,
                                attributed_to: str,
                                forger_id: str = 'SCH-005') -> dict:
    """
    Generate features for a forged manuscript.
    Forger is typically Mink Pavar (SCH-005).
    """
    victim = SCHOLARS[attributed_to]
    forger = SCHOLARS[forger_id]

    # Word count
    word_count = int(np.random.lognormal(mean=8, sigma=0.8))
    word_count = max(500, min(50000, word_count))

    # Sentence length: FORGER's style leaks through
    # Attempts to match victim but imperfectly
    target_length = victim['avg_sentence_length']
    actual_length = target_length * 0.6 + forger['avg_sentence_length'] * 0.4
    avg_sentence_length = actual_length + np.random.normal(0, 2)

    # Word usage: FORGER's school preferences leak through
    word_vector = {}
    if victim['school'] in SCHOOL_VECTORS:
        for term in TERMS:
            victim_base = SCHOOL_VECTORS[victim['school']][term]
            forger_base = SCHOOL_VECTORS[forger['school']][term]
            # Forger's patterns contaminate the forgery
            blended = victim_base * 0.5 + forger_base * 0.5
            word_vector[term] = max(0, min(1, blended + np.random.normal(0, 0.15)))
    else:
        for term in TERMS:
            word_vector[term] = np.random.uniform(0.3, 0.7)

    # Vocabulary richness: forger's style
    vocabulary_richness = (victim['vocabulary_richness'] * 0.4 +
                           forger['vocabulary_richness'] * 0.6 +
                           np.random.normal(0, 0.05))
    vocabulary_richness = max(0.3, min(0.95, vocabulary_richness))

    term_density = sum(word_vector.values()) / len(TERMS)

    # Stylometric variance: HIGHER for forgeries (inconsistent style)
    stylometric_variance = np.random.exponential(0.15) + 0.05

    # Era markers: HIGHER for forgeries (anachronisms)
    era_marker_score = np.random.exponential(0.2) + 0.1

    # Claimed composition date (often before forger was born or during victim's life)
    composition_year = np.random.randint(victim['active_start'], victim['active_end'] + 1)

    return {
        'manuscript_id': f'MS-{manuscript_id:04d}',
        'attributed_author': victim['name'],
        'attributed_school': victim['school'],
        'is_forgery': True,
        'true_author': forger['name'],
        'word_count': word_count,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'vocabulary_richness': round(vocabulary_richness, 3),
        'philosophical_term_density': round(term_density, 3),
        'school_alignment_stone': round(compute_school_alignment(word_vector, 'stone_school'), 3),
        'school_alignment_water': round(compute_school_alignment(word_vector, 'water_school'), 3),
        'school_alignment_pebble': round(compute_school_alignment(word_vector, 'pebble_school'), 3),
        'era_marker_score': round(era_marker_score, 4),
        'stylometric_variance': round(stylometric_variance, 4),
        'composition_year': composition_year
    }


def generate_manuscript_dataset(n_manuscripts: int = 300,
                                 forgery_rate: float = 0.15) -> pd.DataFrame:
    """Generate a full dataset of manuscript features."""

    manuscripts = []
    n_forgeries = int(n_manuscripts * forgery_rate)
    n_authentic = n_manuscripts - n_forgeries

    # Generate authentic manuscripts
    scholar_ids = list(SCHOLARS.keys())
    for i in range(n_authentic):
        scholar_id = np.random.choice(scholar_ids)
        manuscript = generate_authentic_manuscript(i + 1, scholar_id)
        manuscripts.append(manuscript)

    # Generate forgeries (mostly attributed to Grigsu, forged by Mink)
    for i in range(n_forgeries):
        # Most forgeries attributed to Grigsu (stone school target)
        if np.random.random() < 0.7:
            attributed_to = 'SCH-001'  # Grigsu Haldo
        else:
            attributed_to = np.random.choice(['SCH-006', 'SCH-007'])

        manuscript = generate_forged_manuscript(n_authentic + i + 1, attributed_to)
        manuscripts.append(manuscript)

    df = pd.DataFrame(manuscripts)

    # Shuffle so forgeries aren't all at the end
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


if __name__ == '__main__':
    print("Generating manuscript features dataset...")
    df = generate_manuscript_dataset(n_manuscripts=300)

    # Save to data directory
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / 'manuscript_features.csv', index=False)
    print(f"Saved {len(df)} manuscripts to {output_dir / 'manuscript_features.csv'}")

    # Print summary
    print("\n--- Dataset Summary ---")
    print(f"Forgery rate: {df['is_forgery'].mean():.1%}")

    print(f"\nManuscripts by attributed school:")
    print(df['attributed_school'].value_counts())

    print(f"\nAverage features by authenticity:")
    for col in ['avg_sentence_length', 'stylometric_variance', 'era_marker_score']:
        auth_mean = df[~df['is_forgery']][col].mean()
        forg_mean = df[df['is_forgery']][col].mean()
        print(f"  {col}: authentic={auth_mean:.3f}, forgery={forg_mean:.3f}")

    print(f"\nSchool alignment for Grigsu manuscripts:")
    grigsu = df[df['attributed_author'] == 'Grigsu Haldo']
    print(f"  Authentic stone alignment: {grigsu[~grigsu['is_forgery']]['school_alignment_stone'].mean():.3f}")
    print(f"  Forged stone alignment: {grigsu[grigsu['is_forgery']]['school_alignment_stone'].mean():.3f}")
    print(f"  Forged water alignment: {grigsu[grigsu['is_forgery']]['school_alignment_water'].mean():.3f}")
    print("  (Forgeries show contamination from Mink's water-school style)")
