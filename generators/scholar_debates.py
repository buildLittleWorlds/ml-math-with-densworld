"""
Scholar Debates Simulator

Generates synthetic data about philosophical debates in the Capital Archives.
Used throughout the ML Math curriculum for hypothesis testing, p-values, and
the multiple comparisons problem.

World Parameters (from capital ore.txt):
- The Capital hosts debates between philosophical schools
- Three schools: Stone, Water, Pebble
- Senator J and others fund scholarly activities
- Archives are central to Capital intellectual life
- Scholars like Grigsu, Yasho, Boffa, Mink are prominent figures

Teaching Applications:
- Module 1: Hypothesis testing (Do Stone scholars win more debates?)
- Module 1: P-values and significance
- Module 1: Multiple comparisons problem (test 20 hypotheses, find 1 by chance)
- Module 1: Bonferroni correction

IMPORTANT FOR TEACHING:
This dataset is deliberately designed with:
1. A TRUE but small effect (Stone scholars have ~55% win rate vs 50% baseline)
2. Many FALSE patterns that will appear significant by chance
3. Enough noise that students will find "significant" results that are just luck
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# =============================================================================
# WORLD PARAMETERS
# =============================================================================

# Philosophical schools
SCHOOLS = ['stone_school', 'water_school', 'pebble_school']

# School characteristics
SCHOOL_PARAMS = {
    'stone_school': {
        'name': 'Stone School',
        'philosophy': 'permanence, foundation, origin',
        'prestige_modifier': 1.05,  # TRUE but small advantage
        'rhetoric_style': 'formal',
        'founding_year': 780
    },
    'water_school': {
        'name': 'Water School',
        'philosophy': 'dissolution, flowing, residue',
        'prestige_modifier': 1.02,  # Slight advantage
        'rhetoric_style': 'flowing',
        'founding_year': 795
    },
    'pebble_school': {
        'name': 'Pebble School',
        'philosophy': 'compromise, lending, mixed',
        'prestige_modifier': 0.95,  # Slight disadvantage (seen as weak)
        'rhetoric_style': 'mixed',
        'founding_year': 820
    }
}

# Extended scholar database (building on manuscript_features.py)
SCHOLARS = [
    {'id': 'SCH-001', 'name': 'Grigsu Haldo', 'school': 'stone_school',
     'birth_year': 847, 'reputation': 0.9, 'rhetoric_skill': 0.85, 'publications': 45},
    {'id': 'SCH-002', 'name': 'Yasho Krent', 'school': 'water_school',
     'birth_year': 842, 'reputation': 0.85, 'rhetoric_skill': 0.88, 'publications': 52},
    {'id': 'SCH-003', 'name': 'Vagabu Olt', 'school': 'none',
     'birth_year': 851, 'reputation': 0.6, 'rhetoric_skill': 0.75, 'publications': 18},
    {'id': 'SCH-004', 'name': 'Boffa Trent', 'school': 'water_school',
     'birth_year': 849, 'reputation': 0.75, 'rhetoric_skill': 0.72, 'publications': 31},
    {'id': 'SCH-005', 'name': 'Mink Pavar', 'school': 'water_school',
     'birth_year': 855, 'reputation': 0.65, 'rhetoric_skill': 0.92, 'publications': 28},
    {'id': 'SCH-006', 'name': 'Fibon Arrel', 'school': 'pebble_school',
     'birth_year': 810, 'reputation': 0.7, 'rhetoric_skill': 0.68, 'publications': 35},
    {'id': 'SCH-007', 'name': 'Eulr Voss', 'school': 'water_school',
     'birth_year': 790, 'reputation': 0.82, 'rhetoric_skill': 0.78, 'publications': 48},
    {'id': 'SCH-008', 'name': 'Koro Sandis', 'school': 'stone_school',
     'birth_year': 830, 'reputation': 0.78, 'rhetoric_skill': 0.81, 'publications': 39},
    {'id': 'SCH-009', 'name': 'Tova Melk', 'school': 'stone_school',
     'birth_year': 860, 'reputation': 0.68, 'rhetoric_skill': 0.76, 'publications': 22},
    {'id': 'SCH-010', 'name': 'Lunna Gresk', 'school': 'pebble_school',
     'birth_year': 838, 'reputation': 0.72, 'rhetoric_skill': 0.7, 'publications': 33},
    {'id': 'SCH-011', 'name': 'Havro Molt', 'school': 'stone_school',
     'birth_year': 820, 'reputation': 0.88, 'rhetoric_skill': 0.84, 'publications': 58},
    {'id': 'SCH-012', 'name': 'Pilar Senn', 'school': 'water_school',
     'birth_year': 845, 'reputation': 0.73, 'rhetoric_skill': 0.77, 'publications': 27},
    {'id': 'SCH-013', 'name': 'Ursi Dent', 'school': 'pebble_school',
     'birth_year': 852, 'reputation': 0.58, 'rhetoric_skill': 0.65, 'publications': 15},
    {'id': 'SCH-014', 'name': 'Grent Tolliver', 'school': 'stone_school',
     'birth_year': 835, 'reputation': 0.8, 'rhetoric_skill': 0.79, 'publications': 41},
    {'id': 'SCH-015', 'name': 'Willa Harkon', 'school': 'water_school',
     'birth_year': 848, 'reputation': 0.7, 'rhetoric_skill': 0.82, 'publications': 29},
    {'id': 'SCH-016', 'name': 'Ferris Olt', 'school': 'none',
     'birth_year': 862, 'reputation': 0.55, 'rhetoric_skill': 0.6, 'publications': 8},
    {'id': 'SCH-017', 'name': 'Maxa Krent', 'school': 'water_school',
     'birth_year': 870, 'reputation': 0.62, 'rhetoric_skill': 0.74, 'publications': 12},
    {'id': 'SCH-018', 'name': 'Solvar Hent', 'school': 'stone_school',
     'birth_year': 825, 'reputation': 0.85, 'rhetoric_skill': 0.8, 'publications': 47},
]

# Debate venues
VENUES = [
    {'name': 'Grand Archive Hall', 'prestige': 0.95, 'capacity': 500},
    {'name': 'Senate Chamber', 'prestige': 0.9, 'capacity': 200},
    {'name': 'University Hall', 'prestige': 0.75, 'capacity': 300},
    {'name': 'Public Square', 'prestige': 0.5, 'capacity': 1000},
    {'name': 'Archivist Shop', 'prestige': 0.4, 'capacity': 50},
    {'name': 'Senator Huilof Estate', 'prestige': 0.85, 'capacity': 100},
]

# Debate topics
TOPICS = [
    'metaphysics', 'natural_philosophy', 'ethics', 'politics',
    'aesthetics', 'epistemology', 'history', 'cosmology'
]

# Topic advantages by school (small effects)
TOPIC_SCHOOL_ADVANTAGE = {
    'metaphysics': {'stone_school': 0.03, 'water_school': 0.01, 'pebble_school': -0.02},
    'natural_philosophy': {'stone_school': 0.01, 'water_school': 0.02, 'pebble_school': 0.0},
    'ethics': {'stone_school': 0.02, 'water_school': -0.01, 'pebble_school': 0.02},
    'politics': {'stone_school': 0.0, 'water_school': 0.01, 'pebble_school': 0.01},
    'aesthetics': {'stone_school': -0.01, 'water_school': 0.03, 'pebble_school': 0.0},
    'epistemology': {'stone_school': 0.02, 'water_school': 0.0, 'pebble_school': 0.0},
    'history': {'stone_school': 0.01, 'water_school': 0.0, 'pebble_school': 0.01},
    'cosmology': {'stone_school': 0.0, 'water_school': 0.02, 'pebble_school': -0.01},
}


def get_scholar_age(scholar: dict, year: int) -> int:
    """Calculate scholar's age in a given year."""
    return year - scholar['birth_year']


def calculate_win_probability(scholar_a: dict, scholar_b: dict,
                              topic: str, year: int, venue: dict) -> float:
    """
    Calculate probability that scholar_a wins the debate.

    This function encodes the TRUE effects in the data:
    - School prestige matters slightly (~5% effect)
    - Rhetoric skill matters moderately
    - Age/experience matters slightly
    - Publications matter slightly
    - Topic-school match matters slightly

    But there's a LOT of noise, so these effects are hard to detect.
    """
    base_prob = 0.5

    # Get schools
    school_a = scholar_a.get('school', 'none')
    school_b = scholar_b.get('school', 'none')

    # School prestige effect (TRUE but small)
    prestige_a = SCHOOL_PARAMS.get(school_a, {}).get('prestige_modifier', 1.0)
    prestige_b = SCHOOL_PARAMS.get(school_b, {}).get('prestige_modifier', 1.0)
    school_effect = (prestige_a - prestige_b) * 0.3

    # Rhetoric skill effect (TRUE and moderate)
    rhetoric_effect = (scholar_a['rhetoric_skill'] - scholar_b['rhetoric_skill']) * 0.25

    # Age effect (older scholars have slight advantage up to ~60, then decline)
    age_a = get_scholar_age(scholar_a, year)
    age_b = get_scholar_age(scholar_b, year)

    def age_modifier(age):
        if age < 25:
            return -0.05
        elif age < 40:
            return 0.02
        elif age < 55:
            return 0.05
        elif age < 70:
            return 0.02
        else:
            return -0.03

    age_effect = age_modifier(age_a) - age_modifier(age_b)

    # Publication effect (more publications = slight advantage)
    pub_effect = np.tanh((scholar_a['publications'] - scholar_b['publications']) / 30) * 0.08

    # Topic-school match effect
    topic_adv_a = TOPIC_SCHOOL_ADVANTAGE.get(topic, {}).get(school_a, 0)
    topic_adv_b = TOPIC_SCHOOL_ADVANTAGE.get(topic, {}).get(school_b, 0)
    topic_effect = topic_adv_a - topic_adv_b

    # Combine effects
    total_effect = school_effect + rhetoric_effect + age_effect + pub_effect + topic_effect

    # Add significant noise (makes effects hard to detect)
    noise = np.random.normal(0, 0.15)

    # Convert to probability
    prob = base_prob + total_effect + noise
    return np.clip(prob, 0.05, 0.95)


def simulate_debate(debate_id: int, year: int, month: int) -> dict:
    """Simulate a single scholarly debate."""

    # Select two different scholars who were active in this year
    active_scholars = [s for s in SCHOLARS
                       if s['birth_year'] + 25 <= year <= s['birth_year'] + 80]

    if len(active_scholars) < 2:
        return None

    scholar_a, scholar_b = np.random.choice(active_scholars, size=2, replace=False)

    # Select venue (weighted by prestige)
    venue_weights = np.array([v['prestige'] for v in VENUES])
    venue_weights = venue_weights / venue_weights.sum()
    venue = np.random.choice(VENUES, p=venue_weights)

    # Select topic
    topic = np.random.choice(TOPICS)

    # Select judge count (3, 5, or 7)
    judge_count = np.random.choice([3, 5, 7], p=[0.3, 0.5, 0.2])

    # Audience size (depends on venue and topic popularity)
    base_audience = int(venue['capacity'] * np.random.uniform(0.3, 0.9))
    if topic in ['politics', 'ethics']:
        base_audience = int(base_audience * 1.2)
    audience_size = max(20, min(base_audience, venue['capacity']))

    # Calculate outcome
    win_prob_a = calculate_win_probability(scholar_a, scholar_b, topic, year, venue)

    # Simulate the vote
    votes_a = np.random.binomial(judge_count, win_prob_a)
    votes_b = judge_count - votes_a

    if votes_a > votes_b:
        outcome = 'victory_a'
        margin = 'decisive' if votes_a - votes_b >= 3 else 'narrow'
    elif votes_b > votes_a:
        outcome = 'victory_b'
        margin = 'decisive' if votes_b - votes_a >= 3 else 'narrow'
    else:
        outcome = 'draw'
        margin = 'split'

    # Controversy score (higher when close vote or controversial topic)
    base_controversy = 0.3 if margin == 'decisive' else 0.5 if margin == 'narrow' else 0.7
    if topic in ['politics', 'ethics']:
        base_controversy += 0.1
    controversy = np.clip(base_controversy + np.random.normal(0, 0.15), 0, 1)

    # Appeals (more likely when controversial)
    appeals = np.random.binomial(3, controversy * 0.2)

    # Citations in following years (more prestigious = more citations)
    citation_base = (scholar_a['reputation'] + scholar_b['reputation']) / 2 * 20
    citations = max(0, int(citation_base * np.random.lognormal(0, 0.5)))

    # Duration (hours)
    duration = max(1.0, np.random.normal(3.5, 1.5))

    # Generate notes
    notes = generate_debate_notes(scholar_a, scholar_b, outcome, margin, topic)

    return {
        'debate_id': f'DEB{debate_id:03d}',
        'year': year,
        'month': month,
        'venue': venue['name'],
        'topic_category': topic,
        'scholar_a_id': scholar_a['id'],
        'scholar_a_name': scholar_a['name'],
        'scholar_a_school': scholar_a['school'],
        'scholar_a_age': get_scholar_age(scholar_a, year),
        'scholar_a_publications': scholar_a['publications'],
        'scholar_b_id': scholar_b['id'],
        'scholar_b_name': scholar_b['name'],
        'scholar_b_school': scholar_b['school'],
        'scholar_b_age': get_scholar_age(scholar_b, year),
        'scholar_b_publications': scholar_b['publications'],
        'judge_count': judge_count,
        'audience_size': audience_size,
        'outcome': outcome,
        'margin': margin,
        'votes_for_a': votes_a,
        'votes_for_b': votes_b,
        'controversy_score': round(controversy, 2),
        'appeals_filed': appeals,
        'citations_after_5yr': citations,
        'duration_hours': round(duration, 1),
        'notes': notes
    }


def generate_debate_notes(scholar_a: dict, scholar_b: dict,
                          outcome: str, margin: str, topic: str) -> str:
    """Generate narrative notes for the debate."""
    notes = []

    if margin == 'decisive':
        notes.append("Clear victory")
    elif margin == 'narrow':
        notes.append("Closely contested")
    else:
        notes.append("Judges split evenly")

    if scholar_a['school'] != scholar_b['school'] and scholar_a['school'] != 'none' and scholar_b['school'] != 'none':
        notes.append(f"{SCHOOL_PARAMS[scholar_a['school']]['name']} vs {SCHOOL_PARAMS[scholar_b['school']]['name']}")

    if topic == 'politics':
        notes.append("Senate observers present")

    return "; ".join(notes) if notes else "Standard proceedings"


def generate_scholar_debates_dataset(n_debates: int = 250,
                                     start_year: int = 850,
                                     end_year: int = 900) -> pd.DataFrame:
    """
    Generate a full dataset of scholarly debates.

    Designed with specific properties for teaching:
    1. TRUE effect: Stone School scholars win ~55% (vs 50% baseline)
    2. FALSE patterns: Many other factors will appear significant by chance
    3. Multiple comparisons trap: Students who test many hypotheses will find ~5% significant
    """

    debates = []
    debate_id = 1

    for year in range(start_year, end_year + 1):
        # Number of debates this year (varies)
        n_year = np.random.poisson(n_debates / (end_year - start_year + 1))
        n_year = max(1, min(n_year, 15))

        for _ in range(n_year):
            month = np.random.randint(1, 13)
            debate = simulate_debate(debate_id, year, month)

            if debate is not None:
                debates.append(debate)
                debate_id += 1

    df = pd.DataFrame(debates)
    df = df.sort_values(['year', 'month', 'debate_id']).reset_index(drop=True)

    return df


def analyze_for_teaching(df: pd.DataFrame) -> dict:
    """
    Analyze dataset to verify teaching properties.
    This is for validation, not for students to see.
    """

    # Stone school win rate
    stone_as_a = df[df['scholar_a_school'] == 'stone_school']
    stone_wins_a = (stone_as_a['outcome'] == 'victory_a').sum()
    stone_total_a = len(stone_as_a)

    stone_as_b = df[df['scholar_b_school'] == 'stone_school']
    stone_wins_b = (stone_as_b['outcome'] == 'victory_b').sum()
    stone_total_b = len(stone_as_b)

    stone_win_rate = (stone_wins_a + stone_wins_b) / (stone_total_a + stone_total_b) if (stone_total_a + stone_total_b) > 0 else 0

    # Water school win rate
    water_as_a = df[df['scholar_a_school'] == 'water_school']
    water_wins_a = (water_as_a['outcome'] == 'victory_a').sum()
    water_total_a = len(water_as_a)

    water_as_b = df[df['scholar_b_school'] == 'water_school']
    water_wins_b = (water_as_b['outcome'] == 'victory_b').sum()
    water_total_b = len(water_as_b)

    water_win_rate = (water_wins_a + water_wins_b) / (water_total_a + water_total_b) if (water_total_a + water_total_b) > 0 else 0

    return {
        'stone_school_win_rate': stone_win_rate,
        'water_school_win_rate': water_win_rate,
        'total_debates': len(df),
        'draws': (df['outcome'] == 'draw').sum()
    }


if __name__ == '__main__':
    print("Generating scholar debates dataset...")
    print("=" * 60)

    df = generate_scholar_debates_dataset(n_debates=250)

    # Save to data directory
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'scholar_debates.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} debates to {output_path}")

    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"Total debates: {len(df)}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique scholars: {pd.concat([df['scholar_a_id'], df['scholar_b_id']]).nunique()}")

    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())

    print(f"\nDebates by topic:")
    print(df['topic_category'].value_counts())

    print(f"\nDebates by venue:")
    print(df['venue'].value_counts())

    # Analyze for teaching validation
    analysis = analyze_for_teaching(df)
    print("\n--- Teaching Validation (hidden from students) ---")
    print(f"Stone School win rate: {analysis['stone_school_win_rate']:.1%}")
    print(f"Water School win rate: {analysis['water_school_win_rate']:.1%}")
    print(f"Draws: {analysis['draws']} ({analysis['draws']/len(df):.1%})")

    print("\n--- Teaching Applications ---")
    print("1. Hypothesis testing: 'Do Stone scholars win more?' (TRUE effect ~55%)")
    print("2. P-hacking: Test 20 features and find ~1 'significant' by chance")
    print("3. Multiple comparisons: Bonferroni correction demonstration")
    print("4. Effect size: The effect is real but SMALL - hard to detect")
