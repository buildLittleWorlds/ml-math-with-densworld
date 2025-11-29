"""
Expedition Outcomes Simulator

Generates synthetic expedition data based on the Yeller Quarry world parameters.
Used throughout the ML Math curriculum for statistics, regression, and classification examples.

World Parameters (from ore files):
- Expeditions have crews of 8-25 people
- The Boss (red-haired woman) leads with Truck as lieutenant
- Yeller groups (the five) have special roles - very rare
- Creatures attack unpredictably: hornets, wharvers, grimslews
- Success depends on leadership, weather, and luck
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# WORLD PARAMETERS (extracted from ore files)
# =============================================================================

SEASONS = ['spring', 'summer', 'autumn', 'winter']
SEASON_DANGER_MODIFIER = {
    'spring': 1.0,    # Normal
    'summer': 0.8,    # Creatures sluggish in heat
    'autumn': 1.2,    # Creatures preparing for winter, aggressive
    'winter': 1.5     # Desperate creatures, harsh conditions
}

SECTORS = [
    'Western Marsh', 'Eastern Caves', 'Deep Quarry',
    'Surface Flats', 'Miasto Border', 'Dens Edge'
]
SECTOR_DANGER = {
    'Western Marsh': 1.0,
    'Eastern Caves': 1.3,
    'Deep Quarry': 1.8,
    'Surface Flats': 0.6,
    'Miasto Border': 0.7,
    'Dens Edge': 2.0
}
SECTOR_VALUE_MODIFIER = {
    'Western Marsh': 1.0,
    'Eastern Caves': 1.4,
    'Deep Quarry': 2.2,
    'Surface Flats': 0.5,
    'Miasto Border': 0.8,
    'Dens Edge': 2.5
}

CREW_SPECIALTIES = [
    'deep_cave', 'surface_trap', 'general', 'yeller_hunting', 'amalgam_extraction'
]

# Named leaders from the ore (these appear in a small fraction of expeditions)
NOTABLE_LEADERS = [
    ('The Boss', 15, True),   # (name, experience_years, is_exceptional)
    ('Gull', 8, False),
    ('Kerrick', 25, True),
    ('Black Yeller', None, True),  # Unknown experience, exceptional
]


def simulate_expedition(expedition_id: int, year: int) -> dict:
    """Simulate a single expedition with realistic parameters."""

    # Basic expedition parameters
    season = np.random.choice(SEASONS)
    sector = np.random.choice(SECTORS)

    # Crew composition
    crew_size = np.random.randint(8, 26)

    # Leader experience: exponential distribution (most are inexperienced)
    # Mean of ~7 years, but long tail
    leader_experience = max(1, int(np.random.exponential(scale=7)))

    # Is this a Yeller group expedition? (very rare - 10%)
    has_yeller_group = np.random.random() < 0.10

    # Specialty matching sector
    if sector == 'Deep Quarry':
        specialty = np.random.choice(['deep_cave', 'amalgam_extraction', 'general'], p=[0.5, 0.3, 0.2])
    elif sector == 'Surface Flats':
        specialty = np.random.choice(['surface_trap', 'general'], p=[0.7, 0.3])
    else:
        specialty = np.random.choice(CREW_SPECIALTIES)

    # Days in field: normal distribution centered on 21 days
    days_in_field = max(7, int(np.random.normal(21, 7)))

    # Number of creature encounters: Poisson distribution
    base_encounters = 3
    encounter_rate = base_encounters * SEASON_DANGER_MODIFIER[season] * SECTOR_DANGER[sector]
    creature_encounters = np.random.poisson(lam=encounter_rate)

    # Casualties depend on encounters, crew skill, and luck
    casualty_probability_per_encounter = 0.15  # base 15% per encounter

    # Modifiers
    if leader_experience > 10:
        casualty_probability_per_encounter *= 0.7  # experienced leader helps
    if has_yeller_group:
        casualty_probability_per_encounter *= 0.5  # yeller groups are protective
    if specialty == 'deep_cave' and sector == 'Deep Quarry':
        casualty_probability_per_encounter *= 0.8  # specialty match helps

    # Simulate each encounter
    casualties = 0
    for _ in range(creature_encounters):
        if np.random.random() < casualty_probability_per_encounter:
            # 1-3 casualties per dangerous encounter
            casualties += np.random.randint(1, 4)

    # Can't have more casualties than crew
    casualties = min(casualties, crew_size - 1)  # At least 1 survivor to report

    # Catch value: log-normal distribution (most modest, few jackpots)
    base_catch = np.random.lognormal(mean=4, sigma=1)  # median ~55, long tail

    # Modifiers for catch value
    catch_value = base_catch * SECTOR_VALUE_MODIFIER[sector]
    catch_value *= (days_in_field / 21)  # longer expeditions catch more
    catch_value *= (1 - casualties / crew_size)  # casualties reduce effectiveness
    if has_yeller_group:
        catch_value *= 1.5  # yeller groups find better specimens

    # Add some noise
    catch_value *= np.random.uniform(0.7, 1.3)
    catch_value = max(0, catch_value)

    # Success definition: less than 30% casualties AND positive catch value
    success = (casualties < crew_size * 0.3) and (catch_value > 20)

    # Equipment condition at return (1.0 = perfect, 0.0 = destroyed)
    equipment_condition = max(0, 1.0 - (casualties / crew_size) - np.random.uniform(0, 0.3))

    # Morale at return (affected by casualties and success)
    base_morale = 0.7
    morale = base_morale - (casualties / crew_size) * 0.5 + (0.2 if success else -0.1)
    morale = np.clip(morale + np.random.normal(0, 0.1), 0, 1)

    return {
        'expedition_id': f'EXP-{expedition_id:04d}',
        'year': year,
        'season': season,
        'sector': sector,
        'crew_size': crew_size,
        'leader_experience_years': leader_experience,
        'has_yeller_group': has_yeller_group,
        'specialty': specialty,
        'days_in_field': days_in_field,
        'creature_encounters': creature_encounters,
        'casualties': casualties,
        'catch_value': round(catch_value, 2),
        'equipment_condition': round(equipment_condition, 2),
        'morale_at_return': round(morale, 2),
        'success': success
    }


def generate_expedition_dataset(n_expeditions: int = 1000,
                                 start_year: int = 1840,
                                 end_year: int = 1860) -> pd.DataFrame:
    """Generate a full dataset of expedition outcomes."""

    expeditions = []

    for i in range(n_expeditions):
        year = np.random.randint(start_year, end_year + 1)
        expedition = simulate_expedition(i + 1, year)
        expeditions.append(expedition)

    df = pd.DataFrame(expeditions)

    # Sort by year for temporal coherence
    df = df.sort_values(['year', 'expedition_id']).reset_index(drop=True)

    return df


def create_train_test_split(df: pd.DataFrame,
                            test_size: float = 0.2,
                            validation_size: float = 0.1) -> tuple:
    """Create train/test/validation splits."""

    n = len(df)
    indices = np.random.permutation(n)

    test_end = int(n * test_size)
    val_end = test_end + int(n * validation_size)

    test_idx = indices[:test_end]
    val_idx = indices[test_end:val_end]
    train_idx = indices[val_end:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True)
    )


if __name__ == '__main__':
    # Generate the dataset
    print("Generating expedition outcomes dataset...")
    df = generate_expedition_dataset(n_expeditions=1000)

    # Save to data directory
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    # Save full dataset
    df.to_csv(output_dir / 'expedition_outcomes.csv', index=False)
    print(f"Saved {len(df)} expeditions to {output_dir / 'expedition_outcomes.csv'}")

    # Create and save splits
    train_df, test_df, val_df = create_train_test_split(df)
    train_df.to_csv(output_dir / 'expedition_outcomes_train.csv', index=False)
    test_df.to_csv(output_dir / 'expedition_outcomes_test.csv', index=False)
    val_df.to_csv(output_dir / 'expedition_outcomes_val.csv', index=False)
    print(f"Saved splits: train={len(train_df)}, test={len(test_df)}, val={len(val_df)}")

    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"Success rate: {df['success'].mean():.1%}")
    print(f"Average casualties: {df['casualties'].mean():.1f}")
    print(f"Average catch value: {df['catch_value'].mean():.1f}")
    print(f"Expeditions with Yeller groups: {df['has_yeller_group'].mean():.1%}")
    print(f"\nCatch value distribution:")
    print(df['catch_value'].describe())
