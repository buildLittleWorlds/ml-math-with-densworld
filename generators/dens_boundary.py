"""
Dens Boundary Drift Simulator

Generates synthetic mapmaker observation data based on the Dens world parameters.
Used throughout the ML Math curriculum for statistics (signal vs. noise, CLT, confidence intervals).

World Parameters (from dens ore.txt):
- "Yesterday's map is gossip. Tomorrow's map is Senate conspiracy."
- Mapmakers are essential: "Every cousin's uncle has a friend working his hand at mapmaking"
- The Dens boundary shifts constantly, villages sink into densmuck
- The Colonel (before he was a soldier) was a mapmaker near the Dens
- Experienced mapmakers have better precision
- Ground can collapse within a week if densgrit is present

Key Quote:
"Living in earth or by earth that's always shifting, always losing its edges and
forgetting where and what it was, one needs maps like the newspapers of the Capital."

Teaching Applications:
- Module 1: Signal vs. noise, measurement error, CLT, confidence intervals
- Module 3: Finding true boundary position (optimization)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# WORLD PARAMETERS (extracted from dens ore.txt)
# =============================================================================

# Mapmaker characteristics
MAPMAKERS = [
    {'id': 'MAP001', 'name': 'Ferrick Mol', 'experience_years': 22, 'instrument': 'theodolite'},
    {'id': 'MAP002', 'name': 'Young Bastian', 'experience_years': 3, 'instrument': 'measuring_rod'},
    {'id': 'MAP003', 'name': 'Corra Venst', 'experience_years': 15, 'instrument': 'theodolite'},
    {'id': 'MAP004', 'name': 'The Pickbox Man', 'experience_years': 35, 'instrument': 'theodolite'},  # The wandering mapmaker from ore
    {'id': 'MAP005', 'name': 'Ulla Grent', 'experience_years': 8, 'instrument': 'measuring_rod'},
    {'id': 'MAP006', 'name': 'Drogan Pol', 'experience_years': 5, 'instrument': 'pacing'},
    {'id': 'MAP007', 'name': 'Sister Kalma', 'experience_years': 12, 'instrument': 'theodolite'},
    {'id': 'MAP008', 'name': 'Hoff the Younger', 'experience_years': 2, 'instrument': 'pacing'},
    {'id': 'MAP009', 'name': 'Marga Denside', 'experience_years': 18, 'instrument': 'theodolite'},
    {'id': 'MAP010', 'name': 'Pennick Harvor', 'experience_years': 7, 'instrument': 'measuring_rod'},
    {'id': 'MAP011', 'name': 'Old Tunney', 'experience_years': 40, 'instrument': 'theodolite'},
    {'id': 'MAP012', 'name': 'Grit-Eye Salla', 'experience_years': 10, 'instrument': 'measuring_rod'},
    {'id': 'MAP013', 'name': 'Karvo the Silent', 'experience_years': 6, 'instrument': 'pacing'},
    {'id': 'MAP014', 'name': 'Archive Boy', 'experience_years': 1, 'instrument': 'pacing'},
    {'id': 'MAP015', 'name': 'Missus Hollow', 'experience_years': 28, 'instrument': 'theodolite'},
]

# Instrument precision (standard deviation of measurement error in meters)
INSTRUMENT_NOISE = {
    'theodolite': 0.8,      # Most precise
    'measuring_rod': 1.5,   # Medium precision
    'pacing': 3.0           # Least precise (walking and counting)
}

# Weather affects visibility and therefore precision
WEATHER_CONDITIONS = ['clear', 'overcast', 'rain', 'fog']
WEATHER_NOISE_MULTIPLIER = {
    'clear': 1.0,
    'overcast': 1.2,
    'rain': 1.8,
    'fog': 2.5
}

# Seasons affect drift rate (densmuck expands in wet seasons)
SEASONS = ['spring', 'summer', 'autumn', 'winter']
SEASON_DRIFT_MODIFIER = {
    'spring': 1.4,   # Wet season, more drift
    'summer': 0.8,   # Dry, more stable
    'autumn': 1.0,   # Normal
    'winter': 1.3    # Frost heave and spring prep
}

# The Dens boundary exists as a rough circle/blob around center (50, 50)
DENS_CENTER_X = 50.0  # km
DENS_CENTER_Y = 50.0  # km
INITIAL_DENS_RADIUS = 20.0  # km from center to boundary (year 1845)
BOUNDARY_EXPANSION_RATE = 0.15  # km per year (densmuck slowly expands)


def true_boundary_position(x: float, y: float, year: int, month: int) -> float:
    """
    Calculate the TRUE ground stability at a location.
    This is the latent value that mapmakers are trying to measure.

    Returns: stability from 0 (pure densmuck) to 1 (solid ground)
    """
    # Distance from Dens center
    dist_from_center = np.sqrt((x - DENS_CENTER_X)**2 + (y - DENS_CENTER_Y)**2)

    # Boundary expands over time
    years_since_start = year - 1845
    boundary_radius = INITIAL_DENS_RADIUS + BOUNDARY_EXPANSION_RATE * years_since_start

    # Add seasonal variation to boundary
    season_month = month % 12
    if season_month in [2, 3, 4]:  # Spring
        boundary_radius *= 1.05
    elif season_month in [5, 6, 7]:  # Summer
        boundary_radius *= 0.98
    elif season_month in [11, 0, 1]:  # Winter
        boundary_radius *= 1.02

    # Add some spatial irregularity (boundary isn't a perfect circle)
    angle = np.arctan2(y - DENS_CENTER_Y, x - DENS_CENTER_X)
    irregularity = 2.0 * np.sin(3 * angle) + 1.5 * np.cos(5 * angle)  # 2-5 km wobble
    boundary_radius += irregularity

    # Calculate stability based on distance from boundary
    if dist_from_center < boundary_radius - 5:
        # Deep in Dens - pure densmuck
        return 0.0
    elif dist_from_center > boundary_radius + 10:
        # Far from Dens - solid ground
        return 1.0
    else:
        # Transition zone - stability varies smoothly
        normalized_dist = (dist_from_center - (boundary_radius - 5)) / 15.0
        # Sigmoid-like transition
        stability = 1 / (1 + np.exp(-5 * (normalized_dist - 0.5)))
        return float(np.clip(stability, 0, 1))


def calculate_drift(prev_stability: float, current_stability: float,
                    x: float, y: float, prev_x: float, prev_y: float) -> dict:
    """Calculate drift between observations at same location."""

    stability_change = current_stability - prev_stability

    if abs(stability_change) < 0.02:
        return {'detected': False, 'direction': 'none', 'magnitude': 0.0}

    # Estimate drift direction based on Dens center
    dx = x - DENS_CENTER_X
    dy = y - DENS_CENTER_Y

    if stability_change < 0:
        # Ground becoming less stable - Dens expanding toward this point
        if abs(dx) > abs(dy):
            direction = 'W' if dx > 0 else 'E'
        else:
            direction = 'S' if dy > 0 else 'N'
    else:
        # Ground becoming more stable - Dens retreating
        if abs(dx) > abs(dy):
            direction = 'E' if dx > 0 else 'W'
        else:
            direction = 'N' if dy > 0 else 'S'

    # Magnitude in meters (rough estimate)
    magnitude = abs(stability_change) * 50  # ~50m per 0.01 stability change

    return {
        'detected': True,
        'direction': direction,
        'magnitude': round(magnitude, 1)
    }


def simulate_observation(obs_id: int, year: int, month: int,
                         x: float, y: float, mapmaker: dict,
                         weather: str, days_since_survey: int,
                         prev_stability: float = None) -> dict:
    """
    Simulate a single mapmaker observation.

    The mapmaker measures ground stability but introduces measurement error
    based on their experience, instrument, and weather conditions.
    """

    # Get true stability (the signal)
    true_stability = true_boundary_position(x, y, year, month)

    # Calculate measurement noise (the noise)
    base_noise = INSTRUMENT_NOISE[mapmaker['instrument']]

    # Experience reduces noise (experienced mapmakers are more precise)
    experience_factor = 1 / (1 + 0.05 * mapmaker['experience_years'])

    # Weather increases noise
    weather_factor = WEATHER_NOISE_MULTIPLIER[weather]

    # Combined noise standard deviation
    noise_std = base_noise * experience_factor * weather_factor

    # Observed stability = true + noise (but clipped to [0, 1])
    measurement_noise = np.random.normal(0, noise_std * 0.1)  # Scale noise to stability units
    observed_stability = np.clip(true_stability + measurement_noise, 0, 1)

    # Observer's self-assessed confidence (experienced mapmakers know when conditions are bad)
    base_confidence = 0.5 + 0.01 * mapmaker['experience_years']  # More experience = more confident
    confidence = base_confidence / weather_factor  # Bad weather reduces confidence
    confidence = np.clip(confidence + np.random.normal(0, 0.05), 0.1, 0.95)

    # Calculate drift if we have previous observation
    if prev_stability is not None:
        drift_info = calculate_drift(prev_stability, true_stability, x, y, x, y)
    else:
        drift_info = {'detected': False, 'direction': 'none', 'magnitude': 0.0}

    # Determine season from month
    if month in [3, 4, 5]:
        season = 'spring'
    elif month in [6, 7, 8]:
        season = 'summer'
    elif month in [9, 10, 11]:
        season = 'autumn'
    else:
        season = 'winter'

    # Generate field notes (flavor text)
    notes = generate_field_notes(true_stability, observed_stability, weather,
                                 mapmaker, drift_info['detected'])

    return {
        'observation_id': f'OBS{obs_id:04d}',
        'date': f'{year}-{month:02d}-{np.random.randint(1, 29):02d}',
        'year': year,
        'month': month,
        'season': season,
        'location_x': round(x, 2),
        'location_y': round(y, 2),
        'true_stability': round(true_stability, 3),  # Would be hidden in exercises
        'observed_stability': round(observed_stability, 3),
        'measurement_error': round(observed_stability - true_stability, 4),
        'observer_id': mapmaker['id'],
        'observer_name': mapmaker['name'],
        'observer_experience': mapmaker['experience_years'],
        'instrument_type': mapmaker['instrument'],
        'weather_conditions': weather,
        'days_since_last_survey': days_since_survey,
        'drift_detected': drift_info['detected'],
        'drift_direction': drift_info['direction'],
        'drift_magnitude_m': drift_info['magnitude'],
        'confidence_rating': round(confidence, 2),
        'notes': notes
    }


def generate_field_notes(true_stab: float, obs_stab: float, weather: str,
                         mapmaker: dict, drift: bool) -> str:
    """Generate narrative field notes for the observation."""

    notes = []

    # Stability observations
    if obs_stab < 0.2:
        notes.append("Ground feels unsafe, densmuck visible")
    elif obs_stab < 0.4:
        notes.append("Soft ground, footprints sink deep")
    elif obs_stab < 0.6:
        notes.append("Transition zone, patchy stability")
    elif obs_stab < 0.8:
        notes.append("Mostly firm ground with soft patches")
    else:
        notes.append("Solid ground, safe for building")

    # Weather notes
    if weather == 'fog':
        notes.append("visibility poor")
    elif weather == 'rain':
        notes.append("rain affecting measurements")

    # Drift observations
    if drift:
        notes.append("boundary appears to have shifted since last survey")

    # Experienced mapmaker observations
    if mapmaker['experience_years'] > 20:
        if abs(obs_stab - true_stab) > 0.1:
            notes.append("conditions made precise measurement difficult")

    return "; ".join(notes) if notes else "Standard observation"


def generate_dens_boundary_dataset(n_observations: int = 600,
                                   start_year: int = 1845,
                                   end_year: int = 1860) -> pd.DataFrame:
    """
    Generate a full dataset of mapmaker observations.

    The dataset includes:
    - Multiple mapmakers with different skill levels
    - Observations in the transition zone (where uncertainty matters)
    - Temporal patterns (seasonal, annual boundary expansion)
    - Clustering (multiple observations at same location)
    """

    observations = []
    obs_id = 1

    # Generate observation locations in the interesting zone (near boundary)
    # Most observations should be in the transition zone where uncertainty matters
    n_locations = n_observations // 5  # ~5 observations per location on average

    # Generate locations biased toward the boundary region
    locations = []
    for _ in range(n_locations):
        # Sample in polar coordinates centered on Dens
        # Distance from center: biased toward boundary (20-35 km range)
        r = INITIAL_DENS_RADIUS + np.random.normal(5, 8)  # 5 km outside boundary on average
        r = max(INITIAL_DENS_RADIUS - 10, min(r, INITIAL_DENS_RADIUS + 20))

        # Random angle
        theta = np.random.uniform(0, 2 * np.pi)

        x = DENS_CENTER_X + r * np.cos(theta)
        y = DENS_CENTER_Y + r * np.sin(theta)

        locations.append((x, y))

    # Track previous observations at each location for drift calculation
    location_history = {loc: [] for loc in locations}

    # Generate observations over time
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Number of observations this month (varies seasonally)
            n_this_month = np.random.poisson(n_observations / ((end_year - start_year + 1) * 12))
            n_this_month = max(1, min(n_this_month, 30))

            for _ in range(n_this_month):
                # Select a location
                loc_idx = np.random.randint(len(locations))
                x, y = locations[loc_idx]

                # Add small position noise (can't return to exact same spot)
                x += np.random.normal(0, 0.1)
                y += np.random.normal(0, 0.1)

                # Select a mapmaker (weighted by experience - more experienced work more)
                weights = np.array([m['experience_years'] for m in MAPMAKERS])
                weights = weights / weights.sum()
                mapmaker = np.random.choice(MAPMAKERS, p=weights)

                # Weather (weighted toward clear)
                weather = np.random.choice(
                    WEATHER_CONDITIONS,
                    p=[0.5, 0.25, 0.15, 0.10]
                )

                # Days since last survey at this location
                hist = location_history[locations[loc_idx]]
                if hist:
                    last_obs = hist[-1]
                    days_since = (year - last_obs['year']) * 365 + (month - last_obs['month']) * 30
                    prev_stab = last_obs['true_stability']
                else:
                    days_since = 365  # First observation
                    prev_stab = None

                # Generate the observation
                obs = simulate_observation(
                    obs_id, year, month, x, y, mapmaker, weather, days_since, prev_stab
                )

                observations.append(obs)

                # Update history
                location_history[locations[loc_idx]].append({
                    'year': year,
                    'month': month,
                    'true_stability': obs['true_stability']
                })

                obs_id += 1

    df = pd.DataFrame(observations)

    # Sort by date
    df = df.sort_values(['year', 'month', 'observation_id']).reset_index(drop=True)

    return df


if __name__ == '__main__':
    print("Generating Dens boundary observations dataset...")
    print("=" * 60)

    df = generate_dens_boundary_dataset(n_observations=600)

    # Save to data directory
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'dens_boundary_observations.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} observations to {output_path}")

    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"Total observations: {len(df)}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique locations: ~{df.groupby(['location_x', 'location_y']).ngroups}")
    print(f"Unique observers: {df['observer_id'].nunique()}")

    print(f"\nObserved stability distribution:")
    print(df['observed_stability'].describe())

    print(f"\nMeasurement error statistics:")
    print(f"  Mean error: {df['measurement_error'].mean():.4f}")
    print(f"  Std error: {df['measurement_error'].std():.4f}")
    print(f"  (Errors should be roughly normal with mean ~0)")

    print(f"\nDrift detected in {df['drift_detected'].sum()} observations ({df['drift_detected'].mean():.1%})")

    print(f"\nObservations by instrument type:")
    print(df['instrument_type'].value_counts())

    print(f"\nMeasurement error by instrument type:")
    print(df.groupby('instrument_type')['measurement_error'].agg(['mean', 'std']))

    print("\n--- Teaching Applications ---")
    print("1. Signal vs Noise: true_stability is signal, measurement_error is noise")
    print("2. CLT: Average many observations at same location to find true value")
    print("3. Confidence intervals: Use std error to bound estimates")
    print("4. Regression: Predict stability from location, weather, etc.")
