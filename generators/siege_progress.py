"""
Tower of Mirado Siege Progress Simulator

Generates siege data that demonstrates gradient descent concepts.
The Colonel's siege becomes a metaphor for optimization.

World Parameters (from ore files):
- The Colonel has been besieging the Tower for 15-25 years
- The Tower hovers above the desert; unreachable by conventional means
- Progress is measured in failed stratagems
- Supply requisitions to the Capital rarely return with supplies
- Morale decays over time; men desert or die of boredom
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# =============================================================================
# SIEGE WORLD PARAMETERS
# =============================================================================

STRATAGEM_TYPES = [
    'ladder_assault',     # Direct approach - often fails
    'grapple_attempt',    # Technical approach
    'tunnel',             # Slow but sometimes works
    'parley',             # Diplomatic (never works on Tower)
    'catapult',           # Long range
    'waiting',            # Do nothing - morale drops
    'balloon',            # Creative but risky
    'bribery'             # Try to corrupt Tower inhabitants
]

# Each stratagem has different properties in the "loss landscape"
STRATAGEM_EFFECTS = {
    'ladder_assault': {'progress_mean': 0.01, 'progress_std': 0.05, 'morale_cost': -0.02, 'supply_cost': 20},
    'grapple_attempt': {'progress_mean': 0.015, 'progress_std': 0.04, 'morale_cost': -0.01, 'supply_cost': 15},
    'tunnel': {'progress_mean': 0.02, 'progress_std': 0.02, 'morale_cost': -0.03, 'supply_cost': 30},
    'parley': {'progress_mean': 0.0, 'progress_std': 0.01, 'morale_cost': 0.01, 'supply_cost': 5},
    'catapult': {'progress_mean': 0.005, 'progress_std': 0.03, 'morale_cost': 0.0, 'supply_cost': 25},
    'waiting': {'progress_mean': -0.005, 'progress_std': 0.01, 'morale_cost': -0.04, 'supply_cost': 5},
    'balloon': {'progress_mean': 0.03, 'progress_std': 0.08, 'morale_cost': 0.02, 'supply_cost': 40},
    'bribery': {'progress_mean': 0.0, 'progress_std': 0.02, 'morale_cost': 0.0, 'supply_cost': 50}
}


def simulate_siege(n_months: int = 240) -> pd.DataFrame:
    """
    Simulate the siege as an optimization problem.

    The "loss" is (1 - progress_score): we want to minimize distance to breach.
    Each stratagem is a "step" in parameter space.
    The Colonel must estimate gradients without seeing the full landscape.
    """

    records = []

    # Initial state
    year = 1
    month = 1
    personnel = 500
    morale = 0.7
    supplies = 1000
    progress_score = 0.0  # 0 = no progress, 1 = Tower breached

    # The true "loss landscape" has a complex shape
    # Progress is possible but requires the right sequence of actions

    for i in range(n_months):
        # Decide on stratagem (simulating the Colonel's choices)
        # In reality, this would be the "gradient descent step"

        # Colonel's heuristic: try things that worked before, occasionally experiment
        if i == 0:
            stratagem = 'ladder_assault'  # Traditional first attempt
        elif morale < 0.3:
            stratagem = 'parley'  # Desperate times
        elif supplies < 100:
            stratagem = 'waiting'  # Can't do anything
        elif np.random.random() < 0.2:
            # Random exploration (like stochastic gradient descent)
            stratagem = np.random.choice(STRATAGEM_TYPES)
        else:
            # Exploit: choose based on past "success"
            # This is like following the gradient
            weights = []
            for s in STRATAGEM_TYPES:
                effect = STRATAGEM_EFFECTS[s]
                # Prefer high mean progress, low variance
                score = effect['progress_mean'] - 0.5 * effect['progress_std']
                score += 0.01 * effect['morale_cost']  # Consider morale
                if supplies < effect['supply_cost'] * 2:
                    score -= 0.1  # Penalize if low on supplies
                weights.append(max(0.01, score + 0.1))

            weights = np.array(weights) / sum(weights)
            stratagem = np.random.choice(STRATAGEM_TYPES, p=weights)

        effect = STRATAGEM_EFFECTS[stratagem]

        # Apply stratagem effects
        progress_delta = np.random.normal(effect['progress_mean'], effect['progress_std'])

        # Progress is bounded [0, 1] and gets harder as you get closer
        # This creates the "loss landscape" curvature
        difficulty_multiplier = 1 - progress_score * 0.5  # Harder near the goal
        progress_delta *= difficulty_multiplier

        progress_score = np.clip(progress_score + progress_delta, 0, 1)

        # Morale changes
        morale_delta = effect['morale_cost']
        morale_delta += np.random.normal(0, 0.02)  # Random fluctuation
        morale_delta += 0.05 * progress_delta  # Success boosts morale
        morale = np.clip(morale + morale_delta, 0.1, 1.0)

        # Supply changes
        supplies -= effect['supply_cost']
        supplies -= personnel * 0.5  # Monthly consumption

        # Random supply delivery (rare)
        if np.random.random() < 0.1:
            delivery = np.random.randint(100, 500)
            supplies += delivery

        supplies = max(0, supplies)

        # Personnel changes
        if morale < 0.3:
            desertions = np.random.poisson(5)
            personnel -= desertions
        if supplies < 50:
            starvation = np.random.poisson(3)
            personnel -= starvation

        # Random reinforcements (rare)
        if np.random.random() < 0.05:
            reinforcements = np.random.randint(20, 100)
            personnel += reinforcements

        personnel = max(50, personnel)  # Minimum skeleton crew

        # Tower response
        if stratagem == 'parley':
            tower_response = 'silence'
        elif progress_delta > 0.05:
            tower_response = 'repelled'
        elif np.random.random() < 0.1:
            tower_response = 'unknown'
        else:
            tower_response = 'none'

        # Calculate "loss" (what we're trying to minimize)
        loss = 1 - progress_score

        # Estimate "gradient" (the Colonel's sense of which way to go)
        # This is noisy - he can't see the true landscape
        estimated_gradient = -progress_delta + np.random.normal(0, 0.02)

        records.append({
            'year': year,
            'month': month,
            'month_total': i + 1,
            'personnel_count': personnel,
            'morale_index': round(morale, 3),
            'supply_level': round(supplies, 1),
            'stratagem_attempted': stratagem,
            'progress_score': round(progress_score, 4),
            'progress_delta': round(progress_delta, 4),
            'loss': round(loss, 4),
            'estimated_gradient': round(estimated_gradient, 4),
            'tower_response': tower_response
        })

        # Advance time
        month += 1
        if month > 12:
            month = 1
            year += 1

    return pd.DataFrame(records)


if __name__ == '__main__':
    print("Generating siege progress dataset...")
    df = simulate_siege(n_months=240)  # 20 years

    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / 'siege_progress.csv', index=False)
    print(f"Saved {len(df)} months to {output_dir / 'siege_progress.csv'}")

    print("\n--- Dataset Summary ---")
    print(f"Siege duration: {df['year'].max()} years")
    print(f"Final progress: {df['progress_score'].iloc[-1]:.2%}")
    print(f"Final loss: {df['loss'].iloc[-1]:.4f}")

    print(f"\nStratagem frequency:")
    print(df['stratagem_attempted'].value_counts())

    print(f"\nProgress by stratagem:")
    print(df.groupby('stratagem_attempted')['progress_delta'].mean().sort_values(ascending=False).round(4))

    print(f"\nMorale over time:")
    print(f"  Start: {df['morale_index'].iloc[0]:.2f}")
    print(f"  End: {df['morale_index'].iloc[-1]:.2f}")
    print(f"  Min: {df['morale_index'].min():.2f}")
