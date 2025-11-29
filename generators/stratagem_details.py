"""
Stratagem Details Generator

Generates detailed data about individual stratagems in the Colonel's siege of Mirado.
Enhances the siege_progress.csv with granular stratagem-level data for teaching
gradient descent concepts in depth.

World Parameters (from tower of mirado ore.txt):
- The Colonel's siege has lasted nearly 20 years
- The Tower hovers above the desert, defended by unknown means
- Fyncho is the Colonel's second-in-command (Sancho Panza figure)
- Each stratagem is an attempt to make progress toward breaching the Tower
- The Senator in the Capital hopes for victory to boost his political fortunes

Teaching Applications:
- Module 3: Gradient descent as optimization
- Module 3: Learning rate (step size) and its effects
- Module 3: Local minima and getting stuck
- Module 3: Momentum and consecutive similar strategies
- Module 3: Stochastic vs. batch gradient descent
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# =============================================================================
# WORLD PARAMETERS
# =============================================================================

# Stratagem types with detailed properties
STRATAGEMS = {
    'ladder_assault': {
        'description': 'Direct scaling attempt with wooden ladders',
        'risk_level': 0.7,
        'resource_cost': 20,
        'personnel_required': 50,
        'expected_progress': 0.01,
        'variance': 0.05,
        'best_conditions': 'clear_weather',
        'gradient_direction': 'direct'
    },
    'grappling_hook': {
        'description': 'Technical climbing with iron hooks and rope',
        'risk_level': 0.5,
        'resource_cost': 15,
        'personnel_required': 20,
        'expected_progress': 0.015,
        'variance': 0.04,
        'best_conditions': 'low_wind',
        'gradient_direction': 'technical'
    },
    'tunnel_approach': {
        'description': 'Underground excavation toward Tower base',
        'risk_level': 0.4,
        'resource_cost': 30,
        'personnel_required': 80,
        'expected_progress': 0.02,
        'variance': 0.02,
        'best_conditions': 'any',
        'gradient_direction': 'slow_steady'
    },
    'catapult_bombardment': {
        'description': 'Long-range stone and fire attacks',
        'risk_level': 0.3,
        'resource_cost': 25,
        'personnel_required': 15,
        'expected_progress': 0.005,
        'variance': 0.03,
        'best_conditions': 'clear_weather',
        'gradient_direction': 'probing'
    },
    'balloon_ascent': {
        'description': 'Hot air balloon approach to Tower heights',
        'risk_level': 0.9,
        'resource_cost': 40,
        'personnel_required': 5,
        'expected_progress': 0.03,
        'variance': 0.08,
        'best_conditions': 'no_wind',
        'gradient_direction': 'innovative'
    },
    'parley': {
        'description': 'Diplomatic negotiation attempt',
        'risk_level': 0.1,
        'resource_cost': 5,
        'personnel_required': 3,
        'expected_progress': 0.0,
        'variance': 0.01,
        'best_conditions': 'any',
        'gradient_direction': 'lateral'
    },
    'night_raid': {
        'description': 'Stealthy assault under cover of darkness',
        'risk_level': 0.8,
        'resource_cost': 15,
        'personnel_required': 30,
        'expected_progress': 0.02,
        'variance': 0.06,
        'best_conditions': 'new_moon',
        'gradient_direction': 'opportunistic'
    },
    'feint_west': {
        'description': 'Distraction maneuver to west while main force moves east',
        'risk_level': 0.5,
        'resource_cost': 20,
        'personnel_required': 40,
        'expected_progress': 0.01,
        'variance': 0.04,
        'best_conditions': 'clear_weather',
        'gradient_direction': 'indirect'
    },
    'waiting': {
        'description': 'Holding position, conserving resources',
        'risk_level': 0.05,
        'resource_cost': 5,
        'personnel_required': 0,
        'expected_progress': -0.005,
        'variance': 0.01,
        'best_conditions': 'any',
        'gradient_direction': 'none'
    },
    'reconnaissance': {
        'description': 'Scouting and intelligence gathering',
        'risk_level': 0.3,
        'resource_cost': 10,
        'personnel_required': 5,
        'expected_progress': 0.008,
        'variance': 0.015,
        'best_conditions': 'any',
        'gradient_direction': 'gradient_estimation'
    },
}

# Tower response types
TOWER_RESPONSES = ['silence', 'arrows', 'fire', 'negotiation', 'unknown_force']


def simulate_stratagem(stratagem_id: int, year: int, month: int,
                       stratagem_type: str, current_progress: float,
                       morale: float, supplies: float,
                       colonel_confidence: float) -> dict:
    """
    Simulate a single stratagem attempt with detailed gradient descent analogs.
    """

    strat = STRATAGEMS[stratagem_type]

    # Calculate true gradient at current progress
    # The "loss landscape" gets harder near the goal (diminishing returns)
    true_gradient = strat['expected_progress'] * (1 - current_progress * 0.6)

    # Colonel's estimated gradient (noisy)
    gradient_noise = np.random.normal(0, strat['variance'] * 0.5)
    estimated_gradient = true_gradient + gradient_noise

    # Step size (learning rate analog) - larger when confident, smaller when cautious
    base_step_size = 1.0
    if colonel_confidence > 0.7:
        step_size = base_step_size * 1.3  # Aggressive
    elif colonel_confidence < 0.3:
        step_size = base_step_size * 0.6  # Cautious
    else:
        step_size = base_step_size

    # Actual progress = step_size * (true_gradient + noise)
    execution_noise = np.random.normal(0, strat['variance'])
    actual_progress = step_size * (true_gradient + execution_noise)

    # Progress can be negative (setbacks)
    progress_before = current_progress
    progress_after = np.clip(current_progress + actual_progress, 0, 1)
    progress_delta = progress_after - progress_before

    # Gradient error (for teaching: how wrong was the estimate?)
    gradient_error = abs(estimated_gradient - true_gradient)

    # Casualties depend on risk and execution
    base_casualties = int(strat['personnel_required'] * strat['risk_level'] * 0.1)
    casualties = max(0, int(base_casualties + np.random.poisson(base_casualties * 0.3)))

    # Morale impact
    if progress_delta > 0.01:
        morale_impact = 0.05 + np.random.normal(0, 0.02)
    elif progress_delta < -0.005:
        morale_impact = -0.08 + np.random.normal(0, 0.02)
    else:
        morale_impact = -0.02 + np.random.normal(0, 0.01)

    # Tower response
    if stratagem_type == 'parley':
        tower_response = np.random.choice(['silence', 'negotiation'], p=[0.8, 0.2])
    elif stratagem_type in ['balloon_ascent', 'night_raid']:
        tower_response = np.random.choice(TOWER_RESPONSES, p=[0.1, 0.3, 0.3, 0.05, 0.25])
    else:
        tower_response = np.random.choice(['silence', 'arrows', 'fire'], p=[0.4, 0.4, 0.2])

    # Outcome categorization
    if progress_delta > 0.02:
        outcome = 'success'
    elif progress_delta > 0:
        outcome = 'partial'
    elif progress_delta > -0.01:
        outcome = 'failure'
    else:
        outcome = 'disaster'

    # Generate narrative description
    description = generate_stratagem_narrative(
        stratagem_type, outcome, tower_response, casualties, strat
    )

    return {
        'stratagem_id': f'STR{stratagem_id:03d}',
        'year': year,
        'month': month,
        'stratagem_type': stratagem_type,
        'description': description,
        'personnel_committed': strat['personnel_required'],
        'supply_cost': strat['resource_cost'],
        'risk_level': strat['risk_level'],
        'colonel_confidence': round(colonel_confidence, 2),
        # Gradient descent analogs
        'estimated_gradient': round(estimated_gradient, 5),
        'actual_gradient': round(true_gradient, 5),
        'gradient_error': round(gradient_error, 5),
        'step_size': round(step_size, 2),
        'progress_before': round(progress_before, 4),
        'progress_after': round(progress_after, 4),
        'progress_delta': round(progress_delta, 5),
        # Loss function analog
        'loss_before': round(1 - progress_before, 4),
        'loss_after': round(1 - progress_after, 4),
        # Outcomes
        'outcome_category': outcome,
        'casualties': casualties,
        'morale_impact': round(morale_impact, 3),
        'tower_response': tower_response,
        # Meta information
        'gradient_direction': strat['gradient_direction'],
        'was_optimal_direction': estimated_gradient > 0 and true_gradient > 0
    }


def generate_stratagem_narrative(stratagem_type: str, outcome: str,
                                 tower_response: str, casualties: int,
                                 strat: dict) -> str:
    """Generate brief narrative description of the stratagem."""

    narratives = {
        'ladder_assault': {
            'success': "Ladders reached halfway up Tower wall before withdrawal",
            'partial': "Brief foothold gained; men repelled by defenders",
            'failure': "Ladders knocked away by unseen force",
            'disaster': "Mass casualties as Tower responded with fire"
        },
        'grappling_hook': {
            'success': "Hooks found purchase; rope climb achieved new height",
            'partial': "Several hooks held; limited reconnaissance gathered",
            'failure': "Hooks failed to catch on Tower's smooth surface",
            'disaster': "Rope severed by unknown means; climbers fell"
        },
        'tunnel_approach': {
            'success': "Tunnel advanced significantly toward Tower foundation",
            'partial': "Slow progress through rocky substrate",
            'failure': "Tunnel collapsed; work must restart",
            'disaster': "Cave-in trapped tunnel crew"
        },
        'balloon_ascent': {
            'success': "Balloon achieved unprecedented altitude near Tower",
            'partial': "Strong winds limited approach but gathered intelligence",
            'failure': "Balloon caught fire; crew evacuated safely",
            'disaster': "Balloon destroyed by Tower's defenses"
        },
        'parley': {
            'success': "Tower occupants acknowledged our presence",
            'partial': "Silence from Tower but no hostile response",
            'failure': "Parley ignored completely",
            'disaster': "Negotiators detained by unknown parties"
        },
        'night_raid': {
            'success': "Raid achieved surprise; intel gathered before retreat",
            'partial': "Partial infiltration before detection",
            'failure': "Detected early; forced retreat",
            'disaster': "Ambushed; heavy casualties"
        },
        'catapult_bombardment': {
            'success': "Multiple hits observed on Tower exterior",
            'partial': "Some projectiles reached Tower height",
            'failure': "Projectiles fell short of Tower",
            'disaster': "Catapult malfunction injured crew"
        },
        'feint_west': {
            'success': "Distraction worked; gained intelligence on defenses",
            'partial': "Tower responded to feint; main force undetected",
            'failure': "Feint ignored; no advantage gained",
            'disaster': "Both forces detected and repelled"
        },
        'waiting': {
            'success': "Rest improved morale somewhat",
            'partial': "Quiet month; supplies conserved",
            'failure': "Morale declined during inactivity",
            'disaster': "Desertion during idle period"
        },
        'reconnaissance': {
            'success': "Valuable intelligence on Tower defenses gathered",
            'partial': "Scouts returned with limited information",
            'failure': "Scouts unable to approach Tower",
            'disaster': "Scouts captured or lost"
        },
    }

    base_narrative = narratives.get(stratagem_type, {}).get(outcome, "Stratagem attempted")

    if casualties > 10:
        base_narrative += f"; {casualties} casualties"
    elif casualties > 0:
        base_narrative += f"; {casualties} wounded"

    if tower_response == 'unknown_force':
        base_narrative += "; strange lights observed"

    return base_narrative


def generate_stratagem_dataset(n_years: int = 20) -> pd.DataFrame:
    """
    Generate detailed stratagem data over the course of the siege.

    Designed to show:
    1. Colonel's gradient estimates are often wrong
    2. Step size matters (learning rate)
    3. Some directions are better than others
    4. Progress is non-linear (harder near the goal)
    5. Consecutive similar strategies can build momentum (or not)
    """

    stratagems = []
    stratagem_id = 1

    # State variables
    current_progress = 0.0
    morale = 0.7
    supplies = 1000
    colonel_confidence = 0.6

    # Track recent stratagems for momentum calculation
    recent_stratagems = []

    for year in range(1, n_years + 1):
        for month in range(1, 13):
            # Decide how many stratagems this month (usually 0-2)
            if supplies < 50:
                n_stratagems = np.random.choice([0, 1], p=[0.7, 0.3])
            elif morale < 0.3:
                n_stratagems = np.random.choice([0, 1], p=[0.5, 0.5])
            else:
                n_stratagems = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])

            for _ in range(n_stratagems):
                # Colonel's stratagem selection (simulating optimization)
                if supplies < 30:
                    stratagem_type = 'waiting'
                elif morale < 0.2:
                    stratagem_type = 'parley'
                elif colonel_confidence > 0.7 and np.random.random() < 0.3:
                    # Bold move when confident
                    stratagem_type = np.random.choice(['balloon_ascent', 'night_raid', 'ladder_assault'])
                elif np.random.random() < 0.15:
                    # Random exploration
                    stratagem_type = np.random.choice(list(STRATAGEMS.keys()))
                else:
                    # Exploit: prefer strategies with good estimated gradient
                    # But Colonel's estimates are noisy
                    weights = []
                    for st in STRATAGEMS:
                        s = STRATAGEMS[st]
                        # Colonel's (noisy) estimate of value
                        estimated_value = s['expected_progress'] + np.random.normal(0, s['variance'])
                        # Adjust for resource constraints
                        if supplies < s['resource_cost'] * 3:
                            estimated_value *= 0.3
                        weights.append(max(0.01, estimated_value + 0.05))

                    weights = np.array(weights) / sum(weights)
                    stratagem_type = np.random.choice(list(STRATAGEMS.keys()), p=weights)

                # Simulate the stratagem
                result = simulate_stratagem(
                    stratagem_id, year, month, stratagem_type,
                    current_progress, morale, supplies, colonel_confidence
                )

                stratagems.append(result)

                # Update state
                current_progress = result['progress_after']
                morale = np.clip(morale + result['morale_impact'], 0.1, 1.0)
                supplies -= result['supply_cost']

                # Confidence update based on outcome
                if result['outcome_category'] == 'success':
                    colonel_confidence = min(0.9, colonel_confidence + 0.1)
                elif result['outcome_category'] == 'disaster':
                    colonel_confidence = max(0.2, colonel_confidence - 0.15)
                else:
                    colonel_confidence += np.random.normal(0, 0.05)
                    colonel_confidence = np.clip(colonel_confidence, 0.2, 0.9)

                # Track for momentum
                recent_stratagems.append(stratagem_type)
                if len(recent_stratagems) > 5:
                    recent_stratagems.pop(0)

                stratagem_id += 1

            # Monthly supply and morale updates
            supplies += np.random.poisson(10)  # Occasional reinforcements
            supplies = min(supplies, 1500)

            # Small morale decay during inactivity
            if n_stratagems == 0:
                morale = max(0.1, morale - 0.02)

    return pd.DataFrame(stratagems)


if __name__ == '__main__':
    print("Generating stratagem details dataset...")
    print("=" * 60)

    df = generate_stratagem_dataset(n_years=20)

    # Save to data directory
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'stratagem_details.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} stratagems to {output_path}")

    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"Total stratagems: {len(df)}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")

    print(f"\nStratagem types:")
    print(df['stratagem_type'].value_counts())

    print(f"\nOutcome distribution:")
    print(df['outcome_category'].value_counts())

    print(f"\nProgress statistics:")
    print(df['progress_delta'].describe())

    print(f"\nGradient estimation error statistics:")
    print(df['gradient_error'].describe())

    # Analyze gradient descent behavior
    print("\n--- Gradient Descent Analysis ---")
    print(f"Mean estimated gradient: {df['estimated_gradient'].mean():.5f}")
    print(f"Mean actual gradient: {df['actual_gradient'].mean():.5f}")
    print(f"Mean gradient error: {df['gradient_error'].mean():.5f}")
    print(f"Final progress: {df['progress_after'].iloc[-1]:.4f}")
    print(f"Final loss: {df['loss_after'].iloc[-1]:.4f}")

    # Show learning rate effects
    print(f"\nStep size distribution:")
    print(df['step_size'].value_counts())

    # Show which strategies worked best
    print(f"\nMean progress by stratagem type:")
    print(df.groupby('stratagem_type')['progress_delta'].mean().sort_values(ascending=False))

    print("\n--- Teaching Applications ---")
    print("1. Gradient descent: Each stratagem is a 'step' in optimization")
    print("2. Learning rate: step_size shows effect of confidence/aggression")
    print("3. Gradient estimation: Colonel's estimates are noisy (gradient_error)")
    print("4. Loss landscape: Progress gets harder near goal (diminishing returns)")
    print("5. Local minima: Times when all strategies seem equally bad")
