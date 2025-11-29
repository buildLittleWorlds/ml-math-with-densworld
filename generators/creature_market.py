"""
Creature Market Price Simulator

Generates synthetic market data for Yeller Quarry creatures.
Demonstrates log-normal distributions, skewness, and regression concepts.

World Parameters (from ore files):
- Creatures from Yeller Quarry are traded at stall towns
- Prices depend on rarity, danger, condition, and fashion in the Capital
- Some creatures (wharvers, stakdurs) are deadly; others (yeller birds) are pets
- The market is erratic — fashions change, supply fluctuates
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# =============================================================================
# CREATURE DATABASE (from existing creatures.csv + expanded)
# =============================================================================

CREATURES = {
    'CR001': {
        'name': 'Leatherback Burrower',
        'category': 'bird',
        'danger': 3,
        'rarity': 4,
        'base_price': 45,
        'price_volatility': 0.2
    },
    'CR002': {
        'name': 'Stone Spine Lizard',
        'category': 'reptile',
        'danger': 5,
        'rarity': 7,
        'base_price': 305,
        'price_volatility': 0.4
    },
    'CR003': {
        'name': 'Granite Wren',
        'category': 'bird',
        'danger': 1,
        'rarity': 2,
        'base_price': 28,
        'price_volatility': 0.15
    },
    'CR004': {
        'name': 'Swamp Hornet',
        'category': 'insect',
        'danger': 4,
        'rarity': 1,
        'base_price': 5,
        'price_volatility': 0.3
    },
    'CR005': {
        'name': 'Marsh Stalker',
        'category': 'reptile',
        'danger': 6,
        'rarity': 5,
        'base_price': 180,
        'price_volatility': 0.35
    },
    'CR006': {
        'name': 'Glass Beetle',
        'category': 'insect',
        'danger': 2,
        'rarity': 6,
        'base_price': 140,
        'price_volatility': 0.25
    },
    'CR007': {
        'name': 'Yeller Bird',
        'category': 'bird',
        'danger': 0,
        'rarity': 3,
        'base_price': 65,
        'price_volatility': 0.3
    },
    'CR008': {
        'name': 'Wharver',
        'category': 'mammal',
        'danger': 8,
        'rarity': 6,
        'base_price': 420,
        'price_volatility': 0.5
    },
    'CR009': {
        'name': 'Stakdur',
        'category': 'reptile',
        'danger': 9,
        'rarity': 8,
        'base_price': 650,
        'price_volatility': 0.6
    },
    'CR010': {
        'name': 'Grimslew',
        'category': 'mammal',
        'danger': 7,
        'rarity': 7,
        'base_price': 380,
        'price_volatility': 0.45
    },
    'CR011': {
        'name': 'Cave Moth',
        'category': 'insect',
        'danger': 0,
        'rarity': 2,
        'base_price': 12,
        'price_volatility': 0.2
    },
    'CR012': {
        'name': 'Tunnel Eel',
        'category': 'fish',
        'danger': 3,
        'rarity': 4,
        'base_price': 55,
        'price_volatility': 0.25
    },
    'CR013': {
        'name': 'Quarry Toad',
        'category': 'amphibian',
        'danger': 2,
        'rarity': 3,
        'base_price': 35,
        'price_volatility': 0.2
    },
    'CR014': {
        'name': 'Densmuck Crawler',
        'category': 'arthropod',
        'danger': 4,
        'rarity': 5,
        'base_price': 95,
        'price_volatility': 0.3
    },
    'CR015': {
        'name': 'Golden Amalgam Snail',
        'category': 'mollusk',
        'danger': 0,
        'rarity': 9,
        'base_price': 890,
        'price_volatility': 0.7
    }
}

CONDITIONS = {
    'pristine': 1.0,
    'good': 0.85,
    'fair': 0.65,
    'damaged': 0.4,
    'partial': 0.15
}

BUYER_TYPES = [
    'archivist',      # Capital Archives - steady, fair prices
    'senator',        # Private collections - pays premium
    'display',        # Capital Display houses - volume buyer
    'scholar',        # Research - specific creatures
    'trader',         # Resale - negotiates hard
    'collector',      # Hobbyists - unpredictable
    'unknown'         # Anonymous sales
]

DESTINATIONS = [
    'Capital Archives',
    'Senate Zoological',
    'Capital Display',
    'Private Collection',
    'Miasto Market',
    'Scholar Study',
    'Unknown'
]


def simulate_fashion_trend(year: int, week: int, creature_id: str) -> float:
    """
    Simulate fashion trends that affect creature prices.
    Different creatures peak at different times.
    """
    # Each creature has a different phase in the fashion cycle
    creature_idx = int(creature_id[2:])
    phase_offset = creature_idx * 0.7  # Different starting phases

    # 3-year fashion cycle
    time = year + week / 52
    cycle = np.sin(2 * np.pi * (time - 1850) / 3 + phase_offset)

    # Convert to multiplier (0.7 to 1.3)
    return 0.85 + 0.15 * cycle


def simulate_supply_pressure(year: int, week: int, creature_id: str) -> float:
    """
    Simulate supply fluctuations affecting prices.
    Low supply = higher prices.
    """
    # Random walk with mean reversion
    creature = CREATURES[creature_id]

    # Rare creatures have more volatile supply
    volatility = 0.1 + 0.05 * creature['rarity']

    # Seasonal effects (winter = lower supply)
    season_effect = 1.0
    if week < 10 or week > 45:
        season_effect = 1.15  # Winter premium

    noise = np.random.normal(1.0, volatility)
    return max(0.5, min(2.0, noise * season_effect))


def simulate_sale(sale_id: int, year: int, week: int) -> dict:
    """Simulate a single creature sale."""

    # Select creature (weighted by inverse rarity - common creatures sold more often)
    creature_ids = list(CREATURES.keys())
    weights = [10 - CREATURES[cid]['rarity'] for cid in creature_ids]
    weights = np.array(weights) / sum(weights)
    creature_id = np.random.choice(creature_ids, p=weights)
    creature = CREATURES[creature_id]

    # Condition (pristine rare, damaged common)
    condition_weights = [0.1, 0.25, 0.35, 0.2, 0.1]
    condition = np.random.choice(list(CONDITIONS.keys()), p=condition_weights)

    # Buyer type
    buyer_type = np.random.choice(BUYER_TYPES)

    # Destination (correlated with buyer)
    if buyer_type == 'archivist':
        destination = 'Capital Archives'
    elif buyer_type == 'senator':
        destination = np.random.choice(['Senate Zoological', 'Private Collection'])
    elif buyer_type == 'display':
        destination = 'Capital Display'
    elif buyer_type == 'scholar':
        destination = 'Scholar Study'
    elif buyer_type == 'trader':
        destination = 'Miasto Market'
    else:
        destination = np.random.choice(DESTINATIONS)

    # Seller reputation (0-1)
    seller_reputation = np.random.beta(5, 2)  # Skewed toward good reputation

    # Quantity (usually 1, sometimes more for common creatures)
    if creature['rarity'] < 4:
        quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
    else:
        quantity = np.random.choice([1, 2], p=[0.85, 0.15])

    # Calculate price
    base = creature['base_price']

    # Apply condition
    price = base * CONDITIONS[condition]

    # Apply fashion trend
    fashion = simulate_fashion_trend(year, week, creature_id)
    price *= fashion

    # Apply supply pressure
    supply = simulate_supply_pressure(year, week, creature_id)
    price *= supply

    # Buyer type adjustments
    if buyer_type == 'senator':
        price *= 1.25  # Senators pay premium
    elif buyer_type == 'trader':
        price *= 0.85  # Traders negotiate down
    elif buyer_type == 'collector':
        # Collectors are unpredictable
        price *= np.random.uniform(0.8, 1.4)

    # Seller reputation effect
    price *= (0.8 + 0.4 * seller_reputation)

    # Volume discount for multiple
    if quantity > 1:
        price *= (1 - 0.05 * (quantity - 1))

    # Add random market noise
    noise = np.random.normal(1.0, creature['price_volatility'])
    price *= max(0.5, noise)

    # Final price per unit
    price_per_unit = max(1, round(price, 2))

    return {
        'sale_id': f'SALE-{sale_id:05d}',
        'creature_id': creature_id,
        'creature_name': creature['name'],
        'category': creature['category'],
        'danger_rating': creature['danger'],
        'rarity_rating': creature['rarity'],
        'condition': condition,
        'year': year,
        'week': week,
        'seller_reputation': round(seller_reputation, 3),
        'buyer_type': buyer_type,
        'destination': destination,
        'quantity': quantity,
        'price_per_unit': price_per_unit,
        'total_price': round(price_per_unit * quantity, 2)
    }


def generate_market_dataset(n_sales: int = 2000,
                            start_year: int = 1850,
                            end_year: int = 1860) -> pd.DataFrame:
    """Generate a full dataset of creature market sales."""

    sales = []

    for i in range(n_sales):
        year = np.random.randint(start_year, end_year + 1)
        week = np.random.randint(1, 53)
        sale = simulate_sale(i + 1, year, week)
        sales.append(sale)

    df = pd.DataFrame(sales)

    # Sort chronologically
    df = df.sort_values(['year', 'week', 'sale_id']).reset_index(drop=True)

    return df


if __name__ == '__main__':
    print("Generating creature market dataset...")
    df = generate_market_dataset(n_sales=2000)

    # Save to data directory
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / 'creature_market.csv', index=False)
    print(f"Saved {len(df)} sales to {output_dir / 'creature_market.csv'}")

    # Print summary
    print("\n--- Dataset Summary ---")
    print(f"\nPrice distribution (per unit):")
    print(df['price_per_unit'].describe())

    print(f"\nSales by creature category:")
    print(df['category'].value_counts())

    print(f"\nSales by condition:")
    print(df['condition'].value_counts())

    print(f"\nPrice by danger rating:")
    print(df.groupby('danger_rating')['price_per_unit'].mean().round(2))

    # Demonstrate skewness
    print(f"\nPrice skewness: {df['price_per_unit'].skew():.2f}")
    print("(Positive skew = long right tail, typical of market data)")
