# Potential Data Sources for ML Math Curriculum

This document catalogs existing datasets from other Densworld courses that could be reused, and speculates on new datasets that could be generated from the raw ore files.

---

## Part 1: Existing Datasets from Other Courses

### From `yeller-quarry-data-science/data/`

| File | Records | Potential Use in ML Math |
|------|---------|--------------------------|
| `creatures.csv` | 25 | **Vectors & Similarity** — each creature is a point in n-dimensional space (danger, size, rarity, etc.) |
| `crews.csv` | 12 | **Linear Independence** — test if crew features are redundant |
| `traps.csv` | 35 | **Sensitivity Analysis** — how trap parameters affect catch rates |
| `catches.csv` | 50 | **Regression** — predict catch value from creature + trap + crew features |
| `prices.csv` | 75 | **Distributions** — demonstrate skewness, long-tailed market data |
| `incidents.csv` | 30 | **Classification / Confusion Matrices** — predict incident type from conditions |

### From `capital-archives-nlp/data/`

| File | Records | Potential Use in ML Math |
|------|---------|--------------------------|
| `scholars.csv` | 40 | **Dot Product / Similarity** — match scholars to philosophical schools |
| `manuscripts.csv` | 125 | **Sampling** — population vs. sample estimation |
| `manuscript_texts.csv` | 169 sections | **Feature Extraction** — word counts as vectors for regression |
| `debates.csv` | 30 | **Hypothesis Testing** — did debate outcomes depend on school affiliation? |
| `forgery_evidence.csv` | 15 | **Regularization** — feature selection for forgery detection |

---

## Part 2: New Datasets to Generate from Ore Files

The ore files contain ~1.4 million words of narrative. They aren't structured data, but they establish the *rules of the world* — which creatures exist, how expeditions work, what the economy looks like. These rules can seed simulations.

### 2.1 Expedition Outcome Simulator

**Source:** `yeller quarry ore.txt`, `dens ore.txt`

**What the ore establishes:**
- Expeditions have crews of 10-25 people
- The Boss (red-haired woman) leads with Truck as lieutenant
- Yeller groups (the five) have special roles
- Creatures attack unpredictably; hornets, wharvers, grimslews
- Success depends on leadership, weather, and luck

**Simulated Dataset:** `expedition_outcomes.csv`

| Column | Type | Distribution Source |
|--------|------|---------------------|
| `expedition_id` | string | generated |
| `crew_size` | int | uniform(8, 25) |
| `leader_experience_years` | int | exponential(λ=0.15) — most are inexperienced |
| `has_yeller_group` | bool | 10% probability |
| `season` | categorical | uniform across 4 seasons |
| `days_in_field` | int | normal(μ=21, σ=7) |
| `creature_encounters` | int | poisson(λ=3) |
| `casualties` | int | derived from encounters + crew skill |
| `catch_value` | float | log-normal — most expeditions modest, few jackpots |
| `success` | bool | derived from casualties < threshold |

**ML Math Applications:**
- **Module I:** Sample statistics, CLT (average many expeditions)
- **Module II:** Feature vectors for crews
- **Module IV:** Logistic regression for success, bias-variance with train/test splits

---

### 2.2 Manuscript Authenticity Simulator

**Source:** `capital ore.txt`, `capital-archives-nlp/data/`

**What the ore establishes:**
- Three philosophical schools: Stone, Water, Pebble
- Scholars have distinct writing styles
- Forgeries exist — Mink Pavar suspected of forging Grigsu Haldo
- Manuscripts have word frequency patterns tied to era and school

**Simulated Dataset:** `manuscript_features.csv`

| Column | Type | Notes |
|--------|------|-------|
| `manuscript_id` | string | |
| `attributed_author` | string | claimed author |
| `true_author` | string | (hidden for capstone) |
| `is_forgery` | bool | ~15% are forgeries |
| `word_count` | int | |
| `avg_sentence_length` | float | |
| `philosophical_term_density` | float | |
| `school_alignment_stone` | float | dot product with Stone school vector |
| `school_alignment_water` | float | |
| `school_alignment_pebble` | float | |
| `era_marker_score` | float | anachronistic word usage |
| `stylometric_variance` | float | consistency within document |

**ML Math Applications:**
- **Module I:** Hypothesis testing — is `era_marker_score` significantly different for forgeries?
- **Module I:** P-hacking demonstration — test 20 features, find 1 "significant" by chance
- **Module II:** Vectors and similarity — school alignment as dot products
- **Module IV:** Regularization — L1 forces irrelevant features to zero

---

### 2.3 Dens Boundary Drift Simulator

**Source:** `dens ore.txt`

**What the ore establishes:**
- The Dens is a shifting, unstable territory
- Mapmakers are essential because "yesterday's map is gossip"
- The boundary between solid ground and densmuck moves unpredictably
- Villages near the Dens live under constant threat

**Simulated Dataset:** `dens_boundary_observations.csv`

| Column | Type | Notes |
|--------|------|-------|
| `observation_id` | string | |
| `date` | date | |
| `location_x` | float | grid coordinate |
| `location_y` | float | |
| `ground_stability` | float | 0 = pure densmuck, 1 = solid |
| `days_since_last_survey` | int | |
| `drift_direction` | categorical | N/S/E/W/none |
| `drift_magnitude` | float | meters shifted |

**Simulation Logic:**
- Ground stability follows a random walk with drift toward instability near Dens center
- Stability readings have measurement error (noise)
- True boundary position is latent; observations are samples

**ML Math Applications:**
- **Module I:** Signal vs. noise — true boundary position vs. measurement error
- **Module I:** Central Limit Theorem — averaging many observations improves estimate
- **Module III:** Gradient descent — finding the true boundary by stepping in direction of greatest stability change
- **Module III:** Sensitivity — how does error in position estimate affect stability prediction?

---

### 2.4 Tower of Mirado Siege Progress Simulator

**Source:** `tower of mirado ore.txt`, `capital ore.txt`

**What the ore establishes:**
- The Colonel has been besieging the Tower for 15-25 years
- The Tower hovers above the desert; unreachable by conventional means
- Progress is measured in failed stratagems
- Supply requisitions to the Capital rarely return with supplies
- Morale decays over time; men desert or die of boredom

**Simulated Dataset:** `siege_progress.csv`

| Column | Type | Notes |
|--------|------|-------|
| `year` | int | year of siege |
| `month` | int | |
| `personnel_count` | int | decays over time with occasional reinforcements |
| `morale_index` | float | 0-1, trends downward |
| `supply_level` | float | erratic, depends on requisition success |
| `stratagem_attempted` | bool | did Colonel try something this month? |
| `stratagem_type` | categorical | ladder, grapple, tunnel, parley, other |
| `progress_score` | float | latent measure of how close to breach |
| `tower_response` | categorical | silence, repelled, negotiation, unknown |

**Simulation Logic:**
- Progress score is the "loss function" — goal is to minimize distance to breach
- Each stratagem is a "step" — some directions reduce loss, others increase it
- Gradient of loss landscape is unknown to Colonel; he must estimate

**ML Math Applications:**
- **Module III:** Gradient Descent — the siege as optimization
- **Module III:** Loss Landscape — visualize progress score as terrain
- **Module III:** Learning Rate — too aggressive (big attack) vs. too cautious (no progress)
- **Module IV:** Bias-Variance — simple model (just attack repeatedly) vs. complex model (consider all factors)

---

### 2.5 Creature Market Price Simulator

**Source:** `yeller quarry ore.txt`

**What the ore establishes:**
- Creatures from Yeller Quarry are traded at stall towns
- Prices depend on rarity, danger, condition, and fashion in the Capital
- Some creatures (wharvers, stakdurs) are deadly; others (yeller birds) are pets
- The market is erratic — fashions change, supply fluctuates

**Simulated Dataset:** `creature_market.csv`

| Column | Type | Notes |
|--------|------|-------|
| `sale_id` | string | |
| `creature_type` | categorical | ~25 types |
| `condition` | categorical | pristine, damaged, partial |
| `week` | int | week of year |
| `year` | int | |
| `seller_reputation` | float | 0-1 |
| `buyer_type` | categorical | scholar, collector, trader, unknown |
| `sale_price` | float | the target variable |

**Simulation Logic:**
- Base price per creature type (log-normal: most cheap, few expensive)
- Condition multiplier (pristine = 1.0, damaged = 0.5, partial = 0.2)
- Fashion trend: sine wave over years (some creatures peak, then fall)
- Random noise: ±20% for market chaos

**ML Math Applications:**
- **Module I:** Skewness — price distribution is long-tailed
- **Module I:** Variance — prices for rare creatures vary wildly
- **Module IV:** Linear Regression — predict price from features
- **Module IV:** Regularization — which features actually matter?

---

## Part 3: Simulation Architecture

### 3.1 Principle: Ore Files as World Parameters

The ore files don't contain data — they contain the *rules* and *constants* of the world. A simulation extracts these:

```
ORE FILE → WORLD PARAMETERS → SIMULATION → DATASET
```

**Example:** From `yeller quarry ore.txt`, we extract:
- Creature types: wharver, stakdur, grimslew, yeller bird, etc.
- Crew roles: boss, lieutenant, cook, trap-setter, yeller group
- Expedition duration: typically 2-6 weeks
- Danger events: hornets, creature attacks, getting lost

These become the *parameters* of a simulation that generates thousands of synthetic expedition records.

### 3.2 Seeding Randomness with Narrative

To keep data narratively coherent:
- Named characters (Gull, Truck, Polks) can appear in generated data
- Historical events from ore (the Dens campaign, the Colonel's departure) set time boundaries
- Geographic constraints (can't travel faster than X miles/day through marsh) bound possibilities

### 3.3 Generating Train/Test/Validation Splits

For ML exercises, we need:
- **Training set:** 70% of simulated data
- **Test set:** 20% held out
- **Validation set:** 10% for hyperparameter tuning

Because we control the simulation, we can also generate:
- **Concept drift sets:** where the rules change partway through (fashions shift, new creatures discovered)
- **Out-of-distribution sets:** entirely new creature types not in training

---

## Part 4: Priority Order for Dataset Generation

Based on curriculum structure and narrative richness:

1. **Expedition Outcomes** — richest narrative source, teaches most concepts
2. **Creature Market Prices** — natural for regression, easy to simulate
3. **Manuscript Authenticity** — already partially exists in NLP course, extends naturally
4. **Siege Progress** — powerful metaphor for gradient descent, highly memorable
5. **Dens Boundary Drift** — most abstract, save for advanced module

---

## Part 5: Technical Implementation Notes

### 5.1 Simulation in Python

```python
# Pseudocode for expedition simulator

import numpy as np
import pandas as pd

# World parameters extracted from ore
CREATURE_TYPES = ['wharver', 'stakdur', 'grimslew', 'yeller_bird', ...]
CREW_ROLES = ['boss', 'lieutenant', 'cook', 'trap_setter', 'yeller']
DANGER_EVENTS = ['hornets', 'creature_attack', 'lost', 'desertion']

def simulate_expedition():
    crew_size = np.random.randint(8, 26)
    leader_exp = np.random.exponential(scale=7)
    has_yeller = np.random.random() < 0.10
    days = max(7, int(np.random.normal(21, 7)))

    # Simulate encounters
    encounters = np.random.poisson(lam=3)
    casualties = simulate_casualties(encounters, crew_size, leader_exp, has_yeller)

    # Simulate catch
    catch_value = simulate_catch(days, encounters, has_yeller)

    return {
        'crew_size': crew_size,
        'leader_experience': leader_exp,
        'has_yeller_group': has_yeller,
        'days_in_field': days,
        'creature_encounters': encounters,
        'casualties': casualties,
        'catch_value': catch_value,
        'success': casualties < crew_size * 0.3
    }

# Generate 1000 expeditions
expeditions = pd.DataFrame([simulate_expedition() for _ in range(1000)])
```

### 5.2 Validation Against Ore

After simulation, spot-check:
- Do generated expeditions "feel" like ore narratives?
- Are extreme cases plausible? (e.g., 90% casualty rate should be rare but possible)
- Do named characters (if included) behave consistently?

### 5.3 Versioning

Each simulated dataset should be versioned:
- `expedition_outcomes_v1.csv` — initial simulation
- `expedition_outcomes_v2.csv` — after tuning parameters
- `expedition_outcomes_v2_seed42.csv` — reproducible with seed

---

## Appendix: Ore File Quick Reference

| Ore File | Primary Content | Data Generation Potential |
|----------|-----------------|---------------------------|
| `yeller quarry ore.txt` | Expeditions, creatures, The Boss, Yeller groups | HIGH — quantifiable events |
| `capital ore.txt` | Archives, scholars, politics, the Colonel's daughter | MEDIUM — classification, text features |
| `dens ore.txt` | Mapmaking, shifting boundaries, densmuck | MEDIUM — spatial/temporal data |
| `tower of mirado ore.txt` | The siege, the Colonel, Fyncho | MEDIUM — optimization metaphor |
| `dead river ore.txt` | Noir narratives, crime | LOW — less quantifiable |
| `northo ore.txt` | Religious communities | LOW — abstract |
| `densmok ore.txt` | Suburban dynamics | LOW — less action |
| `mirado sticks ore.txt` | Rehabilitation, weariness | LOW — emotional, not quantitative |
| `north town ore.txt` | Children, danger | LOW — sensitive content |
| `capeast ore.txt` | Urban grit | LOW — less developed |
