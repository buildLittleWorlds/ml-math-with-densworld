# ML Math with Densworld: Complete Build-Out Plan

**Created**: 2025-01-28
**Last Updated**: 2025-11-28
**Status**: Phase 1 COMPLETE, Phase 2 COMPLETE (All 4 Modules)
**Reference Course**: `yeller-quarry-data-science/` (10 tutorials, rich narrative, Colab-ready)

---

## Executive Summary

This document provides a comprehensive plan to transform the ML Math course from its current skeletal state into a production-quality course matching the richness of the Yeller Quarry Data Science course.

### Current State Assessment

| Component | Status | Completion |
|-----------|--------|------------|
| Planning & Design | Complete | 100% |
| Module READMEs | Complete | 100% |
| Data Generators | **COMPLETE** | 100% (8 generators, 13 data files) |
| Data Files | **COMPLETE** | 100% (~11,000+ records across files) |
| Module 1 Notebooks | **COMPLETE** | 100% (5 of 5 notebooks) |
| Module 2 Notebooks | **COMPLETE** | 100% (5 of 5 notebooks) |
| Module 3 Notebooks | **COMPLETE** | 100% (4 of 4 notebooks) |
| Module 4 Notebooks | **COMPLETE** | 100% (4 of 4 notebooks) |
| Colab Compatibility | **ALL MODULES COMPLETE** | 100% (18 of 18 notebooks) |
| Narrative Depth | **ALL MODULES COMPLETE** | 100% |

### Key Accomplishments

1. **Phase 1 Complete**: All data generators and CSV files created
2. **Module 1 Complete**: All 5 Statistics & Probability notebooks created with:
   - Colab badges and GitHub raw URL data loading
   - Rich Densworld narrative throughout
   - 4-5 exercises per notebook with narrative framing
   - Summary tables and next lesson previews
3. **Module 2 Complete**: All 5 Linear Algebra notebooks created with:
   - Colab badges and GitHub raw URL data loading
   - "Creatures as Vectors" narrative arc featuring Boffa Trent, Vagabu Olt, The Colonel
   - 4-5 exercises per notebook with narrative framing
   - Summary tables and next lesson previews
4. **Module 3 Complete**: All 4 Calculus notebooks created with:
   - Colab badges and GitHub raw URL data loading
   - "The Colonel's Optimization" narrative arc — 20 years of siege as optimization
   - 4-5 exercises per notebook with narrative framing
   - Summary tables and next lesson previews
5. **Module 4 Complete**: All 4 Applied ML notebooks created with:
   - Colab badges and GitHub raw URL data loading
   - "The Forgery Trial" narrative arc — Mink Pavar's testimony before the Tribunal
   - 4-5 exercises per notebook with narrative framing
   - Summary tables and course completion message

---

## Phase 1: Data Generation (Priority: HIGH)

### Status: ✅ COMPLETE (2025-01-28)

All new data generators have been created and CSV files generated.

**New Generators Created:**
- `generators/dens_boundary.py` → `data/dens_boundary_observations.csv` (590 records)
- `generators/scholar_debates.py` → `data/scholar_debates.csv` (256 records)
- `generators/creature_vectors.py` → `data/creature_vectors.csv` (25 records) + `data/creature_similarity.csv` (300 pairs)
- `generators/stratagem_details.py` → `data/stratagem_details.csv` (162 records)

**Complete Data Inventory:**
| File | Records | Purpose |
|------|---------|---------|
| expedition_outcomes.csv | 1,000 | Regression, classification |
| expedition_outcomes_train/test/val.csv | 700/200/100 | ML train/test splits |
| creature_market.csv | 2,000 | Skewed distributions |
| manuscript_features.csv | 300 | Forgery detection |
| siege_progress.csv | 240 | Gradient descent |
| **dens_boundary_observations.csv** | 590 | Signal vs noise, CLT |
| **scholar_debates.csv** | 256 | Hypothesis testing |
| **creature_vectors.csv** | 25 | Dot products, similarity |
| **creature_similarity.csv** | 300 | Pairwise comparisons |
| **stratagem_details.csv** | 162 | Detailed gradient descent |

---

## Phase 2: Notebook Development (Priority: HIGH)

### Status: Modules 1-3 COMPLETE, Module 4 IN PROGRESS

### Module 1: Statistics & Probability — ✅ COMPLETE (2025-11-28)

All 5 notebooks created with full Colab compatibility and rich narrative.

| Notebook | Status | Description |
|----------|--------|-------------|
| `01-uncertainty-intuition.ipynb` | ✅ Complete | Populations, samples, random variables |
| `02-distributions-as-terrain.ipynb` | ✅ Complete | Normal, skewed, fat-tailed distributions |
| `03-central-limit-theorem.ipynb` | ✅ Complete | CLT, √n rule, sample size planning |
| `04-hypothesis-testing.ipynb` | ✅ Complete | P-values, significance, multiple comparisons, Bonferroni |
| `05-bayesian-classification.ipynb` | ✅ Complete | Bayes' theorem, Naive Bayes, precision/recall, confusion matrices |

**Features implemented:**
- ✅ Colab badges with correct GitHub URLs
- ✅ GitHub raw URL data loading (no local paths)
- ✅ Opening narratives with Densworld quotes
- ✅ Learning objectives in each notebook
- ✅ 4-5 exercises per notebook with narrative framing
- ✅ Summary tables
- ✅ Next lesson previews

---

### Module 2: Linear Algebra (5 notebooks) — ✅ COMPLETE (2025-11-28)

All 5 notebooks created with full Colab compatibility and "Creatures as Vectors" narrative.

| Notebook | Status | Description |
|----------|--------|-------------|
| `01-vectors-dual-view.ipynb` | ✅ Complete | Vectors as data records and points in space |
| `02-vector-norms.ipynb` | ✅ Complete | L1, L2 norms, distance measures, regularization preview |
| `03-dot-product-similarity.ipynb` | ✅ Complete | Dot product, cosine similarity, manuscript forgery detection |
| `04-matrix-transformations.ipynb` | ✅ Complete | Linear transforms, neural network layers as matrices |
| `05-rank-independence.ipynb` | ✅ Complete | Linear independence, rank, multicollinearity, VIF |

**Features implemented:**
- ✅ Colab badges with correct GitHub URLs
- ✅ GitHub raw URL data loading (no local paths)
- ✅ Narrative featuring Boffa Trent, Vagabu Olt, The Colonel, The Pickbox Man
- ✅ Learning objectives in each notebook
- ✅ 4-5 exercises per notebook with narrative framing
- ✅ Summary tables
- ✅ Next lesson previews

**Data Used**: `creature_vectors.csv`, `creature_similarity.csv`, `manuscript_features.csv`, `expedition_outcomes.csv`, `dens_boundary_observations.csv`

---

### Module 3: Calculus (4 notebooks) — ✅ COMPLETE (2025-11-28)

All 4 notebooks created with full Colab compatibility and "The Colonel's Optimization" narrative.

| Notebook | Status | Description |
|----------|--------|-------------|
| `01-derivatives-sensitivity.ipynb` | ✅ Complete | Derivatives as sensitivity, numerical derivatives |
| `02-gradient-compass.ipynb` | ✅ Complete | Partial derivatives, gradients as compass |
| `03-gradient-descent.ipynb` | ✅ Complete | GD algorithm, learning rate, SGD |
| `04-chain-rule-backprop.ipynb` | ✅ Complete | Chain rule, backprop, neural networks from scratch |

**Features implemented:**
- ✅ Colab badges with correct GitHub URLs
- ✅ GitHub raw URL data loading (no local paths)
- ✅ Narrative featuring The Colonel's 20-year siege of the Tower of Mirado
- ✅ Learning objectives in each notebook
- ✅ 4-5 exercises per notebook with narrative framing
- ✅ Summary tables
- ✅ Next lesson previews

**Data Used**: `siege_progress.csv`, `stratagem_details.csv`, `expedition_outcomes.csv`

---

### Module 4: Applied ML (4 notebooks) — ✅ COMPLETE (2025-11-28)

All 4 notebooks created with full Colab compatibility and "The Forgery Trial" narrative.

| Notebook | Status | Description |
|----------|--------|-------------|
| `01-deriving-linear-regression.ipynb` | ✅ Complete | Linear regression derivation, loss function, gradient descent |
| `02-bias-variance-tradeoff.ipynb` | ✅ Complete | Under/overfitting, bias-variance decomposition |
| `03-regularization.ipynb` | ✅ Complete | L1/L2 regularization, Lasso/Ridge, feature selection |
| `04-model-selection-capstone.ipynb` | ✅ Complete | Cross-validation, GridSearchCV, full ML pipeline |

**Features implemented:**
- ✅ Colab badges with correct GitHub URLs
- ✅ GitHub raw URL data loading (no local paths)
- ✅ Narrative featuring Mink Pavar, Eulr Voss, and the Tribunal
- ✅ Learning objectives in each notebook
- ✅ 4-5 exercises per notebook with narrative framing
- ✅ Summary tables
- ✅ Course completion congratulations message

**Data Used**: `manuscript_features.csv`, `expedition_outcomes.csv`, `creature_market.csv`

---

## Phase 3: Colab Conversion (Priority: MEDIUM)

### Status: ✅ ALL COMPLETE

| Module | Status |
|--------|--------|
| Module 1 | ✅ Complete (5/5 notebooks) |
| Module 2 | ✅ Complete (5/5 notebooks) |
| Module 3 | ✅ Complete (4/4 notebooks) |
| Module 4 | ✅ Complete (4/4 notebooks) |

**All notebooks converted:**
- [x] `modules/02-linear-algebra/notebooks/01-vectors-dual-view.ipynb` ✅ Complete
- [x] `modules/03-calculus/notebooks/01-derivatives-sensitivity.ipynb` ✅ Complete
- [x] `modules/04-applied-ml/notebooks/01-deriving-linear-regression.ipynb` ✅ Complete

---

## Phase 4: Narrative Enhancement (Priority: MEDIUM)

### Status: ✅ ALL COMPLETE

Module 1 notebooks now include:
- ✅ Character voices (Vagabu Olt, Mink Pavar, scholars, mapmakers)
- ✅ Opening quotes from Densworld lore
- ✅ Story arc: "The Archives Dilemma" — scholars seeking truth from samples
- ✅ Exercise framing with narrative context

Module 2 notebooks now include:
- ✅ Character voices (Boffa Trent, Vagabu Olt, The Colonel, The Pickbox Man)
- ✅ Opening quotes from Densworld natural philosophy
- ✅ Story arc: "Creatures as Vectors" — cataloguing creatures in feature space
- ✅ Manuscript forgery detection as applied similarity

Module 3 notebooks now include:
- ✅ Character voice: The Colonel as the primary narrator
- ✅ Opening quotes from The Colonel's journals and strategic notes
- ✅ Story arc: "The Colonel's Optimization" — 20 years besieging the Tower of Mirado
- ✅ Gradient descent as navigating an invisible loss landscape
- ✅ Backpropagation as tracing cause and effect through chains of decisions

Module 4 notebooks now include:
- ✅ Character voice: Mink Pavar as the primary narrator
- ✅ Opening quotes from Mink Pavar's testimony
- ✅ Story arc: "The Forgery Trial" — Mink Pavar defending mathematical methods before the Tribunal
- ✅ Eulr Voss as the skeptical rival proposing overly complex models
- ✅ Linear regression derived as evidence for the court
- ✅ Regularization as "taxing complexity"
- ✅ Cross-validation as the final, rigorous test

---

## Appendix A: Updated File Checklist

### Generators — ✅ ALL COMPLETE

- [x] `generators/dens_boundary.py`
- [x] `generators/scholar_debates.py`
- [x] `generators/creature_vectors.py`
- [x] `generators/stratagem_details.py`

### Data Files — ✅ ALL COMPLETE

- [x] `data/dens_boundary_observations.csv`
- [x] `data/scholar_debates.csv`
- [x] `data/creature_vectors.csv`
- [x] `data/creature_similarity.csv`
- [x] `data/stratagem_details.csv`

### Notebooks — Module 1 COMPLETE

**Module 1:** ✅ ALL COMPLETE
- [x] `01-uncertainty-intuition.ipynb` (enhanced)
- [x] `02-distributions-as-terrain.ipynb` (created)
- [x] `03-central-limit-theorem.ipynb` (created)
- [x] `04-hypothesis-testing.ipynb` (created)
- [x] `05-bayesian-classification.ipynb` (created)

**Module 2:** ✅ ALL COMPLETE
- [x] `01-vectors-dual-view.ipynb` (enhanced with Colab + narrative)
- [x] `02-vector-norms.ipynb` (created)
- [x] `03-dot-product-similarity.ipynb` (created)
- [x] `04-matrix-transformations.ipynb` (created)
- [x] `05-rank-independence.ipynb` (created)

**Module 3:** ✅ ALL COMPLETE
- [x] `01-derivatives-sensitivity.ipynb` (enhanced with Colab + narrative)
- [x] `02-gradient-compass.ipynb` (created)
- [x] `03-gradient-descent.ipynb` (created)
- [x] `04-chain-rule-backprop.ipynb` (created)

**Module 4:** ✅ ALL COMPLETE
- [x] `01-deriving-linear-regression.ipynb` (enhanced with Colab + narrative)
- [x] `02-bias-variance-tradeoff.ipynb` (created)
- [x] `03-regularization.ipynb` (created)
- [x] `04-model-selection-capstone.ipynb` (created)

---

## Appendix B: Effort Summary

| Phase | Task | Status |
|-------|------|--------|
| 1 | Data Generators | ✅ Complete |
| 2 | Module 1 Notebooks | ✅ Complete |
| 2 | Module 2 Notebooks (5/5) | ✅ Complete |
| 2 | Module 3 Notebooks (4/4) | ✅ Complete |
| 2 | Module 4 Notebooks (4/4) | ✅ Complete |
| 3 | Colab Conversion (all 18) | ✅ Complete |
| 4 | Narrative Enhancement (all 4 modules) | ✅ Complete |
| — | **COURSE COMPLETE** | ✅ 100% |

---

## Appendix C: Quality Standards

Every notebook must meet these criteria before marking complete:

1. **Runs on Colab** without errors
2. **Loads data from GitHub** URLs
3. **Has opening narrative** from Densworld
4. **States learning objectives** clearly
5. **Explains concepts before formulas**
6. **Uses Densworld data** throughout
7. **Has 4-5 exercises** with narrative framing
8. **Includes summary table**
9. **Previews next lesson**
10. **Follows naming convention**: `XX-topic-name.ipynb`

---

*Last Updated: 2025-11-28*
*Status: COURSE COMPLETE*
