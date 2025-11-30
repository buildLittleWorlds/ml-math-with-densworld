# Welcome to the Mathematical Foundations
## A Course in ML Math with Densworld

![ML Math with Densworld](images/hero.png)

---

*"The Tribunal demanded evidence. 'Show us,' they said, 'how you know this manuscript is false.' I could not simply declare it—I needed mathematics that would stand before the court."*
— Mink Pavar, the Great Forgery Trial of 912

---

### What Is This?

This is a mathematics course disguised as a journey through archives, quarries, and a twenty-year siege.

You'll learn the **mathematical foundations of machine learning**—statistics, linear algebra, calculus, and their application to real algorithms—by exploring problems from the world of Densworld. Instead of abstract examples, you'll work with creature classification, manuscript forgery detection, expedition outcome prediction, and the Colonel's legendary optimization problem.

By the end, you'll understand probability distributions, hypothesis testing, Bayesian reasoning, vector operations, matrix transformations, derivatives, gradients, gradient descent, linear regression derivation, bias-variance tradeoff, and regularization. You'll also know why the Colonel spent twenty years besieging the Tower of Mirado—and what his journals reveal about learning rates.

### Why Learn This Way?

1. **Math needs anchors.** You'll remember what a gradient is because you watched the Colonel step through an invisible landscape, feeling the slope beneath his feet—not because you memorized a definition.

2. **Machine learning math is about intuition.** The formulas matter less than understanding *why* they work. Densworld scenarios give you something to visualize when the symbols get abstract.

3. **The examples are unfamiliar in a useful way.** You can't pattern-match to examples you've seen before. Each problem requires actually applying the math, not recognizing a template.

4. **Multiple perspectives reinforce learning.** Statistics through the Archives' uncertainty, linear algebra through creature feature spaces, calculus through the siege—each module approaches mathematical thinking from a different angle.

---

## The World

**Densworld** contains several regions, each with its own mathematical lessons:

**The Capital Archives** — Millennia of manuscripts, scholarly debates, and philosophical treatises. The Archives face constant uncertainty: which manuscripts are authentic? Which expedition reports are reliable? Statistics becomes the language of evidence.

**Yeller Quarry** — A massive extraction zone where crews trap creatures for sale to Capital merchants. Each creature can be represented as a feature vector—danger rating, depth preference, metal content, market price. Linear algebra becomes the language of similarity and transformation.

**The Tower of Mirado** — A fortress besieged by the Colonel for twenty years. Each day he tried a new approach, measured the outcome, and adjusted. His journals record an optimization problem solved one gradient step at a time. Calculus becomes the language of learning.

**The Great Forgery Trial** — Mink Pavar stood before the Tribunal to defend his mathematical methods. Could stylometric analysis prove a manuscript was forged? The trial required applying statistics, linear algebra, and optimization together. Applied ML is the language of integration.

---

## The Modules

This course is organized into four modules, each with its own narrative arc:

### Module 1: Statistics & Probability
*The Archives Dilemma — Scholars seeking truth from noisy samples*

| # | Lesson | What You'll Learn | The Story |
|---|--------|-------------------|-----------|
| 1 | Uncertainty & Intuition | Probability fundamentals, random variables | The Archives' sampling problem |
| 2 | Distributions as Terrain | Normal, Poisson, Bernoulli distributions | Mapping the space of possible outcomes |
| 3 | Central Limit Theorem | Sample means, standard error | Why the Archives demand detailed reports |
| 4 | Hypothesis Testing | P-values, significance, Type I/II errors | Testing claims about manuscripts |
| 5 | Bayesian Classification | Priors, likelihoods, posteriors | Classifying manuscripts by school |

### Module 2: Linear Algebra
*Creatures as Vectors — Cataloguing creatures in feature space*

| # | Lesson | What You'll Learn | The Story |
|---|--------|-------------------|-----------|
| 1 | Vectors: The Dual View | Geometric vs algebraic vectors | Creatures as points in space |
| 2 | Vector Norms & Distance | L1, L2 norms, distance metrics | How far apart are two creatures? |
| 3 | Dot Product & Similarity | Dot product, cosine similarity | Which creatures are most similar? |
| 4 | Matrix Transformations | Linear maps, matrix multiplication | Transforming the creature catalog |
| 5 | Rank & Independence | Linear independence, rank, null space | Which features are redundant? |

### Module 3: Calculus
*The Colonel's Optimization — 20 years besieging the Tower of Mirado*

| # | Lesson | What You'll Learn | The Story |
|---|--------|-------------------|-----------|
| 1 | Derivatives as Sensitivity | Rates of change, partial derivatives | The Colonel measures the slope |
| 2 | The Gradient Compass | Gradient vectors, directional derivatives | Which way is downhill? |
| 3 | Gradient Descent | The algorithm, learning rate, convergence | Twenty years of stepping carefully |
| 4 | Chain Rule & Backprop | Composite functions, backpropagation | How errors flow backward |

### Module 4: Applied ML
*The Forgery Trial — Mink Pavar defends mathematical methods*

| # | Lesson | What You'll Learn | The Story |
|---|--------|-------------------|-----------|
| 1 | Deriving Linear Regression | Least squares, normal equations | Building the prosecution's model |
| 2 | Bias-Variance Tradeoff | Underfitting, overfitting, model complexity | The Tribunal questions the evidence |
| 3 | Regularization (L1/L2) | Ridge, Lasso, why constraints help | Keeping the model honest |
| 4 | Model Selection (Capstone) | Cross-validation, hyperparameters | The verdict |

---

## How to Use This Course

### Option 1: Google Colab (Recommended)

Click any "Open in Colab" badge in the README. You don't need to install anything.

1. Click the badge
2. Sign in with a Google account
3. Go to **File → Save a copy in Drive** to keep your work
4. Work through the notebook
5. Complete the exercises

### Option 2: Local Installation

If you prefer working locally:

```bash
git clone https://github.com/buildLittleWorlds/ml-math-with-densworld.git
cd ml-math-with-densworld
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

---

## Prerequisites

This course assumes you have:

1. **Basic Python familiarity** — Loops, functions, lists
2. **High school algebra** — Variables, equations, basic graphing
3. **No prior ML or advanced math required** — We build from foundations

If you can solve for x and read a Python function, you're ready.

### Helpful But Not Required

- **Yeller Quarry Data Science** — Familiarity with pandas makes data exercises smoother
- **Capital Archives NLP** — Context for the manuscript examples

---

## Compute Requirements

**Good news:** This entire course runs on free Google Colab without a GPU.

All computations use NumPy and basic Python. Nothing requires deep learning hardware.

---

## The Data

This course uses data from across Densworld:

| File | Records | Description |
|------|---------|-------------|
| `expedition_outcomes.csv` | 1,000 | Expedition results with catch values, casualties |
| `creature_market.csv` | 2,000 | Market transactions with prices |
| `manuscript_features.csv` | 300 | Stylometric features for forgery detection |
| `siege_progress.csv` | 240 | The Colonel's 20-year siege data |
| `creature_vectors.csv` | 25 | Creatures as feature vectors |
| `dens_boundary_observations.csv` | 590 | Signal vs noise measurements |
| `scholar_debates.csv` | 256 | Hypothesis testing data |

---

## Key Characters

**Mink Pavar** — The reclusive scholar who exposed the Great Forgery. Stood before the Tribunal to defend his mathematical methods. His story frames Module 4.

**The Colonel** — Commander who besieged the Tower of Mirado for twenty years. His journals record an optimization problem: each day adjusting tactics based on measured outcomes. His story frames Module 3.

**Boffa Trent** — Natural philosopher who catalogued Quarry creatures. His taxonomy treats creatures as feature vectors. His work frames Module 2.

**Vagabu Olt** — The wandering cartographer whose observations inform expedition planning under uncertainty. His travels frame Module 1.

---

## A Note on Mathematical Rigor

This course prioritizes intuition over formalism. You'll understand *why* gradient descent works before you see the convergence proof. You'll grasp *what* the dot product measures before you memorize the formula.

This is deliberate. Machine learning practice requires mathematical intuition more than theorem-proving skill. If you later pursue theoretical ML, you'll have the conceptual foundation to build on.

That said, the math is real. When we derive linear regression, we actually derive it. When we explain backpropagation, we trace through real computations. The Densworld context adds meaning—it doesn't replace rigor.

---

## Troubleshooting

### "The math is too abstract"
Focus on the Densworld framing. Every formula corresponds to a concrete scenario—creatures as vectors, the Colonel stepping downhill, manuscript similarity scores.

### "The math is too easy"
This course builds toward ML Math, not pure mathematics. If you want more depth, see the "Going Deeper" suggestions at the end of each notebook.

### "I can't visualize high dimensions"
Nobody can, not really. The course uses 2D and 3D visualizations to build intuition, then helps you reason about higher dimensions algebraically.

### "Gradient descent isn't converging"
Check your learning rate. Too large and you'll overshoot. Too small and you'll take forever. The Colonel learned this the hard way.

---

## Course Series Context

This is a **Foundational Course** in the Densworld series:

| Course | Title | Focus |
|--------|-------|-------|
| 1 | Yeller Quarry Data Science | Pandas & Data Analysis |
| 2 | Capital Archives NLP | Natural Language Processing |
| **3** | **ML Math with Densworld** | **Mathematical Foundations** |
| 4 | Journeys and Graphs | Graph Theory & Networks |
| 5 | Densmok CC Analytics | Time Series & Anomaly Detection |
| 6 | Densworld API Course | FastAPI Development |

### What Comes Next

After completing this course, you have the mathematical foundation for:
- **The Forge at Yeller Quarry** — Fine-tuning language models (uses gradient descent, loss functions)
- **The Archivist's Inference Engine** — Understanding model confidence and attention
- **The Wanderer's Experiments** — Evaluating and comparing models

The math in this course is the same math that powers modern AI. The Colonel's gradient steps are what GPT uses to learn.

---

## One Last Thing

Mathematics is not about memorizing formulas. It's about learning to see patterns, to quantify uncertainty, to optimize systematically.

The Colonel spent twenty years besieging the Tower of Mirado. Each day he tried something, measured the outcome, and adjusted. He couldn't see the loss landscape he was navigating—but he could feel the gradient beneath his feet.

That's what this course teaches: how to feel the gradient. How to know which way is downhill. How to step carefully enough to converge but boldly enough to make progress.

Mink Pavar stood before the Tribunal with nothing but mathematics. The words in a manuscript were all he had. The patterns were all he could measure. The conclusion—forgery—was his to defend.

The techniques in this course gave him that power. Now they're yours.

---

*"The mathematics of learning is now yours. Use it wisely."*

*— Mink Pavar, after the verdict*

---

**Ready?**

[Start Module 1, Lesson 1 →](https://colab.research.google.com/github/buildLittleWorlds/ml-math-with-densworld/blob/main/modules/01-statistics-probability/notebooks/01-uncertainty-intuition.ipynb)
