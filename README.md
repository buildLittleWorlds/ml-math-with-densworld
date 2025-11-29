# Mathematical Foundations for Applied Machine Learning

**Program:** Applied Data Science Track
**Prerequisites:** Basic Python Programming
**Estimated Completion Time:** 12-16 Weeks

---

## The Problem We're Solving

The modern data science education landscape is plagued by a "tools-first" mentality. Bootcamps produce graduates who can `import sklearn` and execute code, but crumble when models perform poorly on real-world data. They memorize syntax but lack the mathematical literacy to understand *why* an algorithm fails.

## Our Approach: Semantic Mathematics

This curriculum is built on the thesis that **mathematical intuition is the primary differentiator** between a junior analyst and a senior data scientist.

We don't aim to turn you into an academic mathematician. Instead, we use **Semantic Mathematics** - we rarely introduce a formula without first defining the natural language problem it solves. Equations are treated not as rules to be obeyed, but as sentences to be read.

---

## Course Structure

```
ml-math-with-densworld/
├── modules/
│   ├── 01-statistics-probability/    # The Bedrock
│   ├── 02-linear-algebra/            # The Structure
│   ├── 03-calculus/                  # The Engine
│   └── 04-applied-ml/                # The Synthesis
├── docs/
├── assets/
└── planning.md
```

### Module I: The Bedrock - Statistics & Probability
*Understanding uncertainty, noise, and signal.*

- Populations vs. Samples
- Random Variables & Distributions
- The Central Limit Theorem
- Hypothesis Testing & P-Values
- Bayesian Thinking & Confusion Matrices

### Module II: The Structure - Linear Algebra
*How computers represent and manipulate data.*

- Vectors as Data Points
- The Dot Product as "Similarity"
- Matrix Rank & Linear Independence
- Dimensionality Reduction Foundations

### Module III: The Engine - Calculus
*How models learn and optimize.*

- Derivatives as "Sensitivity"
- The Gradient as a Compass
- Gradient Descent
- The Chain Rule & Backpropagation

### Module IV: Synthesis - Applied Machine Learning
*Combining the three pillars.*

- Deriving Linear Regression from First Principles
- The Bias-Variance Trade-off
- Regularization (L1/L2)
- Model Selection & Evaluation

---

## Learning Outcomes

By the end of this sequence, you will be able to:

1. **Debug Models:** Diagnose overfitting vs. underfitting using statistical metrics
2. **Select Algorithms:** Choose the correct tool based on data distribution, not random trial and error
3. **Interpret Results:** Explain model outputs to stakeholders using concepts of uncertainty and probability

---

## Getting Started

1. Ensure you have Python 3.8+ installed
2. Install dependencies: `pip install -r requirements.txt`
3. Start with Module 1: [01-statistics-probability](modules/01-statistics-probability/)

---

## Recommended External Resources

**Phase 1: Foundations**
- Khan Academy: AP Statistics, Linear Algebra, Differential Calculus

**Phase 2: Visual Intuition**
- 3Blue1Brown: "Essence of Linear Algebra" and "Essence of Calculus"

**Phase 3: Theoretical Depth**
- *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani)
  - Focus: Ch 2 (Statistical Learning), Ch 3 (Linear Regression), Ch 6 (Regularization)
