# Module IV: Synthesis - Applied Machine Learning

*Combining the three pillars to derive algorithms and evaluate performance.*

---

## Why This Matters

This is where everything comes together. The statistics, linear algebra, and calculus you've learned are not separate topics - they are the unified language of machine learning.

In this module, you will:
- **Derive** algorithms from first principles (not just memorize them)
- **Understand** the most important theoretical concept in ML
- **Apply** regularization techniques with mathematical understanding

---

## Lessons

### 1. Deriving Linear Regression
**Notebook:** `01-deriving-linear-regression.ipynb`

We don't memorize the formula; we derive it from the problem statement:
1. **Goal**: Find the line closest to all points
2. **Metric**: Vertical distance (Residuals)
3. **Constraint**: Distances must be positive → Square them
4. **Result**: Minimize the Sum of Squared Residuals (RSS)

### 2. The Bias-Variance Trade-off
**Notebook:** `02-bias-variance-tradeoff.ipynb`

*This is widely considered the single most important theoretical concept in Machine Learning.*

- **Bias (Underfitting)**: Model is too simple
  - Symptom: High Training Error, High Test Error
- **Variance (Overfitting)**: Model is too complex, memorizes noise
  - Symptom: Low Training Error, Huge Test Error
- **The Goal**: Find the "Sweet Spot" of complexity

### 3. Regularization: Taming Complexity
**Notebook:** `03-regularization.ipynb`

Mathematically penalizing complexity to reduce variance:
- **L2 (Ridge)**: Punishes large weights, keeps model stable
- **L1 (Lasso)**: Forces unnecessary weights to zero (automatic feature selection)
- When to use which

### 4. Model Selection & Cross-Validation
**Notebook:** `04-model-selection.ipynb`

- Why you can't evaluate on training data
- The holdout method and its limitations
- K-Fold Cross-Validation
- Choosing hyperparameters responsibly

---

## Learning Objectives

By the end of this module, you will be able to:

1. Derive the linear regression solution from scratch using calculus
2. Explain and diagnose bias vs variance problems in any model
3. Implement L1 and L2 regularization and explain their different effects
4. Properly evaluate models using cross-validation
5. Make principled decisions about model complexity
