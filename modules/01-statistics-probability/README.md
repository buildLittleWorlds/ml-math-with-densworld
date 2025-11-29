# Module I: The Bedrock - Statistics & Probability

*Statistics is the science of distinguishing signal (truth) from noise (randomness). It is the most critical skill for avoiding false discoveries.*

---

## Why This Matters

Every machine learning model makes predictions under uncertainty. Without understanding statistics, you cannot:
- Know if your model is actually better than random guessing
- Determine if a performance improvement is real or just noise
- Avoid fooling yourself with overfitted models

---

## Lessons

### 1. The Intuition of Uncertainty
**Notebook:** `01-uncertainty-intuition.ipynb`

- Populations vs. Samples - we almost never see the "Truth" (Population), only a shadow of it (Sample)
- Every metric you calculate (mean, variance) is an *estimate* with an attached error bar
- Random Variables: moving from algebra ($x=5$) to probability functions

### 2. Distributions as Terrain
**Notebook:** `02-distributions.ipynb`

- The Normal Distribution: why it represents "accumulated independent errors"
- Skewness & Kurtosis: identifying "long tail" data where standard models fail
- Visualizing and recognizing distribution shapes

### 3. The Central Limit Theorem
**Notebook:** `03-central-limit-theorem.ipynb`

- The mathematical bridge from messy real-world data to standard statistical tools
- Why "Squared Error" loss functions work
- Sampling distributions and standard error

### 4. Hypothesis Testing & Significance
**Notebook:** `04-hypothesis-testing.ipynb`

- The P-Value Misconception: it's a measure of *surprise*, not certainty
- The Multiple Comparisons Problem (P-Hacking)
- The Bonferroni Correction

### 5. Bayesian Thinking & Classification
**Notebook:** `05-bayesian-thinking.ipynb`

- Confusion Matrices: beyond simple "Accuracy"
- Sensitivity (Recall) vs. Specificity and their trade-offs
- Bayes' Theorem: updating beliefs based on new evidence

---

## Learning Objectives

By the end of this module, you will be able to:

1. Explain why sample statistics are always estimates with uncertainty
2. Identify when data violates normality assumptions
3. Correctly interpret p-values and avoid common statistical fallacies
4. Calculate and explain precision, recall, and the confusion matrix
5. Apply Bayes' Theorem to update probabilities
