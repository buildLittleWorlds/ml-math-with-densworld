# Module III: The Engine - Calculus

*Calculus in Machine Learning is almost entirely focused on Optimization: the process of sliding a bad model toward a good model.*

---

## Why This Matters

Every time you train a machine learning model, calculus is running under the hood. Without understanding it, you cannot:
- Understand why learning rates matter
- Debug models that won't converge
- Grasp how deep learning actually works
- Make informed decisions about optimization hyperparameters

---

## A Different Approach to Calculus

Forget the geometry you learned in school ("slope of tangent lines").

In ML, we think of derivatives as **sensitivity analysis**:

> "If I nudge input $x$ slightly, how violently does the error react?"

---

## Lessons

### 1. Derivatives as Sensitivity
**Notebook:** `01-derivatives-sensitivity.ipynb`

- Reframing calculus from geometry to sensitivity analysis
- Understanding what the derivative actually tells us
- The connection to model training

### 2. The Gradient: Your Compass
**Notebook:** `02-gradient-compass.ipynb`

- What is a gradient? (A vector of partial derivatives)
- The "foggy mountain" intuition
- How to know which direction is "downhill"

### 3. Gradient Descent: Learning to Walk Downhill
**Notebook:** `03-gradient-descent.ipynb`

- The core optimization algorithm of machine learning
- Learning rate: how big should each step be?
- Visualizing the loss landscape
- Common pitfalls (local minima, saddle points)

### 4. The Chain Rule: Layers of Functions
**Notebook:** `04-chain-rule-backprop.ipynb`

- Why neural networks are "nested functions"
- The Chain Rule: combining sensitivities
- Backpropagation demystified

---

## Learning Objectives

By the end of this module, you will be able to:

1. Interpret derivatives as sensitivity measures, not just slopes
2. Calculate and interpret gradients for multi-variable functions
3. Implement gradient descent from scratch
4. Explain how backpropagation uses the chain rule
5. Diagnose common optimization problems (learning rate issues, vanishing gradients)
