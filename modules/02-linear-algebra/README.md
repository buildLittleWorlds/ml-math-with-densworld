# Module II: The Structure - Linear Algebra

*If Statistics is the philosophy of data, Linear Algebra is the syntax. It describes how massive datasets are stored and manipulated in high-dimensional space.*

---

## Why This Matters

Every dataset you work with is a matrix. Every machine learning model is fundamentally doing linear algebra operations. Without understanding these operations, you cannot:
- Understand why your neural network has a certain architecture
- Debug dimensionality problems in your models
- Grasp how recommendation systems find similar users/items
- Understand dimensionality reduction techniques like PCA

---

## Lessons

### 1. Vectors: Arrows vs. Data Points
**Notebook:** `01-vectors-dual-view.ipynb`

- The Computer Science View: A row in a database (e.g., `[Age, Height, Income]`)
- The Physics View: A coordinate in n-dimensional space
- Why both views matter for understanding ML

### 2. Vector Norms: Measuring Distance
**Notebook:** `02-vector-norms.ipynb`

- L2 (Euclidean): "As the crow flies" distance
- L1 (Manhattan): "Taxicab" grid movement
- Why different norms matter for regularization

### 3. The Dot Product: The "Similarity" Detector
**Notebook:** `03-dot-product-similarity.ipynb`

- The Mechanics: $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$
- The Intuition: A measure of alignment
  - High Positive = Pointing same direction (Similar)
  - Zero = Perpendicular (Unrelated/Orthogonal)
  - Negative = Pointing opposite directions
- Application: Recommender systems and word embeddings

### 4. Matrices as Transformations
**Notebook:** `04-matrix-transformations.ipynb`

- Matrices as functions that transform vectors
- Visualizing what matrix multiplication "does"
- The connection to neural network layers

### 5. Matrix Rank & Linear Independence
**Notebook:** `05-rank-independence.ipynb`

- Linear Independence: Does a new feature add information?
- Why "Low Rank" matrices break regression models
- Introduction to dimensionality reduction

---

## Learning Objectives

By the end of this module, you will be able to:

1. Interpret a data point as both a database row and a vector in space
2. Calculate L1 and L2 distances and explain when to use each
3. Use the dot product to measure similarity between vectors
4. Visualize and explain what matrix multiplication does geometrically
5. Determine if features are linearly independent
