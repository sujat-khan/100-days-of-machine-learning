# Bagging, Pasting, Random Subspaces, and Random Patches

## Overview
Ensemble learning with `BaggingClassifier` (or `BaggingRegressor`) allows sampling both **instances (rows)** and **features (columns)** to train diverse base estimators. Depending on how sampling is configured across rows and columns, we get four primary sampling techniques:

1. **Bagging**
2. **Pasting**
3. **Random Subspaces**
4. **Random Patches**

---

## 1. Sampling Rows: Bagging vs. Pasting

| Feature | Bagging (Bootstrap Aggregating) | Pasting |
| :--- | :--- | :--- |
| **Sampling Strategy** | With replacement (`bootstrap=True`) | Without replacement (`bootstrap=False`) |
| **Duplicate Instances** | A single data point can appear multiple times in the same model's training subset. | Each data point can appear at most once per model's training subset. |
| **Out-of-Bag (OOB) Evaluation** | **Supported** — ~37% of instances left out per model can be used for evaluation. | **Not Supported** |
| **Diversity** | Higher diversity, lower variance (reduces overfitting). | Slightly lower diversity, slightly higher variance. |

---

## 2. Sampling Features & Rows: Random Subspaces vs. Random Patches

### Random Subspaces
- **Concept**: Keeps **all instances (rows)** but samples a random subset of **features (columns)**.
- **When to use**: High-dimensional datasets (e.g., image features, genomic data) where features outnumber instances or feature correlation is high.
- **Parameters**: `max_samples=1.0`, `max_features < 1.0` (or integer count of features).

### Random Patches
- **Concept**: Samples random subsets of **both instances (rows)** AND **features (columns)**.
- **When to use**: Large datasets with high dimensions (many samples and many features) to speed up training while preserving estimator diversity.
- **Parameters**: `max_samples < 1.0`, `max_features < 1.0`.

---

## Summary Matrix of Ensemble Variants

| Method | Row Subsampling (`max_samples`) | Row Bootstrap (`bootstrap`) | Feature Subsampling (`max_features`) | Feature Bootstrap (`bootstrap_features`) |
| :--- | :---: | :---: | :---: | :---: |
| **Bagging** | `< 1.0` | `True` | `1.0` | `False` |
| **Pasting** | `< 1.0` | `False` | `1.0` | `False` |
| **Random Subspaces** | `1.0` | `False` | `< 1.0` | `True` or `False` |
| **Random Patches** | `< 1.0` | `True` or `False` | `< 1.0` | `True` or `False` |

---

## Code Examples (`scikit-learn`)

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Bagging (Row sampling with replacement)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.5,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

# 2. Pasting (Row sampling without replacement)
pasting = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.5,
    bootstrap=False,
    random_state=42
)

# 3. Random Subspaces (All rows, random subset of features)
random_subspaces = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=1.0,         # Use all instances
    max_features=0.7,        # Use 70% of features randomly
    bootstrap_features=True,
    random_state=42
)

# 4. Random Patches (Subsets of BOTH rows and features)
random_patches = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.5,         # 50% of instances
    max_features=0.7,        # 70% of features
    bootstrap=True,
    bootstrap_features=True,
    random_state=42
)
```
