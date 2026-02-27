# Day 3 — NumPy Arrays & Vectorization

> **Goal:** Understand why NumPy exists, how to use arrays efficiently, and the core preprocessing techniques used in ML.

---

## 1. What is a NumPy Array?

A NumPy array is a **fixed-type, multi-dimensional container** for numerical data.

| Feature | Python List | NumPy Array |
|---|---|---|
| Types | Mixed (int, str, etc.) | Fixed (all same type) |
| Speed | Slow (Python loop overhead) | Fast (C-compiled) |
| Memory | Higher | Lower |
| Math ops | Manual loops | Built-in vectorized ops |

```python
import numpy as np

arr = np.array([1, 2, 3])
print(arr.shape)   # (3,)
print(arr.dtype)   # int64
```

> 💡 **Always check `.shape` and `.dtype`** — most ML bugs come from shape mismatches.

---

## 2. Creating Arrays

```python
np.array([[1,2],[3,4]])       # From a list
np.zeros((2, 2))              # All zeros
np.ones((2, 3), dtype=int)    # All ones
np.arange(2, 8, 2)            # [2, 4, 6] — like range()
np.linspace(2, 8, 5)          # [2. 3.5 5. 6.5 8.] — evenly spaced
np.empty((2, 2))              # Uninitialized (garbage values — use with caution)
np.random.randn(5, 5)         # Random from standard normal distribution
```

---

## 3. Indexing & Slicing

```python
arr = np.array([[1,2,3,4],
                [7,8,9,5]])

arr[0]        # First row  → [1 2 3 4]
arr[0, 3]     # Row 0, Col 3 → 4
arr[0, :]     # All columns of row 0
arr[:, 3]     # All rows, column 3 → [4 5]
arr[0:2, 1:3] # Rows 0-1, Cols 1-2 → [[2,3],[8,9]]
arr[::2]      # Every 2nd row
```

### ⚠️ Views vs Copies — Important!

```python
x = arr[0:2, 0:2]        # This is a VIEW — changes affect original arr!
x = arr[0:2, 0:2].copy() # This is a COPY — safe to modify independently
```

> In ML pipelines, always `.copy()` when you don't want to alter the source data.

---

## 4. Boolean Filtering

```python
a = np.array([1, 2, 3, 4, 5])
print(a[a > 3])   # [4 5]
```

This is heavily used in ML for filtering datasets by condition (e.g. remove outliers, select class labels).

---

## 5. Vectorization

Instead of looping element by element, NumPy applies operations **across the whole array at once** using C-compiled code under the hood.

```python
# ❌ Slow — Python loop
for i in range(len(a)):
    a[i] += 10

# ✅ Fast — Vectorized
a = a + 10
```

> **Why it matters:** For large datasets (millions of rows), vectorization can be 100x faster than Python loops. This is the foundation of efficient ML training.

---

## 6. Broadcasting

Broadcasting lets NumPy perform operations on arrays of **different shapes** without copying data.

```python
arr = np.array([[1,2,3],
                [4,5,6]])   # Shape (2, 3)

row = np.array([10, 20, 30])  # Shape (3,)

print(arr + row)
# [[11 22 33]
#  [14 25 36]]
```

### Broadcasting Rules
1. If arrays have different numbers of dimensions, the smaller shape is **padded with 1s on the left**.
2. Dimensions are compatible if they are **equal** or one of them is **1**.

---

## 7. Aggregation Methods

```python
arr = np.array([[2,3,4],[5,6,7]])

np.sum(arr)              # Sum of all elements
np.sum(arr, axis=0)      # Column-wise sum → [7, 9, 11]
np.sum(arr, axis=1)      # Row-wise sum    → [9, 18]
np.mean(arr)             # Overall mean
np.std(arr)              # Standard deviation
np.min(arr, axis=0)      # Min per column
np.max(arr)              # Max overall
```

> 💡 **axis=0 → collapse rows (operate down columns). axis=1 → collapse columns (operate across rows).** This trips up almost everyone at first.

---

## 8. Mini Project — Normalisation vs Standardisation

### Why it matters in ML
Raw data has different scales (e.g. age: 0–100, salary: 0–100,000). Most ML models perform better when features are on a **similar scale**.

### Normalisation (Min-Max Scaling)
Squishes values to the range **[0, 1]**.

```
x_norm = (x - x_min) / (x_max - x_min)
```

Use when: you need **bounded values**, e.g. for neural network inputs or image pixels.

### Standardisation (Z-Score Scaling)
Recenters data to **mean ≈ 0, std ≈ 1**.

```
x_std = (x - mean) / std
```

Use when: your model **assumes normally distributed input**, e.g. linear regression, logistic regression.

### Code

```python
def normaliseArray(arr):
    arr = arr.astype(float)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_min == arr_max:          # Edge case: all same values
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def standardiseArray(arr):
    arr = arr.astype(float)
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    if arr_std == 0:                # Edge case: no variance
        return np.zeros_like(arr)
    return (arr - arr_mean) / arr_std

arr = np.random.randn(5, 5)
arr1 = normaliseArray(arr)
arr2 = standardiseArray(arr)

print(f"Original     — Mean: {np.mean(arr):.4f},  Std: {np.std(arr):.4f}")
print(f"Normalised   — Mean: {np.mean(arr1):.4f}, Std: {np.std(arr1):.4f}")
print(f"Standardised — Mean: {np.mean(arr2):.4f}, Std: {np.std(arr2):.4f}")
```

### What to expect in output
- Normalised mean: somewhere between 0 and 1 (not predictable)
- Standardised mean: **~0.0000**, std: **~1.0000** — always

---

## Quick Recall Cheatsheet

| Concept | One-liner |
|---|---|
| NumPy array | Fixed-type, C-speed, multi-dimensional |
| `.shape` / `.dtype` | Always check these first |
| Slicing | `arr[rows, cols]` — think row first, column second |
| View vs Copy | Default is view — use `.copy()` to be safe |
| Boolean filter | `arr[arr > 3]` — used heavily in data cleaning |
| Vectorization | No loops — apply ops to whole array at once |
| Broadcasting | NumPy stretches compatible shapes automatically |
| axis=0 | Operate **down** the rows (per column) |
| axis=1 | Operate **across** the columns (per row) |
| Normalise | Bounds to [0,1] — use for bounded inputs |
| Standardise | Mean~0, Std~1 — use for regression models |

---

