# Day 4 — Pandas Basics + Dataset Loading

> **Goal:** Load, explore, and extract insights from real-world tabular data using Pandas. Understand the core data structures and operations that form the foundation of every ML pipeline.

---

## 1. What is Pandas?

Pandas is built **on top of NumPy** and adds labeled rows and columns to arrays — making tabular data easy to work with.

| Structure | What it is | NumPy equivalent |
|---|---|---|
| **Series** | Single column with labels | 1D array with index |
| **DataFrame** | Full table (rows + columns) | 2D array with row & column labels |

```python
import pandas as pd

# Series — one column
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# DataFrame — full table
df = pd.read_csv('file.csv')
```

> 💡 Pandas is not a replacement for NumPy — it sits on top of it. A DataFrame column is just a NumPy array with a label.

---

## 2. Loading a Dataset

```python
# From a CSV file
df = pd.read_csv('file.csv')

# From a URL
df = pd.read_csv('https://raw.githubusercontent.com/...')
```

---

## 3. First Look — Always Do This First

Before doing anything with a dataset, understand its shape and quality.

```python
df.shape            # (rows, columns)
df.head()           # First 5 rows
df.tail()           # Last 5 rows
df.columns          # Column names
df.dtypes           # Data type per column
df.info()           # Summary — nulls, types, memory usage
df.describe()       # Stats for numeric columns (mean, std, min, max)
```

> 💡 `df.info()` and `df.isnull().sum()` are the two most important first steps in any real ML project. Always check the quality of your data before touching it.

---

## 4. Selecting Data

```python
df['title']                         # Single column → returns a Series
df[['title', 'type', 'country']]    # Multiple columns → returns a DataFrame

df[df['type'] == 'Movie']           # Filter rows by condition
df[df['release_year'] > 2019]       # Filter by numeric condition
```

> 💡 Boolean filtering in Pandas works exactly like NumPy — `df[df['col'] > value]` is the same mental model as `arr[arr > value]`.

---

## 5. Handling Missing Values

Real datasets always have missing values. Know your options:

```python
df.isnull().sum()                           # Count nulls per column

df.dropna()                                 # Drop ALL rows with any null — loses data, use carefully
df.dropna(subset=['title', 'type'])         # Only drop rows where specific columns are null

df['country'].fillna('Unknown', inplace=True)     # Fill nulls with a default value
df['director'].fillna('Not Given', inplace=True)
```

### When to drop vs fill?
- **Drop** — when the row is unusable without that value (e.g. missing target label)
- **Fill** — when you can make a reasonable assumption (e.g. unknown country, median age)

---

## 6. Aggregations & Grouping

```python
df['type'].value_counts()                       # Count of each unique value
df['release_year'].mean()                       # Mean of a column
df.groupby('type')['release_year'].mean()       # Mean release year, split by type
df['country'].value_counts().head(10)           # Top 10 most common countries
```

---

## 7. Handling Date Columns

Date columns load as **strings** by default — convert them before doing any date math.

```python
# Step 1 — check the raw type (usually 'object')
print(df['date_added'].dtype)

# Step 2 — convert to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
# errors='coerce' → unparseable dates become NaT (Not a Time) instead of crashing

# Step 3 — extract date parts
df['date_added'].dt.year     # Extract year
df['date_added'].dt.month    # Extract month
df['date_added'].dt.day      # Extract day

# Filter by year
df[df['date_added'].dt.year > 2018].shape[0]   # Count rows after 2018
```

---

## 8. Handling Multi-Value Cells

A common real-world issue — multiple values stored in one cell as a comma-separated string.

```
# Raw data looks like:
"United States, India, United Kingdom"
```

Using `value_counts()` directly undercounts because it treats the whole string as one value.

```python
# ❌ Undercounts — treats "United States, India" as one entry
df['country'].value_counts().head(1)

# ✅ Correct — splits, explodes, then counts
df['country'].str.split(', ').explode().str.strip().value_counts().head(1)
```

### What each step does
| Step | What it does |
|---|---|
| `.str.split(', ')` | Splits each cell into a list |
| `.explode()` | Turns each list item into its own row |
| `.str.strip()` | Removes accidental whitespace |
| `.value_counts()` | Counts each value independently |

---

## 9. Percentages with value_counts()

```python
# Raw counts
df['type'].value_counts()

# As percentages — clean one-liner
df['type'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
```

`normalize=True` converts counts to proportions (0–1). Multiply by 100 for percentages.

---

## 10. Mini Project — Netflix Dataset EDA

**Dataset:** Netflix Movies & TV Shows  
**3 questions answered with Pandas:**

```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/netflix_titles.csv')

# --- Q1: How many titles were added after 2018? ---
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
titles_after_2018 = df[df['date_added'].dt.year > 2018].shape[0]~
print(f"Titles added after 2018: {titles_after_2018}")

# --- Q2: Which country has the most content? ---
top_country = df['country'].str.split(', ').explode().str.strip().value_counts().head(1)
print(f"Top country: {top_country}")

# --- Q3: Percentage of Movies vs TV Shows ---
percentages = df['type'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
print(percentages)
```

---

## Quick Recall Cheatsheet

| Concept | One-liner |
|---|---|
| DataFrame | Table with labeled rows and columns, built on NumPy |
| `df.info()` | Always run first — shows nulls, types, shape |
| `df['col']` | Returns a Series |
| `df[['col1','col2']]` | Returns a DataFrame |
| Boolean filter | `df[df['col'] == value]` — same as NumPy |
| `dropna()` | Removes rows with nulls — use carefully |
| `fillna()` | Fills nulls with a default value |
| `value_counts()` | Frequency of each unique value |
| `groupby()` | Aggregate by category |
| `pd.to_datetime()` | Convert string → datetime |
| `.dt.year` | Extract year from datetime column |
| `.explode()` | Unpack multi-value cells into separate rows |
| `normalize=True` | Converts counts to proportions in value_counts() |

---

