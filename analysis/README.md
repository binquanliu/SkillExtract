# Improved Baseline Analysis for Fig2c & Fig2d

This directory contains an improved implementation of baseline analysis for comparing machine learning models against rigorous baselines. The code addresses common issues in random baseline implementations and provides a comprehensive set of controls.

## 📋 Table of Contents

- [Key Improvements](#key-improvements)
- [Files](#files)
- [Quick Start](#quick-start)
- [Baseline Methods](#baseline-methods)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Understanding the Output](#understanding-the-output)
- [Common Issues](#common-issues)

## 🚀 Key Improvements

### Original Code Issues

The original code had several problems:

1. **❌ Fixed Random Seed**: Used `np.random.seed(42)` for all random baselines, making them not truly random
2. **❌ Single Trial**: Only one random trial, no way to measure variability
3. **❌ Uncontrolled Normalization**: Applied L2 normalization without checking if real data was normalized
4. **❌ Missing Variables**: Many undefined variables and incomplete context

### Improvements in This Version

1. **✅ True Random Baselines**: Multiple trials with different seeds
2. **✅ Matched Random Baselines**: Control for statistical properties of real data
3. **✅ Cross-Trial Variability**: Measure and report variance across random trials
4. **✅ Comprehensive Baselines**: 6 different baseline types
5. **✅ Complete Documentation**: Fully documented with examples
6. **✅ Modular Design**: Separate utility functions for reusability

## 📁 Files

```
analysis/
├── README.md                           # This file
├── analysis_utils.py                   # Core utility functions
├── fig2_baseline_analysis_improved.py  # Main analysis script
├── example_config.py                   # Configuration template
└── results/                            # Output directory (created automatically)
    ├── Fig2c_improved.png
    ├── Fig2c_improved_data.csv
    ├── Fig2d_improved.png
    └── Fig2d_improved_data.csv
```

## 🏃 Quick Start

### Step 1: Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn scipy joblib
pip install himalaya  # For ridge regression models
```

### Step 2: Prepare Your Data

You need three components:

1. **Participant Ratings** (`dfdata_use`): DataFrame with shape `(n_participants, n_items)`
2. **Feature Bags** (`X_bags_list`): List of length `n_items`, each element has shape `(n_reasons, n_dims)`
3. **Item IDs** (`use_items`): List of item identifiers

### Step 3: Run the Analysis

```bash
cd analysis
python fig2_baseline_analysis_improved.py
```

**Note**: The default script uses dummy data. See [Configuration](#configuration) to use your own data.

## 📊 Baseline Methods

### 1. Intercept Only (Baseline 1)

**What it does**: Predicts the mean rating from the training set.

**Purpose**: Tests if the model does better than just predicting the average.

**Expected performance**: Usually near 0 correlation.

```python
# In each CV fold:
prediction = mean(training_ratings)
```

### 2. True Random Vectors (Baseline 2)

**What it does**: Generates completely random feature vectors (multiple trials).

**Purpose**: Tests if any random features can produce correlations by chance.

**Improvements over original**:
- Multiple trials (default: 10) with different random seeds
- Reports cross-trial variability
- Optional L2 normalization matching real data

```python
# Multiple trials
for trial in range(n_trials):
    X_random = generate_random_vectors(seed=trial)
    results[trial] = fit_model(X_random, y)

# Average across trials
final_correlation = mean(results)
```

### 3. Matched Random Vectors (Baseline 3)

**What it does**: Generates random vectors that match the statistical properties of real data (mean, std, normalization).

**Purpose**: Controls for low-level statistical properties while removing semantic content.

**Key insight**: If this performs similarly to real data, your features may not contain meaningful semantic information.

```python
# Compute statistics of real data
real_mean = mean(all_real_vectors)
real_std = std(all_real_vectors)

# Generate matched random vectors
X_matched = random.randn(...) * real_std + real_mean
```

### 4. Shuffled Item-Reason Association (Baseline 4)

**What it does**: Randomly shuffles the correspondence between items and their feature bags.

**Purpose**: Tests if the specific item-feature mapping matters, or if any mapping would work.

**Interpretation**: If performance drops significantly, the specific associations are important.

```python
# Shuffle the item-feature mapping
shuffled_idx = random.permutation(len(X_bags))
X_shuffled = [X_bags[i] for i in shuffled_idx]
```

### 5. Domain-Label Baseline (Baseline 5)

**What it does**: Clusters items into domains (using K-means) and uses one-hot domain labels as features.

**Purpose**: Tests if coarse domain-level categorization is sufficient.

**Similar to**: RIASEC categories in career research (6 domains).

```python
# Cluster items
kmeans = KMeans(n_clusters=6)
domains = kmeans.fit_predict(item_embeddings)

# Create one-hot features
X_domain = one_hot_encode(domains)
```

### 6. Permutation Test (Baseline 6) - Optional

**What it does**: Shuffles the target variable (y) while keeping features intact.

**Purpose**: Creates a null distribution for significance testing.

**Warning**: Very computationally expensive (disabled by default).

```python
# For each participant
for perm in range(100):
    y_shuffled = random.permutation(y)
    result = fit_model(X, y_shuffled)
```

## ⚙️ Configuration

### Method 1: Modify `example_config.py`

```python
# Copy the example config
cp example_config.py my_config.py

# Edit my_config.py
class AnalysisConfig:
    N_CV_FOLDS = 5
    N_RANDOM_TRIALS = 10
    METHOD_NAME = "MyModel"
    # ... etc
```

### Method 2: Inline Configuration

```python
from fig2_baseline_analysis_improved import Config, run_complete_baseline_analysis

# Load your data
dfdata_use = pd.read_csv("my_ratings.csv")
X_bags_list = np.load("my_embeddings.npy", allow_pickle=True)

# Configure
config = Config()
config.N_RANDOM_TRIALS = 20  # More trials for stability
config.METHOD_NAME = "BERT-Embeddings"

# Run analysis
results = run_complete_baseline_analysis(dfdata_use, X_bags_list, use_items, config)
```

## 📈 Usage Examples

### Example 1: Basic Analysis

```python
import pandas as pd
import numpy as np
from fig2_baseline_analysis_improved import (
    Config,
    run_complete_baseline_analysis,
    plot_fig2c,
    plot_fig2d
)

# Load data
dfdata_use = pd.read_csv("participant_ratings.csv", index_col=0)
X_bags_list = np.load("embeddings.npy", allow_pickle=True).tolist()
use_items = list(dfdata_use.columns)

# Configure
config = Config()
config.N_RANDOM_TRIALS = 10
config.METHOD_NAME = "MyModel"

# Run baselines
baseline_results = run_complete_baseline_analysis(
    dfdata_use, X_bags_list, use_items, config
)

# Load main model results (from your previous analysis)
correls_main = pd.read_csv("my_model_results.csv")['correlation'].values

# Plot
plot_fig2c(correls_main, config.METHOD_NAME, "Fig2c.png")
plot_fig2d(correls_main, baseline_results, config.METHOD_NAME, "Fig2d.png")
```

### Example 2: Custom Baseline

```python
from analysis_utils import pearsonr_safe, standardize_y
from joblib import Parallel, delayed

def my_custom_baseline(p_idx, row, X_bags):
    """Your custom baseline implementation"""
    y = row.to_numpy(dtype=float)
    y_std, mu, sd = standardize_y(y)

    # Your baseline logic here
    y_pred = ...

    corr = pearsonr_safe(y_std, y_pred)
    return [p_idx, corr, 0.0]

# Run it
results = Parallel(n_jobs=-1)(
    delayed(my_custom_baseline)(p_idx, row, X_bags_list)
    for p_idx, (_, row) in enumerate(dfdata_use.iterrows())
)

correls = [r[1] for r in results]
```

### Example 3: Sensitivity Analysis

```python
# Test different numbers of random trials
for n_trials in [5, 10, 20, 50]:
    config.N_RANDOM_TRIALS = n_trials

    results = run_random_baseline(
        dfdata_use, n_items, n_reasons, n_dims,
        config.ALPHAS, config.N_CV_FOLDS, False,
        config.N_JOBS, n_trials=n_trials
    )

    print(f"Trials={n_trials}: Mean={np.mean(results[0]):.4f}, "
          f"Cross-trial SD={results[1]:.4f}")
```

## 🔍 Understanding the Output

### Console Output

```
[BASELINE 2] Running True Random Vector Model (10 trials)...
  Trial 1/10 (seed=100)...
    Mean Corr: 0.0234 ± 0.1123
  Trial 2/10 (seed=101)...
    Mean Corr: -0.0156 ± 0.1089
  ...

  [Summary]
    Mean Correlation: 0.0089 ± 0.1056
    Cross-trial variability: 0.0234  ← Important: measures random noise
```

**Key metrics:**
- **Mean Correlation**: Average across participants
- **Cross-trial variability**: How much the random baseline varies between trials (should be reported!)

### Statistical Tests

```
ONE-SAMPLE T-TESTS (vs 0)
======================================================================
MyModel                  : t= 12.345, p=1.23e-15, n=100
True Random              : t=  0.567, p=5.71e-01, n=100  ← Not significant
Matched Random           : t=  0.823, p=4.12e-01, n=100  ← Not significant
...

PAIRED T-TESTS (MyModel vs Baselines)
======================================================================
vs True Random       : t= 11.234, p=2.34e-14, Δ=+0.1234  ← Significant improvement
vs Matched Random    : t=  9.876, p=5.67e-12, Δ=+0.0987
...
```

### Figures

**Fig2c**: Distribution of correlations for your main model
- Should show mostly positive correlations
- Mean should be significantly > 0

**Fig2d**: Bar chart comparing all baselines
- Your model should be the rightmost (highest) bar
- Error bars show standard error
- Random baselines should be near 0

## ❓ Common Issues

### Issue 1: Random Baseline Shows High Correlation

**Symptom**: Random vectors give correlation > 0.1

**Possible causes**:
1. Not enough CV folds (overfitting)
2. Too few items (high variance)
3. Data leakage

**Solution**:
```python
config.N_CV_FOLDS = 10  # Increase folds
config.N_RANDOM_TRIALS = 20  # More trials to confirm
```

### Issue 2: Matched Random ≈ Real Data

**Symptom**: Matched random baseline performs similar to your model

**Interpretation**: Your features may not contain semantic information beyond their statistical properties

**Action**: Check if embeddings are meaningful
```python
from analysis_utils import check_data_normalization

is_norm, avg_norm = check_data_normalization(X_bags_list)
print(f"Normalized: {is_norm}, Avg norm: {avg_norm}")

# Visualize embeddings
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform([bag.mean(axis=0) for bag in X_bags_list])
plt.scatter(X_2d[:, 0], X_2d[:, 1])
```

### Issue 3: All Baselines Near Zero, Main Model Also Low

**Symptom**: Everything shows near-zero correlation

**Possible causes**:
1. Task is very difficult
2. Features are not informative
3. Not enough data

**Diagnostics**:
```python
# Check data variance
print(f"Rating variance: {dfdata_use.values.var():.4f}")
print(f"Rating range: {dfdata_use.values.min():.2f} to {dfdata_use.values.max():.2f}")

# Check for NaNs
print(f"NaN count: {dfdata_use.isna().sum().sum()}")

# Check feature variance
feature_vars = [bag.var() for bag in X_bags_list]
print(f"Feature variance: {np.mean(feature_vars):.4f} ± {np.std(feature_vars):.4f}")
```

### Issue 4: Shuffled Items ≈ Real Items

**Symptom**: Shuffling item-feature associations doesn't hurt performance

**Interpretation**: The model is not learning item-specific associations; it may be exploiting dataset biases

**Action**: Investigate potential confounds
```python
# Check if certain items are systematically rated higher
item_means = dfdata_use.mean(axis=0)
print(f"Item mean range: {item_means.min():.2f} to {item_means.max():.2f}")

# Check participant biases
participant_means = dfdata_use.mean(axis=1)
print(f"Participant mean range: {participant_means.min():.2f} to {participant_means.max():.2f}")
```

### Issue 5: Out of Memory Errors

**Symptom**: Script crashes with memory error

**Solutions**:
```python
# Reduce parallel jobs
config.N_JOBS = 4  # Instead of -1

# Process in batches
def process_in_batches(func, items, batch_size=10):
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = [func(item) for item in batch]
        results.extend(batch_results)
    return results

# Disable permutation test
config.RUN_PERMUTATION_TEST = False
```

## 📝 Citation

If you use this code in your research, please cite appropriately and describe the baseline methods in your paper.

### Recommended Description for Methods Section

> "We compared our model against six baselines: (1) Intercept-only, predicting the mean rating; (2) True random vectors, averaging over 10 trials with different random seeds to estimate chance performance; (3) Matched random vectors, controlling for the statistical properties (mean, standard deviation, and normalization) of the real embeddings; (4) Shuffled item-feature associations, testing whether specific item-feature mappings matter; (5) Domain labels, using K-means clustering to create coarse-grained categories; and (6) Permutation tests, shuffling target variables to create a null distribution. Statistical significance was assessed using paired t-tests comparing the main model to each baseline."

## 📧 Support

For issues or questions:
1. Check the [Common Issues](#common-issues) section
2. Review the code comments in `analysis_utils.py`
3. Run with dummy data first to verify installation

## 📄 License

MIT License - feel free to use and modify for your research.

---

**Last updated**: 2026-01-01
**Version**: 2.0 (Improved)
