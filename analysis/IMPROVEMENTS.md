# Improvements Summary: Baseline Analysis Code

## 🔴 Critical Issues Fixed in Original Code

### 1. **Fixed Random Seed Issue** ⚠️ CRITICAL

**Original Code:**
```python
np.random.seed(42)  # ❌ Fixed seed
X_bags_random = []
for item in use_items:
    random_matrix = np.random.randn(n_reasons, n_dims)
    ...
```

**Problem:**
- Generated the **exact same** "random" vectors every time
- Not truly random - defeats the purpose of a random baseline
- Cannot assess variability due to randomness

**Improved Version:**
```python
# Multiple trials with different seeds
all_trial_correls = []
for trial in range(n_random_trials):
    np.random.seed(100 + trial)  # ✓ Different seed each trial
    X_bags_random = generate_random_bags(...)
    results = run_model(X_bags_random)
    all_trial_correls.append(results)

# Average across trials
correls_mean = np.mean(all_trial_correls, axis=0)
cross_trial_std = np.std(all_trial_correls, axis=0).mean()  # ✓ Report variability!
```

**Impact:** Original code likely **underestimated** the variance of random baseline.

---

### 2. **Uncontrolled Normalization** ⚠️ MODERATE

**Original Code:**
```python
# Always normalizes random vectors
for i in range(n_reasons):
    norm = np.linalg.norm(random_matrix[i])
    if norm > 0:
        random_matrix[i] = random_matrix[i] / norm  # ❌ Always L2-normalize
```

**Problem:**
- Normalizes random vectors without checking if real data is normalized
- Creates unfair comparison if real data is not normalized
- L2-normalization imposes structure (all vectors on unit sphere)

**Improved Version:**
```python
# Auto-detect normalization from real data
is_normalized, avg_norm = check_data_normalization(X_bags_list)

# Generate random bags matching real data properties
X_bags_random = generate_random_bags(
    n_items, n_reasons, n_dims,
    normalize=is_normalized  # ✓ Match real data
)
```

**Impact:** Fair comparison between random and real embeddings.

---

### 3. **Single Trial - No Variance Estimate** ⚠️ CRITICAL

**Original Code:**
```python
# Run model once with random vectors
results_random = Parallel(...)(
    delayed(process_participant_corrected)(...)
    for p_idx, (_, row) in enumerate(dfdata_use.iterrows())
)

correls_random = [r[1] for r in results_random]
print(f"Mean Corr: {np.nanmean(correls_random):.4f}")  # ❌ No variance across trials
```

**Problem:**
- Single random trial = unreliable estimate
- Cannot distinguish signal from noise
- No confidence intervals for random baseline

**Improved Version:**
```python
# Multiple trials
all_trial_correls = []
for trial in range(10):  # ✓ 10 trials
    ... run trial ...
    all_trial_correls.append(correls)

# Report mean AND variance
correls_mean = np.mean(all_trial_correls, axis=0)
cross_trial_std = np.std(all_trial_correls, axis=0).mean()

print(f"Mean: {np.mean(correls_mean):.4f}")
print(f"Cross-trial variability: {cross_trial_std:.4f}")  # ✓ Report this!
```

**Impact:** Can now assess whether main model significantly exceeds random chance.

---

### 4. **Missing Matched Random Baseline** ⚠️ MODERATE

**Original Code:**
- Only had "pure random" baseline
- Did not control for statistical properties of real embeddings

**Improved Version:**
```python
def generate_matched_random_bags(X_bags_list, seed=None):
    """Generate random bags matching statistical properties of real data"""

    # Compute real data statistics
    all_vectors = np.vstack([bag.flatten() for bag in X_bags_list])
    real_mean = np.mean(all_vectors)
    real_std = np.std(all_vectors)

    # Generate matched random
    random_matrix = np.random.randn(n_reasons, n_dims) * real_std + real_mean

    # Match normalization if applicable
    if is_normalized:
        # ... normalize to match avg_norm ...
```

**Impact:** Can now test if real embeddings work better than noise with same statistics.

---

### 5. **Incomplete Variable Definitions** ⚠️ HIGH

**Original Code:**
- References undefined variables: `dfdata_use`, `X_bags_list`, `use_items`
- Missing function: `process_participant_corrected()`
- Undefined constants: `n_reasons`, `n_dims`, `ALPHAS`, etc.

**Improved Version:**
- Complete, runnable code
- Example data loading in `example_config.py`
- All functions fully implemented
- Comprehensive documentation

---

## ✅ New Features Added

### 1. **Modular Utility Functions**

Created `analysis_utils.py` with reusable functions:
- `pearsonr_safe()`: Safe correlation with NaN handling
- `standardize_y()`: Standardization with safety checks
- `check_data_normalization()`: Auto-detect normalization
- `generate_random_bags()`: Flexible random generation
- `generate_matched_random_bags()`: Statistics-matched random
- `fit_group_ridge_cv()`: Unified model fitting
- `run_statistical_tests()`: Comprehensive statistical analysis

### 2. **Additional Baselines**

Added baselines missing from original:
- **Matched Random**: Controls for statistical properties
- **Permutation Test**: Most stringent null hypothesis test (optional)

### 3. **Comprehensive Documentation**

- **README.md**: 400+ lines of documentation
  - Usage examples
  - Common issues and solutions
  - Interpretation guide
  - Methods section template for papers

- **IMPROVEMENTS.md**: This file - explains all changes

- **example_config.py**: Template for user configuration

### 4. **Statistical Rigor**

```python
# Comprehensive statistical tests
run_statistical_tests(correls_main, baseline_dict)

# Reports:
# - One-sample t-tests (vs 0) for each method
# - Paired t-tests (main vs each baseline)
# - Effect sizes (mean differences)
# - Cross-trial variability for random baselines
```

### 5. **Better Visualization**

- Improved `plot_fig2c()`: Correlation distribution
- Improved `plot_fig2d()`: Baseline comparison with better colors
- Saves both figures and raw data (CSV)

---

## 📊 Comparison Table

| Feature | Original Code | Improved Code |
|---------|---------------|---------------|
| Random seed | Fixed (42) | Multiple (100+trial) |
| Number of trials | 1 | 10 (configurable) |
| Cross-trial variance | ❌ Not reported | ✅ Reported |
| Normalization check | ❌ Always normalizes | ✅ Auto-detects |
| Matched random | ❌ No | ✅ Yes |
| Permutation test | ❌ No | ✅ Yes (optional) |
| Statistical tests | ⚠️ Basic | ✅ Comprehensive |
| Documentation | ❌ Minimal | ✅ Extensive |
| Runnable example | ❌ No | ✅ Yes |
| Error handling | ⚠️ Basic | ✅ Robust |

---

## 🎯 Key Recommendations

### For Original Code Users

If you've been using the original code, you should:

1. **Re-run analyses with multiple random trials**
   ```python
   config.N_RANDOM_TRIALS = 10  # Minimum 10
   ```

2. **Report cross-trial variability**
   ```python
   print(f"Random baseline: {mean:.4f} (cross-trial SD: {std:.4f})")
   ```

3. **Use matched random baseline**
   - Tests if your embeddings are better than noise with same statistics
   - More stringent than pure random

4. **Check normalization**
   ```python
   is_normalized, avg_norm = check_data_normalization(X_bags_list)
   print(f"Data normalized: {is_normalized}")
   ```

### For Paper Writing

If reporting baseline comparisons, include:

✅ **Do include:**
- Number of random trials (e.g., "averaged over 10 random trials")
- Cross-trial standard deviation
- Both pure random and matched random baselines
- Paired t-tests with effect sizes

❌ **Don't:**
- Use single random trial
- Use fixed random seed without reporting it
- Compare normalized random vs non-normalized real data
- Only report mean without variance

### Example Methods Section

> "We compared our model against six baselines. For random baselines, we averaged results over 10 trials with different random seeds (seeds 100-109) to obtain stable estimates. The true random baseline used unconstrained random vectors, while the matched random baseline generated random vectors with the same mean, standard deviation, and L2-normalization as the real embeddings. We report both the mean correlation and cross-trial standard deviation (SD_trial) to quantify random variability. Statistical significance was assessed using paired t-tests."

---

## 🔬 Scientific Justification

### Why Multiple Random Trials Matter

**Single trial:**
- Random seed = 42: correlation = 0.023
- Random seed = 43: correlation = -0.015
- Random seed = 44: correlation = 0.031

**Issue:** Which one is "the" random baseline? All are equally valid!

**Solution:** Average over many trials
- Mean over 10 trials: 0.012 ± 0.018
- Now we have a distribution, not a point estimate

### Why Matched Random Matters

**Scenario:**
- Real embeddings: mean=0, std=0.3, normalized=True
- Pure random: mean=0, std=1.0, normalized=False

**Problem:** Different scales → unfair comparison

**Solution:** Matched random
- Same mean, std, and normalization as real data
- Only semantic content is removed

**Interpretation:**
- If matched random ≈ real: embeddings may not be semantic
- If real >> matched random: embeddings contain meaningful information

---

## 📈 Expected Results

### Healthy Pattern

```
Intercept Only:     r = 0.00 ± 0.05
True Random:        r = 0.01 ± 0.02  (cross-trial SD: 0.02)
Matched Random:     r = 0.02 ± 0.03  (cross-trial SD: 0.02)
Shuffled Items:     r = 0.05 ± 0.04
Domain Labels:      r = 0.15 ± 0.08
Your Model:         r = 0.35 ± 0.12  ✓ Clearly best
```

### Warning Signs

**Pattern 1: High Random Baseline**
```
True Random:        r = 0.12 ± 0.03  ⚠️ Too high!
Your Model:         r = 0.18 ± 0.05  ⚠️ Only slightly better
```
→ Possible overfitting, data leakage, or insufficient CV

**Pattern 2: Matched ≈ Real**
```
Matched Random:     r = 0.30 ± 0.05
Your Model:         r = 0.32 ± 0.06  ⚠️ Not much better
```
→ Embeddings may not contain semantic information

**Pattern 3: Shuffled ≈ Real**
```
Shuffled Items:     r = 0.28 ± 0.07
Your Model:         r = 0.30 ± 0.08  ⚠️ Item associations don't matter
```
→ Model may be exploiting dataset biases, not learning item-specific features

---

## 🚀 Migration Guide

### Step 1: Install Improved Version

```bash
cd SkillExtract/analysis
pip install -r requirements.txt
```

### Step 2: Adapt Your Data

```python
# Old code (incomplete)
# X_bags_random = [...]
# results_random = Parallel(...)(...)

# New code (complete)
from example_config import load_data, AnalysisConfig
from fig2_baseline_analysis_improved import run_complete_baseline_analysis

dfdata_use, X_bags_list, use_items = load_data()  # Replace with your data
config = AnalysisConfig()

results = run_complete_baseline_analysis(dfdata_use, X_bags_list, use_items, config)
```

### Step 3: Update Analysis

```python
# Old: Single random trial
# correls_random = [...]

# New: Multiple trials with variance
correls_random = results['random']['correlations']
cross_trial_std = results['random']['cross_trial_std']

print(f"Random baseline: {np.mean(correls_random):.4f}")
print(f"Cross-trial SD: {cross_trial_std:.4f}")  # NEW - report this!
```

### Step 4: Update Figures

```python
from fig2_baseline_analysis_improved import plot_fig2c, plot_fig2d

# Automatically includes all baselines
plot_fig2c(correls_main, "MyModel", "Fig2c.png")
plot_fig2d(correls_main, results, "MyModel", "Fig2d.png")
```

---

## 📞 Support

If you encounter issues:

1. Check `README.md` for usage examples
2. Run `test_analysis.py` to verify installation
3. Review code comments in `analysis_utils.py`
4. Try with dummy data first (included in examples)

---

**Version**: 2.0 (Improved)
**Date**: 2026-01-01
**Author**: Improved Baseline Analysis
