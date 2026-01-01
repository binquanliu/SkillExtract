# Quick Start Guide

Get started with the improved baseline analysis in 5 minutes!

## ⚡ 1-Minute Setup

```bash
# Navigate to analysis directory
cd SkillExtract/analysis

# Install dependencies
pip install -r requirements.txt

# Run test with dummy data
python fig2_baseline_analysis_improved.py
```

Done! Check the generated figures: `Fig2c_improved.png` and `Fig2d_improved.png`

---

## 🎯 Use With Your Data (5 minutes)

### Option A: Modify the main script directly

Open `fig2_baseline_analysis_improved.py` and find the section marked:

```python
# ========================================================================
# STEP 1: Load your data
# ========================================================================
```

Replace the dummy data with your actual data:

```python
# Load your data
dfdata_use = pd.read_csv("your_participant_ratings.csv", index_col=0)
X_bags_list = np.load("your_embeddings.npy", allow_pickle=True).tolist()
use_items = list(dfdata_use.columns)

# Also load your main model results
correls_main = pd.read_csv("your_model_results.csv")['correlation'].values
```

Then run:
```bash
python fig2_baseline_analysis_improved.py
```

### Option B: Use as a library

Create your own script:

```python
from fig2_baseline_analysis_improved import Config, run_complete_baseline_analysis
import pandas as pd
import numpy as np

# 1. Load your data
dfdata_use = pd.read_csv("ratings.csv", index_col=0)
X_bags_list = np.load("embeddings.npy", allow_pickle=True).tolist()

# 2. Configure
config = Config()
config.N_RANDOM_TRIALS = 10  # Number of random trials
config.METHOD_NAME = "MyModel"

# 3. Run baselines
results = run_complete_baseline_analysis(
    dfdata_use, X_bags_list, list(dfdata_use.columns), config
)

# 4. Generate figures (with your main model results)
from fig2_baseline_analysis_improved import plot_fig2c, plot_fig2d

correls_main = ...  # Your main model correlations
plot_fig2c(correls_main, config.METHOD_NAME, "Fig2c.png")
plot_fig2d(correls_main, results, config.METHOD_NAME, "Fig2d.png")
```

---

## 📊 Understanding the Output

### Console Output

```
[BASELINE 2] Running True Random Vector Model (10 trials)...
  Trial 1/10 (seed=100)...
    Mean Corr: 0.0234 ± 0.1123
  ...
  [Summary]
    Mean Correlation: 0.0089 ± 0.1056
    Cross-trial variability: 0.0234  ← IMPORTANT: Report this in your paper!
```

### Files Generated

- `Fig2c_improved.png` - Histogram of correlations for your model
- `Fig2c_improved_data.csv` - Raw correlation values
- `Fig2d_improved.png` - Bar chart comparing all baselines
- `Fig2d_improved_data.csv` - Summary statistics table

### Key Metrics in Fig2d

```
Model               Mean    SD      SE
True Random         0.01   0.10   0.01   ← Should be near 0
Matched Random      0.02   0.11   0.01   ← Should be near 0
Intercept Only      0.00   0.08   0.01   ← Should be near 0
Shuffled Items      0.05   0.12   0.02   ← Should be low
Domain Labels       0.15   0.14   0.02   ← Moderate
MyModel             0.35   0.18   0.03   ← Should be highest ✓
```

---

## 🔍 Common First-Time Issues

### Issue: Import Error

```python
ModuleNotFoundError: No module named 'himalaya'
```

**Fix:**
```bash
pip install himalaya
# or
pip install -r requirements.txt
```

### Issue: Wrong Data Shape

```python
AssertionError: Wrong bag shape
```

**Fix:** Check your data format:
```python
print(f"Ratings shape: {dfdata_use.shape}")  # Should be (n_participants, n_items)
print(f"Num bags: {len(X_bags_list)}")       # Should equal n_items
print(f"Bag shape: {X_bags_list[0].shape}")  # Should be (n_reasons, n_dims)
```

### Issue: Too Slow

**Fix:** Reduce number of trials for testing:
```python
config.N_RANDOM_TRIALS = 3  # Instead of 10
config.N_JOBS = -1  # Use all CPU cores
```

---

## 📝 Reporting Results (Copy-Paste for Your Paper)

### Methods Section

> We compared our model against six baselines: (1) Intercept-only, predicting the mean rating; (2) True random vectors, averaging over 10 trials with different random seeds to estimate chance performance; (3) Matched random vectors, controlling for statistical properties (mean, SD, normalization) of real embeddings; (4) Shuffled item-feature associations; (5) Domain labels from K-means clustering (k=6); and (6) Cross-validation with leave-one-out (LOOCV) for all models. Statistical significance was assessed using paired t-tests comparing our model to each baseline.

### Results Section

> Our model achieved a mean correlation of r = 0.35 (SD = 0.18), significantly outperforming all baselines (all p < 0.001, paired t-tests). The true random baseline yielded r = 0.01 ± 0.10 (cross-trial SD = 0.02), and the matched random baseline yielded r = 0.02 ± 0.11 (cross-trial SD = 0.02), confirming that our model's performance exceeds chance. The shuffled items baseline (r = 0.05 ± 0.12) and domain labels baseline (r = 0.15 ± 0.14) showed that both item-specific features and fine-grained embeddings contribute to model performance.

### Figure Caption

> **Figure 2c.** Distribution of leave-one-out cross-validation correlations across participants (N = 100). Red dashed line indicates mean correlation (r = 0.35).
>
> **Figure 2d.** Comparison of model performance against baselines. Error bars represent standard error. True Random and Matched Random baselines are averaged over 10 independent trials with different random seeds.

---

## 🎓 Next Steps

1. **Read the full documentation**: See `README.md` for detailed explanations

2. **Understand the improvements**: See `IMPROVEMENTS.md` for what was fixed

3. **Customize for your needs**: Modify `example_config.py`

4. **Add custom baselines**: See examples in `fig2_baseline_analysis_improved.py`

---

## 💡 Pro Tips

1. **Start with 3 random trials** for testing, then increase to 10+ for final analysis
2. **Always report cross-trial variability** for random baselines
3. **Use matched random** as your primary random baseline (more stringent)
4. **Check data normalization** before running analysis
5. **Save intermediate results** in case of crashes

---

Need help? Check:
- `README.md` - Full documentation
- `IMPROVEMENTS.md` - What was fixed
- `test_analysis.py` - Verify installation
- Code comments - Detailed inline documentation

**Happy analyzing! 🚀**
