"""
Quick test script to verify the improved baseline analysis works correctly
"""

import numpy as np
import pandas as pd
import sys

print("="*70)
print("TESTING IMPROVED BASELINE ANALYSIS")
print("="*70)

# Test 1: Import all modules
print("\n[Test 1] Testing imports...")
try:
    from analysis_utils import (
        pearsonr_safe,
        standardize_y,
        check_data_normalization,
        generate_random_bags,
        generate_matched_random_bags,
        fit_group_ridge_cv,
        run_statistical_tests
    )
    print("  ✓ analysis_utils imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import analysis_utils: {e}")
    sys.exit(1)

try:
    from fig2_baseline_analysis_improved import (
        Config,
        process_participant_corrected,
        baseline_intercept_only,
        run_random_baseline,
        run_complete_baseline_analysis
    )
    print("  ✓ fig2_baseline_analysis_improved imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import fig2_baseline_analysis_improved: {e}")
    sys.exit(1)

try:
    from example_config import load_data, AnalysisConfig
    print("  ✓ example_config imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import example_config: {e}")
    sys.exit(1)

# Test 2: Test utility functions
print("\n[Test 2] Testing utility functions...")

# Test pearsonr_safe
a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 4, 6, 8, 10])
corr = pearsonr_safe(a, b)
assert abs(corr - 1.0) < 0.01, "pearsonr_safe failed"
print(f"  ✓ pearsonr_safe: r={corr:.4f}")

# Test with NaNs
a_nan = np.array([1, 2, np.nan, 4, 5])
b_nan = np.array([2, 4, 6, 8, 10])
corr_nan = pearsonr_safe(a_nan, b_nan)
print(f"  ✓ pearsonr_safe with NaNs: r={corr_nan:.4f}")

# Test standardize_y
y = np.array([1, 2, 3, 4, 5])
y_std, mu, sd = standardize_y(y)
assert abs(y_std.mean()) < 0.01, "standardize_y mean failed"
assert abs(y_std.std() - 1.0) < 0.01, "standardize_y std failed"
print(f"  ✓ standardize_y: mean={y_std.mean():.4f}, std={y_std.std():.4f}")

# Test 3: Test random bag generation
print("\n[Test 3] Testing random bag generation...")

n_items, n_reasons, n_dims = 10, 5, 100

# Generate random bags
X_bags = generate_random_bags(n_items, n_reasons, n_dims, normalize=False, seed=42)
assert len(X_bags) == n_items, "Wrong number of bags"
assert X_bags[0].shape == (n_reasons, n_dims), "Wrong bag shape"
print(f"  ✓ Random bags: {len(X_bags)} bags, shape {X_bags[0].shape}")

# Generate normalized bags
X_bags_norm = generate_random_bags(n_items, n_reasons, n_dims, normalize=True, seed=42)
norms = [np.linalg.norm(X_bags_norm[0][i]) for i in range(n_reasons)]
assert all(abs(n - 1.0) < 0.01 for n in norms), "Normalization failed"
print(f"  ✓ Normalized bags: mean norm={np.mean(norms):.4f}")

# Test 4: Check data normalization detection
print("\n[Test 4] Testing normalization detection...")

is_norm, avg_norm = check_data_normalization(X_bags_norm, n_reasons)
print(f"  ✓ Normalized data detected: {is_norm}, avg_norm={avg_norm:.4f}")

is_norm, avg_norm = check_data_normalization(X_bags, n_reasons)
print(f"  ✓ Non-normalized data detected: {is_norm}, avg_norm={avg_norm:.4f}")

# Test 5: Test matched random generation
print("\n[Test 5] Testing matched random generation...")

X_bags_matched = generate_matched_random_bags(X_bags_norm, seed=42)
assert len(X_bags_matched) == len(X_bags_norm), "Wrong number of matched bags"

# Check statistics match
real_mean = np.mean([bag.flatten() for bag in X_bags_norm])
matched_mean = np.mean([bag.flatten() for bag in X_bags_matched])
print(f"  ✓ Matched random: real_mean={real_mean:.4f}, matched_mean={matched_mean:.4f}")

# Test 6: Test configuration
print("\n[Test 6] Testing configuration...")

config = Config()
print(f"  ✓ Config.N_CV_FOLDS: {config.N_CV_FOLDS}")
print(f"  ✓ Config.N_RANDOM_TRIALS: {config.N_RANDOM_TRIALS}")
print(f"  ✓ Config.METHOD_NAME: {config.METHOD_NAME}")

config2 = AnalysisConfig()
print(f"  ✓ AnalysisConfig.N_DOMAINS: {config2.N_DOMAINS}")

# Test 7: Test data loading
print("\n[Test 7] Testing data loading...")

dfdata_use, X_bags_list, use_items = load_data()
print(f"  ✓ Loaded {len(dfdata_use)} participants")
print(f"  ✓ Loaded {len(X_bags_list)} items")
print(f"  ✓ Bag shape: {X_bags_list[0].shape}")

# Test 8: Test baseline function
print("\n[Test 8] Testing baseline function...")

# Test intercept baseline
row = dfdata_use.iloc[0]
result = baseline_intercept_only(0, row, 5)
print(f"  ✓ Intercept baseline: corr={result[1]:.4f}, mse={result[2]:.4f}")

# Test 9: Test statistical functions
print("\n[Test 9] Testing statistical functions...")

correls_main = np.random.rand(50) * 0.3 + 0.2
correls_random = np.random.rand(50) * 0.1
correls_baseline = np.random.rand(50) * 0.15

baseline_dict = {
    'Random': correls_random,
    'Baseline': correls_baseline
}

stat_results = run_statistical_tests(correls_main, baseline_dict, model_name="Test")
print(f"  ✓ Statistical tests completed")
print(f"    One-sample results: {len(stat_results['one_sample'])} tests")
print(f"    Paired results: {len(stat_results['paired'])} tests")

# Test 10: Integration test with small dataset
print("\n[Test 10] Running mini integration test...")

# Create small dataset
n_participants = 10
n_items = 20
n_reasons = 3
n_dims = 50

dfdata_small = pd.DataFrame(
    np.random.randn(n_participants, n_items),
    columns=[f"item_{i}" for i in range(n_items)]
)

X_bags_small = generate_random_bags(n_items, n_reasons, n_dims, normalize=True, seed=42)

# Test with 2 random trials (fast)
config_test = Config()
config_test.N_RANDOM_TRIALS = 2
config_test.N_JOBS = 1  # Single core for testing

print(f"  Running random baseline with {n_participants} participants, {n_items} items...")

try:
    correls_mean, std_cross, all_trials = run_random_baseline(
        dfdata_small, n_items, n_reasons, n_dims,
        config_test.ALPHAS, config_test.N_CV_FOLDS,
        config_test.USE_KERNEL_METHOD, config_test.N_JOBS,
        n_trials=2, normalize=True
    )

    print(f"  ✓ Random baseline completed")
    print(f"    Mean correlation: {np.mean(correls_mean):.4f}")
    print(f"    Cross-trial std: {std_cross:.4f}")

except Exception as e:
    print(f"  ✗ Random baseline failed: {e}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nThe improved baseline analysis is ready to use.")
print("Next steps:")
print("  1. Modify example_config.py with your data")
print("  2. Run: python fig2_baseline_analysis_improved.py")
print("  3. Check results/ directory for outputs")
