"""
Improved Baseline Analysis for Fig2c & Fig2d

This script provides a rigorous baseline comparison with:
1. True random vector baselines (multiple trials)
2. Matched random baselines (same statistical properties)
3. Permutation tests
4. Intercept-only baseline
5. Shuffled item-reason associations
6. Domain-label baseline

Author: Improved version
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp, ttest_rel
import warnings
warnings.filterwarnings('ignore')

from himalaya.ridge import GroupRidgeCV
from himalaya.kernel_ridge import MultipleKernelRidgeCV

# Import utility functions
from analysis_utils import (
    pearsonr_safe,
    standardize_y,
    check_data_normalization,
    generate_random_bags,
    generate_matched_random_bags,
    fit_group_ridge_cv,
    run_statistical_tests
)


# ============================================================================
# CONFIGURATION - UPDATE THESE VARIABLES BASED ON YOUR DATA
# ============================================================================

class Config:
    """Configuration parameters for the analysis"""

    # Data parameters (MUST BE SET)
    # dfdata_use = pd.DataFrame(...)  # Your rating data (participants x items)
    # X_bags_list = [...]  # List of feature bags (one per item)
    # use_items = [...]  # List of item identifiers

    # Model parameters
    N_CV_FOLDS = 5
    ALPHAS = np.logspace(-2, 10, 20)
    USE_KERNEL_METHOD = False  # True for MultipleKernelRidgeCV, False for GroupRidgeCV

    # Parallel processing
    N_JOBS = -1  # -1 uses all cores

    # Random baseline parameters
    N_RANDOM_TRIALS = 10  # Number of random trials (increase for more stable estimates)
    N_PERMUTATION_TESTS = 100  # Number of permutations per participant

    # Domain clustering
    N_DOMAINS = 6  # Number of domains for domain-label baseline

    # Output
    METHOD_NAME = "Himalaya"  # Name for your method
    OUTPUT_DIR = "./results"


# ============================================================================
# Helper Function: Process Single Participant
# ============================================================================

def process_participant_corrected(p_idx, row, X_bags, alphas, cv_folds, use_kernel=False):
    """
    Process a single participant with Himalaya models.

    Parameters
    ----------
    p_idx : int
        Participant index
    row : pd.Series
        Participant's ratings for all items
    X_bags : list of np.ndarray
        Feature bags for each item
    alphas : array-like
        Regularization parameters
    cv_folds : int
        Number of cross-validation folds
    use_kernel : bool
        Whether to use kernel ridge (True) or group ridge (False)

    Returns
    -------
    list : [p_idx, correlation, mse]
    """
    y = row.to_numpy(dtype=float)
    y_std, mu, sd = standardize_y(y)

    try:
        if use_kernel:
            # Kernel Ridge approach
            results = fit_kernel_ridge_cv(X_bags, y_std, alphas, cv_folds, return_predictions=False)
            corr = results['correlation']
            mse = results['mse']

        else:
            # Group Ridge approach
            # Concatenate all bags into feature matrix
            n_reasons = X_bags[0].shape[0]
            n_dims = X_bags[0].shape[1]

            X_concat = np.array([bag.flatten() for bag in X_bags])
            groups = np.repeat(np.arange(n_reasons), n_dims)

            results = fit_group_ridge_cv(X_concat, y_std, groups, alphas, cv_folds)
            corr = results['correlation']
            mse = results['mse']

        return [p_idx, corr, mse]

    except Exception as e:
        print(f"[ERROR] Participant {p_idx}: {e}")
        return [p_idx, 0.0, 1.0]


# ============================================================================
# BASELINE 1: Intercept Only (Mean Prediction)
# ============================================================================

def baseline_intercept_only(p_idx, row, cv_folds):
    """
    Baseline: predict the mean rating from training set.

    This is the simplest possible baseline - just predict the average.
    """
    y = row.to_numpy(dtype=float)
    y_std, mu, sd = standardize_y(y)

    y_pred_all = np.zeros_like(y_std)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(y_std):
        y_train = y_std[train_idx]
        train_mean = np.mean(y_train)
        y_pred_all[test_idx] = train_mean

    corr = pearsonr_safe(y_std, y_pred_all)
    mse = mean_squared_error(y_std, y_pred_all)

    return [p_idx, corr, mse]


# ============================================================================
# BASELINE 2: True Random Vector Model (Multiple Trials)
# ============================================================================

def run_random_baseline(dfdata_use, n_items, n_reasons, n_dims, alphas, cv_folds,
                        use_kernel, n_jobs, n_trials=10, normalize=False):
    """
    Run random vector baseline with multiple trials.

    Parameters
    ----------
    n_trials : int
        Number of random trials to average over
    normalize : bool
        Whether to normalize random vectors (should match real data)

    Returns
    -------
    correls_mean : list
        Average correlations across trials
    correls_std : float
        Standard deviation across trials (measure of random variability)
    all_trial_correls : list of lists
        Correlations from each trial
    """
    print(f"\n[BASELINE 2] Running True Random Vector Model ({n_trials} trials)...")

    # Check if we should normalize
    if normalize is None:
        # Auto-detect (will be done in first trial)
        normalize = False

    all_trial_correls = []
    all_trial_mses = []

    for trial in range(n_trials):
        # Use different seed for each trial
        seed = 100 + trial

        # Generate random bags
        X_bags_random = generate_random_bags(
            n_items=n_items,
            n_reasons=n_reasons,
            n_dims=n_dims,
            normalize=normalize,
            seed=seed
        )

        print(f"  Trial {trial+1}/{n_trials} (seed={seed})...")

        # Run model with random vectors
        results_random = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_participant_corrected)(
                p_idx, row, X_bags_random, alphas, cv_folds, use_kernel
            )
            for p_idx, (_, row) in enumerate(dfdata_use.iterrows())
        )

        trial_correls = [r[1] for r in results_random]
        trial_mses = [r[2] for r in results_random]

        all_trial_correls.append(trial_correls)
        all_trial_mses.append(trial_mses)

        print(f"    Mean Corr: {np.nanmean(trial_correls):.4f} ± {np.nanstd(trial_correls):.4f}")

    # Average across trials (for each participant)
    correls_mean = np.mean(all_trial_correls, axis=0).tolist()
    mses_mean = np.mean(all_trial_mses, axis=0).tolist()

    # Measure cross-trial variability
    correls_cross_trial_std = np.std(all_trial_correls, axis=0).mean()

    print(f"\n  [Summary]")
    print(f"    Mean Correlation: {np.nanmean(correls_mean):.4f} ± {np.nanstd(correls_mean):.4f}")
    print(f"    Cross-trial variability: {correls_cross_trial_std:.4f}")

    return correls_mean, correls_cross_trial_std, all_trial_correls


# ============================================================================
# BASELINE 3: Matched Random Baseline
# ============================================================================

def run_matched_random_baseline(dfdata_use, X_bags_list, alphas, cv_folds,
                                use_kernel, n_jobs, n_trials=10):
    """
    Random baseline that matches statistical properties of real data.

    This controls for mean, std, and normalization of the real embeddings.
    """
    print(f"\n[BASELINE 3] Running Matched Random Vector Model ({n_trials} trials)...")

    all_trial_correls = []

    for trial in range(n_trials):
        seed = 200 + trial

        # Generate matched random bags
        X_bags_matched = generate_matched_random_bags(X_bags_list, seed=seed)

        print(f"  Trial {trial+1}/{n_trials} (seed={seed})...")

        results_matched = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_participant_corrected)(
                p_idx, row, X_bags_matched, alphas, cv_folds, use_kernel
            )
            for p_idx, (_, row) in enumerate(dfdata_use.iterrows())
        )

        trial_correls = [r[1] for r in results_matched]
        all_trial_correls.append(trial_correls)

        print(f"    Mean Corr: {np.nanmean(trial_correls):.4f}")

    correls_mean = np.mean(all_trial_correls, axis=0).tolist()
    correls_cross_trial_std = np.std(all_trial_correls, axis=0).mean()

    print(f"\n  [Summary]")
    print(f"    Mean Correlation: {np.nanmean(correls_mean):.4f} ± {np.nanstd(correls_mean):.4f}")
    print(f"    Cross-trial variability: {correls_cross_trial_std:.4f}")

    return correls_mean, correls_cross_trial_std, all_trial_correls


# ============================================================================
# BASELINE 4: Shuffled Item-Reason Association
# ============================================================================

def run_shuffled_baseline(dfdata_use, X_bags_list, alphas, cv_folds,
                         use_kernel, n_jobs, n_trials=5):
    """
    Shuffle the correspondence between items and their feature bags.

    This tests whether the specific item-reason associations matter.
    """
    print(f"\n[BASELINE 4] Running Shuffled Item-Reason Model ({n_trials} trials)...")

    all_trial_correls = []

    for trial in range(n_trials):
        seed = 300 + trial
        np.random.seed(seed)

        # Shuffle item-bag correspondence
        shuffled_indices = np.random.permutation(len(X_bags_list))
        X_bags_shuffled = [X_bags_list[i] for i in shuffled_indices]

        print(f"  Trial {trial+1}/{n_trials} (seed={seed})...")

        results_shuffled = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_participant_corrected)(
                p_idx, row, X_bags_shuffled, alphas, cv_folds, use_kernel
            )
            for p_idx, (_, row) in enumerate(dfdata_use.iterrows())
        )

        trial_correls = [r[1] for r in results_shuffled]
        all_trial_correls.append(trial_correls)

        print(f"    Mean Corr: {np.nanmean(trial_correls):.4f}")

    correls_mean = np.mean(all_trial_correls, axis=0).tolist()

    print(f"\n  [Summary]")
    print(f"    Mean Correlation: {np.nanmean(correls_mean):.4f} ± {np.nanstd(correls_mean):.4f}")

    return correls_mean, all_trial_correls


# ============================================================================
# BASELINE 5: Domain-Label Model
# ============================================================================

def run_domain_baseline(dfdata_use, X_bags_list, n_reasons, alphas, cv_folds, n_jobs, n_domains=6):
    """
    Cluster items into domains and use one-hot domain labels as features.

    This tests whether coarse domain-level information is sufficient.
    """
    print(f"\n[BASELINE 5] Running Domain-Label Model ({n_domains} domains)...")

    # Cluster items using K-means
    np.random.seed(42)
    mean_embeddings = np.array([bag.mean(axis=0) for bag in X_bags_list])
    kmeans = KMeans(n_clusters=n_domains, random_state=42, n_init=10)
    item_domains = kmeans.fit_predict(mean_embeddings)

    print(f"  Domain distribution: {np.bincount(item_domains)}")

    # Create domain one-hot features
    X_bags_domain = []
    for item_idx in range(len(X_bags_list)):
        domain_matrix = np.zeros((n_reasons, n_domains))
        domain_idx = item_domains[item_idx]
        domain_matrix[:, domain_idx] = 1.0
        X_bags_domain.append(domain_matrix)

    # Fit models
    results_domain = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(process_participant_domain)(
            p_idx, row, X_bags_domain, n_reasons, n_domains, alphas, cv_folds
        )
        for p_idx, (_, row) in enumerate(dfdata_use.iterrows())
    )

    correls_domain = [r[1] for r in results_domain]
    mses_domain = [r[2] for r in results_domain]

    print(f"  Mean Corr: {np.nanmean(correls_domain):.4f} ± {np.nanstd(correls_domain):.4f}")

    return correls_domain, mses_domain


def process_participant_domain(p_idx, row, X_bags_domain, n_reasons, n_domains, alphas, cv_folds):
    """Helper function for domain baseline"""
    y = row.to_numpy(dtype=float)
    y_std, mu, sd = standardize_y(y)

    X_concat = np.array([bag.flatten() for bag in X_bags_domain])
    groups = np.repeat(np.arange(n_reasons), n_domains)

    try:
        results = fit_group_ridge_cv(X_concat, y_std, groups, alphas, cv_folds)
        return [p_idx, results['correlation'], results['mse']]
    except Exception as e:
        print(f"[ERROR] Participant {p_idx}: {e}")
        return [p_idx, 0.0, 1.0]


# ============================================================================
# BASELINE 6: Permutation Test
# ============================================================================

def run_permutation_baseline(dfdata_use, X_bags_list, alphas, cv_folds,
                            use_kernel, n_jobs, n_perms_per_participant=100):
    """
    Permutation test: shuffle y values to create null distribution.

    This is the most stringent baseline - it preserves the feature structure
    but breaks the relationship with the target variable.
    """
    print(f"\n[BASELINE 6] Running Permutation Test ({n_perms_per_participant} perms/participant)...")
    print("  WARNING: This may take a long time!")

    def permutation_single_participant(p_idx, row):
        y = row.to_numpy(dtype=float)
        perm_corrs = []

        for perm in range(n_perms_per_participant):
            # Shuffle y
            np.random.seed(1000 + p_idx * 1000 + perm)
            y_shuffled = np.random.permutation(y)

            # Create shuffled row
            row_shuffled = pd.Series(y_shuffled, index=row.index)

            # Run model
            result = process_participant_corrected(
                p_idx, row_shuffled, X_bags_list, alphas, cv_folds, use_kernel
            )
            perm_corrs.append(result[1])

        # Return mean and std of permutation distribution
        return [p_idx, np.mean(perm_corrs), np.std(perm_corrs)]

    results_perm = Parallel(n_jobs=max(1, n_jobs // 2), verbose=2)(
        delayed(permutation_single_participant)(p_idx, row)
        for p_idx, (_, row) in enumerate(dfdata_use.iterrows())
    )

    correls_perm = [r[1] for r in results_perm]
    stds_perm = [r[2] for r in results_perm]

    print(f"  Mean Corr: {np.nanmean(correls_perm):.4f} ± {np.nanstd(correls_perm):.4f}")
    print(f"  Mean within-participant std: {np.mean(stds_perm):.4f}")

    return correls_perm, stds_perm


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_complete_baseline_analysis(dfdata_use, X_bags_list, use_items, config):
    """
    Run complete baseline analysis with all baselines.

    Parameters
    ----------
    dfdata_use : pd.DataFrame
        Participant ratings (rows=participants, cols=items)
    X_bags_list : list of np.ndarray
        Feature bags for each item
    use_items : list
        Item identifiers
    config : Config
        Configuration object

    Returns
    -------
    results_dict : dict
        Complete results dictionary
    """
    print("\n" + "="*70)
    print("FIG2c & FIG2d - IMPROVED BASELINE ANALYSIS")
    print("="*70)

    # Extract dimensions
    n_participants = len(dfdata_use)
    n_items = len(X_bags_list)
    n_reasons = X_bags_list[0].shape[0]
    n_dims = X_bags_list[0].shape[1]

    print(f"\nData Summary:")
    print(f"  Participants: {n_participants}")
    print(f"  Items: {n_items}")
    print(f"  Reasons per item: {n_reasons}")
    print(f"  Dimensions per reason: {n_dims}")

    # Check normalization
    is_normalized, avg_norm = check_data_normalization(X_bags_list, n_reasons)
    print(f"  Data is L2-normalized: {is_normalized}")
    print(f"  Average vector norm: {avg_norm:.4f}")

    # Run main model first (assumed to be already computed)
    # correls_main = [...]  # Load or compute your main model results

    results = {}

    # ========================================================================
    # Baseline 1: Intercept Only
    # ========================================================================
    print("\n[BASELINE 1] Running Intercept Only Model...")
    results_intercept = Parallel(n_jobs=config.N_JOBS, verbose=2)(
        delayed(baseline_intercept_only)(p_idx, row, config.N_CV_FOLDS)
        for p_idx, (_, row) in enumerate(dfdata_use.iterrows())
    )
    correls_intercept = [r[1] for r in results_intercept]
    mses_intercept = [r[2] for r in results_intercept]
    print(f"  Mean Corr: {np.nanmean(correls_intercept):.4f} ± {np.nanstd(correls_intercept):.4f}")

    results['intercept'] = {
        'correlations': correls_intercept,
        'mses': mses_intercept
    }

    # ========================================================================
    # Baseline 2: True Random Vectors
    # ========================================================================
    correls_random, std_random, trials_random = run_random_baseline(
        dfdata_use, n_items, n_reasons, n_dims,
        config.ALPHAS, config.N_CV_FOLDS, config.USE_KERNEL_METHOD,
        config.N_JOBS, n_trials=config.N_RANDOM_TRIALS,
        normalize=is_normalized
    )

    results['random'] = {
        'correlations': correls_random,
        'cross_trial_std': std_random,
        'all_trials': trials_random
    }

    # ========================================================================
    # Baseline 3: Matched Random Vectors
    # ========================================================================
    correls_matched, std_matched, trials_matched = run_matched_random_baseline(
        dfdata_use, X_bags_list,
        config.ALPHAS, config.N_CV_FOLDS, config.USE_KERNEL_METHOD,
        config.N_JOBS, n_trials=config.N_RANDOM_TRIALS
    )

    results['matched_random'] = {
        'correlations': correls_matched,
        'cross_trial_std': std_matched,
        'all_trials': trials_matched
    }

    # ========================================================================
    # Baseline 4: Shuffled Items
    # ========================================================================
    correls_shuffled, trials_shuffled = run_shuffled_baseline(
        dfdata_use, X_bags_list,
        config.ALPHAS, config.N_CV_FOLDS, config.USE_KERNEL_METHOD,
        config.N_JOBS, n_trials=5
    )

    results['shuffled'] = {
        'correlations': correls_shuffled,
        'all_trials': trials_shuffled
    }

    # ========================================================================
    # Baseline 5: Domain Labels
    # ========================================================================
    correls_domain, mses_domain = run_domain_baseline(
        dfdata_use, X_bags_list, n_reasons,
        config.ALPHAS, config.N_CV_FOLDS, config.N_JOBS,
        n_domains=config.N_DOMAINS
    )

    results['domain'] = {
        'correlations': correls_domain,
        'mses': mses_domain
    }

    # ========================================================================
    # Baseline 6: Permutation Test (Optional - very slow)
    # ========================================================================
    # Uncomment if you want to run permutation tests
    # correls_perm, stds_perm = run_permutation_baseline(
    #     dfdata_use, X_bags_list,
    #     config.ALPHAS, config.N_CV_FOLDS, config.USE_KERNEL_METHOD,
    #     config.N_JOBS, n_perms_per_participant=100
    # )
    # results['permutation'] = {
    #     'correlations': correls_perm,
    #     'stds': stds_perm
    # }

    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_fig2c(correls_main, method_name, output_path="Fig2c.png"):
    """Plot Fig2c: Correlation distribution for main model"""
    from scipy.stats import ttest_1samp

    t_result = ttest_1samp(correls_main, 0)
    print(f"\nOne-sample t-test against 0 ({method_name}):")
    print(f"  t={t_result.statistic:.6f}, p={t_result.pvalue:.6e}, df={len(correls_main)-1}")

    # Save data
    pd.Series(correls_main, name="corr").to_csv(
        output_path.replace(".png", "_data.csv"), index=False
    )

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(correls_main, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel("LOOCV Correlation", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"C: {method_name} Correlation Distribution", fontsize=14, fontweight='bold')
    plt.axvline(np.nanmean(correls_main), color='red', linestyle='--',
                linewidth=2, label=f'Mean={np.nanmean(correls_main):.3f}')
    plt.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[OK] {output_path} saved")
    plt.close()


def plot_fig2d(correls_main, baseline_results, method_name, output_path="Fig2d.png"):
    """Plot Fig2d: Baseline comparison"""

    # Build comparison dataframe
    baselines = [
        ["True Random", np.nanmean(baseline_results['random']['correlations']),
         np.nanstd(baseline_results['random']['correlations']),
         len(baseline_results['random']['correlations'])],

        ["Matched Random", np.nanmean(baseline_results['matched_random']['correlations']),
         np.nanstd(baseline_results['matched_random']['correlations']),
         len(baseline_results['matched_random']['correlations'])],

        ["Intercept Only", np.nanmean(baseline_results['intercept']['correlations']),
         np.nanstd(baseline_results['intercept']['correlations']),
         len(baseline_results['intercept']['correlations'])],

        ["Shuffled Items", np.nanmean(baseline_results['shuffled']['correlations']),
         np.nanstd(baseline_results['shuffled']['correlations']),
         len(baseline_results['shuffled']['correlations'])],

        ["Domain Labels", np.nanmean(baseline_results['domain']['correlations']),
         np.nanstd(baseline_results['domain']['correlations']),
         len(baseline_results['domain']['correlations'])],

        [f"{method_name}", np.nanmean(correls_main),
         np.nanstd(correls_main), len(correls_main)],
    ]

    dfb = pd.DataFrame(baselines, columns=["Model", "Mean", "SD", "N"])
    dfb["SE"] = dfb["SD"] / (dfb["N"] ** 0.5)

    # Save data
    dfb.to_csv(output_path.replace(".png", "_data.csv"), index=False)

    # Plot
    dfb_indexed = dfb.set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#C73E1D', '#F18F01', '#A23B72', '#6A994E', '#FFB627', '#2E86AB']

    dfb_indexed["Mean"].plot(
        kind="bar",
        yerr=dfb_indexed["SE"],
        capsize=4,
        error_kw=dict(ecolor="black", lw=1),
        ax=ax,
        rot=45,
        color=colors[:len(dfb)],
        alpha=0.8,
        edgecolor='black'
    )

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Average LOOCV Correlation", fontsize=12)
    plt.title("D: Baseline Comparison", fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[OK] {output_path} saved")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("BASELINE COMPARISON SUMMARY")
    print("="*70)
    print(dfb.to_string(index=False))

    return dfb


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage - YOU MUST UPDATE THIS WITH YOUR ACTUAL DATA
    """

    # ========================================================================
    # STEP 1: Load your data
    # ========================================================================
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    # TODO: Replace these with your actual data loading code
    # Example:
    # dfdata_use = pd.read_csv("participant_ratings.csv", index_col=0)
    # X_bags_list = np.load("feature_bags.npy", allow_pickle=True)
    # use_items = list(dfdata_use.columns)

    # For demonstration, create dummy data
    print("\n⚠️  WARNING: Using dummy data for demonstration")
    print("    Please replace with your actual data!\n")

    n_participants = 50
    n_items = 100
    n_reasons = 5
    n_dims = 300

    # Dummy rating data
    dfdata_use = pd.DataFrame(
        np.random.randn(n_participants, n_items),
        columns=[f"item_{i}" for i in range(n_items)]
    )

    # Dummy feature bags
    X_bags_list = [
        np.random.randn(n_reasons, n_dims)
        for _ in range(n_items)
    ]

    use_items = list(dfdata_use.columns)

    # ========================================================================
    # STEP 2: Configure analysis
    # ========================================================================
    config = Config()
    config.N_RANDOM_TRIALS = 3  # Reduced for demo (use 10+ in real analysis)
    config.METHOD_NAME = "Himalaya"

    # ========================================================================
    # STEP 3: Run main model (example - replace with your actual model)
    # ========================================================================
    print("\n" + "="*70)
    print("RUNNING MAIN MODEL")
    print("="*70)

    # TODO: Replace with your actual main model results
    # For demo, generate dummy correlations
    correls_main = np.random.rand(n_participants) * 0.3 + 0.2  # Dummy: 0.2-0.5 range

    print(f"Main model mean correlation: {np.nanmean(correls_main):.4f}")

    # ========================================================================
    # STEP 4: Run baseline analysis
    # ========================================================================
    baseline_results = run_complete_baseline_analysis(
        dfdata_use, X_bags_list, use_items, config
    )

    # ========================================================================
    # STEP 5: Statistical tests
    # ========================================================================
    baseline_dict = {
        'True Random': baseline_results['random']['correlations'],
        'Matched Random': baseline_results['matched_random']['correlations'],
        'Intercept Only': baseline_results['intercept']['correlations'],
        'Shuffled Items': baseline_results['shuffled']['correlations'],
        'Domain Labels': baseline_results['domain']['correlations'],
    }

    stat_results = run_statistical_tests(
        correls_main, baseline_dict, model_name=config.METHOD_NAME
    )

    # ========================================================================
    # STEP 6: Generate figures
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    plot_fig2c(correls_main, config.METHOD_NAME, "Fig2c_improved.png")
    dfb = plot_fig2d(correls_main, baseline_results, config.METHOD_NAME, "Fig2d_improved.png")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - Fig2c_improved.png")
    print("  - Fig2c_improved_data.csv")
    print("  - Fig2d_improved.png")
    print("  - Fig2d_improved_data.csv")
