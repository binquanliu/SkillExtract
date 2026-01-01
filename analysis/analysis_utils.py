"""
Utility functions for baseline analysis
Provides helper functions for correlation calculation, model fitting, and statistical tests
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp, ttest_rel
from himalaya.ridge import GroupRidgeCV
from himalaya.kernel_ridge import MultipleKernelRidgeCV


def pearsonr_safe(a, b):
    """
    Compute Pearson correlation coefficient with safety checks.

    Parameters
    ----------
    a, b : array-like
        Input arrays

    Returns
    -------
    float
        Pearson correlation coefficient, or 0.0 if computation fails
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Remove NaN and Inf values
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    if a.size == 0 or b.size == 0:
        return 0.0

    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(a, b)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0


def standardize_y(y):
    """
    Standardize target variable with safety checks.

    Parameters
    ----------
    y : array-like
        Target variable

    Returns
    -------
    y_standardized : np.ndarray
        Standardized target variable
    mu : float
        Mean of original y
    sd : float
        Standard deviation of original y
    """
    y = np.asarray(y, dtype=float)
    mu, sd = np.nanmean(y), np.nanstd(y)

    if sd < 1e-8:
        sd = 1.0

    y_standardized = (y - mu) / sd
    y_standardized[~np.isfinite(y_standardized)] = 0.0

    return y_standardized, mu, sd


def check_data_normalization(X_bags_list, n_reasons=None, threshold=0.01):
    """
    Check if data is normalized by examining vector norms.

    Parameters
    ----------
    X_bags_list : list of np.ndarray
        List of feature bags
    n_reasons : int, optional
        Number of reasons to check (default: all)
    threshold : float
        Relative tolerance for norm checking

    Returns
    -------
    is_normalized : bool
        True if data appears to be L2-normalized
    mean_norm : float
        Average L2 norm of vectors
    """
    if len(X_bags_list) == 0:
        return False, 0.0

    sample_bag = X_bags_list[0]
    if n_reasons is None:
        n_reasons = sample_bag.shape[0]

    norms = [np.linalg.norm(sample_bag[i]) for i in range(min(n_reasons, sample_bag.shape[0]))]
    mean_norm = np.mean(norms)
    is_normalized = np.allclose(norms, 1.0, rtol=threshold)

    return is_normalized, mean_norm


def generate_random_bags(n_items, n_reasons, n_dims, normalize=False, seed=None):
    """
    Generate random feature bags.

    Parameters
    ----------
    n_items : int
        Number of items
    n_reasons : int
        Number of reasons per item
    n_dims : int
        Dimensionality of each vector
    normalize : bool
        Whether to L2-normalize each vector
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    X_bags_random : list of np.ndarray
        List of random feature bags
    """
    if seed is not None:
        np.random.seed(seed)

    X_bags_random = []
    for _ in range(n_items):
        random_matrix = np.random.randn(n_reasons, n_dims)

        if normalize:
            for i in range(n_reasons):
                norm = np.linalg.norm(random_matrix[i])
                if norm > 1e-8:
                    random_matrix[i] = random_matrix[i] / norm

        X_bags_random.append(random_matrix)

    return X_bags_random


def generate_matched_random_bags(X_bags_list, seed=None):
    """
    Generate random bags that match the statistical properties of real data.

    Parameters
    ----------
    X_bags_list : list of np.ndarray
        Real feature bags to match
    seed : int, optional
        Random seed

    Returns
    -------
    X_bags_random : list of np.ndarray
        Random bags with matched statistics
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute statistics of real data
    all_vectors = np.vstack([bag.reshape(-1, bag.shape[-1]) for bag in X_bags_list])
    real_mean = np.mean(all_vectors)
    real_std = np.std(all_vectors)

    # Check if data is normalized
    is_normalized, avg_norm = check_data_normalization(X_bags_list)

    X_bags_random = []
    for bag in X_bags_list:
        n_reasons, n_dims = bag.shape

        # Generate with matched mean and std
        random_matrix = np.random.randn(n_reasons, n_dims) * real_std + real_mean

        # Match normalization if applicable
        if is_normalized:
            for i in range(n_reasons):
                norm = np.linalg.norm(random_matrix[i])
                if norm > 1e-8:
                    random_matrix[i] = random_matrix[i] * (avg_norm / norm)

        X_bags_random.append(random_matrix)

    return X_bags_random


def fit_group_ridge_cv(X, y, groups, alphas, cv_folds, return_predictions=False):
    """
    Fit Group Ridge regression with cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target variable
    groups : np.ndarray
        Group assignments
    alphas : list
        Regularization parameters to try
    cv_folds : int
        Number of CV folds
    return_predictions : bool
        Whether to return out-of-sample predictions

    Returns
    -------
    results : dict
        Dictionary containing correlation, MSE, best alpha, and optionally predictions
    """
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Standardize target
        y_std, y_mu, y_sd = standardize_y(y)
        y_2d = y_std.reshape(-1, 1)

        # Fit model to find best alpha
        model = GroupRidgeCV(
            groups=groups,
            alphas=alphas,
            cv=cv_folds,
            solver="conjugate_gradient",
            solver_params={"max_iter": 10000, "tol": 1e-6}
        )
        model.fit(X_scaled, y_2d)
        best_alpha = model.best_alphas_[0]

        # Get out-of-sample predictions
        y_pred_all = np.zeros_like(y_std)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train = y_std[train_idx].reshape(-1, 1)

            # Refit with best alpha
            model_fold = GroupRidgeCV(
                groups=groups,
                alphas=[best_alpha],
                cv=2,
                solver="conjugate_gradient",
                solver_params={"max_iter": 10000, "tol": 1e-6}
            )
            model_fold.fit(X_train, y_train)
            y_pred_all[test_idx] = model_fold.predict(X_test).ravel()

        # Compute metrics
        corr = pearsonr_safe(y_std, y_pred_all)
        mse = mean_squared_error(y_std, y_pred_all)

        results = {
            'correlation': corr,
            'mse': mse,
            'best_alpha': best_alpha
        }

        if return_predictions:
            results['predictions'] = y_pred_all
            results['y_true'] = y_std

        return results

    except Exception as e:
        print(f"[ERROR] fit_group_ridge_cv: {e}")
        return {
            'correlation': 0.0,
            'mse': 1.0,
            'best_alpha': alphas[0] if alphas else 1.0
        }


def fit_kernel_ridge_cv(X_bags, y, alphas, cv_folds, return_predictions=False):
    """
    Fit Multiple Kernel Ridge regression with cross-validation.

    Parameters
    ----------
    X_bags : list of np.ndarray
        List of feature bags (kernels)
    y : np.ndarray
        Target variable
    alphas : list
        Regularization parameters
    cv_folds : int
        Number of CV folds
    return_predictions : bool
        Whether to return predictions

    Returns
    -------
    results : dict
        Dictionary containing correlation, MSE, and optionally predictions
    """
    try:
        # Standardize target
        y_std, y_mu, y_sd = standardize_y(y)
        y_2d = y_std.reshape(-1, 1)

        # Prepare kernels
        kernels = []
        for bag in X_bags:
            K = bag @ bag.T
            kernels.append(K)

        # Fit model
        model = MultipleKernelRidgeCV(
            kernels="precomputed",
            alphas=alphas,
            cv=cv_folds,
            solver="conjugate_gradient",
            solver_params={"max_iter": 10000, "tol": 1e-6}
        )
        model.fit(kernels, y_2d)

        # Get predictions (simplified - would need proper LOOCV for kernels)
        y_pred_all = model.predict(kernels).ravel()

        corr = pearsonr_safe(y_std, y_pred_all)
        mse = mean_squared_error(y_std, y_pred_all)

        results = {
            'correlation': corr,
            'mse': mse
        }

        if return_predictions:
            results['predictions'] = y_pred_all
            results['y_true'] = y_std

        return results

    except Exception as e:
        print(f"[ERROR] fit_kernel_ridge_cv: {e}")
        return {
            'correlation': 0.0,
            'mse': 1.0
        }


def run_statistical_tests(correls_main, baseline_dict, model_name="Main Model"):
    """
    Run comprehensive statistical tests comparing main model to baselines.

    Parameters
    ----------
    correls_main : array-like
        Correlations from main model
    baseline_dict : dict
        Dictionary mapping baseline names to correlation arrays
    model_name : str
        Name of the main model

    Returns
    -------
    results : dict
        Dictionary containing all test results
    """
    results = {}

    # One-sample t-tests against 0
    print("\n" + "="*70)
    print("ONE-SAMPLE T-TESTS (vs 0)")
    print("="*70)

    one_sample_results = {}

    # Test main model
    t_stat, p_val = ttest_1samp(correls_main, 0)
    print(f"{model_name:25s}: t={t_stat:7.3f}, p={p_val:.4e}, n={len(correls_main)}")
    one_sample_results[model_name] = {'t': t_stat, 'p': p_val, 'n': len(correls_main)}

    # Test each baseline
    for name, correls in baseline_dict.items():
        t_stat, p_val = ttest_1samp(correls, 0)
        print(f"{name:25s}: t={t_stat:7.3f}, p={p_val:.4e}, n={len(correls)}")
        one_sample_results[name] = {'t': t_stat, 'p': p_val, 'n': len(correls)}

    results['one_sample'] = one_sample_results

    # Paired t-tests: Main vs Baselines
    print("\n" + "="*70)
    print(f"PAIRED T-TESTS ({model_name} vs Baselines)")
    print("="*70)

    paired_results = {}

    for name, baseline_correls in baseline_dict.items():
        t_stat, p_val = ttest_rel(correls_main, baseline_correls)
        mean_diff = np.nanmean(np.array(correls_main) - np.array(baseline_correls))
        print(f"vs {name:20s}: t={t_stat:7.3f}, p={p_val:.4e}, Δ={mean_diff:+.4f}")
        paired_results[name] = {
            't': t_stat,
            'p': p_val,
            'mean_diff': mean_diff
        }

    results['paired'] = paired_results

    return results
