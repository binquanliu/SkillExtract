"""
Example Configuration File for Baseline Analysis

Copy this file and modify it to match your experiment setup.
"""

import numpy as np
import pandas as pd


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """
    Load your experimental data.

    Returns
    -------
    dfdata_use : pd.DataFrame
        Participant ratings (rows=participants, cols=items)
    X_bags_list : list of np.ndarray
        Feature bags for each item, each with shape (n_reasons, n_dims)
    use_items : list
        List of item identifiers (should match dfdata_use columns)
    """

    # ========================================================================
    # METHOD 1: Load from files
    # ========================================================================

    # Example: Load participant ratings
    # dfdata_use = pd.read_csv("data/participant_ratings.csv", index_col=0)

    # Example: Load pre-computed embeddings
    # X_bags_list = np.load("data/item_embeddings.npy", allow_pickle=True).tolist()

    # Example: Load item list
    # use_items = list(dfdata_use.columns)


    # ========================================================================
    # METHOD 2: Compute embeddings on the fly
    # ========================================================================

    # Example: Load raw data and compute embeddings
    # import json
    # with open("data/items.json", "r") as f:
    #     items_data = json.load(f)
    #
    # # Compute embeddings for each item
    # from your_embedding_model import get_embeddings
    # X_bags_list = []
    # for item in items_data:
    #     reasons = item['reasons']  # List of reason texts
    #     embeddings = get_embeddings(reasons)  # Shape: (n_reasons, n_dims)
    #     X_bags_list.append(embeddings)


    # ========================================================================
    # METHOD 3: Dummy data for testing
    # ========================================================================

    print("⚠️  Loading DUMMY DATA for testing purposes")

    n_participants = 50
    n_items = 100
    n_reasons = 5
    n_dims = 300

    # Create dummy ratings
    dfdata_use = pd.DataFrame(
        np.random.randn(n_participants, n_items),
        columns=[f"item_{i}" for i in range(n_items)]
    )

    # Create dummy embeddings (normalized)
    X_bags_list = []
    for _ in range(n_items):
        bag = np.random.randn(n_reasons, n_dims)
        # L2 normalize each vector
        for i in range(n_reasons):
            norm = np.linalg.norm(bag[i])
            if norm > 0:
                bag[i] = bag[i] / norm
        X_bags_list.append(bag)

    use_items = list(dfdata_use.columns)

    return dfdata_use, X_bags_list, use_items


# ============================================================================
# MODEL PARAMETERS
# ============================================================================

class AnalysisConfig:
    """Configuration for baseline analysis"""

    # ========================================================================
    # Cross-validation settings
    # ========================================================================
    N_CV_FOLDS = 5  # Number of folds for cross-validation

    # ========================================================================
    # Regularization parameters
    # ========================================================================
    # For ridge regression - test a range of alpha values
    ALPHAS = np.logspace(-2, 10, 20)

    # Alternative: manually specify alphas
    # ALPHAS = [0.01, 0.1, 1.0, 10, 100, 1000]

    # ========================================================================
    # Model selection
    # ========================================================================
    # False = GroupRidgeCV (default, faster)
    # True = MultipleKernelRidgeCV (slower, may be more accurate)
    USE_KERNEL_METHOD = False

    # ========================================================================
    # Parallel processing
    # ========================================================================
    N_JOBS = -1  # -1 = use all CPU cores
                 # 1 = single core (for debugging)
                 # 4 = use 4 cores (or any other number)

    # ========================================================================
    # Random baseline parameters
    # ========================================================================
    # Number of random trials to average over
    # Higher = more stable estimates but slower
    # Recommended: 10-20 for final analysis, 3-5 for testing
    N_RANDOM_TRIALS = 10

    # Number of permutations per participant (for permutation test)
    # Only used if you enable permutation test
    # Recommended: 100-1000
    N_PERMUTATION_TESTS = 100

    # ========================================================================
    # Domain clustering
    # ========================================================================
    # Number of domains for K-means clustering
    # Similar to RIASEC (6 domains) in career research
    N_DOMAINS = 6

    # ========================================================================
    # Output settings
    # ========================================================================
    # Name for your method (used in plots and file names)
    METHOD_NAME = "MyModel"

    # Directory for saving results
    OUTPUT_DIR = "./results"

    # ========================================================================
    # Analysis options
    # ========================================================================
    # Run full permutation test? (very slow, usually not needed)
    RUN_PERMUTATION_TEST = False

    # Save individual trial results?
    SAVE_ALL_TRIALS = True

    # DPI for saved figures
    FIGURE_DPI = 300


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test configuration loading"""

    print("Testing configuration...")

    # Load data
    dfdata_use, X_bags_list, use_items = load_data()

    # Print summary
    print("\nData Summary:")
    print(f"  Participants: {len(dfdata_use)}")
    print(f"  Items: {len(X_bags_list)}")
    print(f"  Reasons per item: {X_bags_list[0].shape[0]}")
    print(f"  Dimensions: {X_bags_list[0].shape[1]}")

    # Create config
    config = AnalysisConfig()

    print("\nConfiguration:")
    print(f"  CV Folds: {config.N_CV_FOLDS}")
    print(f"  Alpha range: {config.ALPHAS[0]:.2e} to {config.ALPHAS[-1]:.2e}")
    print(f"  Random trials: {config.N_RANDOM_TRIALS}")
    print(f"  Method: {config.METHOD_NAME}")

    print("\n✓ Configuration loaded successfully!")
