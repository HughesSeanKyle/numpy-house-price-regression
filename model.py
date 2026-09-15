"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    X_clean = X.copy()
    for j in range(X_clean.shape[1]):
        col = X_clean[:, j]
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            mean_val = np.nanmean(col)
            # If the entire column is NaN, default to 0.0
            if np.isnan(mean_val):
                mean_val = 0.0
            X_clean[nan_mask, j] = mean_val
    return X_clean

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    iqr = q75 - q25
    lower_bounds = q25 - k * iqr
    upper_bounds = q75 + k * iqr
    return lower_bounds, upper_bounds

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, lower, upper)
    pass

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    ratio = numerator / (denominator + eps)
    return ratio
    pass

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    return np.column_stack([X, col])
    pass

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    unique_cats = np.unique(labels)  # np.unique auto-sorts alphabetically
    encoding = np.zeros((len(labels), len(unique_cats)))
    for idx, cat in enumerate(unique_cats):
        encoding[labels == cat, idx] = 1.0
    return encoding
    pass

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    # Avoid zero division if standard deviation is zero
    stds[stds == 0.0] = 1.0
    return means, stds
    pass

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std
    pass

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])
    pass

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    # rng = np.random.default_rng(seed)
    rng = np.random.RandomState(seed)
    return rng.permutation(n_samples)
    pass

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[0:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx
    pass

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return X[indices], y[indices]
    pass

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    return np.linalg.solve(X.T @ X, X.T @ y)
    pass

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta
    pass

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return float(np.mean(np.abs(y_true - y_pred)))
    pass

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pass

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))
    pass

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    residuals = y_true - y_pred
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "median_abs": float(np.median(np.abs(residuals))),
    }
    pass

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X_imputed = impute_nan_with_mean(X)
    lower_b, upper_b = compute_iqr_bounds(X_imputed, iqr_k)
    X_clipped = clip_columns(X_imputed, lower_b, upper_b)
    return X_clipped
    pass

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # 1. Isolate the explicit 1-D numerator and denominator columns for your updated step 004
    num_col = X_num[:, ratio_num_idx]
    den_col = X_num[:, ratio_den_idx]
    
    # 2. Compute the 1-D ratio vector
    ratio_feat = make_ratio_feature(num_col, den_col)
    
    # 3. Append the new column (ensure it handles or reshapes if append_column expects 2D)
    # If append_column expects a 2D array, use ratio_feat.reshape(-1, 1)
    X_final = append_column(X_num, ratio_feat.reshape(-1, 1))
    
    # 4. Conditionally include categorical fields if they are provided by the harness
    if cat_labels is not None and len(cat_labels) > 0:
        one_hot_feat = one_hot_encode(cat_labels)
        X_final = np.hstack([X_final, one_hot_feat])
        
    return X_final

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    shuffled_idx = make_shuffled_indices(len(X), seed)
    tr_idx, val_idx, te_idx = partition_indices(shuffled_idx, train_ratio, val_ratio)
    
    X_train, y_train = subset_xy(X, y, tr_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, te_idx)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    """
    Fits standardisation arrays on the training features only, applies them 
    across all sets, prepends a bias column, and outputs the transformed dictionary 
    alongside the calibration vectors.
    
    Returns a triple: (std_splits, mean, std)
    """
    # 1. Extract the raw arrays from the splits dictionary
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    X_test  = splits["X_test"]
    
    # 2. Fit standardisation statistics on the training features only
    mean, std = fit_standardizer(X_train)
    
    # 3. Apply the transformation to every feature matrix fold
    X_train_scaled = apply_standardizer(X_train, mean, std)
    X_val_scaled   = apply_standardizer(X_val, mean, std)
    X_test_scaled  = apply_standardizer(X_test, mean, std)
    
    # 4. Prepend the intercept bias column to each scaled feature space
    # Create a new dictionary to house the standardized and biased splits
    std_splits = {
        "X_train": add_bias_column(X_train_scaled),
        "y_train": splits["y_train"],
        "X_val":   add_bias_column(X_val_scaled),
        "y_val":   splits["y_val"],
        "X_test":  add_bias_column(X_test_scaled),
        "y_test":  splits["y_test"]
    }
    
    # 5. Return the exact triple format requested by the platform
    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "residual_summary": residual_summary(y_true, y_pred)
    }

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

