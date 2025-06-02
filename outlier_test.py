import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from typing import Union, Tuple, Optional


def detect_outliers(
    data: np.ndarray,
    contamination: float = 0.1,
    n_neighbors: int = 20,
    algorithm: str = "auto",
    metric: str = "minkowski",
) -> np.ndarray:
    """
    Detect outliers in the data using Local Outlier Factor.

    Args:
        data: Input data, shape (n_samples, n_features)
        contamination: The proportion of outliers in the data set
        n_neighbors: Number of neighbors to use
        algorithm: Algorithm used to compute the nearest neighbors
        metric: Metric used for the distance computation

    Returns:
        Boolean mask of the same shape as data's first dimension,
        True for outliers and False for inliers
    """
    # Reshape data if it's 1D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Initialize the LOF model
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        algorithm=algorithm,
        metric=metric,
    )

    # Fit the model and predict
    # -1 for outliers and 1 for inliers
    y_pred = lof.fit_predict(data)

    # Convert to boolean mask (True for outliers)
    mask = y_pred == -1

    print(
        f"Detected {np.sum(mask)} outliers out of {len(data)} samples ({np.sum(mask) / len(data) * 100:.2f}%)"
    )

    return mask


def remove_outliers(
    data: np.ndarray,
    contamination: float = 0.1,
    n_neighbors: int = 20,
    algorithm: str = "auto",
    metric: str = "minkowski",
) -> np.ndarray:
    """
    Remove outliers from the data using Local Outlier Factor.

    Args:
        data: Input data, shape (n_samples, n_features)
        contamination: The proportion of outliers in the data set
        n_neighbors: Number of neighbors to use
        algorithm: Algorithm used to compute the nearest neighbors
        metric: Metric used for the distance computation

    Returns:
        Data with outliers removed
    """
    # Get outlier mask
    outlier_mask = detect_outliers(
        data=data,
        contamination=contamination,
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric,
    )

    # Remove outliers
    clean_data = data[~outlier_mask]

    return clean_data


def replace_outliers(
    data: np.ndarray,
    contamination: float = 0.1,
    n_neighbors: int = 20,
    algorithm: str = "auto",
    metric: str = "minkowski",
    replacement_method: str = "mean",
    window_size: int = 5,
) -> np.ndarray:
    """
    Replace outliers in the data using Local Outlier Factor.

    Args:
        data: Input data, shape (n_samples, n_features)
        contamination: The proportion of outliers in the data set
        n_neighbors: Number of neighbors to use
        algorithm: Algorithm used to compute the nearest neighbors
        metric: Metric used for the distance computation
        replacement_method: Method to replace outliers
                          ('mean', 'median', 'interpolate', 'rolling_mean')
        window_size: Size of the window for rolling statistics (if applicable)

    Returns:
        Data with outliers replaced
    """
    # Create a copy of the data
    clean_data = data.copy()

    # Get outlier mask
    outlier_mask = detect_outliers(
        data=data,
        contamination=contamination,
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric,
    )

    # No outliers detected
    if not np.any(outlier_mask):
        return clean_data

    # Replace outliers based on specified method
    if replacement_method == "mean":
        # Replace with mean of non-outliers
        if data.ndim > 1:
            for i in range(data.shape[1]):
                col_mean = np.mean(data[~outlier_mask, i])
                clean_data[outlier_mask, i] = col_mean
        else:
            mean_value = np.mean(data[~outlier_mask])
            clean_data[outlier_mask] = mean_value

    elif replacement_method == "median":
        # Replace with median of non-outliers
        if data.ndim > 1:
            for i in range(data.shape[1]):
                col_median = np.median(data[~outlier_mask, i])
                clean_data[outlier_mask, i] = col_median
        else:
            median_value = np.median(data[~outlier_mask])
            clean_data[outlier_mask] = median_value

    elif replacement_method == "interpolate":
        # Linear interpolation
        indices = np.arange(len(data))
        if data.ndim > 1:
            for i in range(data.shape[1]):
                clean_data[:, i] = np.interp(
                    indices, indices[~outlier_mask], data[~outlier_mask, i]
                )
        else:
            clean_data = np.interp(indices, indices[~outlier_mask], data[~outlier_mask])

    elif replacement_method == "rolling_mean":
        # Replace with rolling mean
        from scipy.ndimage import uniform_filter1d

        if data.ndim > 1:
            for i in range(data.shape[1]):
                # Calculate rolling mean excluding the outliers
                temp = data[:, i].copy()
                # Fill outliers with NaN temporarily
                temp[outlier_mask] = np.nan
                # Use convolution to get rolling mean, ignoring NaNs
                mask = ~np.isnan(temp)
                rolling_mean = np.zeros_like(temp)
                rolling_mean[mask] = uniform_filter1d(temp[mask], size=window_size)
                # Replace outliers with rolling mean
                clean_data[outlier_mask, i] = rolling_mean[outlier_mask]
        else:
            # Calculate rolling mean excluding the outliers
            temp = data.copy()
            # Fill outliers with NaN temporarily
            temp[outlier_mask] = np.nan
            # Use convolution to get rolling mean, ignoring NaNs
            mask = ~np.isnan(temp)
            rolling_mean = np.zeros_like(temp)
            rolling_mean[mask] = uniform_filter1d(temp[mask], size=window_size)
            # Replace outliers with rolling mean
            clean_data[outlier_mask] = rolling_mean[outlier_mask]

    return clean_data


def clean_time_series(
    data: np.ndarray,
    time_axis: int = 0,
    feature_axis: int = 1,
    contamination: float = 0.1,
    n_neighbors: int = 20,
    treatment: str = "replace",
    replacement_method: str = "interpolate",
) -> np.ndarray:
    """
    Clean time series data by detecting and treating outliers.

    Args:
        data: Time series data, shape depends on time_axis and feature_axis
        time_axis: Axis representing time dimension (default 0)
        feature_axis: Axis representing features dimension (default 1)
        contamination: The proportion of outliers expected
        n_neighbors: Number of neighbors to use for LOF
        treatment: 'remove' or 'replace' outliers
        replacement_method: If replacing, method to use ('mean', 'median', 'interpolate', 'rolling_mean')

    Returns:
        Cleaned time series data
    """
    # Ensure data is numpy array
    data = np.asarray(data)

    # For multivariate time series, clean each feature independently
    if data.ndim > 1:
        # Transpose if needed to make feature axis the last axis
        if time_axis == 0 and feature_axis == 1:
            # Shape is already (time, features)
            pass
        elif time_axis == 1 and feature_axis == 0:
            # Shape is (features, time), transpose to (time, features)
            data = data.T
        else:
            raise ValueError(
                "Currently only supports 2D time series with time and feature axes"
            )

        # Clean each feature
        cleaned_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            feature_data = data[:, i]

            if treatment == "remove":
                cleaned_feature = remove_outliers(
                    feature_data,
                    contamination=contamination,
                    n_neighbors=min(n_neighbors, len(feature_data) - 1),
                )
                # Need to handle different lengths after removal
                cleaned_data[: len(cleaned_feature), i] = cleaned_feature
                if len(cleaned_feature) < len(data):
                    # Mark remaining positions as NaN or a special value
                    cleaned_data[len(cleaned_feature) :, i] = np.nan
            else:  # replace
                cleaned_feature = replace_outliers(
                    feature_data,
                    contamination=contamination,
                    n_neighbors=min(n_neighbors, len(feature_data) - 1),
                    replacement_method=replacement_method,
                )
                cleaned_data[:, i] = cleaned_feature

        # Transpose back if needed
        if time_axis == 1 and feature_axis == 0:
            cleaned_data = cleaned_data.T
    else:
        # For univariate time series
        if treatment == "remove":
            cleaned_data = remove_outliers(
                data,
                contamination=contamination,
                n_neighbors=min(n_neighbors, len(data) - 1),
            )
        else:  # replace
            cleaned_data = replace_outliers(
                data,
                contamination=contamination,
                n_neighbors=min(n_neighbors, len(data) - 1),
                replacement_method=replacement_method,
            )

    return cleaned_data


if __name__ == "__main__":
    # Example usage
    # Generate sample time series with outliers
    np.random.seed(42)
    n_samples = 1000
    time = np.linspace(0, 10, n_samples)
    signal = np.sin(time) + 0.1 * np.random.randn(n_samples)

    # Add outliers
    outlier_idx = np.random.choice(n_samples, size=50, replace=False)
    signal[outlier_idx] += np.random.randn(len(outlier_idx)) * 3

    # Clean the time series
    cleaned_signal = clean_time_series(
        signal,
        contamination=0.05,
        treatment="replace",
        replacement_method="interpolate",
    )

    # Plot the results
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, "b-", alpha=0.5, label="Original signal")
        plt.plot(time, cleaned_signal, "r-", label="Cleaned signal")
        plt.legend()
        plt.title("Time Series Outlier Removal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.savefig("outlier_cleaning_example.png")
        plt.close()
        print("Example plot saved as 'outlier_cleaning_example.png'")
    except ImportError:
        print("Matplotlib not available for plotting")
