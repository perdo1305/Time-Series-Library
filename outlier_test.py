import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from typing import Union, Tuple, Optional
import pandas as pd
import os


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


def process_csv(
    csv_path: str,
    date_column: str = None,
    target_columns: Union[list, str] = None,
    contamination: float = 0.1,
    treatment: str = "replace",
    replacement_method: str = "interpolate",
    output_path: str = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Process a CSV file to detect and clean outliers in time series data.
    
    Args:
        csv_path: Path to the CSV file
        date_column: Name of the datetime column (optional)
        target_columns: List of column names to process (if None, process all numeric columns)
        contamination: The proportion of outliers expected
        treatment: 'remove' or 'replace' outliers
        replacement_method: If replacing, method to use ('mean', 'median', 'interpolate', 'rolling_mean')
        output_path: Path to save the cleaned CSV (if None, will use original filename with '_cleaned' suffix)
        plot: Whether to create and save plots comparing original and cleaned data
    
    Returns:
        DataFrame with cleaned data
    """
    print(f"Loading data from {csv_path}")
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    print(f"Loaded data with shape: {df.shape}")
    
    # Convert date column to datetime if provided
    if date_column is not None and date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            print(f"Converted {date_column} to datetime")
        except Exception as e:
            print(f"Warning: Could not convert {date_column} to datetime: {e}")
            date_column = None
    
    # Determine which columns to process
    if target_columns is None:
        # Process all numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target_columns = numeric_cols
        print(f"No target columns specified, using all numeric columns: {target_columns}")
    elif isinstance(target_columns, str):
        # Convert single column name to list
        target_columns = [target_columns]
    
    # Ensure all target columns exist in the dataframe
    target_columns = [col for col in target_columns if col in df.columns]
    if not target_columns:
        print("Error: No valid target columns found in the data")
        return df
    
    print(f"Processing columns: {target_columns}")
    
    # Create a copy of the original dataframe for cleaning
    cleaned_df = df.copy()
    
    # Process each target column
    for column in target_columns:
        print(f"\nProcessing column: {column}")
        data = df[column].values
        
        # Skip columns with all NaN values
        if np.all(np.isnan(data)):
            print(f"Column {column} contains all NaN values, skipping")
            continue
        
        # Fill NaN values with interpolation for processing
        if np.any(np.isnan(data)):
            print(f"Column {column} contains {np.sum(np.isnan(data))} NaN values, filling with interpolation")
            data = pd.Series(data).interpolate().fillna(method='bfill').fillna(method='ffill').values
        
        # Apply outlier cleaning
        cleaned_data = clean_time_series(
            data=data,
            contamination=contamination,
            treatment=treatment,
            replacement_method=replacement_method,
        )
        
        # Update the cleaned dataframe
        cleaned_df[column] = cleaned_data
        
        # Create plot if requested
        if plot:
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 6))
                
                # Plot original data
                if date_column is not None:
                    plt.plot(df[date_column], df[column], 'b-', alpha=0.5, label='Original')
                    plt.plot(df[date_column], cleaned_df[column], 'r-', label='Cleaned')
                else:
                    plt.plot(df[column].values, 'b-', alpha=0.5, label='Original')
                    plt.plot(cleaned_df[column].values, 'r-', label='Cleaned')
                
                plt.legend()
                plt.title(f'Outlier Cleaning: {column}')
                plt.xlabel('Time' if date_column is None else date_column)
                plt.ylabel(column)
                
                # Create the output directory if it doesn't exist
                plot_dir = os.path.dirname(csv_path) if output_path is None else os.path.dirname(output_path)
                plot_dir = plot_dir if plot_dir else '.'
                os.makedirs(plot_dir, exist_ok=True)
                
                # Generate plot filename
                plot_filename = os.path.join(
                    plot_dir, 
                    f"{os.path.splitext(os.path.basename(csv_path))[0]}_{column}_cleaned.png"
                )
                
                plt.savefig(plot_filename)
                plt.close()
                print(f"Plot saved as {plot_filename}")
            except ImportError:
                print("Matplotlib not available for plotting")
            except Exception as e:
                print(f"Error creating plot: {e}")
    
    # Save the cleaned data if an output path is provided
    if output_path is None and treatment != 'remove':
        # Generate default output path
        dirname = os.path.dirname(csv_path)
        basename = os.path.basename(csv_path)
        filename, ext = os.path.splitext(basename)
        output_path = os.path.join(dirname, f"{filename}_cleaned{ext}")
    
    if output_path is not None and treatment != 'remove':
        try:
            cleaned_df.to_csv(output_path, index=False)
            print(f"\nCleaned data saved to {output_path}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")
    
    return cleaned_df


if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process CSV files to detect and clean outliers in time series data.')
    
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--date-column', type=str, default=None, help='Name of the datetime column (optional)')
    parser.add_argument('--target-columns', type=str, nargs='+', default=None, 
                        help='List of column names to process (if not provided, all numeric columns will be processed)')
    parser.add_argument('--contamination', type=float, default=0.1, 
                        help='The proportion of outliers expected (default: 0.1)')
    parser.add_argument('--treatment', type=str, default='replace', choices=['remove', 'replace'],
                        help='Whether to remove or replace outliers (default: replace)')
    parser.add_argument('--replacement-method', type=str, default='interpolate', 
                        choices=['mean', 'median', 'interpolate', 'rolling_mean'],
                        help='Method to replace outliers (default: interpolate)')
    parser.add_argument('--output-path', type=str, default=None, 
                        help='Path to save the cleaned CSV (if not provided, will use original filename with "_cleaned" suffix)')
    parser.add_argument('--no-plot', action='store_true', 
                        help='Disable plotting (plots are enabled by default)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the CSV file
    process_csv(
        csv_path=args.csv_path,
        date_column=args.date_column,
        target_columns=args.target_columns,
        contamination=args.contamination,
        treatment=args.treatment,
        replacement_method=args.replacement_method,
        output_path=args.output_path,
        plot=not args.no_plot,
    )
    
    # If no arguments were provided, show example with sample data
    if len(vars(parser.parse_args([args.csv_path]))) == 1:
        print("\n\nExample usage with sample data:")
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
