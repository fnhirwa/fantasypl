"""Time-series transformations for FPL data."""

import pandas as pd
import numpy as np


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = [3, 5, 10],
    agg_funcs: list[str] = ['mean', 'std'],
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Add rolling window features to dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time-series data
    columns : list[str]
        Columns to compute rolling features for
    windows : list[int]
        Window sizes for rolling computation
    agg_funcs : list[str]
        Aggregation functions ('mean', 'std', 'min', 'max', 'sum')
    min_periods : int
        Minimum observations in window
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling features
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for window in windows:
            for func in agg_funcs:
                feature_name = f"{col}_rolling_{window}_{func}"
                df[feature_name] = df[col].rolling(
                    window=window,
                    min_periods=min_periods
                ).agg(func)
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] = [1, 2, 3, 7]
) -> pd.DataFrame:
    """
    Add lagged features to dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list[str]
        Columns to create lags for
    lags : list[int]
        Lag periods
        
    Returns
    -------
    pd.DataFrame
        DataFrame with lagged features
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for lag in lags:
            feature_name = f"{col}_lag_{lag}"
            df[feature_name] = df[col].shift(lag)
    
    return df


def add_ewma_features(
    df: pd.DataFrame,
    columns: list[str],
    alphas: list[float] = [0.3, 0.5, 0.7]
) -> pd.DataFrame:
    """
    Add exponentially weighted moving average features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list[str]
        Columns to compute EWMA for
    alphas : list[float]
        Smoothing factors (0 < alpha < 1)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with EWMA features
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for alpha in alphas:
            feature_name = f"{col}_ewma_{int(alpha*100)}"
            df[feature_name] = df[col].ewm(alpha=alpha, adjust=False).mean()
    
    return df


def add_trend_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = [5, 10]
) -> pd.DataFrame:
    """
    Add trend features (slope) using linear regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list[str]
        Columns to compute trends for
    windows : list[int]
        Window sizes for trend calculation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with trend features
    """
    df = df.copy()
    
    def calculate_slope(series):
        """Calculate slope of linear fit."""
        if len(series) < 2 or series.isna().all():
            return np.nan
        x = np.arange(len(series))
        y = series.values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return slope
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for window in windows:
            feature_name = f"{col}_trend_{window}"
            df[feature_name] = df[col].rolling(
                window=window,
                min_periods=2
            ).apply(calculate_slope, raw=False)
    
    return df


def add_diff_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: list[int] = [1, 2]
) -> pd.DataFrame:
    """
    Add difference features (current - previous).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list[str]
        Columns to compute differences for
    periods : list[int]
        Difference periods
        
    Returns
    -------
    pd.DataFrame
        DataFrame with difference features
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for period in periods:
            feature_name = f"{col}_diff_{period}"
            df[feature_name] = df[col].diff(periods=period)
    
    return df


def add_consistency_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 5
) -> pd.DataFrame:
    """
    Add consistency measures (coefficient of variation).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list[str]
        Columns to measure consistency for
    window : int
        Window size
        
    Returns
    -------
    pd.DataFrame
        DataFrame with consistency features
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        feature_name = f"{col}_consistency_{window}"
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, min_periods=1).std()
        
        # Coefficient of variation (lower = more consistent)
        df[feature_name] = rolling_std / (rolling_mean + 1e-6)
    
    return df
