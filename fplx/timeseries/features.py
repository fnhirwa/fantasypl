"""Feature engineering pipeline for FPL time-series data."""

import pandas as pd
from typing import Optional
import logging

from fplx.timeseries.transforms import (
    add_rolling_features,
    add_lag_features,
    add_ewma_features,
    add_trend_features,
    add_diff_features,
    add_consistency_features,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for player time-series data.
    
    Parameters
    ----------
    config : Optional[Dict]
        Feature configuration dictionary
    """
    
    DEFAULT_CONFIG = {
        'rolling_windows': [3, 5, 10],
        'lag_periods': [1, 2, 3],
        'ewma_alphas': [0.3, 0.5],
        'trend_windows': [5, 10],
        'key_columns': ['points', 'minutes', 'xG', 'xA', 'bonus'],
    }
    
    def __init__(self, config: Optional[dict] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input player timeseries data
            
        Returns
        -------
        pd.DataFrame
            Transformed data with engineered features
        """
        df = df.copy()
        
        # Identify available columns
        key_cols = [c for c in self.config['key_columns'] if c in df.columns]
        
        if not key_cols:
            logger.warning("No key columns found for feature engineering")
            return df
        
        # Apply transformations
        logger.info("Adding rolling features...")
        df = add_rolling_features(
            df,
            columns=key_cols,
            windows=self.config['rolling_windows'],
            agg_funcs=['mean', 'std']
        )
        
        logger.info("Adding lag features...")
        df = add_lag_features(
            df,
            columns=key_cols,
            lags=self.config['lag_periods']
        )
        
        logger.info("Adding EWMA features...")
        df = add_ewma_features(
            df,
            columns=key_cols,
            alphas=self.config['ewma_alphas']
        )
        
        logger.info("Adding trend features...")
        df = add_trend_features(
            df,
            columns=key_cols,
            windows=self.config['trend_windows']
        )
        
        logger.info("Adding difference features...")
        df = add_diff_features(
            df,
            columns=key_cols,
            periods=[1]
        )
        
        logger.info("Adding consistency features...")
        df = add_consistency_features(
            df,
            columns=['minutes', 'points'],
            window=5
        )
        
        return df
    
    def get_feature_names(self, base_columns: list[str]) -> list[str]:
        """
        Get list of all generated feature names.
        
        Parameters
        ----------
        base_columns : list[str]
            Base column names
            
        Returns
        -------
        list[str]
            Generated feature names
        """
        features = []
        
        for col in base_columns:
            # Rolling features
            for window in self.config['rolling_windows']:
                features.extend([
                    f"{col}_rolling_{window}_mean",
                    f"{col}_rolling_{window}_std",
                ])
            
            # Lag features
            for lag in self.config['lag_periods']:
                features.append(f"{col}_lag_{lag}")
            
            # EWMA features
            for alpha in self.config['ewma_alphas']:
                features.append(f"{col}_ewma_{int(alpha*100)}")
            
            # Trend features
            for window in self.config['trend_windows']:
                features.append(f"{col}_trend_{window}")
            
            # Diff features
            features.append(f"{col}_diff_1")
        
        # Consistency features
        features.extend([
            'minutes_consistency_5',
            'points_consistency_5',
        ])
        
        return features
    
    def create_future_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Create features for future predictions.

        This method extends the historical data by `horizon` periods,
        applies the full feature engineering pipeline, and returns
        the newly created future feature set.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data
        horizon : int
            Number of future gameweeks to predict
            
        Returns
        -------
        pd.DataFrame
            DataFrame with features for future gameweeks
        """
        if df.empty:
            return pd.DataFrame()

        # Create future placeholders by repeating the last known data point
        last_row = df.iloc[-1:].copy()
        
        # Avoid duplicating index if it's a timestamp or gameweek
        is_numeric_index = pd.api.types.is_numeric_dtype(df.index)
        if isinstance(df.index, pd.DatetimeIndex) or is_numeric_index:
            last_index = df.index[-1]
            future_index = pd.RangeIndex(start=last_index + 1, stop=last_index + 1 + horizon)
            last_row.index = [future_index[0]] # Temporarily align for concat
        else:
            future_index = pd.RangeIndex(start=len(df), stop=len(df) + horizon)

        future_rows = pd.concat([last_row] * horizon, ignore_index=True)
        if isinstance(df.index, pd.DatetimeIndex) or is_numeric_index:
             future_rows.index = future_index

        # Combine historical and future data
        combined_df = pd.concat([df, future_rows])
        
        # Run the full feature engineering pipeline on the combined data
        # This ensures that rolling/lag features are calculated correctly
        # based on the historical context.
        engineered_df = self.fit_transform(combined_df)
        
        # Return only the future part
        return engineered_df.tail(horizon)
