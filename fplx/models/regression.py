"""ML regression models for FPL prediction."""

import numpy as np
import pandas as pd
import logging

try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from fplx.models.base import BaseModel
from fplx.models.rolling_cv import RollingCV

logger = logging.getLogger(__name__)


class RegressionModel(BaseModel):
    """
    Machine learning regression model for FPL predictions.
    
    Adapted from the MLSP project's regressor patterns.
    
    Parameters
    ----------
    model_type : str
        Type of model: 'ridge', 'xgboost', 'lightgbm'
    initial_train_size : int
        Size of initial training window
    test_size : int
        Forecast horizon
    step : int
        Rolling window step size
    """
    
    def __init__(
        self,
        model_type: str = 'ridge',
        initial_train_size: int = 10,
        test_size: int = 1,
        step: int = 1,
        **model_kwargs
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn, xgboost, or lightgbm not available. Install with: pip install fplx[ml]")
        
        self.model_type = model_type
        self.cv = RollingCV(initial_train_size, test_size, step)
        self.model = self._create_model(model_type, **model_kwargs)
        self.predictions = []
        self.true_values = []
        self.feature_importance = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        """Fit the model."""
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Generate predictions."""
        # Ensure the prediction data has the same columns as the training data
        if self.feature_names_:
            X_pred = X.reindex(columns=self.feature_names_, fill_value=0)
        else:
            X_pred = X
        return self.model.predict(X_pred)
    
    def _create_model(self, model_type: str, **kwargs):
        """Create the underlying model."""
        if model_type == 'ridge':
            return Ridge(alpha=1.0, **kwargs)
        elif model_type == 'lasso':
            return Lasso(alpha=0.1, **kwargs)
        elif model_type == 'xgboost':
            params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42,
            }
            params.update(kwargs)
            return XGBRegressor(**params)
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': -1,
            }
            params.update(kwargs)
            return LGBMRegressor(**params)
        else:
            logger.warning(f"Unknown model type {model_type}, using Ridge")
            return Ridge(alpha=1.0)
    
    def fit_predict(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        verbose: bool = False
    ) -> pd.Series:
        """
        Fit model and generate predictions using rolling CV.
        
        Parameters
        ----------
        y : pd.Series
            Target time series (points to predict)
        X : pd.DataFrame
            Feature matrix
        verbose : bool
            Print progress
            
        Returns
        -------
        pd.Series
            Predictions aligned with test indices
        """
        X_vals = X.values
        y_vals = y.values
        
        self.predictions = []
        self.true_values = []
        pred_indices = []
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X_vals)):
            X_train, X_test = X_vals[train_idx], X_vals[test_idx]
            y_train, y_test = y_vals[train_idx], y_vals[test_idx]
            
            # Handle NaN values
            valid_train = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            if valid_train.sum() < 5:
                if verbose:
                    logger.warning(f"Fold {fold}: insufficient valid training data")
                continue
            
            X_train_clean = X_train[valid_train]
            y_train_clean = y_train[valid_train]
            
            # Fit model
            self.model.fit(X_train_clean, y_train_clean)
            
            # Predict
            valid_test = ~np.isnan(X_test).any(axis=1)
            if not valid_test.any():
                continue
            
            X_test_clean = X_test[valid_test]
            y_pred = self.model.predict(X_test_clean)
            
            self.predictions.extend(y_pred)
            self.true_values.extend(y_test[valid_test])
            pred_indices.extend(test_idx[valid_test])
            
            if verbose:
                rmse = np.sqrt(mean_squared_error(y_test[valid_test], y_pred))
                logger.info(f"Fold {fold}: RMSE = {rmse:.3f}")
        
        # Create prediction series
        pred_series = pd.Series(
            self.predictions,
            index=pred_indices,
            name='predicted_points'
        )
        
        return pred_series
    
    def predict_next(self, X: pd.DataFrame) -> float:
        """
        Predict next value given features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (single row for next gameweek)
            
        Returns
        -------
        float
            Predicted points
        """
        if X.empty or self.model is None:
            return 0.0
        
        X_vals = X.values
        if np.isnan(X_vals).any():
            # Impute with mean
            X_vals = np.nan_to_num(X_vals, nan=0.0)
        
        pred = self.model.predict(X_vals)
        return max(0, pred[0])
    
    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models).
        
        Parameters
        ----------
        feature_names : list[str]
            Names of features
            
        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if self.model_type in ['xgboost', 'lightgbm']:
            importance = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return df
        else:
            logger.warning("Feature importance only available for tree-based models")
            return pd.DataFrame()
    
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate model performance.
        
        Returns
        -------
        dict[str, float]
            Dictionary of metrics
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        true_values = np.array(self.true_values)
        
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mae = np.mean(np.abs(true_values - predictions))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions),
        }
