"""Baseline heuristic models for FPL prediction."""

import logging

import pandas as pd

from fplx.models.base import BaseModel

logger = logging.getLogger(__name__)


class BaselineModel(BaseModel):
    """
    Baseline model using simple heuristics.

    Methods:
    - Rolling average of points
    - Weighted recent form
    - Form-based prediction
    """

    def __init__(self, method: str = "rolling_mean", window: int = 5):
        """
        Initialize baseline model.

        Parameters
        ----------
        method : str
            Prediction method: 'rolling_mean', 'ewma', 'last_value'
        window : int
            Window size for rolling calculations
        """
        self.method = method
        self.window = window
        self.predictions = {}

    def fit(self, X, y=None):
        """Fit the model (no-op for baseline)."""
        return self

    def predict(self, X: pd.DataFrame) -> float:
        """
        Predict next gameweek points for a player.

        Parameters
        ----------
        X : pd.DataFrame
            Player historical data

        Returns
        -------
        float
            Predicted points
        """
        if X.empty or "points" not in X.columns:
            return 0.0

        points = X["points"]

        if self.method == "rolling_mean":
            return self._rolling_mean(points)
        if self.method == "ewma":
            return self._ewma(points)
        if self.method == "last_value":
            return points.iloc[-1]
        logger.warning(f"Unknown method {self.method}, using rolling_mean")
        return self._rolling_mean(points)

    def _rolling_mean(self, points: pd.Series) -> float:
        """Calculate rolling mean."""
        if len(points) >= self.window:
            return points.tail(self.window).mean()
        return points.mean()

    def _ewma(self, points: pd.Series, alpha: float = 0.3) -> float:
        """Calculate exponentially weighted moving average."""
        return points.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

    def batch_predict(self, players_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """
        Predict for multiple players.

        Parameters
        ----------
        players_data : dict[str, pd.DataFrame]
            Dictionary mapping player ID to their data

        Returns
        -------
        dict[str, float]
            Dictionary of predictions
        """
        predictions = {}
        for player_id, data in players_data.items():
            predictions[player_id] = self.predict(data)

        self.predictions = predictions
        return predictions


class FormBasedModel(BaselineModel):
    """
    Enhanced baseline using form indicators.
    """

    def predict(self, X: pd.DataFrame) -> float:
        """
        Predict based on form with adjustments.

        Parameters
        ----------
        player_data : pd.DataFrame
            Player historical data

        Returns
        -------
        float
            Predicted points
        """
        if X.empty:
            return 0.0

        base_prediction = super().predict(X)

        # Apply adjustments
        latest = X.iloc[-1]

        # Minutes adjustment: if playing less, reduce prediction
        if "minutes" in latest:
            if latest["minutes"] < 60:
                base_prediction *= 0.7

        # Trend adjustment
        if "points_trend_5" in latest:
            trend = latest["points_trend_5"]
            if trend > 0.5:
                base_prediction *= 1.1  # Positive trend bonus
            elif trend < -0.5:
                base_prediction *= 0.9  # Negative trend penalty

        return max(0, base_prediction)
