"""Statistical performance signals."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StatsSignal:
    """
    Generate performance signals from statistical data.

    Combines multiple statistical indicators into a unified score.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """
        Initialize with custom weights for different stats.

        Parameters
        ----------
        weights : Optional[dict[str, float]]
            Weights for different statistics
        """
        self.weights = weights or {
            "points_mean": 0.3,
            "xG_mean": 0.15,
            "xA_mean": 0.15,
            "minutes_consistency": 0.2,
            "form_trend": 0.2,
        }

    def compute_signal(self, player_data: pd.DataFrame) -> float:
        """
        Compute aggregated signal score from player statistics.

        Parameters
        ----------
        player_data : pd.DataFrame
            Player historical data with engineered features

        Returns
        -------
        float
            Aggregated signal score (0-100)
        """
        if player_data.empty:
            return 0.0

        # Get latest row (most recent data)
        latest = player_data.iloc[-1]

        score = 0.0

        # Points form (rolling mean)
        if "points_rolling_5_mean" in latest:
            points_component = (
                latest["points_rolling_5_mean"] * self.weights["points_mean"]
            )
            score += points_component

        # xG contribution
        if "xG_rolling_5_mean" in latest:
            xg_component = latest["xG_rolling_5_mean"] * 10 * self.weights["xG_mean"]
            score += xg_component

        # xA contribution
        if "xA_rolling_5_mean" in latest:
            xa_component = latest["xA_rolling_5_mean"] * 10 * self.weights["xA_mean"]
            score += xa_component

        # Minutes consistency (inverse of coefficient of variation)
        if "minutes_consistency_5" in latest:
            consistency = 1.0 / (1.0 + latest["minutes_consistency_5"])
            consistency_component = (
                consistency * 10 * self.weights["minutes_consistency"]
            )
            score += consistency_component

        # Form trend
        if "points_trend_5" in latest:
            trend = latest["points_trend_5"]
            # Normalize trend: positive trend is good
            trend_component = max(0, trend) * 5 * self.weights["form_trend"]
            score += trend_component

        return max(0, score)

    def batch_compute(self, players_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """
        Compute signals for multiple players.

        Parameters
        ----------
        players_data : dict[str, pd.DataFrame]
            Dictionary mapping player ID/name to their data

        Returns
        -------
        dict[str, float]
            Dictionary of player signals
        """
        signals = {}
        for player_id, data in players_data.items():
            signals[player_id] = self.compute_signal(data)

        return signals
