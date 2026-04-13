"""Ensemble models combining multiple predictors."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble combining multiple models with weighted averaging.

    Parameters
    ----------
    models : list
        List of model instances
    weights : Optional[list[float]]
        Weights for each model (must sum to 1)
    """

    def __init__(self, models: list, weights: Optional[list[float]] = None):
        self.models = models

        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1")
            self.weights = weights

    def predict(self, player_data: pd.DataFrame) -> float:
        """
        Ensemble prediction for a single player.

        Parameters
        ----------
        player_data : pd.DataFrame
            Player historical data

        Returns
        -------
        float
            Ensemble prediction
        """
        predictions = []

        for model in self.models:
            try:
                pred = model.predict(player_data)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {type(model).__name__} failed: {e}")
                predictions.append(0.0)

        # Weighted average
        ensemble_pred = sum(p * w for p, w in zip(predictions, self.weights))
        return max(0, ensemble_pred)

    def batch_predict(self, players_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """
        Ensemble predictions for multiple players.

        Parameters
        ----------
        players_data : Dict[str, pd.DataFrame]
            Dictionary mapping player ID to their data

        Returns
        -------
        Dict[str, float]
            Dictionary of ensemble predictions
        """
        predictions = {}

        for player_id, data in players_data.items():
            predictions[player_id] = self.predict(data)

        return predictions


class AdaptiveEnsemble(EnsembleModel):
    """Adaptive ensemble that adjusts weights based on recent performance."""

    def __init__(self, models: list, learning_rate: float = 0.1):
        super().__init__(models)
        self.learning_rate = learning_rate
        self.model_errors = [[] for _ in models]

    def update_weights(self):
        """Update weights based on recent errors."""
        if not any(self.model_errors):
            return

        # Calculate inverse error scores
        avg_errors = []
        for errors in self.model_errors:
            if errors:
                avg_errors.append(np.mean(errors[-5:]))  # Last 5 predictions
            else:
                avg_errors.append(1.0)

        # Inverse error weighting
        inv_errors = [1.0 / (e + 1e-6) for e in avg_errors]
        total = sum(inv_errors)
        new_weights = [ie / total for ie in inv_errors]

        # Smooth update
        self.weights = [
            (1 - self.learning_rate) * old + self.learning_rate * new
            for old, new in zip(self.weights, new_weights)
        ]

        # Renormalize
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
