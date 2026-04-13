"""Metrics for evaluating inference accuracy and optimization quality.

Part I (18-662) metrics: prediction accuracy, calibration, ablation.
Part II (18-660) metrics: actual points, optimality gap, consistency.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """
    Collects and computes inference evaluation metrics.

    Usage:
        metrics = InferenceMetrics()
        for each player-gameweek:
            metrics.add(predicted_mean, predicted_var, actual_points)
        report = metrics.compute()
    """

    predicted_means: list[float] = field(default_factory=list)
    predicted_vars: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)

    # Per-model ablation tracking
    model_predictions: dict[str, list[float]] = field(default_factory=dict)

    def add(
        self,
        predicted_mean: float,
        predicted_var: float,
        actual: float,
        model_preds: dict[str, float] | None = None,
    ):
        """Record a single prediction-actual pair."""
        self.predicted_means.append(predicted_mean)
        self.predicted_vars.append(predicted_var)
        self.actuals.append(actual)

        if model_preds:
            for name, pred in model_preds.items():
                if name not in self.model_predictions:
                    self.model_predictions[name] = []
                self.model_predictions[name].append(pred)

    def compute(self) -> dict:
        """Compute all inference metrics."""
        preds = np.array(self.predicted_means)
        varis = np.array(self.predicted_vars)
        acts = np.array(self.actuals)

        if len(preds) == 0:
            return {}

        errors = preds - acts

        report = {
            "n_predictions": len(preds),
            "mse": float(np.mean(errors**2)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "mean_bias": float(np.mean(errors)),
        }

        # Calibration: what fraction of actuals fall within 95% CI?
        stds = np.sqrt(np.maximum(varis, 1e-8))
        lower_95 = preds - 1.96 * stds
        upper_95 = preds + 1.96 * stds
        in_ci = (acts >= lower_95) & (acts <= upper_95)
        report["calibration_95"] = float(np.mean(in_ci))

        # Also check 50% CI
        lower_50 = preds - 0.674 * stds
        upper_50 = preds + 0.674 * stds
        in_ci_50 = (acts >= lower_50) & (acts <= upper_50)
        report["calibration_50"] = float(np.mean(in_ci_50))

        # Mean predicted std (average uncertainty)
        report["mean_predicted_std"] = float(np.mean(stds))

        # Log-likelihood under Gaussian predictive distribution
        # log p(y | mu, sigma^2) = -0.5 * (log(2*pi*sigma^2) + (y-mu)^2/sigma^2)
        ll = -0.5 * (np.log(2 * np.pi * np.maximum(varis, 1e-8)) + errors**2 / np.maximum(varis, 1e-8))
        report["mean_log_likelihood"] = float(np.mean(ll))

        # Per-model ablation MSE
        ablation = {}
        for name, model_preds in self.model_predictions.items():
            mp = np.array(model_preds)
            if len(mp) == len(acts):
                ablation[name] = {
                    "mse": float(np.mean((mp - acts) ** 2)),
                    "mae": float(np.mean(np.abs(mp - acts))),
                }
        if ablation:
            report["ablation"] = ablation

        return report


@dataclass
class OptimizationMetrics:
    """
    Collects and computes optimization evaluation metrics.

    Tracks actual points earned per gameweek under different strategies,
    and compares against oracle (hindsight-optimal).

    Usage:
        metrics = OptimizationMetrics()
        for each gameweek:
            metrics.add_gameweek(gw, actual_points, oracle_points)
        report = metrics.compute()
    """

    # {strategy_name: [points_per_gw]}
    strategy_points: dict[str, list[float]] = field(default_factory=dict)
    oracle_points: list[float] = field(default_factory=list)
    gameweeks: list[int] = field(default_factory=list)

    def add_gameweek(
        self,
        gw: int,
        strategy_results: dict[str, float],
        oracle: float,
    ):
        """
        Record actual points for one gameweek across strategies.

        Parameters
        ----------
        gw : int
            Gameweek number.
        strategy_results : dict[str, float]
            {strategy_name: actual_points_earned}
        oracle : float
            Best possible points with hindsight.
        """
        self.gameweeks.append(gw)
        self.oracle_points.append(oracle)

        for name, pts in strategy_results.items():
            if name not in self.strategy_points:
                self.strategy_points[name] = []
            self.strategy_points[name].append(pts)

    def compute(self) -> dict:
        """Compute optimization metrics for all strategies."""
        oracle = np.array(self.oracle_points)
        report = {
            "n_gameweeks": len(self.gameweeks),
            "oracle_total": float(np.sum(oracle)),
            "oracle_mean_per_gw": float(np.mean(oracle)) if len(oracle) > 0 else 0.0,
            "strategies": {},
        }

        for name, pts_list in self.strategy_points.items():
            pts = np.array(pts_list)
            total = float(np.sum(pts))
            mean_gw = float(np.mean(pts)) if len(pts) > 0 else 0.0
            std_gw = float(np.std(pts)) if len(pts) > 0 else 0.0

            # Optimality gap: (oracle - strategy) / oracle
            gaps = (oracle[: len(pts)] - pts) / np.maximum(oracle[: len(pts)], 1e-6)
            mean_gap = float(np.mean(gaps)) if len(gaps) > 0 else 0.0

            # Worst-case: minimum points in any single gameweek
            worst_gw = float(np.min(pts)) if len(pts) > 0 else 0.0

            # Consistency: coefficient of variation
            cv = std_gw / mean_gw if mean_gw > 0 else 0.0

            report["strategies"][name] = {
                "total_points": total,
                "mean_per_gw": mean_gw,
                "std_per_gw": std_gw,
                "cv": cv,
                "worst_gw_points": worst_gw,
                "mean_optimality_gap": mean_gap,
                "pct_of_oracle": total / float(np.sum(oracle)) * 100 if np.sum(oracle) > 0 else 0,
            }

        return report


__all__ = ["InferenceMetrics", "OptimizationMetrics"]
