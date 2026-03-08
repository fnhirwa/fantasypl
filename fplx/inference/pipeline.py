"""Per-player inference pipeline orchestrator.

This is the single entry point that FPLModel.fit() calls for each player.
It coordinates HMM, Kalman Filter, signal injection, and fusion.

Usage:
    pipeline = PlayerInferencePipeline()
    pipeline.ingest_observations(points_array)
    pipeline.inject_news("Player ruled out for 3 weeks", timestep=20)
    pipeline.inject_fixture_difficulty(difficulty=4.5, timestep=21)
    results = pipeline.run()
    ep_mean, ep_var = pipeline.predict_next()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fplx.inference.fusion import fuse_estimates, fuse_sequences
from fplx.inference.hmm import (
    DEFAULT_EMISSION_PARAMS,
    DEFAULT_INITIAL_DIST,
    DEFAULT_TRANSITION_MATRIX,
    HMMInference,
)
from fplx.inference.kalman import KalmanFilter

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for inference pipeline outputs."""

    # HMM outputs
    filtered_beliefs: np.ndarray       # (T, N) forward-filtered state posteriors
    smoothed_beliefs: np.ndarray       # (T, N) forward-backward smoothed posteriors
    viterbi_path: np.ndarray           # (T,) most likely state sequence
    hmm_predicted_mean: float = 0.0    # one-step-ahead E[Y_{T+1}] from HMM
    hmm_predicted_var: float = 0.0     # one-step-ahead Var[Y_{T+1}] from HMM

    # Kalman outputs
    kalman_filtered: np.ndarray = field(default_factory=lambda: np.array([]))
    kalman_uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    kf_predicted_mean: float = 0.0     # one-step-ahead E[X_{T+1}] from KF
    kf_predicted_var: float = 0.0      # one-step-ahead Var[X_{T+1}] from KF

    # Fused outputs
    fused_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    fused_var: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_mean: float = 0.0        # final one-step-ahead fused E[P]
    predicted_var: float = 0.0         # final one-step-ahead fused Var[P]


# News signal (HMM/KF perturbation mapping)
# maps news categories to HMM state boosts and KF process noise multipliers
NEWS_PERTURBATION_MAP = {
    "unavailable": {
        "state_boost": {0: 10.0, 1: 2.0},  # strongly boost Injured, slightly Slump
        "kalman_shock": 5.0,                 # large process noise increase
    },
    "doubtful": {
        "state_boost": {0: 3.0, 1: 2.0},
        "kalman_shock": 2.0,
    },
    "rotation": {
        "state_boost": {1: 2.0, 2: 1.5},   # boost Slump/Average
        "kalman_shock": 1.5,
    },
    "positive": {
        "state_boost": {3: 2.0, 4: 1.5},   # boost Good/Star
        "kalman_shock": 1.0,                 # no extra noise for good news
    },
    "neutral": {
        "state_boost": {},
        "kalman_shock": 1.0,
    },
}

# Fixture difficulty (observation noise factor)
# difficulty in [1, 5]: 1=easiest, 5=hardest
FIXTURE_NOISE_MAP = {
    1: 0.8,   # easy fixture → less noise, more predictable
    2: 0.9,
    3: 1.0,   # neutral
    4: 1.2,
    5: 1.5,   # hard fixture → more noise, less predictable
}


def _classify_news(availability: float, minutes_risk: float) -> str:
    """
    Classify a NewsSignal output into a perturbation category.

    This bridges the existing NewsSignal.generate_signal() output format
    to the perturbation map above.

    Parameters
    ----------
    availability : float
        From NewsSignal (0=out, 1=available).
    minutes_risk : float
        From NewsSignal (0=no risk, 1=high risk).

    Returns
    -------
    str
        One of: "unavailable", "doubtful", "rotation", "positive", "neutral"
    """
    if availability <= 0.1:
        return "unavailable"
    if availability <= 0.6:
        return "doubtful"
    if minutes_risk >= 0.5:
        return "rotation"
    # Distinguish "no news" (avail=1.0, risk=0.0) from "positive news" (avail=0.9)
    # The existing NewsParser returns avail=1.0 for empty text and avail=0.9
    # for positive patterns like "back in training".
    if availability >= 0.85 and availability < 1.0 and minutes_risk <= 0.1:
        return "positive"
    return "neutral"


def _difficulty_to_noise_factor(difficulty: float) -> float:
    """Interpolate fixture difficulty to observation noise factor."""
    difficulty = max(1.0, min(5.0, difficulty))
    # Linear interpolation
    lower = int(difficulty)
    upper = min(lower + 1, 5)
    frac = difficulty - lower
    return (1 - frac) * FIXTURE_NOISE_MAP[lower] + frac * FIXTURE_NOISE_MAP[upper]


class PlayerInferencePipeline:
    """
    Orchestrates HMM + Kalman inference for a single player.

    Parameters
    ----------
    hmm_params : dict, optional
        Override HMM parameters: transition_matrix, emission_params, initial_dist.
    kf_params : dict, optional
        Override Kalman parameters: Q, R, x0, P0.
    """

    def __init__(
        self,
        hmm_params: Optional[dict] = None,
        kf_params: Optional[dict] = None,
    ):
        hmm_params = hmm_params or {}
        kf_params = kf_params or {}

        self.hmm = HMMInference(
            transition_matrix=hmm_params.get("transition_matrix"),
            emission_params=hmm_params.get("emission_params"),
            initial_dist=hmm_params.get("initial_dist"),
        )
        self.kf = KalmanFilter(
            process_noise=kf_params.get("process_noise", 1.0),
            observation_noise=kf_params.get("observation_noise", 4.0),
            initial_state_mean=kf_params.get("initial_state_mean", 4.0),
            initial_state_covariance=kf_params.get("initial_state_covariance", 2.0),
        )

        self.observations: Optional[np.ndarray] = None
        self._result: Optional[InferenceResult] = None

    def ingest_observations(self, points: np.ndarray):
        """
        Set the player's historical points sequence.

        Parameters
        ----------
        points : np.ndarray, shape (T,)
            Weekly points history.
        """
        self.observations = np.asarray(points, dtype=float)
        self._result = None  # invalidate cached result

    def inject_news(
        self,
        news_signal: dict,
        timestep: int,
    ):
        """
        Inject a news signal into the inference at a specific gameweek.

        Bridges from existing NewsSignal.generate_signal() output format.

        Parameters
        ----------
        news_signal : dict
            Output from NewsSignal.generate_signal(). Must contain:
            'availability', 'minutes_risk', 'confidence'.
        timestep : int
            The gameweek index to apply the perturbation.
        """
        category = _classify_news(
            news_signal.get("availability", 1.0),
            news_signal.get("minutes_risk", 0.0),
        )
        confidence = news_signal.get("confidence", 0.6)
        perturbation = NEWS_PERTURBATION_MAP[category]

        # Inject into HMM
        if perturbation["state_boost"]:
            self.hmm.inject_news_perturbation(
                timestep=timestep,
                state_boost=perturbation["state_boost"],
                confidence=confidence,
            )

        # Inject into Kalman
        if perturbation["kalman_shock"] != 1.0:
            self.kf.inject_process_shock(
                timestep=timestep,
                multiplier=perturbation["kalman_shock"],
            )

    def inject_fixture_difficulty(self, difficulty: float, timestep: int):
        """
        Inject fixture difficulty into Kalman observation noise.

        Parameters
        ----------
        difficulty : float
            Fixture difficulty score (1-5, from FixtureSignal).
        timestep : int
            The gameweek index.
        """
        noise_factor = _difficulty_to_noise_factor(difficulty)
        self.kf.inject_observation_noise(timestep=timestep, factor=noise_factor)

    def run(self) -> InferenceResult:
        """
        Run full inference pipeline: HMM + Kalman + Fusion.

        Returns
        -------
        InferenceResult
            All inference outputs.
        """
        if self.observations is None or len(self.observations) == 0:
            raise RuntimeError("No observations ingested. Call ingest_observations().")

        obs = self.observations

        # HMM
        alpha, _ = self.hmm.forward(obs)
        gamma = self.hmm.forward_backward(obs)
        viterbi_path = self.hmm.viterbi(obs)
        hmm_pred_mean, hmm_pred_var, _ = self.hmm.predict_next(obs)

        # Kalman
        kf_x, kf_P = self.kf.filter(obs)
        kf_pred_mean, kf_pred_var = self.kf.predict_next()

        # Fusion (full sequence, smoothed)
        fused_mean, fused_var = fuse_sequences(
            gamma, kf_x, kf_P, self.hmm.emission_params
        )

        # Fused one-step-ahead prediction
        pred_mean, pred_var = fuse_estimates(
            hmm_pred_mean, hmm_pred_var,
            kf_pred_mean, kf_pred_var,
        )

        self._result = InferenceResult(
            filtered_beliefs=alpha,
            smoothed_beliefs=gamma,
            viterbi_path=viterbi_path,
            hmm_predicted_mean=hmm_pred_mean,
            hmm_predicted_var=hmm_pred_var,
            kalman_filtered=kf_x,
            kalman_uncertainty=kf_P,
            kf_predicted_mean=kf_pred_mean,
            kf_predicted_var=kf_pred_var,
            fused_mean=fused_mean,
            fused_var=fused_var,
            predicted_mean=pred_mean,
            predicted_var=pred_var,
        )

        return self._result

    def predict_next(self) -> tuple[float, float]:
        """
        Get the fused one-step-ahead forecast.

        Returns
        -------
        expected_points : float
        variance : float
        """
        if self._result is None:
            self.run()
        return self._result.predicted_mean, self._result.predicted_var

    def learn_parameters(self, n_iter: int = 20):
        """
        Run Baum-Welch to learn HMM parameters from current observations.

        Call this before run() if you want data-driven parameters.
        """
        if self.observations is None:
            raise RuntimeError("No observations. Call ingest_observations() first.")
        self.hmm.fit(self.observations, n_iter=n_iter)


__all__ = ["PlayerInferencePipeline", "InferenceResult"]
