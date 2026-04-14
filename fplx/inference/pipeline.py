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
from fplx.inference.hmm import HMMInference
from fplx.inference.kalman import KalmanFilter

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for inference pipeline outputs."""

    # HMM outputs
    filtered_beliefs: np.ndarray  # (T, N) forward-filtered state posteriors
    smoothed_beliefs: np.ndarray  # (T, N) forward-backward smoothed posteriors
    viterbi_path: np.ndarray  # (T,) most likely state sequence
    hmm_predicted_mean: float = 0.0  # one-step-ahead E[Y_{T+1}] from HMM
    hmm_predicted_var: float = 0.0  # one-step-ahead Var[Y_{T+1}] from HMM

    # Kalman outputs
    kalman_filtered: np.ndarray = field(default_factory=lambda: np.array([]))
    kalman_uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    kf_predicted_mean: float = 0.0  # one-step-ahead E[X_{T+1}] from KF
    kf_predicted_var: float = 0.0  # one-step-ahead Var[X_{T+1}] from KF

    # Fused outputs
    fused_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    fused_var: np.ndarray = field(default_factory=lambda: np.array([]))
    fusion_alpha: Optional[float] = None  # used for calibrated-alpha fusion mode
    predicted_mean: float = 0.0  # final one-step-ahead fused E[P]
    predicted_var: float = 0.0  # final one-step-ahead fused Var[P]


# News signal defaults (HMM/KF perturbation mapping)
# maps news categories to HMM state boosts and KF process noise multipliers
DEFAULT_NEWS_PERTURBATION_MAP = {
    "unavailable": {
        "state_boost": {0: 10.0, 1: 2.0},  # strongly boost Injured, slightly Slump
        "kalman_shock": 5.0,  # large process noise increase
    },
    "doubtful": {
        "state_boost": {0: 3.0, 1: 2.0},
        "kalman_shock": 2.0,
    },
    "rotation": {
        "state_boost": {1: 2.0, 2: 1.5},  # boost Slump/Average
        "kalman_shock": 1.5,
    },
    "positive": {
        "state_boost": {3: 2.0, 4: 1.5},  # boost Good/Star
        "kalman_shock": 1.0,  # no extra noise for good news
    },
    "neutral": {
        "state_boost": {},
        "kalman_shock": 1.0,
    },
}

DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS = {
    "unavailable_max_availability": 0.1,
    "doubtful_max_availability": 0.6,
    "rotation_min_minutes_risk": 0.5,
    "positive_min_availability": 0.85,
    "positive_max_availability": 1.0,
    "positive_max_minutes_risk": 0.1,
}

DEFAULT_NEWS_PARAMS = {
    "classification_thresholds": DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS,
    "perturbation_map": DEFAULT_NEWS_PERTURBATION_MAP,
    "default_confidence": 0.6,
}

DEFAULT_FUSION_PARAMS = {
    "default_alpha": 0.8,
    "alpha_floor": 0.5,
    "grid_step": 0.05,
    "min_history": 8,
}

# Fixture difficulty (observation noise factor)
# difficulty in [1, 5]: 1=easiest, 5=hardest
FIXTURE_NOISE_MAP = {
    1: 0.8,  # easy fixture → less noise, more predictable
    2: 0.9,
    3: 1.0,  # neutral
    4: 1.2,
    5: 1.5,  # hard fixture → more noise, less predictable
}


def _classify_news(
    availability: float,
    minutes_risk: float,
    thresholds: Optional[dict] = None,
) -> str:
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
    thresholds = thresholds or DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS

    unavailable_max_availability = float(
        thresholds.get(
            "unavailable_max_availability",
            DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS["unavailable_max_availability"],
        )
    )
    doubtful_max_availability = float(
        thresholds.get(
            "doubtful_max_availability",
            DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS["doubtful_max_availability"],
        )
    )
    rotation_min_minutes_risk = float(
        thresholds.get(
            "rotation_min_minutes_risk",
            DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS["rotation_min_minutes_risk"],
        )
    )
    positive_min_availability = float(
        thresholds.get(
            "positive_min_availability",
            DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS["positive_min_availability"],
        )
    )
    positive_max_availability = float(
        thresholds.get(
            "positive_max_availability",
            DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS["positive_max_availability"],
        )
    )
    positive_max_minutes_risk = float(
        thresholds.get(
            "positive_max_minutes_risk",
            DEFAULT_NEWS_CLASSIFICATION_THRESHOLDS["positive_max_minutes_risk"],
        )
    )

    if availability <= unavailable_max_availability:
        return "unavailable"
    if availability <= doubtful_max_availability:
        return "doubtful"
    if minutes_risk >= rotation_min_minutes_risk:
        return "rotation"
    # Distinguish "no news" (avail=1.0, risk=0.0) from "positive news" (avail=0.9)
    # The existing NewsParser returns avail=1.0 for empty text and avail=0.9
    # for positive patterns like "back in training".
    if (
        availability >= positive_min_availability
        and availability < positive_max_availability
        and minutes_risk <= positive_max_minutes_risk
    ):
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


def _merge_nested_dicts(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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
        hmm_variance_floor: float = 1.0,
        news_params: Optional[dict] = None,
        fusion_mode: str = "precision",
        fusion_params: Optional[dict] = None,
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
        self.hmm_variance_floor = max(float(hmm_variance_floor), 1e-6)
        self.news_params = _merge_nested_dicts(DEFAULT_NEWS_PARAMS, news_params or {})
        self.fusion_mode = fusion_mode
        self.fusion_params = _merge_nested_dicts(DEFAULT_FUSION_PARAMS, fusion_params or {})
        if self.fusion_mode not in {"precision", "calibrated_alpha"}:
            raise ValueError(
                f"Unknown fusion_mode '{self.fusion_mode}'. Expected one of: 'precision', 'calibrated_alpha'."
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
            self.news_params.get("classification_thresholds"),
        )
        confidence = news_signal.get(
            "confidence",
            float(self.news_params.get("default_confidence", 0.6)),
        )

        perturbation_map = self.news_params.get("perturbation_map", DEFAULT_NEWS_PERTURBATION_MAP)
        perturbation = perturbation_map.get(
            category,
            perturbation_map.get("neutral", {"state_boost": {}, "kalman_shock": 1.0}),
        )

        # Inject into HMM
        state_boost = perturbation.get("state_boost", {})
        if state_boost:
            self.hmm.inject_news_perturbation(
                timestep=timestep,
                state_boost=state_boost,
                confidence=confidence,
            )

        # Inject into Kalman
        kalman_shock = float(perturbation.get("kalman_shock", 1.0))
        if kalman_shock != 1.0:
            self.kf.inject_process_shock(
                timestep=timestep,
                multiplier=kalman_shock,
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

    def _hmm_sequence_moments(self, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute HMM mean/variance sequences from state posteriors."""
        n_states = gamma.shape[1]
        state_means = np.array([self.hmm.emission_params[s][0] for s in range(n_states)])
        state_vars = np.array([
            max(self.hmm.emission_params[s][1] ** 2, self.hmm_variance_floor) for s in range(n_states)
        ])

        hmm_mean = gamma @ state_means
        hmm_var = gamma @ state_vars + gamma @ (state_means**2) - hmm_mean**2
        hmm_var = np.maximum(hmm_var, self.hmm_variance_floor)
        return hmm_mean, hmm_var

    def _estimate_fusion_alpha(self, observations: np.ndarray) -> float:
        """Estimate alpha in y = alpha*KF + (1-alpha)*HMM via rolling validation."""
        default_alpha = float(self.fusion_params.get("default_alpha", 0.8))
        default_alpha = float(np.clip(default_alpha, 0.0, 1.0))

        # Floor: KF always gets at least this weight. The HMM's 5-state
        # discretization loses information relative to the KF's continuous
        # tracking, so we never let HMM dominate the fusion.
        alpha_floor = float(self.fusion_params.get("alpha_floor", 0.5))

        min_history = int(self.fusion_params.get("min_history", 8))
        min_history = max(3, min_history)

        grid_step = float(self.fusion_params.get("grid_step", 0.05))
        grid_step = min(max(grid_step, 0.01), 0.5)

        if len(observations) <= min_history:
            return default_alpha

        hmm_preds = []
        kf_preds = []
        actuals = []

        for t in range(min_history, len(observations)):
            prefix = observations[:t]
            target = observations[t]

            hmm_mu, _, _ = self.hmm.predict_next(prefix)

            kf_tmp = self.kf.copy_with_overrides(max_timestep=t)

            kf_tmp.filter(prefix)
            kf_mu, _ = kf_tmp.predict_next()

            if np.isfinite(hmm_mu) and np.isfinite(kf_mu) and np.isfinite(target):
                hmm_preds.append(hmm_mu)
                kf_preds.append(kf_mu)
                actuals.append(target)

        if len(actuals) < 3:
            return default_alpha

        hmm_preds = np.asarray(hmm_preds)
        kf_preds = np.asarray(kf_preds)
        actuals = np.asarray(actuals)

        alphas = np.arange(alpha_floor, 1.0 + 1e-9, grid_step)
        mses = []
        for alpha in alphas:
            pred = alpha * kf_preds + (1.0 - alpha) * hmm_preds
            mses.append(np.mean((pred - actuals) ** 2))

        best_idx = int(np.argmin(mses))
        return float(alphas[best_idx])

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

        fusion_alpha = None
        if self.fusion_mode == "calibrated_alpha":
            fusion_alpha = self._estimate_fusion_alpha(obs)
            hmm_seq_mean, hmm_seq_var = self._hmm_sequence_moments(gamma)

            fused_mean = fusion_alpha * kf_x + (1.0 - fusion_alpha) * hmm_seq_mean
            fused_var = fusion_alpha**2 * np.maximum(kf_P, 1e-6) + (1.0 - fusion_alpha) ** 2 * np.maximum(
                hmm_seq_var, self.hmm_variance_floor
            )

            pred_mean = fusion_alpha * kf_pred_mean + (1.0 - fusion_alpha) * hmm_pred_mean
            pred_var = fusion_alpha**2 * max(kf_pred_var, 1e-6) + (1.0 - fusion_alpha) ** 2 * max(
                hmm_pred_var, self.hmm_variance_floor
            )
        else:
            # Fusion (full sequence, smoothed)
            # Apply an HMM variance floor so HMM does not become unrealistically
            # overconfident and dominate precision-weighted fusion.
            emission_params_for_fusion = {
                s: (mu, max(std, np.sqrt(self.hmm_variance_floor)))
                for s, (mu, std) in self.hmm.emission_params.items()
            }
            fused_mean, fused_var = fuse_sequences(gamma, kf_x, kf_P, emission_params_for_fusion)

            # Fused one-step-ahead prediction
            pred_mean, pred_var = fuse_estimates(
                hmm_pred_mean,
                max(hmm_pred_var, self.hmm_variance_floor),
                kf_pred_mean,
                kf_pred_var,
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
            fusion_alpha=fusion_alpha,
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
