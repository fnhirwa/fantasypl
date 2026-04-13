"""Kalman Filter for continuous player point potential tracking.

State model:    x_{t+1} = x_t + w_t,    w_t ~ N(0, Q_t)
Observation:    y_t     = x_t + v_t,     v_t ~ N(0, R_t)

Supports per-timestep noise overrides so that:
- News shocks (injury) → inflate Q_t (true form can jump suddenly)
- Fixture difficulty → inflate R_t (harder opponents → noisier observations)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    1D Kalman Filter for tracking latent point potential.

    Parameters
    ----------
    process_noise : float
        Default process noise variance (form drift rate).
    observation_noise : float
        Default observation noise variance (weekly point noise).
    initial_state_mean : float
        Initial state estimate.
    initial_state_covariance : float
        Initial state uncertainty (variance).
    """

    def __init__(
        self,
        process_noise: float = 1.0,
        observation_noise: float = 4.0,
        initial_state_mean: float = 4.0,
        initial_state_covariance: float = 2.0,
    ):
        self.default_process_noise = process_noise
        self.default_observation_noise = observation_noise
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance

        # Per-timestep noise overrides
        self._process_noise_overrides: dict[int, float] = {}
        self._observation_noise_overrides: dict[int, float] = {}

        # Stored results after filtering
        self.filtered_state_means: Optional[np.ndarray] = None
        self.filtered_state_covariances: Optional[np.ndarray] = None
        self.kalman_gains: Optional[np.ndarray] = None  # Kalman gains

    def inject_process_shock(self, timestep: int, multiplier: float):
        """
        Inflate process noise at a specific timestep.

        Use when news indicates a sudden form change (injury, transfer).
        process_noise_t = default_process_noise * multiplier.

        Parameters
        ----------
        timestep : int
            Gameweek index.
        multiplier : float
            Process noise multiplier (>1 = more uncertainty about form drift).
        """
        self._process_noise_overrides[timestep] = self.default_process_noise * multiplier

    def inject_observation_noise(self, timestep: int, factor: float):
        """
        Adjust observation noise at a specific timestep.

        Use for fixture difficulty: harder opponents → less predictable points.
        observation_noise_t = default_observation_noise * factor.

        Parameters
        ----------
        timestep : int
            Gameweek index.
        factor : float
            Observation noise factor (>1 = harder fixture, noisier observation).
        """
        self._observation_noise_overrides[timestep] = self.default_observation_noise * factor

    def clear_overrides(self):
        """Remove all per-timestep noise overrides."""
        self._process_noise_overrides.clear()
        self._observation_noise_overrides.clear()

    def get_process_noise_override(self, timestep: int) -> Optional[float]:
        """Return explicit process noise override at timestep, if any."""
        return self._process_noise_overrides.get(timestep)

    def set_noise_overrides(
        self,
        process_noise_overrides: dict[int, float],
        observation_noise_overrides: dict[int, float],
    ):
        """Replace per-timestep noise overrides."""
        self._process_noise_overrides = dict(process_noise_overrides)
        self._observation_noise_overrides = dict(observation_noise_overrides)

    def copy_with_overrides(self, max_timestep: Optional[int] = None) -> "KalmanFilter":
        """Create a parameter-identical filter with copied noise overrides.

        Parameters
        ----------
        max_timestep : int, optional
            If provided, only overrides for timesteps <= max_timestep are copied.
        """
        copied = KalmanFilter(
            process_noise=self.default_process_noise,
            observation_noise=self.default_observation_noise,
            initial_state_mean=self.initial_state_mean,
            initial_state_covariance=self.initial_state_covariance,
        )

        if max_timestep is None:
            proc = dict(self._process_noise_overrides)
            obs = dict(self._observation_noise_overrides)
        else:
            proc = {k: v for k, v in self._process_noise_overrides.items() if k <= max_timestep}
            obs = {k: v for k, v in self._observation_noise_overrides.items() if k <= max_timestep}

        copied.set_noise_overrides(proc, obs)

        return copied

    def _get_process_noise(self, timestep: int) -> float:
        return self._process_noise_overrides.get(timestep, self.default_process_noise)

    def _get_observation_noise(self, timestep: int) -> float:
        return self._observation_noise_overrides.get(timestep, self.default_observation_noise)

    def filter(self, observations: np.ndarray):
        """
        Run Kalman filter on observations with per-timestep noise.

        Parameters
        ----------
        observations : np.ndarray, shape (num_timesteps,)

        Returns
        -------
        filtered_state_means : np.ndarray, shape (num_timesteps,)
            Filtered state estimates (posterior mean).
        filtered_state_covariances : np.ndarray, shape (num_timesteps,)
            Filtered state uncertainties (posterior variance).
        """
        num_timesteps = len(observations)
        filtered_state_means = np.zeros(num_timesteps)
        filtered_state_covariances = np.zeros(num_timesteps)
        kalman_gains = np.zeros(num_timesteps)

        predicted_state_mean = self.initial_state_mean
        predicted_state_covariance = self.initial_state_covariance

        for t in range(num_timesteps):
            process_noise_t = self._get_process_noise(t)
            observation_noise_t = self._get_observation_noise(t)

            # Predict
            if t > 0:
                predicted_state_mean = filtered_state_means[t - 1]
                predicted_state_covariance = filtered_state_covariances[t - 1] + process_noise_t

            # Update
            y = observations[t]
            innovation = y - predicted_state_mean
            innovation_covariance = predicted_state_covariance + observation_noise_t  # Innovation covariance
            kalman_gain = predicted_state_covariance / innovation_covariance  # Kalman gain

            filtered_state_means[t] = predicted_state_mean + kalman_gain * innovation
            filtered_state_covariances[t] = (1 - kalman_gain) * predicted_state_covariance
            kalman_gains[t] = kalman_gain

        self.filtered_state_means = filtered_state_means
        self.filtered_state_covariances = filtered_state_covariances
        self.kalman_gains = kalman_gains

        return filtered_state_means, filtered_state_covariances

    def predict_next(self) -> tuple[float, float]:
        """
        Predict next observation with uncertainty.

        Returns the predictive distribution for Y_{t+1} (the observation),
        not X_{t+1} (the latent state). This ensures consistency with the
        HMM predict_next which also returns observation-level variance.

        Var[Y_{t+1}] = Var[X_{t+1}|y_{1:t}] + R
                     = (P_t + Q) + R

        Must call filter() first.

        Returns
        -------
        predicted_mean : float
            E[Y_{t+1} | y_{1:t}].
        predicted_var : float
            Var[Y_{t+1} | y_{1:t}] (observation-level, includes R).
        """
        if self.filtered_state_means is None or self.filtered_state_covariances is None:
            raise RuntimeError("Must call filter() before predict_next().")

        num_timesteps = len(self.filtered_state_means)
        next_process_noise = self._get_process_noise(num_timesteps)
        next_observation_noise = self._get_observation_noise(num_timesteps)

        predicted_mean = self.filtered_state_means[-1]
        # State-level: P_{t+1|t} = P_{t|t} + Q
        state_var = self.filtered_state_covariances[-1] + next_process_noise
        # Observation-level: Var[Y] = P_{t+1|t} + R
        predicted_var = state_var + next_observation_noise

        return predicted_mean, predicted_var

    def smooth(self, observations: np.ndarray):
        """
        Run RTS smoother (backward pass after forward Kalman filter).

        Parameters
        ----------
        observations : np.ndarray, shape (num_timesteps,)

        Returns
        -------
        smoothed_state_means : np.ndarray, shape (num_timesteps,)
            Smoothed state estimates.
        smoothed_state_covariances : np.ndarray, shape (num_timesteps,)
            Smoothed state uncertainties.
        """
        filtered_state_means, filtered_state_covariances = self.filter(observations)
        num_timesteps = len(observations)

        smoothed_state_means = np.zeros(num_timesteps)
        smoothed_state_covariances = np.zeros(num_timesteps)

        smoothed_state_means[-1] = filtered_state_means[-1]
        smoothed_state_covariances[-1] = filtered_state_covariances[-1]

        for t in range(num_timesteps - 2, -1, -1):
            next_process_noise = self._get_process_noise(t + 1)
            predicted_state_covariance = filtered_state_covariances[t] + next_process_noise

            # Smoother gain
            if predicted_state_covariance > 0:
                smoother_gain = filtered_state_covariances[t] / predicted_state_covariance
            else:
                smoother_gain = 0.0

            smoothed_state_means[t] = filtered_state_means[t] + smoother_gain * (
                smoothed_state_means[t + 1] - filtered_state_means[t]
            )
            smoothed_state_covariances[t] = filtered_state_covariances[t] + smoother_gain**2 * (
                smoothed_state_covariances[t + 1] - predicted_state_covariance
            )

        return smoothed_state_means, smoothed_state_covariances


__all__ = ["KalmanFilter"]
