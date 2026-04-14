"""Position-aware multivariate-emission HMM for player form inference.

Uses position-specific feature vectors extracted from the full vaastav dataset:

  GK:  [saves/90, xGC/90, clean_sheet, bonus, mins_frac]
  DEF: [xG, xA, xGC/90, clean_sheet, influence/100, bonus, mins_frac]
  MID: [xG, xA, creativity/100, threat/100, bonus, mins_frac]
  FWD: [xG, xA, threat/100, bonus, mins_frac]

Each state emits a multivariate Gaussian with diagonal covariance.
Baum-Welch learns per-player emission parameters from their history.

The minutes_fraction feature (0 or ~1) lets the HMM identify the Injured
state from the feature vector alone, without NLP news signals.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from fplx.inference.enriched import compute_xpoints

logger = logging.getLogger(__name__)

STATE_NAMES = ["Injured", "Slump", "Average", "Good", "Star"]
N_STATES = 5

# Domain-specific projection to reduce sample complexity.
# We compress rich features into:
#   xPts: structurally scored expected points from enriched.compute_xpoints
#   mins_frac: availability/playing-time signal to separate Injured state
POSITION_FEATURES = {
    "GK": ["xPts", "mins_frac"],
    "DEF": ["xPts", "mins_frac"],
    "MID": ["xPts", "mins_frac"],
    "FWD": ["xPts", "mins_frac"],
}

# FPL scoring rules
GOAL_PTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
GC_PTS = {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0}
ASSIST_PTS = 3

# Default emission parameters per position
# Format: {position: {state_idx: (mean_vector, var_vector)}}
# These are starting points; Baum-Welch adapts them per player.


def _default_emissions(position: str) -> tuple[np.ndarray, np.ndarray]:
    """Generate default 2D emission means and vars [xPts, mins_frac]."""
    # Position-aware expected-points ladder while keeping the same latent semantics.
    star_anchor = {"GK": 6.8, "DEF": 6.5, "MID": 7.5, "FWD": 7.2}.get(position, 7.0)

    means = np.zeros((N_STATES, 2))
    vars_ = np.full((N_STATES, 2), 0.01)

    # Injured
    means[0] = [0.0, 0.0]
    vars_[0] = [0.05, 0.001]

    # Slump, Average, Good, Star
    means[1] = [0.25 * star_anchor, 0.75]
    means[2] = [0.45 * star_anchor, 0.95]
    means[3] = [0.68 * star_anchor, 1.00]
    means[4] = [1.00 * star_anchor, 1.00]

    vars_[1] = [0.8, 0.05]
    vars_[2] = [1.2, 0.02]
    vars_[3] = [1.8, 0.01]
    vars_[4] = [2.4, 0.005]

    return means, vars_


DEFAULT_TRANSITION = np.array([
    [0.60, 0.25, 0.10, 0.05, 0.00],
    [0.05, 0.50, 0.35, 0.08, 0.02],
    [0.02, 0.10, 0.55, 0.25, 0.08],
    [0.02, 0.05, 0.15, 0.55, 0.23],
    [0.01, 0.02, 0.07, 0.30, 0.60],
])
DEFAULT_INITIAL = np.array([0.05, 0.10, 0.50, 0.25, 0.10])


def _safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(0).values


def build_feature_matrix(timeseries: pd.DataFrame, position: str) -> np.ndarray:
    """
    Extract position-specific feature matrix from player timeseries.

    Parameters
    ----------
    timeseries : pd.DataFrame
        Player gameweek history from vaastav dataset.
    position : str
        GK, DEF, MID, or FWD.

    Returns
    -------
    np.ndarray, shape (T, D) where D depends on position.
    """
    n = len(timeseries)
    features = np.zeros((n, 2))

    mins = _safe_col(timeseries, "minutes")
    features[:, 1] = np.clip(mins / 90.0, 0.0, 1.0)  # mins_frac

    # Domain-specific projection from rich event space to structural xPts.
    features[:, 0] = compute_xpoints(timeseries, position)
    return features


class MultivariateHMM:
    """
    Position-aware HMM with multivariate diagonal Gaussian emissions.

    Parameters
    ----------
    position : str
        GK, DEF, MID, FWD. Determines feature set and default emissions.
    """

    def __init__(
        self,
        position: str = "MID",
        transition_matrix: Optional[np.ndarray] = None,
        initial_dist: Optional[np.ndarray] = None,
    ):
        self.position = position
        self.means, self.vars = _default_emissions(position)

        # Priors for MAP-style regularization in Baum-Welch.
        self.prior_means = self.means.copy()
        self.prior_vars = self.vars.copy()
        self.prior_A = (
            transition_matrix.copy() if transition_matrix is not None else DEFAULT_TRANSITION.copy()
        )

        self.A = self.prior_A.copy()
        self.pi = initial_dist.copy() if initial_dist is not None else DEFAULT_INITIAL.copy()
        self.n_states = N_STATES
        self.n_features = self.means.shape[1]
        self._transition_overrides: dict[int, np.ndarray] = {}

    def _emission_log_prob(self, obs: np.ndarray, state: int) -> float:
        mu = self.means[state]
        var = np.maximum(self.vars[state], 1e-8)
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (obs - mu) ** 2 / var)

    def _emission_prob_vector(self, obs: np.ndarray) -> np.ndarray:
        lp = np.array([self._emission_log_prob(obs, s) for s in range(self.n_states)])
        lp -= np.max(lp)
        return np.exp(lp)

    def _get_A(self, t: int) -> np.ndarray:
        return self._transition_overrides.get(t, self.A)

    def inject_news_perturbation(self, timestep: int, state_boost: dict, confidence: float = 1.0):
        """Perturb transition matrix at timestep (same API as scalar HMM)."""
        A_p = self.A.copy()
        for src in range(self.n_states):
            for tgt, boost in state_boost.items():
                A_p[src, tgt] *= 1.0 + confidence * (boost - 1.0)
            s = A_p[src].sum()
            if s > 0:
                A_p[src] /= s
        self._transition_overrides[timestep] = A_p

    def forward(self, observations: np.ndarray):
        """Forward algorithm. observations: (T, D)."""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        scale = np.zeros(T)
        b = self._emission_prob_vector(observations[0])
        alpha[0] = self.pi * b
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]
        for t in range(1, T):
            b = self._emission_prob_vector(observations[t])
            alpha[t] = (alpha[t - 1] @ self._get_A(t)) * b
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
        return alpha, scale

    def forward_backward(self, observations: np.ndarray) -> np.ndarray:
        """Smoothed posteriors P(S_t | y_{1:T})."""
        T = len(observations)
        alpha, scale = self.forward(observations)
        beta = np.zeros((T, self.n_states))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            b_next = self._emission_prob_vector(observations[t + 1])
            beta[t] = self._get_A(t + 1) @ (b_next * beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]
        gamma = alpha * beta
        rs = gamma.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        return gamma / rs

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Most likely state sequence."""
        T = len(observations)
        log_d = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        log_d[0] = np.log(self.pi + 1e-300) + np.array([
            self._emission_log_prob(observations[0], s) for s in range(self.n_states)
        ])
        for t in range(1, T):
            log_A = np.log(self._get_A(t) + 1e-300)
            log_b = np.array([self._emission_log_prob(observations[t], s) for s in range(self.n_states)])
            for s in range(self.n_states):
                c = log_d[t - 1] + log_A[:, s]
                psi[t, s] = np.argmax(c)
                log_d[t, s] = c[psi[t, s]] + log_b[s]
        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(log_d[T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def predict_next_features(self, observations: np.ndarray):
        """
        Predict next gameweek's feature vector.

        Returns mean, var (per feature), and state distribution.
        """
        alpha, _ = self.forward(observations)
        next_dist = alpha[-1] @ self._get_A(len(observations))
        mean = next_dist @ self.means
        var = next_dist @ self.vars + next_dist @ (self.means**2) - mean**2
        return mean, np.maximum(var, 1e-8), next_dist

    def _expected_points_from_state_dist(self, state_dist: np.ndarray) -> float:
        """Map state distribution to expected points via projected xPts feature."""
        xpts_idx = POSITION_FEATURES[self.position].index("xPts")
        expected_points = float(state_dist @ self.means[:, xpts_idx])
        return max(0.0, expected_points)

    def one_step_point_predictions(self, observations: np.ndarray) -> np.ndarray:
        """One-step-ahead point predictions for each historical timestep.

        Returns array preds where preds[t] predicts points at timestep t,
        using information up to t-1 (preds[0] is NaN).
        """
        T = len(observations)
        preds = np.full(T, np.nan)
        if T < 2:
            return preds

        alpha, _ = self.forward(observations)
        for t in range(1, T):
            pred_dist = alpha[t - 1] @ self._get_A(t)
            preds[t] = self._expected_points_from_state_dist(pred_dist)
        return preds

    def predict_next_points(self, observations: np.ndarray) -> tuple[float, float]:
        """
        Convert predicted features → expected FPL points.

        Uses FPL scoring rules applied to predicted feature rates.
        """
        feat_mean, feat_var, _ = self.predict_next_features(observations)
        feat_names = POSITION_FEATURES[self.position]
        xpts_idx = feat_names.index("xPts")

        ep = max(0.0, float(feat_mean[xpts_idx]))
        var_pts = float(max(feat_var[xpts_idx], 1e-6) + 1.0)  # residual floor
        return ep, var_pts

    def fit(
        self,
        observations: np.ndarray,
        n_iter: int = 20,
        tol: float = 1e-4,
        prior_weight: float = 0.85,
    ):
        """Baum-Welch EM with MAP-style prior interpolation.

        Parameters
        ----------
        observations : np.ndarray
            Feature matrix with shape (T, D).
        n_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance on log-likelihood.
        prior_weight : float
            Weight on prior parameters in [0, 1]. Higher values increase
            regularization toward position-level default emissions/transitions.
        """
        T = observations.shape[0]
        prev_ll = -np.inf
        prior_weight = float(np.clip(prior_weight, 0.0, 1.0))

        for _ in range(n_iter):
            alpha, scale = self.forward(observations)

            # Backward pass with scaling aligned to forward()
            beta = np.zeros((T, self.n_states))
            beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                b_next = self._emission_prob_vector(observations[t + 1])
                beta[t] = self._get_A(t + 1) @ (b_next * beta[t + 1])
                if scale[t + 1] > 0:
                    beta[t] /= scale[t + 1]

            gamma = alpha * beta
            rs = gamma.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            gamma /= rs

            # M-step: initial
            self.pi = np.maximum(gamma[0], 1e-10)
            self.pi /= self.pi.sum()

            # M-step: transitions
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                b_next = self._emission_prob_vector(observations[t + 1])
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self._get_A(t + 1)[i, j] * b_next[j] * beta[t + 1, j]
                xs = xi[t].sum()
                if xs > 0:
                    xi[t] /= xs
            for i in range(self.n_states):
                d = gamma[:-1, i].sum()
                if d > 1e-10:
                    mle_A = xi[:, i, :].sum(axis=0) / d
                    self.A[i] = prior_weight * self.prior_A[i] + (1.0 - prior_weight) * mle_A
                rs = self.A[i].sum()
                if rs > 0:
                    self.A[i] /= rs

            # M-step: emissions
            for s in range(self.n_states):
                w = gamma[:, s]
                ws = w.sum()
                if ws > 1e-10:
                    mle_mu = np.average(observations, axis=0, weights=w)
                    diff = observations - mle_mu
                    mle_var = np.average(diff**2, axis=0, weights=w)
                    self.means[s] = prior_weight * self.prior_means[s] + (1.0 - prior_weight) * mle_mu
                    self.vars[s] = np.maximum(
                        prior_weight * self.prior_vars[s] + (1.0 - prior_weight) * mle_var,
                        1e-4,
                    )

            ll = np.sum(np.log(scale + 1e-300))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        return self


__all__ = [
    "MultivariateHMM",
    "build_feature_matrix",
    "STATE_NAMES",
    "POSITION_FEATURES",
    "N_STATES",
]
