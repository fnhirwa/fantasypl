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

logger = logging.getLogger(__name__)

STATE_NAMES = ["Injured", "Slump", "Average", "Good", "Star"]
N_STATES = 5

# Position-specific feature definitions
POSITION_FEATURES = {
    "GK": ["saves_per90", "xGC_per90", "clean_sheet", "bonus", "mins_frac"],
    "DEF": ["xG", "xA", "xGC_per90", "clean_sheet", "influence_norm", "bonus", "mins_frac"],
    "MID": ["xG", "xA", "creativity_norm", "threat_norm", "bonus", "mins_frac"],
    "FWD": ["xG", "xA", "threat_norm", "bonus", "mins_frac"],
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
    """Generate default emission means and vars for a position."""
    n_feat = len(POSITION_FEATURES[position])
    # All-zeros for Injured state
    means = np.zeros((N_STATES, n_feat))
    vars_ = np.full((N_STATES, n_feat), 0.01)

    if position == "GK":
        # [saves_per90, xGC_per90, cs, bonus, mins]
        means[1] = [2.0, 1.5, 0.15, 0.2, 0.75]  # Slump
        means[2] = [3.0, 1.2, 0.30, 0.5, 0.95]  # Average
        means[3] = [3.5, 0.8, 0.45, 1.2, 1.00]  # Good
        means[4] = [4.5, 0.5, 0.55, 2.0, 1.00]  # Star
        vars_[0] = [0.01, 0.01, 0.01, 0.01, 0.001]
        vars_[1] = [1.0, 0.5, 0.10, 0.3, 0.05]
        vars_[2] = [1.5, 0.4, 0.15, 0.5, 0.02]
        vars_[3] = [1.5, 0.3, 0.15, 0.8, 0.01]
        vars_[4] = [2.0, 0.2, 0.15, 1.0, 0.005]
    elif position == "DEF":
        # [xG, xA, xGC_per90, cs, influence_norm, bonus, mins]
        means[1] = [0.02, 0.02, 1.5, 0.15, 0.15, 0.2, 0.75]
        means[2] = [0.04, 0.05, 1.2, 0.30, 0.25, 0.5, 0.95]
        means[3] = [0.08, 0.08, 0.8, 0.40, 0.40, 1.0, 1.00]
        means[4] = [0.15, 0.12, 0.5, 0.50, 0.55, 1.8, 1.00]
        vars_[0] = [0.001] * 7
        vars_[1] = [0.005, 0.005, 0.5, 0.10, 0.05, 0.3, 0.05]
        vars_[2] = [0.010, 0.008, 0.4, 0.15, 0.08, 0.5, 0.02]
        vars_[3] = [0.020, 0.015, 0.3, 0.15, 0.10, 0.8, 0.01]
        vars_[4] = [0.040, 0.025, 0.2, 0.15, 0.12, 1.0, 0.005]
    elif position == "MID":
        # [xG, xA, creativity_norm, threat_norm, bonus, mins]
        means[1] = [0.05, 0.03, 0.10, 0.10, 0.2, 0.75]
        means[2] = [0.15, 0.10, 0.25, 0.25, 0.5, 0.95]
        means[3] = [0.30, 0.18, 0.40, 0.40, 1.2, 1.00]
        means[4] = [0.55, 0.28, 0.60, 0.60, 2.0, 1.00]
        vars_[0] = [0.001] * 6
        vars_[1] = [0.010, 0.005, 0.05, 0.05, 0.3, 0.05]
        vars_[2] = [0.020, 0.012, 0.08, 0.08, 0.5, 0.02]
        vars_[3] = [0.050, 0.025, 0.10, 0.10, 0.8, 0.01]
        vars_[4] = [0.100, 0.050, 0.12, 0.12, 1.0, 0.005]
    else:  # FWD
        # [xG, xA, threat_norm, bonus, mins]
        means[1] = [0.08, 0.03, 0.15, 0.2, 0.75]
        means[2] = [0.25, 0.08, 0.35, 0.5, 0.95]
        means[3] = [0.45, 0.15, 0.55, 1.2, 1.00]
        means[4] = [0.75, 0.25, 0.75, 2.0, 1.00]
        vars_[0] = [0.001] * 5
        vars_[1] = [0.015, 0.005, 0.08, 0.3, 0.05]
        vars_[2] = [0.040, 0.012, 0.10, 0.5, 0.02]
        vars_[3] = [0.080, 0.025, 0.12, 0.8, 0.01]
        vars_[4] = [0.150, 0.050, 0.15, 1.0, 0.005]

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
    feat_names = POSITION_FEATURES.get(position, POSITION_FEATURES["MID"])
    D = len(feat_names)
    features = np.zeros((n, D))

    mins = _safe_col(timeseries, "minutes")
    mins_frac = np.clip(mins / 90.0, 0, 1)

    for i, fname in enumerate(feat_names):
        if fname == "mins_frac":
            features[:, i] = mins_frac
        elif fname == "xG":
            features[:, i] = _safe_col(timeseries, "xG")
        elif fname == "xA":
            features[:, i] = _safe_col(timeseries, "xA")
        elif fname == "bonus":
            features[:, i] = _safe_col(timeseries, "bonus")
        elif fname == "clean_sheet":
            features[:, i] = _safe_col(timeseries, "clean_sheets")
        elif fname == "saves_per90":
            saves = _safe_col(timeseries, "saves")
            # Normalize to per-90: saves / (minutes/90), avoid div-by-zero
            features[:, i] = np.where(mins > 0, saves / np.maximum(mins / 90.0, 0.1), 0)
        elif fname == "xGC_per90":
            xgc = _safe_col(timeseries, "expected_goals_conceded")
            if np.all(xgc == 0):
                xgc = _safe_col(timeseries, "goals_conceded")
            features[:, i] = np.where(mins > 0, xgc / np.maximum(mins / 90.0, 0.1), 0)
        elif fname == "influence_norm":
            features[:, i] = _safe_col(timeseries, "influence") / 100.0
        elif fname == "creativity_norm":
            features[:, i] = _safe_col(timeseries, "creativity") / 100.0
        elif fname == "threat_norm":
            features[:, i] = _safe_col(timeseries, "threat") / 100.0

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
        self.A = transition_matrix.copy() if transition_matrix is not None else DEFAULT_TRANSITION.copy()
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

    def predict_next_points(self, observations: np.ndarray) -> tuple[float, float]:
        """
        Convert predicted features → expected FPL points.

        Uses FPL scoring rules applied to predicted feature rates.
        """
        feat_mean, feat_var, state_dist = self.predict_next_features(observations)
        feat_names = POSITION_FEATURES[self.position]

        # P(plays) from state distribution
        p_plays = 1.0 - state_dist[0]

        # Map feature predictions to point components
        def _get(name):
            if name in feat_names:
                return feat_mean[feat_names.index(name)]
            return 0.0

        def _get_var(name):
            if name in feat_names:
                return feat_var[feat_names.index(name)]
            return 0.0

        # Appearance
        appearance = 2.0 * p_plays

        # Goals
        xg = _get("xG")
        goal_comp = xg * GOAL_PTS[self.position]

        # Assists
        xa = _get("xA")
        assist_comp = xa * ASSIST_PTS

        # Clean sheet
        cs_rate = _get("clean_sheet")
        cs_comp = cs_rate * CS_PTS[self.position]

        # Goals conceded penalty
        xgc = _get("xGC_per90")
        gc_comp = (xgc / 2.0) * GC_PTS[self.position] * p_plays

        # Bonus
        bonus = _get("bonus")

        # Saves (GK)
        saves_comp = 0.0
        if self.position == "GK":
            saves_rate = _get("saves_per90")
            saves_comp = saves_rate / 3.0 * p_plays

        ep = appearance + goal_comp + assist_comp + cs_comp + gc_comp + bonus + saves_comp
        ep = max(0.0, ep)

        # Variance propagation through scoring rules
        gp = GOAL_PTS[self.position]
        var_pts = (
            gp**2 * _get_var("xG")
            + ASSIST_PTS**2 * _get_var("xA")
            + CS_PTS[self.position] ** 2 * _get_var("clean_sheet")
            + _get_var("bonus")
            + 1.0
        )  # residual floor

        return ep, float(var_pts)

    def fit(self, observations: np.ndarray, n_iter: int = 20, tol: float = 1e-4):
        """Baum-Welch EM for multivariate diagonal Gaussian emissions."""
        T = observations.shape[0]
        prev_ll = -np.inf

        for iteration in range(n_iter):
            alpha, scale = self.forward(observations)
            gamma = self.forward_backward(observations)

            # M-step: initial
            self.pi = np.maximum(gamma[0], 1e-10)
            self.pi /= self.pi.sum()

            # M-step: transitions
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                b_next = self._emission_prob_vector(observations[t + 1])
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self._get_A(t + 1)[i, j] * b_next[j]
                xs = xi[t].sum()
                if xs > 0:
                    xi[t] /= xs
            for i in range(self.n_states):
                d = gamma[:-1, i].sum()
                if d > 1e-10:
                    for j in range(self.n_states):
                        self.A[i, j] = xi[:, i, j].sum() / d
                rs = self.A[i].sum()
                if rs > 0:
                    self.A[i] /= rs

            # M-step: emissions
            for s in range(self.n_states):
                w = gamma[:, s]
                ws = w.sum()
                if ws > 1e-10:
                    mu = np.average(observations, axis=0, weights=w)
                    diff = observations - mu
                    var = np.average(diff**2, axis=0, weights=w)
                    self.means[s] = mu
                    self.vars[s] = np.maximum(var, 1e-4)

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
