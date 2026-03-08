"""Hidden Markov Model for player form state inference.

Implements:
- Forward algorithm (online filtering)
- Forward-Backward (offline smoothing)
- Viterbi decoding (most likely state sequence)
- Dynamic transition matrix perturbation (news signal injection)
- Baum-Welch parameter learning (EM)
- One-step-ahead prediction with uncertainty
"""

import logging
from typing import Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# default form states
STATE_NAMES = ["Injured", "Slump", "Average", "Good", "Star"]
N_STATES = len(STATE_NAMES)

# Emission parameters: mean and std of points per state
# Injured players score ~0-1, Stars score ~6-10
DEFAULT_EMISSION_PARAMS = {
    0: (0.5, 0.5),  # Injured:  mean=0.5,  std=0.5
    1: (2.0, 1.0),  # Slump:   mean=2.0,  std=1.0
    2: (4.0, 1.5),  # Average: mean=4.0,  std=1.5
    3: (6.0, 1.5),  # Good:    mean=6.0,  std=1.5
    4: (8.5, 2.0),  # Star:    mean=8.5,  std=2.0
}

# Transition matrix (row = from, col = to)
# Key property: states are "sticky" (high self-transition) with realistic drift
DEFAULT_TRANSITION_MATRIX = np.array([
    # Inj, Slump, Avg, Good, Star
    [0.60, 0.25, 0.10, 0.05, 0.00],  # Injured (likely stays injured or slumps)
    [0.05, 0.50, 0.35, 0.08, 0.02],  # Slump   (drifts toward average)
    [0.02, 0.10, 0.55, 0.25, 0.08],  # Average  (mostly stable)
    [0.02, 0.05, 0.15, 0.55, 0.23],  # Good     (mostly stable)
    [0.01, 0.02, 0.07, 0.30, 0.60],  # Star     (sticky at top)
])

DEFAULT_INITIAL_DIST = np.array([0.05, 0.10, 0.50, 0.25, 0.10])


class HMMInference:
    """
    Hidden Markov Model for discrete player form states.

    Supports dynamic transition matrix perturbation so that external
    signals (news, injuries) can shift state probabilities mid-sequence.

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (N, N), optional
        transition_matrix[i,j] = P(S_{t+1}=j | S_t=i). Rows must sum to 1.
    emission_params : dict, optional
        {state_index: (mean, std)} for Gaussian emissions.
    initial_dist : np.ndarray, shape (N,), optional
        Prior over initial state.
    """

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        emission_params: Optional[dict] = None,
        initial_dist: Optional[np.ndarray] = None,
    ):
        self.transition_matrix = (
            transition_matrix.copy() if transition_matrix is not None else DEFAULT_TRANSITION_MATRIX.copy()
        )
        self.emission_params = emission_params or dict(DEFAULT_EMISSION_PARAMS)
        self.pi = initial_dist.copy() if initial_dist is not None else DEFAULT_INITIAL_DIST.copy()
        self.n_states = len(self.pi)

        # per-timestep transition overrides (for news injection)
        # key: timestep t, Value: modified transition_matrix matrix for that step
        self._transition_overrides: dict[int, np.ndarray] = {}

    def inject_news_perturbation(
        self,
        timestep: int,
        state_boost: dict[int, float],
        confidence: float = 1.0,
    ):
        """
        Perturb transition matrix at a specific timestep based on news.

        For each source state, the transition probability toward boosted
        target states is multiplied by the boost factor (scaled by confidence),
        then the row is renormalized.

        Parameters
        ----------
        timestep : int
            The gameweek at which the perturbation applies.
        state_boost : dict[int, float]
            {target_state: multiplicative_boost}. E.g., {0: 10.0} means
            "10x more likely to transition to Injured."
        confidence : float
            Scales the perturbation. 0 = no effect, 1 = full effect.
        """
        perturbed_transition_matrix = self.transition_matrix.copy()

        for source_state in range(self.n_states):
            for target_state, boost in state_boost.items():
                # scale boost by confidence: effective_boost = 1 + confidence*(boost-1)
                effective_boost = 1.0 + confidence * (boost - 1.0)
                perturbed_transition_matrix[source_state, target_state] *= effective_boost

            # renormalize row
            row_sum = perturbed_transition_matrix[source_state].sum()
            if row_sum > 0:
                perturbed_transition_matrix[source_state] /= row_sum

        self._transition_overrides[timestep] = perturbed_transition_matrix

    def clear_perturbations(self):
        """Remove all per-timestep transition overrides."""
        self._transition_overrides.clear()

    def _get_transition_matrix(self, timestep: int) -> np.ndarray:
        """Get transition matrix for a given timestep (may be perturbed)."""
        return self._transition_overrides.get(timestep, self.transition_matrix)

    def _emission_prob(self, observation: float, state: int) -> float:
        """P(y_t | S_t = state) under Gaussian emission."""
        mu, sigma = self.emission_params[state]
        return norm.pdf(observation, loc=mu, scale=sigma)

    def _emission_vector(self, observation: float) -> np.ndarray:
        """Emission probabilities for all states given one observation."""
        return np.array([self._emission_prob(observation, s) for s in range(self.n_states)])

    def forward(self, observations: np.ndarray):
        """
        Forward algorithm with dynamic transition matrices.

        Parameters
        ----------
        observations : np.ndarray, shape (num_timesteps,)

        Returns
        -------
        forward_messages : np.ndarray, shape (num_timesteps, N)
            Normalized forward messages. forward_messages[t] = P(S_t | y_1:t)
        scale : np.ndarray, shape (num_timesteps,)
            Per-timestep normalization constants.
        """
        num_timesteps = len(observations)
        forward_messages = np.zeros((num_timesteps, self.n_states))
        scale = np.zeros(num_timesteps)

        # t = 0
        b = self._emission_vector(observations[0])
        forward_messages[0] = self.pi * b
        scale[0] = forward_messages[0].sum()
        if scale[0] > 0:
            forward_messages[0] /= scale[0]

        # t = 1..num_timesteps-1
        for t in range(1, num_timesteps):
            transition_matrix_t = self._get_transition_matrix(t)
            b = self._emission_vector(observations[t])
            forward_messages[t] = (forward_messages[t - 1] @ transition_matrix_t) * b
            scale[t] = forward_messages[t].sum()
            if scale[t] > 0:
                forward_messages[t] /= scale[t]

        return forward_messages, scale

    def forward_backward(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute smoothed posteriors P(S_t | y_1:num_timesteps).

        Parameters
        ----------
        observations : np.ndarray, shape (num_timesteps,)

        Returns
        -------
        smoothed_posteriors : np.ndarray, shape (num_timesteps, N)
            smoothed_posteriors[t, s] = P(S_t=s | y_1:num_timesteps)
        """
        num_timesteps = len(observations)
        forward_messages, scale = self.forward(observations)

        backward_messages = np.zeros((num_timesteps, self.n_states))
        backward_messages[num_timesteps - 1] = 1.0

        for t in range(num_timesteps - 2, -1, -1):
            transition_matrix_t_plus_1 = self._get_transition_matrix(t + 1)
            b_next = self._emission_vector(observations[t + 1])
            backward_messages[t] = transition_matrix_t_plus_1 @ (b_next * backward_messages[t + 1])
            if scale[t + 1] > 0:
                backward_messages[t] /= scale[t + 1]

        smoothed_posteriors = forward_messages * backward_messages
        row_sums = smoothed_posteriors.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        smoothed_posteriors /= row_sums

        return smoothed_posteriors

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """
        Most likely state sequence via Viterbi decoding.

        Parameters
        ----------
        observations : np.ndarray, shape (num_timesteps,)

        Returns
        -------
        best_path : np.ndarray of int, shape (num_timesteps,)
        """
        num_timesteps = len(observations)
        log_pi = np.log(self.pi + 1e-300)

        log_probabilities = np.zeros((num_timesteps, self.n_states))
        backpointers = np.zeros((num_timesteps, self.n_states), dtype=int)

        b0 = self._emission_vector(observations[0])
        log_probabilities[0] = log_pi + np.log(b0 + 1e-300)

        for t in range(1, num_timesteps):
            transition_matrix_t = self._get_transition_matrix(t)
            log_transition_matrix_t = np.log(transition_matrix_t + 1e-300)
            b = self._emission_vector(observations[t])
            for s in range(self.n_states):
                candidates = log_probabilities[t - 1] + log_transition_matrix_t[:, s]
                backpointers[t, s] = np.argmax(candidates)
                log_probabilities[t, s] = candidates[backpointers[t, s]] + np.log(b[s] + 1e-300)

        best_path = np.zeros(num_timesteps, dtype=int)
        best_path[num_timesteps - 1] = np.argmax(log_probabilities[num_timesteps - 1])
        for t in range(num_timesteps - 2, -1, -1):
            best_path[t] = backpointers[t + 1, best_path[t + 1]]

        return best_path

    def predict_next(self, observations: np.ndarray) -> tuple[float, float, np.ndarray]:
        """
        Predict next timestep's points distribution.

        Runs forward algorithm, then propagates one step ahead via
        the transition matrix.

        Parameters
        ----------
        observations : np.ndarray, shape (num_timesteps,)

        Returns
        -------
        expected_points : float
            E[Y_{num_timesteps+1} | y_1:num_timesteps]
        variance : float
            Var[Y_{num_timesteps+1} | y_1:num_timesteps] (from law of total variance)
        next_state_dist : np.ndarray, shape (N,)
            P(S_{num_timesteps+1} | y_1:num_timesteps)
        """
        forward_messages, _ = self.forward(observations)
        current_belief = forward_messages[-1]  # P(S_num_timesteps | y_1:num_timesteps)

        num_timesteps = len(observations)
        next_transition_matrix = self._get_transition_matrix(num_timesteps)  # transition for next step
        next_state_dist = (
            current_belief @ next_transition_matrix
        )  # P(S_{num_timesteps+1} | y_1:num_timesteps)

        state_means = np.array([self.emission_params[s][0] for s in range(self.n_states)])
        state_vars = np.array([self.emission_params[s][1] ** 2 for s in range(self.n_states)])

        expected_points = next_state_dist @ state_means

        # law of total variance: Var = E[Var|S] + Var[E|S]
        variance = next_state_dist @ state_vars + next_state_dist @ (state_means**2) - expected_points**2

        return expected_points, max(0.0, variance), next_state_dist

    def fit(
        self,
        observations: np.ndarray,
        n_iter: int = 20,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        """
        Learn transition matrix and emission parameters via Baum-Welch EM.

        Parameters
        ----------
        observations : np.ndarray, shape (num_timesteps,)
            Training sequence.
        n_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance on log-likelihood.
        verbose : bool
            Print progress.

        Returns
        -------
        self
        """
        num_timesteps = len(observations)
        prev_log_likelihood = -np.inf

        for iteration in range(n_iter):
            # E-step
            forward_messages, scale = self.forward(observations)
            smoothed_posteriors = self.forward_backward(observations)

            # transition_posteriors: P(S_t=i, S_{t+1}=j | y_1:num_timesteps) for transition re-estimation
            transition_posteriors = np.zeros((num_timesteps - 1, self.n_states, self.n_states))
            for t in range(num_timesteps - 1):
                transition_matrix_t_plus_1 = self._get_transition_matrix(t + 1)
                b_next = self._emission_vector(observations[t + 1])

                # Beta from backward (recompute minimal)
                # Use smoothed_posteriors and forward_messages to derive transition_posteriors directly
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        transition_posteriors[t, i, j] = (
                            forward_messages[t, i] * transition_matrix_t_plus_1[i, j] * b_next[j]
                        )

                xi_sum = transition_posteriors[t].sum()
                if xi_sum > 0:
                    transition_posteriors[t] /= xi_sum

            # M-step
            # Re-estimate initial distribution
            self.pi = smoothed_posteriors[0]

            # Re-estimate transition matrix
            for i in range(self.n_states):
                denom = smoothed_posteriors[:-1, i].sum()
                if denom > 0:
                    for j in range(self.n_states):
                        self.transition_matrix[i, j] = transition_posteriors[:, i, j].sum() / denom
                # Renormalize
                row_sum = self.transition_matrix[i].sum()
                if row_sum > 0:
                    self.transition_matrix[i] /= row_sum

            # re-estimate emission parameters
            for s in range(self.n_states):
                weights = smoothed_posteriors[:, s]
                w_sum = weights.sum()
                if w_sum > 1e-10:
                    mu = np.average(observations, weights=weights)
                    var = np.average((observations - mu) ** 2, weights=weights)
                    sigma = max(np.sqrt(var), 0.1)  # floor to prevent collapse
                    self.emission_params[s] = (mu, sigma)

            # log-likelihood
            log_likelihood = np.sum(np.log(scale + 1e-300))
            if verbose:
                logger.info("EM iteration %d: LL = %.4f", iteration, log_likelihood)

            if abs(log_likelihood - prev_log_likelihood) < tol:
                if verbose:
                    logger.info("Converged at iteration %d", iteration)
                break
            prev_log_likelihood = log_likelihood

        return self


__all__ = [
    "HMMInference",
    "STATE_NAMES",
    "N_STATES",
    "DEFAULT_EMISSION_PARAMS",
    "DEFAULT_TRANSITION_MATRIX",
    "DEFAULT_INITIAL_DIST",
]
