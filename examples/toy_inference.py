"""
FPLX Inference Toy Example
===========================
Demonstrates the core inference pipeline from the FPLX proposal:
1. Synthetic data generation with known ground-truth latent states
2. HMM inference (Forward-Backward + Viterbi) for discrete form states
3. Kalman Filter for continuous point potential tracking
4. Fusion of HMM + Kalman outputs
5. Comparison against baseline (rolling average)

The synthetic player experiences: stable form to hot streak to injury to recovery
This tests whether inference can detect regime shifts in noisy observations.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

np.random.seed(42)


# GENERATIVE MODEL (Ground Truth)
# States: 0=Injured, 1=Slump, 2=Average, 3=Good, 4=Star
STATE_NAMES = ["Injured", "Slump", "Average", "Good", "Star"]
N_STATES = len(STATE_NAMES)

# Emission parameters: mean and std of points per state
# Injured players score ~0-1, Stars score ~6-10
EMISSION_PARAMS = {
    0: (0.5, 0.5),  # Injured:  mean=0.5,  std=0.5
    1: (2.0, 1.0),  # Slump:   mean=2.0,  std=1.0
    2: (4.0, 1.5),  # Average: mean=4.0,  std=1.5
    3: (6.0, 1.5),  # Good:    mean=6.0,  std=1.5
    4: (8.5, 2.0),  # Star:    mean=8.5,  std=2.0
}

# Transition matrix (row = from, col = to)
# Key property: states are "sticky" (high self-transition) with realistic drift
TRANSITION_MATRIX = np.array([
    # Inj, Slump, Avg, Good, Star
    [0.60, 0.25, 0.10, 0.05, 0.00],  # Injured (likely stays injured or slumps)
    [0.05, 0.50, 0.35, 0.08, 0.02],  # Slump   (drifts toward average)
    [0.02, 0.10, 0.55, 0.25, 0.08],  # Average  (mostly stable)
    [0.02, 0.05, 0.15, 0.55, 0.23],  # Good     (mostly stable)
    [0.01, 0.02, 0.07, 0.30, 0.60],  # Star     (sticky at top)
])

# Initial state distribution
INITIAL_DIST = np.array([0.05, 0.10, 0.50, 0.25, 0.10])


def generate_synthetic_player(n_weeks=38, shock_week=20, shock_state=0):
    """
    Generate synthetic FPL player data with a forced regime shift.

    For the first `shock_week` weeks, states evolve naturally via the transition matrix.
    At `shock_week`, the player is forced into `shock_state` (e.g., Injured).
    Then states evolve naturally again (recovery).

    Returns
    -------
    true_states : np.ndarray of int, shape (n_weeks,)
    observations : np.ndarray of float, shape (n_weeks,)
    """
    true_states = np.zeros(n_weeks, dtype=int)
    observations = np.zeros(n_weeks)

    # sample initial state
    true_states[0] = np.random.choice(N_STATES, p=INITIAL_DIST)

    for t in range(1, n_weeks):
        if t == shock_week:
            # force a regime shift (e.g., injury)
            true_states[t] = shock_state
        else:
            true_states[t] = np.random.choice(
                N_STATES, p=TRANSITION_MATRIX[true_states[t - 1]]
            )

    # generate observations from emission model
    for t in range(n_weeks):
        mu, sigma = EMISSION_PARAMS[true_states[t]]
        observations[t] = max(0, np.random.normal(mu, sigma))  # Points >= 0

    return true_states, observations


# HMM INFERENCE: Forward-Backward + Viterbi
class HMMInference:
    """
    Hidden Markov Model inference for discrete form states.

    Implements:
    - Forward algorithm (filtering: P(S_t | y_1:t))
    - Forward-Backward algorithm (smoothing: P(S_t | y_1:T))
    - Viterbi algorithm (most likely state sequence)
    """

    def __init__(self, transition_matrix, emission_params, initial_dist):
        """
        Parameters
        ----------
        transition_matrix : np.ndarray, shape (N, N)
            A[i,j] = P(S_{t+1}=j | S_t=i)
        emission_params : dict
            {state: (mean, std)} for Gaussian emissions
        initial_dist : np.ndarray, shape (N,)
            Prior over initial state
        """
        self.A = transition_matrix
        self.emission_params = emission_params
        self.pi = initial_dist
        self.n_states = len(initial_dist)

    def _emission_prob(self, observation, state):
        """P(y_t | S_t = state) under Gaussian emission."""
        mu, sigma = self.emission_params[state]
        return norm.pdf(observation, loc=mu, scale=sigma)

    def _emission_vector(self, observation):
        """Emission probabilities for all states given one observation."""
        return np.array([
            self._emission_prob(observation, s) for s in range(self.n_states)
        ])

    def forward(self, observations):
        """
        Forward algorithm: compute filtered beliefs P(S_t | y_1:t).

        Returns
        -------
        alpha : np.ndarray, shape (T, N)
            alpha[t, s] = P(S_t=s, y_1:t) (unnormalized)
        scale : np.ndarray, shape (T,)
            Normalization constants (for numerical stability)
        """
        time_steps = len(observations)
        alpha = np.zeros((time_steps, self.n_states))
        scale = np.zeros(time_steps)

        # t=0
        b = self._emission_vector(observations[0])
        alpha[0] = self.pi * b
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]

        # t=1..T-1
        for t in range(1, time_steps):
            b = self._emission_vector(observations[t])
            alpha[t] = (alpha[t - 1] @ self.A) * b
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        return alpha, scale

    def forward_backward(self, observations):
        """
        Forward-Backward algorithm: compute smoothed beliefs P(S_t | y_1:T).

        Returns
        -------
        gamma : np.ndarray, shape (T, N)
            gamma[t, s] = P(S_t=s | y_1:T)
        """
        time_steps = len(observations)
        alpha, scale = self.forward(observations)

        # Backward pass
        beta = np.zeros((time_steps, self.n_states))
        beta[time_steps - 1] = 1.0

        for t in range(time_steps - 2, -1, -1):
            b_next = self._emission_vector(observations[t + 1])
            beta[t] = self.A @ (b_next * beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        # Combine: gamma[t] = alpha[t] * beta[t], then normalize
        gamma = alpha * beta
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        gamma /= row_sums

        return gamma

    def viterbi(self, observations):
        """
        Viterbi algorithm: most likely state sequence.

        Returns
        -------
        best_path : np.ndarray of int, shape (T,)
        """
        time_steps = len(observations)
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # delta[t, s] = max log-prob of any path ending in state s at time t
        delta = np.zeros((time_steps, self.n_states))
        psi = np.zeros((time_steps, self.n_states), dtype=int)

        b0 = self._emission_vector(observations[0])
        delta[0] = log_pi + np.log(b0 + 1e-300)

        for t in range(1, time_steps):
            b = self._emission_vector(observations[t])
            for s in range(self.n_states):
                candidates = delta[t - 1] + log_A[:, s]
                psi[t, s] = np.argmax(candidates)
                delta[t, s] = candidates[psi[t, s]] + np.log(b[s] + 1e-300)

        # Backtrack
        best_path = np.zeros(time_steps, dtype=int)
        best_path[time_steps - 1] = np.argmax(delta[time_steps - 1])
        for t in range(time_steps - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]

        return best_path



# KALMAN FILTER for continuous point potential
class KalmanFilter:
    """
    1D Kalman Filter for tracking a player's true point potential.

    State model:    x_{t+1} = x_t + w_t,    w_t ~ N(0, Q)
    Observation:    y_t     = x_t + v_t,     v_t ~ N(0, R)

    Parameters
    ----------
    Q : float
        Process noise variance (how fast true form drifts)
    R : float
        Observation noise variance (how noisy weekly points are)
    x0 : float
        Initial state estimate
    P0 : float
        Initial state uncertainty (variance)
    """

    def __init__(self, Q=1.0, R=4.0, x0=4.0, P0=2.0):
        self.Q = Q
        self.R = R
        self.x0 = x0
        self.P0 = P0

    def filter(self, observations):
        """
        Run Kalman filter on a sequence of observations.

        Returns
        -------
        x_est : np.ndarray, shape (T,)
            Filtered state estimates (posterior mean)
        P_est : np.ndarray, shape (T,)
            Filtered state uncertainties (posterior variance)
        """
        T = len(observations)
        x_est = np.zeros(T)
        P_est = np.zeros(T)

        # initialize
        x_pred = self.x0
        P_pred = self.P0

        for t in range(T):
            # predict step
            # x_{t|t-1} = x_{t-1|t-1}  (random walk model)
            # P_{t|t-1} = P_{t-1|t-1} + Q
            if t > 0:
                x_pred = x_est[t - 1]
                P_pred = P_est[t - 1] + self.Q

            # Update step
            y = observations[t]
            innovation = y - x_pred
            S = P_pred + self.R  # Innovation covariance
            K = P_pred / S  # Kalman gain

            x_est[t] = x_pred + K * innovation
            P_est[t] = (1 - K) * P_pred

        return x_est, P_est



#Combined HMM + Kalman
def fuse_predictions(hmm_gamma, kalman_x, kalman_P, emission_params):
    """
    Fuse HMM smoothed posteriors with Kalman filtered estimates.

    Strategy: Compute HMM-weighted expected points (from state posteriors),
    then combine with Kalman estimate using inverse-variance weighting.

    Parameters
    ----------
    hmm_gamma : np.ndarray, shape (T, N)
        Smoothed state posteriors from HMM
    kalman_x : np.ndarray, shape (T,)
        Kalman filtered point estimates
    kalman_P : np.ndarray, shape (T,)
        Kalman filtered uncertainties
    emission_params : dict
        {state: (mean, std)}

    Returns
    -------
    fused_mean : np.ndarray, shape (T,)
    fused_var : np.ndarray, shape (T,)
    """
    time_steps = len(kalman_x)
    n_states = hmm_gamma.shape[1]

    # HMM expected points: sum_s P(S_t=s|data) * mu_s
    state_means = np.array([emission_params[s][0] for s in range(n_states)])
    state_vars = np.array([emission_params[s][1] ** 2 for s in range(n_states)])

    hmm_mean = hmm_gamma @ state_means
    # HMM variance: law of total variance
    hmm_var = hmm_gamma @ state_vars + hmm_gamma @ (state_means**2) - hmm_mean**2
    hmm_var = np.maximum(hmm_var, 0.01)  # floor

    # Inverse-variance weighting (optimal linear combination of two Gaussians)
    w_kalman = hmm_var / (hmm_var + kalman_P)
    w_hmm = kalman_P / (hmm_var + kalman_P)

    fused_mean = w_kalman * kalman_x + w_hmm * hmm_mean
    fused_var = 1.0 / (1.0 / hmm_var + 1.0 / kalman_P)

    return fused_mean, fused_var

# BASELINE: Rolling average
def rolling_average(observations, window=5):
    """Simple rolling average baseline."""
    time_steps = len(observations)
    predictions = np.zeros(time_steps)
    for t in range(time_steps):
        start = max(0, t - window + 1)
        predictions[t] = observations[start : t + 1].mean()
    return predictions

def run_experiment():
    """Run full inference pipeline on synthetic data and visualize results."""
    # Generate synthetic player: 38 weeks, injury at week 20
    N_WEEKS = 38
    SHOCK_WEEK = 20
    true_states, observations = generate_synthetic_player(
        n_weeks=N_WEEKS,
        shock_week=SHOCK_WEEK,
        shock_state=0,  # 0 = Injured
    )
    print("=" * 70)
    print("FPLX Inference Simulated Example")
    print("=" * 70)
    print(f"Generated {N_WEEKS} weeks of synthetic data.")
    print(f"Forced injury at week {SHOCK_WEEK}.")
    print(f"True states: {[STATE_NAMES[s] for s in true_states]}")
    print(f"Observations (points): {np.round(observations, 1)}")
    print()

    # HMM Inference
    hmm = HMMInference(TRANSITION_MATRIX, EMISSION_PARAMS, INITIAL_DIST)

    # filtering (online: only uses past data)
    alpha, scale = hmm.forward(observations)
    hmm_filtered_states = np.argmax(alpha, axis=1)

    # smoothing (offline: uses all data)
    gamma = hmm.forward_backward(observations)
    hmm_smoothed_states = np.argmax(gamma, axis=1)

    # viterbi (most likely path)
    viterbi_path = hmm.viterbi(observations)

    # Kalman Filter
    kf = KalmanFilter(Q=1.0, R=4.0, x0=4.0, P0=2.0)
    kalman_x, kalman_P = kf.filter(observations)

    # fusion
    fused_mean, fused_var = fuse_predictions(gamma, kalman_x, kalman_P, EMISSION_PARAMS)
    fused_std = np.sqrt(fused_var)

    # Baseline
    baseline = rolling_average(observations, window=5)

    # Metrics
    # Ground-truth expected points per state
    true_expected = np.array([EMISSION_PARAMS[s][0] for s in true_states])

    mse_baseline = np.mean((baseline - true_expected) ** 2)
    mse_kalman = np.mean((kalman_x - true_expected) ** 2)
    mse_fused = np.mean((fused_mean - true_expected) ** 2)

    # HMM state recovery accuracy
    acc_filtered = np.mean(hmm_filtered_states == true_states)
    acc_smoothed = np.mean(hmm_smoothed_states == true_states)
    acc_viterbi = np.mean(viterbi_path == true_states)

    print("--- Prediction MSE (vs true expected points per state) ---")
    print(f"  Rolling Average (baseline):  {mse_baseline:.3f}")
    print(f"  Kalman Filter:               {mse_kalman:.3f}")
    print(f"  Fused (HMM+Kalman):          {mse_fused:.3f}")
    print()
    print("--- HMM State Recovery Accuracy ---")
    print(f"  Forward (filtered):   {acc_filtered:.1%}")
    print(f"  Forward-Backward:     {acc_smoothed:.1%}")
    print(f"  Viterbi:              {acc_viterbi:.1%}")
    print()

    # visualization
    weeks = np.arange(1, N_WEEKS + 1)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        "FPLX Inference Pipeline — Synthetic Player (Injury at GW20)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # panel 1: Observations + True expected + Predictions
    ax1 = axes[0]
    ax1.scatter(
        weeks,
        observations,
        color="#888",
        alpha=0.6,
        s=30,
        label="Observed points",
        zorder=3,
    )
    ax1.plot(
        weeks, true_expected, "k--", lw=1.5, label="True expected (oracle)", zorder=2
    )
    ax1.plot(
        weeks,
        baseline,
        color="#e74c3c",
        lw=1.2,
        alpha=0.8,
        label=f"Rolling avg (window=5)",
    )
    ax1.plot(weeks, kalman_x, color="#2980b9", lw=1.5, label="Kalman filtered")
    ax1.fill_between(
        weeks,
        kalman_x - 1.96 * np.sqrt(kalman_P),
        kalman_x + 1.96 * np.sqrt(kalman_P),
        alpha=0.15,
        color="#2980b9",
        label="Kalman 95% CI",
    )
    ax1.plot(weeks, fused_mean, color="#27ae60", lw=2, label="Fused (HMM+Kalman)")
    ax1.fill_between(
        weeks,
        fused_mean - 1.96 * fused_std,
        fused_mean + 1.96 * fused_std,
        alpha=0.15,
        color="#27ae60",
        label="Fused 95% CI",
    )
    ax1.axvline(SHOCK_WEEK + 1, color="red", ls=":", alpha=0.5, label="Injury shock")
    ax1.set_ylabel("Points")
    ax1.set_title("Point Estimates vs. Ground Truth")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.set_ylim(-1, 14)
    ax1.grid(True, alpha=0.3)

    # panel 2: True states vs Viterbi path
    ax2 = axes[1]
    ax2.step(weeks, true_states, where="mid", color="black", lw=2, label="True state")
    ax2.step(
        weeks,
        viterbi_path,
        where="mid",
        color="#8e44ad",
        lw=1.5,
        ls="--",
        label="Viterbi path",
    )
    ax2.step(
        weeks,
        hmm_filtered_states,
        where="mid",
        color="#2980b9",
        lw=1,
        ls=":",
        alpha=0.7,
        label="Forward (filtered)",
    )
    ax2.set_ylabel("State")
    ax2.set_yticks(range(N_STATES))
    ax2.set_yticklabels(STATE_NAMES)
    ax2.set_title("HMM State Recovery")
    ax2.axvline(SHOCK_WEEK + 1, color="red", ls=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # panel 3: HMM smoothed posteriors (heatmap)
    ax3 = axes[2]
    im = ax3.imshow(
        gamma.T,
        aspect="auto",
        cmap="YlOrRd",
        extent=[0.5, N_WEEKS + 0.5, -0.5, N_STATES - 0.5],
        origin="lower",
        interpolation="nearest",
    )
    ax3.set_ylabel("State")
    ax3.set_yticks(range(N_STATES))
    ax3.set_yticklabels(STATE_NAMES)
    ax3.set_title("HMM Smoothed Posteriors P(S_t | all data)")
    ax3.axvline(SHOCK_WEEK + 1, color="white", ls=":", lw=2)
    plt.colorbar(im, ax=ax3, label="Probability", shrink=0.8)

    # panel 4: Uncertainty comparison
    ax4 = axes[3]
    ax4.plot(weeks, np.sqrt(kalman_P), color="#2980b9", lw=1.5, label="Kalman std")
    ax4.plot(weeks, fused_std, color="#27ae60", lw=2, label="Fused std")
    # HMM entropy as uncertainty measure
    hmm_entropy = -np.sum(gamma * np.log(gamma + 1e-300), axis=1) / np.log(N_STATES)
    ax4.plot(
        weeks,
        hmm_entropy * 3,
        color="#8e44ad",
        lw=1.5,
        ls="--",
        label="HMM entropy (scaled)",
    )
    ax4.axvline(SHOCK_WEEK + 1, color="red", ls=":", alpha=0.5, label="Injury shock")
    ax4.set_xlabel("Gameweek")
    ax4.set_ylabel("Uncertainty")
    ax4.set_title("Uncertainty Quantification")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("fplx_inference_toy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved to fplx_inference_toy.png")

    # summary table
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Baseline':>10} {'Kalman':>10} {'Fused':>10}")
    print("-" * 70)
    print(
        f"{'MSE vs true expected':<35} {mse_baseline:>10.3f} {mse_kalman:>10.3f} "
        f"{mse_fused:>10.3f}"
    )

    # MSE around shock window (weeks 18-25)
    shock_slice = slice(17, 25)
    mse_b_shock = np.mean((baseline[shock_slice] - true_expected[shock_slice]) ** 2)
    mse_k_shock = np.mean((kalman_x[shock_slice] - true_expected[shock_slice]) ** 2)
    mse_f_shock = np.mean((fused_mean[shock_slice] - true_expected[shock_slice]) ** 2)
    print(
        f"{'MSE around shock (GW18-25)':<35} {mse_b_shock:>10.3f} {mse_k_shock:>10.3f} "
        f"{mse_f_shock:>10.3f}"
    )

    print("-" * 70)
    print(f"{'HMM Accuracy (Viterbi)':<35} {acc_viterbi:>10.1%}")
    print(f"{'HMM Accuracy (Smoothed)':<35} {acc_smoothed:>10.1%}")
    print("=" * 70)

    return {
        "observations": observations,
        "true_states": true_states,
        "true_expected": true_expected,
        "kalman_x": kalman_x,
        "kalman_P": kalman_P,
        "fused_mean": fused_mean,
        "fused_var": fused_var,
        "gamma": gamma,
        "viterbi_path": viterbi_path,
        "baseline": baseline,
    }


if __name__ == "__main__":
    results = run_experiment()
