"""Fusion of HMM and Kalman Filter outputs.

Combines discrete state posteriors (HMM) with continuous estimates (Kalman)
using inverse-variance weighting — optimal under Gaussian independence.
"""

import numpy as np


def fuse_estimates(
    hmm_mean: float,
    hmm_var: float,
    kf_mean: float,
    kf_var: float,
) -> tuple[float, float]:
    """
    Fuse a single HMM estimate with a single Kalman estimate.

    Uses inverse-variance weighting:
        fused_mean = (hmm_mean/hmm_var + kf_mean/kf_var) / (1/hmm_var + 1/kf_var)
        fused_var  = 1 / (1/hmm_var + 1/kf_var)

    Parameters
    ----------
    hmm_mean : float
        HMM expected points (from state posterior weighted emission means).
    hmm_var : float
        HMM variance (law of total variance over state posterior).
    kf_mean : float
        Kalman filtered point estimate.
    kf_var : float
        Kalman filtered uncertainty (posterior variance).

    Returns
    -------
    fused_mean : float
    fused_var : float
    """
    hmm_var = max(hmm_var, 1e-6)
    kf_var = max(kf_var, 1e-6)

    precision_hmm = 1.0 / hmm_var
    precision_kf = 1.0 / kf_var
    total_precision = precision_hmm + precision_kf

    fused_mean = (precision_hmm * hmm_mean + precision_kf * kf_mean) / total_precision
    fused_var = 1.0 / total_precision

    return fused_mean, fused_var


def fuse_sequences(
    hmm_gamma: np.ndarray,
    kalman_x: np.ndarray,
    kalman_P: np.ndarray,
    emission_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse full sequences of HMM posteriors and Kalman estimates.

    Parameters
    ----------
    hmm_gamma : np.ndarray, shape (T, N)
        Smoothed state posteriors from HMM.
    kalman_x : np.ndarray, shape (T,)
        Kalman filtered estimates.
    kalman_P : np.ndarray, shape (T,)
        Kalman filtered uncertainties.
    emission_params : dict
        {state_index: (mean, std)} from HMM.

    Returns
    -------
    fused_mean : np.ndarray, shape (T,)
    fused_var : np.ndarray, shape (T,)
    """
    T = len(kalman_x)
    n_states = hmm_gamma.shape[1]

    state_means = np.array([emission_params[s][0] for s in range(n_states)])
    state_vars = np.array([emission_params[s][1] ** 2 for s in range(n_states)])

    # HMM expected value and variance per timestep
    hmm_mean = hmm_gamma @ state_means
    hmm_var = (
        hmm_gamma @ state_vars
        + hmm_gamma @ (state_means ** 2)
        - hmm_mean ** 2
    )
    hmm_var = np.maximum(hmm_var, 1e-6)
    kalman_P_safe = np.maximum(kalman_P, 1e-6)

    # Inverse-variance weighting
    precision_hmm = 1.0 / hmm_var
    precision_kf = 1.0 / kalman_P_safe
    total_precision = precision_hmm + precision_kf

    fused_mean = (precision_hmm * hmm_mean + precision_kf * kalman_x) / total_precision
    fused_var = 1.0 / total_precision

    return fused_mean, fused_var


__all__ = ["fuse_estimates", "fuse_sequences"]
