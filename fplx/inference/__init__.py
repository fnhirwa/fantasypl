"""Probabilistic inference modules for FPLX."""

from fplx.inference.enriched import batch_enriched_predict, compute_xpoints, enriched_predict
from fplx.inference.fusion import fuse_estimates, fuse_sequences
from fplx.inference.hmm import HMMInference
from fplx.inference.kalman import KalmanFilter
from fplx.inference.multivariate_hmm import MultivariateHMM, build_feature_matrix
from fplx.inference.pipeline import InferenceResult, PlayerInferencePipeline

__all__ = [
    "HMMInference",
    "KalmanFilter",
    "fuse_estimates",
    "fuse_sequences",
    "PlayerInferencePipeline",
    "InferenceResult",
    "enriched_predict",
    "batch_enriched_predict",
    "compute_xpoints",
    "MultivariateHMM",
    "build_feature_matrix",
]
