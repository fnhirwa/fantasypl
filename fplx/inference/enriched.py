"""Feature-enriched prediction using vaastav dataset fields.

Decomposes points into underlying components predicted from stable rates:
  E[P] = avail * (appearance + xG*goal_pts + xA*3 + cs*cs_pts + bonus + saves/3 + gc_penalty)

Each component uses EWMA on the underlying rate, which is less noisy
than EWMA on total_points.
"""

import numpy as np
import pandas as pd

GOAL_PTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
GC_PTS = {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0}
ASSIST_PTS = 3


def _ewma(values: np.ndarray, alpha: float = 0.3) -> float:
    if len(values) == 0:
        return 0.0
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return float(result)


def _safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(0).values


def compute_xpoints(timeseries: pd.DataFrame, position: str) -> np.ndarray:
    """Compute per-GW expected points from underlying rates."""
    n = len(timeseries)
    if n == 0:
        return np.array([])

    mins = _safe_col(timeseries, "minutes")
    played = mins > 0
    appearance = np.where(mins >= 60, 2.0, np.where(played, 1.0, 0.0))

    xg = _safe_col(timeseries, "xG")
    if np.all(xg == 0):
        xg = _safe_col(timeseries, "goals")
    goal_comp = xg * GOAL_PTS.get(position, 4)

    xa = _safe_col(timeseries, "xA")
    if np.all(xa == 0):
        xa = _safe_col(timeseries, "assists")
    assist_comp = xa * ASSIST_PTS

    cs = _safe_col(timeseries, "clean_sheets")
    cs_comp = cs * CS_PTS.get(position, 0)

    gc = _safe_col(timeseries, "goals_conceded")
    gc_comp = np.floor(gc / 2.0) * GC_PTS.get(position, 0)

    bonus_comp = _safe_col(timeseries, "bonus")

    saves_comp = np.zeros(n)
    if position == "GK":
        saves_comp = np.floor(_safe_col(timeseries, "saves") / 3.0)

    return (appearance + goal_comp + assist_comp + cs_comp + gc_comp + bonus_comp + saves_comp) * played


def enriched_predict(
    timeseries: pd.DataFrame,
    position: str,
    alpha: float = 0.3,
    lookback: int = 10,
) -> tuple[float, float]:
    """
    Predict expected points from underlying performance rates.

    Returns (expected_points, variance).
    """
    if timeseries.empty or "minutes" not in timeseries.columns:
        return 0.0, 4.0

    ts = timeseries.tail(lookback).copy()
    mins = _safe_col(ts, "minutes")
    played_mask = mins > 0
    n_played = played_mask.sum()

    if n_played < 2:
        return 0.0, 4.0

    avail = float(played_mask[-min(3, len(played_mask)) :].mean())
    if avail < 0.1:
        return 0.0, 1.0

    xpts = compute_xpoints(ts, position)
    played_xpts = xpts[played_mask]
    conditional_ep = max(0.0, _ewma(played_xpts, alpha))

    # Variance from residuals
    if "points" in ts.columns:
        pts = _safe_col(ts, "points")
        played_pts = pts[played_mask]
        residuals = played_pts - played_xpts
        var_estimate = float(np.var(residuals)) + 1.0
    else:
        var_estimate = 4.0

    ep = conditional_ep * avail
    var_out = avail * var_estimate + avail * (1 - avail) * conditional_ep**2
    return ep, var_out


def batch_enriched_predict(players, alpha: float = 0.3):
    ep, ev = {}, {}
    for p in players:
        mu, var = enriched_predict(p.timeseries, p.position, alpha=alpha)
        ep[p.id] = mu
        ev[p.id] = var
    return ep, ev


__all__ = ["enriched_predict", "batch_enriched_predict", "compute_xpoints"]
