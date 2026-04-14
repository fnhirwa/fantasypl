"""Feature-enriched prediction using xG, xA, BPS, clean sheets, saves.

Two modes:
  1. enriched_predict(): single-number prediction for squad selection
  2. compute_xpoints(): per-gameweek "expected points" series for HMM/KF input

xPoints replaces raw points as the HMM/KF observation signal. Instead of
seeing [5, 0, 2, 8], the HMM sees [3.2, 0, 4.1, 4.8] — what points SHOULD
have been based on underlying performance. This removes outcome noise
(lucky/unlucky finishes) so the HMM tracks true form, not luck.
"""

import numpy as np
import pandas as pd

# FPL scoring rules (exact)
GOAL_PTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_PTS = 3
GC_PTS = {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0}  # per 2 conceded


def _ewma(values: np.ndarray, alpha: float = 0.3) -> float:
    if len(values) == 0:
        return 0.0
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return float(result)


def _safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """Get column as float array, filling missing with 0."""
    if col not in df.columns:
        return np.zeros(len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(0).values


def compute_xpoints(timeseries: pd.DataFrame, position: str) -> np.ndarray:
    """
    Compute per-gameweek expected points from underlying performance metrics.

    For each GW, computes what a player's points SHOULD have been based on:
      - xG (not actual goals) × position goal value
      - xA (not actual assists) × 3
      - clean sheet (actual, since there's no xCS)
      - bonus (actual)
      - saves (GK)
      - appearance points from minutes
      - goals conceded penalty

    This is the denoised signal the HMM should track instead of raw points.

    Parameters
    ----------
    timeseries : pd.DataFrame
        Must contain 'minutes'. Uses xG, xA, clean_sheets, bonus, etc.
    position : str
        GK, DEF, MID, FWD.

    Returns
    -------
    np.ndarray
        xPoints per gameweek (same length as timeseries).
    """
    n = len(timeseries)
    if n == 0:
        return np.array([])

    mins = _safe_col(timeseries, "minutes")
    played = mins > 0

    # Appearance points
    appearance = np.where(mins >= 60, 2.0, np.where(played, 1.0, 0.0))

    # Goals component: use xG if available, else actual goals
    xg = _safe_col(timeseries, "xG")
    if np.all(xg == 0) and "goals" in timeseries.columns:
        xg = _safe_col(timeseries, "goals")
    goal_comp = xg * GOAL_PTS.get(position, 4)

    # Assists component: use xA if available, else actual assists
    xa = _safe_col(timeseries, "xA")
    if np.all(xa == 0) and "assists" in timeseries.columns:
        xa = _safe_col(timeseries, "assists")
    assist_comp = xa * ASSIST_PTS

    # Clean sheet component
    cs = _safe_col(timeseries, "clean_sheets")
    cs_comp = cs * CS_PTS.get(position, 0)

    # Goals conceded penalty
    gc = _safe_col(timeseries, "goals_conceded")
    gc_comp = np.floor(gc / 2.0) * GC_PTS.get(position, 0)

    # Bonus
    bonus_comp = _safe_col(timeseries, "bonus")

    # Saves (GK only)
    saves_comp = np.zeros(n)
    if position == "GK":
        saves = _safe_col(timeseries, "saves")
        saves_comp = np.floor(saves / 3.0)  # 1pt per 3 saves

    # Combine
    xpts = appearance + goal_comp + assist_comp + cs_comp + gc_comp + bonus_comp + saves_comp

    # Zero out GWs where player didn't play (no points if no minutes)

    return xpts * played


def enriched_predict(
    timeseries: pd.DataFrame,
    position: str,
    alpha: float = 0.3,
    lookback: int = 10,
) -> tuple[float, float]:
    """
    Predict expected points from underlying performance metrics.

    Parameters
    ----------
    timeseries : pd.DataFrame
    position : str
    alpha : float
        EWMA decay.
    lookback : int
        Max recent GWs.

    Returns
    -------
    expected_points : float
    variance_estimate : float
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

    # Compute xPoints for the window
    xpts = compute_xpoints(ts, position)

    # EWMA on xPoints from played GWs only
    played_xpts = xpts[played_mask]
    conditional_ep = _ewma(played_xpts, alpha)
    conditional_ep = max(0.0, conditional_ep)

    # Variance from residuals of xPoints vs actual points
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
    """Run enriched prediction for all players."""
    ep, ev = {}, {}
    for p in players:
        mu, var = enriched_predict(p.timeseries, p.position, alpha=alpha)
        ep[p.id] = mu
        ev[p.id] = var
    return ep, ev


__all__ = ["enriched_predict", "batch_enriched_predict", "compute_xpoints"]
