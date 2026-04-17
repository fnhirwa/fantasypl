"""Fixture-aware enriched prediction with semi-variance for downside risk.

Improvements over base enriched:
  - Cards, own goals, penalties (negative pts previously unmodeled)
  - Home/away adjustment from player history
  - Opponent strength adjustment from player history
  - Ensemble with FPL's xP when available
  - Semi-variance: only penalize downside deviation below E[P]
  - Longer lookback with exponential decay (more data, recency bias)
"""

import numpy as np
import pandas as pd

GOAL_PTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
GC_PTS = {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0}
ASSIST_PTS = 3


def _ewma(values, alpha=0.3):
    if len(values) == 0:
        return 0.0
    r = float(values[0])
    for v in values[1:]:
        r = alpha * float(v) + (1 - alpha) * r
    return r


def _ewma_decay(values, alpha=0.3):
    """EWMA that returns per-element weights for variance computation."""
    n = len(values)
    if n == 0:
        return 0.0, np.array([])
    weights = np.array([(1 - alpha) ** (n - 1 - i) * alpha for i in range(n)])
    weights[0] += (1 - alpha) ** n  # initial weight
    weights /= weights.sum()
    return float(np.dot(weights, values)), weights


def _safe_col(df, col):
    if col not in df.columns:
        return np.zeros(len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(0).values


def compute_xpoints(timeseries, position):
    """Compute per-GW expected points from ALL underlying components."""
    n = len(timeseries)
    if n == 0:
        return np.array([])

    mins = _safe_col(timeseries, "minutes")
    played = mins > 0
    appearance = np.where(mins >= 60, 2.0, np.where(played, 1.0, 0.0))

    xg = _safe_col(timeseries, "xG")
    if np.all(xg == 0):
        xg = _safe_col(timeseries, "goals").astype(float)
    xa = _safe_col(timeseries, "xA")
    if np.all(xa == 0):
        xa = _safe_col(timeseries, "assists").astype(float)

    goal_c = xg * GOAL_PTS.get(position, 4)
    assist_c = xa * ASSIST_PTS
    cs_c = _safe_col(timeseries, "clean_sheets") * CS_PTS.get(position, 0)
    gc_c = np.floor(_safe_col(timeseries, "goals_conceded") / 2.0) * GC_PTS.get(position, 0)
    bonus_c = _safe_col(timeseries, "bonus")

    saves_c = np.zeros(n)
    if position == "GK":
        saves_c = np.floor(_safe_col(timeseries, "saves") / 3.0)

    yc = _safe_col(timeseries, "yellow_cards") * (-1)
    rc = _safe_col(timeseries, "red_cards") * (-3)
    og = _safe_col(timeseries, "own_goals") * (-2)
    pm = _safe_col(timeseries, "penalties_missed") * (-2)
    ps = np.zeros(n)
    if position == "GK":
        ps = _safe_col(timeseries, "penalties_saved") * 5

    return (
        appearance + goal_c + assist_c + cs_c + gc_c + bonus_c + saves_c + yc + rc + og + pm + ps
    ) * played


def _home_away_factor(timeseries):
    """Home/away scoring multipliers from history."""
    if "was_home" not in timeseries.columns or "points" not in timeseries.columns:
        return 1.0, 1.0
    pts = _safe_col(timeseries, "points")
    mins = _safe_col(timeseries, "minutes")
    wh = _safe_col(timeseries, "was_home")
    played = mins > 0
    if played.sum() < 4:
        return 1.0, 1.0
    overall = pts[played].mean()
    if overall < 0.5:
        return 1.0, 1.0
    h = pts[played & (wh > 0.5)]
    a = pts[played & (wh < 0.5)]
    hf = np.clip(h.mean() / overall, 0.8, 1.3) if len(h) >= 2 else 1.0
    af = np.clip(a.mean() / overall, 0.8, 1.3) if len(a) >= 2 else 1.0
    return float(hf), float(af)


def _opponent_mult(timeseries, opp_id):
    """Scoring multiplier against a specific opponent."""
    if "opponent_team" not in timeseries.columns or opp_id <= 0:
        return 1.0
    pts = _safe_col(timeseries, "points")
    mins = _safe_col(timeseries, "minutes")
    opp = _safe_col(timeseries, "opponent_team").astype(int)
    played = mins > 0
    overall = pts[played].mean() if played.sum() > 0 else 2.0
    if overall < 0.5:
        return 1.0
    mask = played & (opp == opp_id)
    if mask.sum() < 1:
        return 1.0
    return float(np.clip(pts[mask].mean() / overall, 0.6, 1.6))


def enriched_predict(timeseries, position, alpha=0.3, lookback=15, upcoming_fixture=None):
    """
    Predict expected points with fixture awareness and semi-variance.

    Parameters
    ----------
    timeseries : pd.DataFrame
    position : str
    alpha : float
        EWMA decay.
    lookback : int
        Max recent GWs (increased from 10 to 15 for more data).
    upcoming_fixture : dict, optional
        {"was_home": bool, "opponent_team": int, "xP": float}

    Returns
    -------
    expected_points : float
    variance : float
    downside_risk : float  (semi-deviation below E[P])
    """
    if timeseries.empty or "minutes" not in timeseries.columns:
        return 0.0, 4.0, 0.0

    ts = timeseries.tail(lookback).copy()
    mins = _safe_col(ts, "minutes")
    played_mask = mins > 0
    n_played = int(played_mask.sum())

    if n_played < 2:
        return 0.0, 4.0, 0.0

    avail = float(played_mask[-min(3, len(played_mask)) :].mean())
    if avail < 0.1:
        return 0.0, 1.0, 0.0

    # xPoints from all components
    xpts = compute_xpoints(ts, position)
    played_xpts = xpts[played_mask]

    # EWMA on played xPoints
    conditional_ep = max(0.0, _ewma(played_xpts, alpha))

    # Fixture adjustments
    fixture_mult = 1.0
    if upcoming_fixture:
        hf, af = _home_away_factor(timeseries)
        fixture_mult = hf if upcoming_fixture.get("was_home", False) else af
        opp_id = upcoming_fixture.get("opponent_team", 0)
        if opp_id > 0:
            fixture_mult *= _opponent_mult(timeseries, opp_id)
    conditional_ep *= fixture_mult

    # Ensemble with xP
    if upcoming_fixture and upcoming_fixture.get("xP", 0) > 0:
        conditional_ep = 0.7 * conditional_ep + 0.3 * upcoming_fixture["xP"]

    # Variance and semi-variance from residuals
    downside_risk = 0.0
    if "points" in ts.columns:
        pts = _safe_col(ts, "points")
        played_pts = pts[played_mask]
        residuals = played_pts - played_xpts
        var_estimate = float(np.var(residuals)) + 1.0

        # Semi-variance: only negative residuals (actual < expected)
        neg_residuals = residuals[residuals < 0]
        if len(neg_residuals) >= 2:
            downside_risk = float(np.sqrt(np.mean(neg_residuals**2)))
        else:
            downside_risk = float(np.sqrt(var_estimate)) * 0.5
    else:
        var_estimate = 4.0
        downside_risk = 1.0

    ep = conditional_ep * avail
    var_out = avail * var_estimate + avail * (1 - avail) * conditional_ep**2
    dr_out = downside_risk * avail

    return ep, var_out, dr_out


def batch_enriched_predict(players, alpha=0.3, fixture_info=None):
    """Run enriched prediction for all players. Returns ep, var, downside_risk dicts."""
    ep, ev, dr = {}, {}, {}
    for p in players:
        fix = fixture_info.get(p.id) if fixture_info else None
        mu, var, dsr = enriched_predict(p.timeseries, p.position, alpha=alpha, upcoming_fixture=fix)
        ep[p.id] = mu
        ev[p.id] = var
        dr[p.id] = dsr
    return ep, ev, dr


__all__ = ["enriched_predict", "batch_enriched_predict", "compute_xpoints"]
