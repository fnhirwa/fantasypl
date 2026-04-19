"""Double Gameweek (DGW) detection, timeseries aggregation, and prediction scaling.

A Double Gameweek occurs when a team plays two Premier League fixtures in the same
FPL gameweek. From the perspective of the inference pipeline and optimizer, this has
two distinct effects:

1. **Historical timeseries (training/inference input)**
   The vaastav dataset stores each fixture as a separate row. A DGW player therefore
   has *two* rows sharing the same ``gameweek`` value. If not aggregated, the HMM
   will see them as two sequential timesteps with single-game-calibrated emissions,
   causing the model to misinterpret a large total (e.g. 14 pts from two good games)
   as a single "Star" observation when it is actually two "Good" observations.

2. **Forward prediction (next-GW forecast for ILP)**
   When the *upcoming* gameweek is a DGW, a player plays twice. Their expected FPL
   points should be approximately 2× the single-game prediction (under independence),
   and their variance should also scale accordingly.

Usage
-----
>>> from fplx.data.double_gameweek import (
...     detect_dgw_gameweeks,
...     aggregate_dgw_timeseries,
...     scale_predictions_for_dgw,
...     get_fixture_counts_from_bootstrap,
... )
"""

from __future__ import annotations

import contextlib
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Additive stat columns: sum across fixtures within a DGW gameweek
_ADDITIVE_COLS = [
    "points",
    "minutes",
    "goals",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "transfers_in",
    "transfers_out",
]

# Rate/expected stat columns: average across fixtures within a DGW gameweek
_RATE_COLS = [
    "xG",
    "xA",
    "xP",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals_conceded",
]

# Context columns: take value from last fixture (prices don't change mid-GW)
_LAST_COLS = [
    "value",
    "selected",
    "opponent_team",
    "was_home",
]


def detect_dgw_gameweeks(timeseries: pd.DataFrame) -> dict[int, int]:
    """Return a mapping of {gameweek: n_fixtures} for a single player's timeseries.

    A gameweek with n_fixtures > 1 is a Double (or Triple) Gameweek.

    Parameters
    ----------
    timeseries : pd.DataFrame
        Per-fixture timeseries as returned by ``VaastavLoader.build_player_objects``.
        Must contain a ``gameweek`` column.

    Returns
    -------
    dict[int, int]
        ``{gameweek_number: fixture_count}`` for all gameweeks in the data.
        Gameweeks with a single fixture have value 1.

    Examples
    --------
    >>> counts = detect_dgw_gameweeks(player.timeseries)
    >>> dgw_gws = [gw for gw, n in counts.items() if n > 1]
    """
    if timeseries.empty or "gameweek" not in timeseries.columns:
        return {}
    return timeseries.groupby("gameweek").size().to_dict()


def aggregate_dgw_timeseries(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fixture rows into one normalised row per gameweek.

    This is the **single place** where Double Gameweek handling lives.  All
    downstream consumers (inference pipeline, enriched predictor, MV-HMM,
    Kalman Filter) always receive exactly one row per FPL decision period and
    never need to be aware of DGWs.

    For a DGW gameweek (``n_fixtures == 2``):

    * **Additive stats** (goals, minutes, bonus, …) are **summed** to reflect
      the total accumulated across both matches.
    * **Per-fixture normalisation** is applied to ``points`` and to every
      additive stat that forms an inference feature.  The normalised column is
      stored alongside the raw total:

      .. code-block:: text

          points         # raw total (used for scoring / oracle)
          points_norm    # per-fixture average (used by inference / HMM)

      The HMM emission distributions are calibrated on ``points_norm``, so a
      DGW observation of 10 total points (``points_norm = 5``) is correctly
      interpreted as an "Average" game rather than misidentified as a "Star"
      event (8.5 pts single-game emission mean).

    * **Rate / expected stats** (xG, xA, …) are averaged — they already
      represent per-match rates.

    * **Context columns** (price, opponent) take the last-fixture value.

    For a single-fixture gameweek (``n_fixtures == 1``) the row is returned
    unchanged and ``points_norm == points``.

    Parameters
    ----------
    timeseries : pd.DataFrame
        Raw per-fixture timeseries (may contain duplicate ``gameweek`` values
        for DGW players).

    Returns
    -------
    pd.DataFrame
        One row per gameweek, sorted ascending by ``gameweek``.
        New columns added:
        - ``n_fixtures``  : int, number of fixtures played that round
        - ``points_norm`` : float, per-fixture normalised points
    """
    if timeseries.empty or "gameweek" not in timeseries.columns:
        return timeseries.copy()

    gw_counts = timeseries.groupby("gameweek").size()
    has_multi = (gw_counts > 1).any()

    if not has_multi:
        ts = timeseries.copy()
        ts["n_fixtures"] = 1
        ts["points_norm"] = ts["points"] if "points" in ts.columns else 0.0
        return ts.sort_values("gameweek").reset_index(drop=True)

    agg_rows = []
    for gw, grp in timeseries.groupby("gameweek"):
        n = len(grp)
        row: dict = {"gameweek": gw, "n_fixtures": n}

        # ── Additive stats: sum across fixtures ───────────────────────────
        for col in _ADDITIVE_COLS:
            if col in grp.columns:
                row[col] = pd.to_numeric(grp[col], errors="coerce").fillna(0.0).sum()

        # ── Per-fixture normalisation of inference-facing columns ─────────
        # points_norm is what the HMM / enriched predictor trains on.
        # All other additive stat norms follow the same pattern.
        pts_total = row.get("points", 0.0)
        row["points_norm"] = pts_total / n if n > 0 else 0.0

        for col in _ADDITIVE_COLS:
            if col in row and col != "points":
                row[f"{col}_norm"] = row[col] / n if n > 0 else 0.0

        # ── Rate / expected stats: average across fixtures ────────────────
        for col in _RATE_COLS:
            if col in grp.columns:
                row[col] = pd.to_numeric(grp[col], errors="coerce").mean()

        # ── Context: last fixture value ───────────────────────────────────
        for col in _LAST_COLS:
            if col in grp.columns:
                row[col] = grp[col].iloc[-1]

        # Remaining columns: last value
        handled = set(_ADDITIVE_COLS + _RATE_COLS + _LAST_COLS + ["gameweek"])
        for col in grp.columns:
            if col not in handled:
                with contextlib.suppress(Exception):
                    row[col] = grp[col].iloc[-1]
        agg_rows.append(row)

    result = pd.DataFrame(agg_rows).sort_values("gameweek").reset_index(drop=True)

    for col in result.columns:
        if col != "gameweek":
            with contextlib.suppress(Exception):
                result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def scale_predictions_for_dgw(
    expected_points: dict[int, float],
    variances: dict[int, float],
    downside_risks: dict[int, float],
    fixture_counts: dict[int, int],
    variance_mode: str = "additive",
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Scale single-game predictions to account for a Double Gameweek.

    For a player with ``n`` fixtures in the upcoming gameweek:

    - Expected points: ``E[P_total] = n * E[P_single]``
    - Variance (additive, under independence): ``Var[P_total] = n * Var[P_single]``
    - Downside risk: ``DR_total = sqrt(n) * DR_single``

    This is exact under independence of the two match performances. The
    independence assumption is acceptable because FPL points in different
    matches are only weakly correlated (shared clean sheet probability for the
    same game counts for both defenders, but that is captured in the single-game
    variance estimate).

    Parameters
    ----------
    expected_points : dict[int, float]
        Single-game expected points per player id.
    variances : dict[int, float]
        Single-game predictive variance per player id.
    downside_risks : dict[int, float]
        Single-game semi-deviation per player id.
    fixture_counts : dict[int, int]
        Number of upcoming fixtures per player id (1 for SGW, 2 for DGW).
        Players absent from this dict are assumed to have 1 fixture.
    variance_mode : str
        ``"additive"`` (default): ``Var[P_total] = n * Var[P_single]`` — correct
        under independence.
        ``"conservative"``: multiply variance by ``n^2`` to account for possible
        correlation (e.g. both games against the same strong opponent).

    Returns
    -------
    ep_scaled, var_scaled, dr_scaled : tuple of dicts
        Scaled prediction dicts with the same keys as the inputs.

    Notes
    -----
    Blank gameweek (BGW) players (``n = 0``) receive ``E[P] = 0``,
    ``Var[P] = 0.1``, ``DR = 0``. The optimizer will naturally exclude them
    since their expected points are zero.

    Examples
    --------
    >>> ep_scaled, var_scaled, dr_scaled = scale_predictions_for_dgw(
    ...     expected_points, variances, downside_risks, fixture_counts
    ... )
    """
    ep_out: dict[int, float] = {}
    var_out: dict[int, float] = {}
    dr_out: dict[int, float] = {}

    for pid, ep in expected_points.items():
        n = fixture_counts.get(pid, 1)

        if n == 0:
            # Blank gameweek — player has no fixture
            ep_out[pid] = 0.0
            var_out[pid] = 0.1
            dr_out[pid] = 0.0
            continue

        var = variances.get(pid, 4.0)
        dr = downside_risks.get(pid, var**0.5 / 2**0.5)

        ep_out[pid] = ep * n

        if variance_mode == "additive":
            var_out[pid] = var * n
        else:  # conservative
            var_out[pid] = var * n * n

        # Semi-deviation scales as sqrt(n) under independence
        dr_out[pid] = dr * (n**0.5)

    return ep_out, var_out, dr_out


def get_fixture_counts_from_bootstrap(
    bootstrap: dict,
    target_gw: int,
) -> dict[int, int]:
    """Derive per-player fixture counts for a gameweek from FPL bootstrap data.

    Parses the ``fixtures`` list in the bootstrap-static response to count how
    many fixtures each team plays in ``target_gw``. Returns a player-level
    mapping derived from each player's ``team`` id.

    Parameters
    ----------
    bootstrap : dict
        Full bootstrap-static API response containing ``"fixtures"`` and
        ``"elements"`` lists.
    target_gw : int
        The gameweek to inspect.

    Returns
    -------
    dict[int, int]
        ``{player_id: n_fixtures}`` for all players. Players whose team has no
        fixture in ``target_gw`` (BGW) receive 0.
    """
    fixtures = bootstrap.get("fixtures", [])
    elements = bootstrap.get("elements", [])

    # Count fixtures per team in target_gw
    team_fixture_counts: dict[int, int] = {}
    for fix in fixtures:
        if fix.get("event") != target_gw:
            continue
        h = fix.get("team_h")
        a = fix.get("team_a")
        if h is not None:
            team_fixture_counts[h] = team_fixture_counts.get(h, 0) + 1
        if a is not None:
            team_fixture_counts[a] = team_fixture_counts.get(a, 0) + 1

    # Map player → team → fixture count
    player_counts: dict[int, int] = {}
    for elem in elements:
        pid = elem["id"]
        team = elem.get("team")
        player_counts[pid] = team_fixture_counts.get(team, 1)

    n_dgw = sum(1 for t, n in team_fixture_counts.items() if n > 1)
    n_bgw = sum(1 for t, n in team_fixture_counts.items() if n == 0)
    if n_dgw:
        logger.info("GW%d: %d teams with DGW, %d teams with BGW.", target_gw, n_dgw, n_bgw)

    return player_counts


def get_fixture_counts_from_vaastav(
    loader,
    target_gw: int,
) -> dict[int, int]:
    """Derive per-player fixture counts for a historical gameweek from vaastav data.

    Uses the merged_gw CSV to count how many rows each player has for
    ``target_gw``. This is the ground-truth fixture count for backtesting.

    Parameters
    ----------
    loader : VaastavLoader
        An initialised loader instance.
    target_gw : int
        Gameweek to inspect.

    Returns
    -------
    dict[int, int]
        ``{player_id: n_fixtures}`` — 1 for SGW, 2 for DGW, 0 if no fixture.
    """
    df = loader.load_gameweek(target_gw)
    if df.empty:
        return {}
    counts = df.groupby("element").size()
    return counts.to_dict()


__all__ = [
    "detect_dgw_gameweeks",
    "aggregate_dgw_timeseries",
    "scale_predictions_for_dgw",
    "get_fixture_counts_from_bootstrap",
    "get_fixture_counts_from_vaastav",
]
