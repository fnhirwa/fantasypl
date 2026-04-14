"""Dataset utilities for Temporal Fusion Transformer (TFT).

This module converts vaastav merged gameweek data into a global panel format
compatible with `pytorch_forecasting.TimeSeriesDataSet`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplx.inference.enriched import compute_xpoints


def build_tft_panel(merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Build TFT panel dataframe from merged gameweek data.

    Output schema includes:
    - group_id: player identifier
    - time_idx: gameweek index
    - static categoricals: position, team
    - known covariates: fixture_difficulty, is_home
    - unknown covariates: xPts, mins_frac, news_sentiment, actual_points
    """
    df = merged_gw.copy()

    rename_map = {
        "element": "group_id",
        "gameweek": "time_idx",
        "points": "actual_points",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "group_id" not in df.columns and "element" in merged_gw.columns:
        df["group_id"] = merged_gw["element"]
    if "time_idx" not in df.columns and "gameweek" in merged_gw.columns:
        df["time_idx"] = merged_gw["gameweek"]
    if "actual_points" not in df.columns and "points" in merged_gw.columns:
        df["actual_points"] = merged_gw["points"]

    df["group_id"] = pd.to_numeric(df["group_id"], errors="coerce").astype("Int64")
    df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce")
    df["actual_points"] = pd.to_numeric(df["actual_points"], errors="coerce").fillna(0.0)

    if "position" not in df.columns:
        df["position"] = "MID"
    if "team" not in df.columns:
        df["team"] = "Unknown"
    df["position"] = df["position"].astype(str)
    df["team"] = df["team"].astype(str)

    # Known future covariates.
    if "was_home" in df.columns:
        df["is_home"] = pd.to_numeric(df["was_home"], errors="coerce").fillna(0.0)
    else:
        df["is_home"] = 0.0

    if "fixture_difficulty" in df.columns:
        df["fixture_difficulty"] = pd.to_numeric(df["fixture_difficulty"], errors="coerce").fillna(3.0)
    else:
        df["fixture_difficulty"] = 3.0

    if "minutes" in df.columns:
        mins = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
        df["mins_frac"] = np.clip(mins / 90.0, 0.0, 1.0)
    else:
        df["mins_frac"] = 0.0

    # Placeholder until historical NLP news pipeline is fully integrated.
    df["news_sentiment"] = 0.0

    # Structural xPts projection per player-position trajectory.
    xpts = np.zeros(len(df), dtype=float)
    for _, grp in df.groupby("group_id", dropna=True):
        grp_sorted = grp.sort_values("time_idx")
        pos = str(grp_sorted["position"].iloc[0])
        x_vals = compute_xpoints(grp_sorted, pos)
        xpts[grp_sorted.index.to_numpy()] = x_vals
    df["xPts"] = xpts

    keep_cols = [
        "group_id",
        "time_idx",
        "position",
        "team",
        "fixture_difficulty",
        "is_home",
        "xPts",
        "mins_frac",
        "news_sentiment",
        "actual_points",
    ]
    df = df[keep_cols].dropna(subset=["group_id", "time_idx"]).copy()
    df["group_id"] = df["group_id"].astype(int)
    df["time_idx"] = df["time_idx"].astype(int)
    return df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)


def make_tft_datasets(
    panel_df: pd.DataFrame,
    training_cutoff: int,
    encoder_length: int = 15,
    prediction_length: int = 1,
):
    """Create TFT training and prediction datasets.

    Requires optional dependency `pytorch-forecasting`.
    """
    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
    except ImportError as e:
        raise ImportError(
            "TFT dataset creation requires pytorch-forecasting. "
            "Install with: pip install pytorch-forecasting lightning"
        ) from e

    train_df = panel_df[panel_df["time_idx"] <= training_cutoff].copy()
    train_df["actual_points"] = (
        pd.to_numeric(train_df["actual_points"], errors="coerce").fillna(0.0).astype(float)
    )

    # Ensure encoder length is feasible for available history.
    hist_len = train_df.groupby("group_id")["time_idx"].nunique()
    if hist_len.empty:
        raise ValueError("No training data available for TFT dataset creation.")

    max_possible_encoder = int(hist_len.max() - prediction_length)
    if max_possible_encoder < 1:
        raise ValueError(
            "Insufficient history to build TFT windows. Increase training cutoff or reduce prediction length."
        )

    eff_encoder_length = min(int(encoder_length), max_possible_encoder)
    min_required_len = eff_encoder_length + prediction_length

    valid_ids = hist_len[hist_len >= min_required_len].index
    train_df = train_df[train_df["group_id"].isin(valid_ids)].copy()
    if train_df.empty:
        raise ValueError(
            "No groups have enough history after encoder-length adjustment. "
            f"Required per-group length: {min_required_len}."
        )

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="actual_points",
        group_ids=["group_id"],
        min_encoder_length=eff_encoder_length,
        max_encoder_length=eff_encoder_length,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,
        static_categoricals=["position", "team"],
        time_varying_known_reals=["time_idx", "fixture_difficulty", "is_home"],
        time_varying_unknown_reals=["xPts", "mins_frac", "news_sentiment", "actual_points"],
        categorical_encoders={
            "position": NaNLabelEncoder(add_nan=True),
            "team": NaNLabelEncoder(add_nan=True),
        },
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    pred_df = panel_df[panel_df["group_id"].isin(valid_ids)].copy()
    pred_df["actual_points"] = (
        pd.to_numeric(pred_df["actual_points"], errors="coerce").fillna(0.0).astype(float)
    )

    prediction = TimeSeriesDataSet.from_dataset(
        training,
        pred_df,
        predict=True,
        stop_randomization=True,
    )

    return training, prediction


__all__ = ["build_tft_panel", "make_tft_datasets"]
