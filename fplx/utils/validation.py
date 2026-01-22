"""Data validation utilities."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """
    Validate that dataframe has required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    required_columns : list[str]
        Required column names

    Returns
    -------
    bool
        True if valid
    """
    missing = set(required_columns) - set(df.columns)

    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False

    return True


def check_data_quality(
    df: pd.DataFrame, max_missing_pct: float = 0.3
) -> dict[str, float]:
    """
    Check data quality and report issues.

    Parameters
    ----------
    df : pd.DataFrame
        Data to check
    max_missing_pct : float
        Maximum acceptable percentage of missing values

    Returns
    -------
    Dict[str, float]
        Quality metrics
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_pct = missing_cells / total_cells if total_cells > 0 else 0

    # Per-column missing
    col_missing = df.isna().mean()
    problematic_cols = col_missing[col_missing > max_missing_pct].index.tolist()

    metrics = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "missing_percentage": missing_pct * 100,
        "problematic_columns": problematic_cols,
    }

    if missing_pct > max_missing_pct:
        logger.warning(f"High missing data: {missing_pct * 100:.2f}%")

    if problematic_cols:
        logger.warning(f"Columns with high missing data: {problematic_cols}")

    return metrics


def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Impute missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Data with missing values
    strategy : str
        Imputation strategy: 'mean', 'median', 'forward_fill', 'zero'

    Returns
    -------
    pd.DataFrame
        Data with imputed values
    """
    df = df.copy()

    if strategy == "mean":
        df = df.fillna(df.mean())
    elif strategy == "median":
        df = df.fillna(df.median())
    elif strategy == "forward_fill":
        df = df.fillna(method="ffill")
    elif strategy == "zero":
        df = df.fillna(0)
    else:
        logger.warning(f"Unknown strategy {strategy}, using mean")
        df = df.fillna(df.mean())

    return df


__all__ = ["validate_data", "check_data_quality", "impute_missing"]
