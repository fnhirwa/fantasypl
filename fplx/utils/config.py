"""Configuration management."""

import json
from pathlib import Path
from typing import Any, Optional


class Config:
    """
    Configuration manager for FPLX.

    Parameters
    ----------
    config : Optional[Dict]
        Configuration dictionary
    """

    DEFAULT_CONFIG = {
        "model_type": "baseline",
        "optimizer": "greedy",
        "budget": 100.0,
        "horizon": 1,
        "formation": "auto",
        "feature_engineering": {
            "rolling_windows": [3, 5, 10],
            "lag_periods": [1, 2, 3],
            "ewma_alphas": [0.3, 0.5],
        },
        "signals": {
            "stats_weights": {
                "points_mean": 0.3,
                "xG_mean": 0.15,
                "xA_mean": 0.15,
                "minutes_consistency": 0.2,
                "form_trend": 0.2,
            },
        },
        "inference": {
            "hmm_variance_floor": 1.0,
            "fusion_mode": "precision",
            "fusion_params": {},
            "hmm_params": {},
            "kf_params": {},
            "news_params": {},
            "mvhmm_params": {
                "prior_weight": 0.85,
                "n_iter": 15,
                "lookback": 8,
                "min_history": 3,
            },
            "tft_params": {
                "checkpoint": None,
                "encoder_length": 15,
            },
        },
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self._update_nested(self.config, config)

    def _update_nested(self, base: dict, update: dict):
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._update_nested(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Parameters
        ----------
        key : str
            Configuration key (supports nested keys with '.')
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Parameters
        ----------
        key : str
            Configuration key (supports nested keys with '.')
        value : Any
            Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def load_from_file(self, filepath: Path):
        """
        Load configuration from JSON file.

        Parameters
        ----------
        filepath : Path
            Path to configuration file
        """
        with open(filepath) as f:
            file_config = json.load(f)

        self._update_nested(self.config, file_config)

    def save_to_file(self, filepath: Path):
        """
        Save configuration to JSON file.

        Parameters
        ----------
        filepath : Path
            Path to save configuration
        """
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)

    def to_dict(self) -> dict:
        """
        Get configuration as dictionary.

        Returns
        -------
        Dict
            Configuration dictionary
        """
        return self.config.copy()
