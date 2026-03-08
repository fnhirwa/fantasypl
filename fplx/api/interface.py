"""High-level API interface for FPLX."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from fplx.core.player import Player
from fplx.core.squad import Squad
from fplx.data.loaders import FPLDataLoader
from fplx.data.news_collector import NewsCollector
from fplx.inference.pipeline import PlayerInferencePipeline
from fplx.models.baseline import BaselineModel, FormBasedModel
from fplx.models.ensemble import EnsembleModel
from fplx.models.regression import RegressionModel
from fplx.selection.optimizer import GreedyOptimizer, ILPOptimizer
from fplx.signals.fixtures import FixtureSignal
from fplx.signals.news import NewsSignal
from fplx.signals.stats import StatsSignal
from fplx.timeseries.features import FeatureEngineer
from fplx.utils.config import Config

logger = logging.getLogger(__name__)


class FPLModel:
    """
    High-level interface for FPL analysis and squad optimization.

    This is the main user-facing API. It orchestrates data loading,
    feature engineering, model fitting, and squad optimization.

    Parameters
    ----------
    budget : float
        Maximum squad budget (default 100.0)
    horizon : int
        Prediction horizon in gameweeks (default 1)
    formation : str
        Desired formation, or "auto" for optimization
    config : Optional[Dict]
        Custom configuration

    Examples
    --------
    >>> from fplx import FPLModel
    >>> model = FPLModel(budget=100, horizon=1)
    >>> model.load_data()
    >>> model.fit()
    >>> squad = model.select_best_11()
    >>> squad.summary()
    """

    def __init__(
        self,
        budget: float = 100.0,
        horizon: int = 1,
        formation: str = "auto",
        config: Optional[dict] = None,
    ):
        self.budget = budget
        self.horizon = horizon
        self.formation = formation
        self.config = Config(config)

        # core components are instantiated on-demand
        self._data_loader = None
        self._feature_engineer = None
        self._news_collector = None
        self._stats_signal = None
        self._news_signal = None
        self._fixture_signal = None
        self._model = None
        self._optimizer = None

        # Data containers
        self.players: list[Player] = []
        self.players_data: dict[int, pd.DataFrame] = {}
        self.expected_points: dict[int, float] = {}
        self.expected_variance: dict[int, float] = {}
        self.current_gameweek: int = 1

    @property
    def data_loader(self):
        if self._data_loader is None:
            self._data_loader = FPLDataLoader(**self.config.get("data_loader", {}))
        return self._data_loader

    @property
    def feature_engineer(self):
        if self._feature_engineer is None:
            self._feature_engineer = FeatureEngineer(**self.config.get("feature_engineer", {}))
        return self._feature_engineer

    @property
    def news_collector(self):
        if self._news_collector is None:
            self._news_collector = NewsCollector(cache_dir=self.config.get("news_cache_dir"))
        return self._news_collector

    @property
    def stats_signal(self):
        if self._stats_signal is None:
            self._stats_signal = StatsSignal(**self.config.get("stats_signal", {}))
        return self._stats_signal

    @property
    def news_signal(self):
        if self._news_signal is None:
            self._news_signal = NewsSignal(**self.config.get("news_signal", {}))
        return self._news_signal

    @property
    def fixture_signal(self):
        if self._fixture_signal is None:
            self._fixture_signal = FixtureSignal(**self.config.get("fixture_signal", {}))
        return self._fixture_signal

    @property
    def model(self):
        if self._model is None:
            self._model = self._create_model()
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = self._create_optimizer()
        return self._optimizer

    def _create_model(self):
        """Factory method for creating the prediction model."""
        model_type = self.config.get("model_type", "baseline")
        model_config = self.config.get("model_config", {})

        if model_type == "baseline":
            return BaselineModel(**model_config)
        if model_type == "form_based":
            return FormBasedModel(**model_config)
        if model_type in ["ridge", "xgboost", "lightgbm"]:
            return RegressionModel(model_type=model_type, **model_config)
        if model_type == "ensemble":
            return EnsembleModel(**model_config)
        logger.warning(f"Unknown model type {model_type}, using baseline.")
        return BaselineModel()

    def _create_optimizer(self):
        """Factory method for creating the squad optimizer."""
        optimizer_type = self.config.get("optimizer", "greedy")
        optimizer_config = self.config.get("optimizer_config", {})
        optimizer_config["budget"] = self.budget

        if optimizer_type == "greedy":
            return GreedyOptimizer(**optimizer_config)
        if optimizer_type == "ilp":
            return ILPOptimizer(**optimizer_config)
        logger.warning(f"Unknown optimizer {optimizer_type}, using greedy.")
        return GreedyOptimizer(**optimizer_config)

    def load_data(self, source: str = "api", path: Optional[Union[str, Path]] = None) -> None:
        """
        Load player and fixture data.

        Parameters
        ----------
        source : str
            Data source: 'api' or 'local'
        path : Optional[Union[str, Path]]
            Path to local data (if source is 'local')
        """
        logger.info(f"Loading data from {source}...")
        if source == "api":
            bootstrap_data = self.data_loader.fetch_bootstrap_data()
            self.players = self.data_loader.load_players(bootstrap_data)

            # Determine current gameweek from bootstrap events
            for event in bootstrap_data.get("events", []):
                if event.get("is_current"):
                    self.current_gameweek = event["id"]
                    break

            # Collect per-gameweek news snapshots for inference
            self.news_collector.collect_from_bootstrap(bootstrap_data, self.current_gameweek)

        elif source == "local":
            if path is None:
                raise ValueError("Path must be provided for local data source.")
            self.players = self.data_loader.load_from_csv(path)

        logger.info(f"Loaded {len(self.players)} players.")

        # load detailed time-series for each player
        for player in self.players:
            # This is a simplification; in reality, you'd fetch this
            # or have it in your local data.
            self.players_data[player.id] = player.timeseries

    def fit(self) -> None:
        """
        Fit the prediction model.

        Uses the probabilistic inference pipeline (HMM + Kalman + Fusion)
        when model_type is 'inference'. Falls back to the original
        feature engineering pipeline for baseline/ML models.
        """
        if not self.players:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        model_type = self.config.get("model_type", "baseline")
        logger.info(f"Fitting model '{model_type}'...")

        if model_type == "inference":
            self._fit_inference()
        else:
            self._fit_legacy(model_type)

        logger.info("Model fitting complete.")

    def _fit_inference(self) -> None:
        """
        Inference pipeline: HMM + Kalman + signal injection + fusion.

        For each player:
        1. Ingest historical points into PlayerInferencePipeline
        2. Inject news signal (from NewsCollector) into HMM/KF
        3. Inject fixture difficulty into KF observation noise
        4. Run inference → fused (mean, variance)
        """
        all_expected_points = {}
        all_expected_variance = {}

        hmm_params = self.config.get("inference.hmm_params", {})
        kf_params = self.config.get("inference.kf_params", {})

        for player in self.players:
            # 1. Extract points history
            if player.timeseries.empty or "points" not in player.timeseries.columns:
                all_expected_points[player.id] = 0.0
                all_expected_variance[player.id] = 1.0
                continue

            points = player.timeseries["points"].values.astype(float)
            if len(points) < 2:
                all_expected_points[player.id] = 0.0
                all_expected_variance[player.id] = 1.0
                continue

            # 2. Create per-player inference pipeline
            pipeline = PlayerInferencePipeline(
                hmm_params=hmm_params if hmm_params else None,
                kf_params=kf_params if kf_params else None,
            )
            pipeline.ingest_observations(points)

            # 3. Inject news signal at current gameweek
            snapshot = self.news_collector.get_player_news(player.id, self.current_gameweek)
            if snapshot:
                enriched_text = snapshot.to_news_signal_input()
                if enriched_text:
                    signal = self.news_signal.generate_signal(enriched_text)
                    pipeline.inject_news(signal, timestep=len(points) - 1)

            # 4. Inject fixture difficulty (if fixture data available)
            # TODO: wire FixtureSignal.compute_fixture_difficulty() here
            # once upcoming fixture data is loaded in load_data()
            # 5. Run inference and get fused prediction
            pipeline.run()
            ep_mean, ep_var = pipeline.predict_next()

            all_expected_points[player.id] = max(0.0, ep_mean)
            all_expected_variance[player.id] = ep_var

        self.expected_points = all_expected_points
        self.expected_variance = all_expected_variance

    def _fit_legacy(self, model_type: str) -> None:
        """Original feature engineering + model prediction pipeline."""
        all_expected_points = {}

        if model_type not in ["baseline", "form_based"]:
            all_features = []
            all_targets = []
            for player in self.players:
                if not player.timeseries.empty:
                    features = self.feature_engineer.fit_transform(player.timeseries)
                    target = player.timeseries["points"]
                    aligned_features, aligned_target = features.align(target, join="inner", axis=0)
                    all_features.append(aligned_features)
                    all_targets.append(aligned_target)

            if all_features:
                X_train = pd.concat(all_features)
                y_train = pd.concat(all_targets)
                X_train_numeric = X_train.select_dtypes(include=np.number)
                self.model.fit(X_train_numeric, y_train)

        for player in self.players:
            if model_type in ["baseline", "form_based"]:
                self.model.fit(player.timeseries)
                predictions = self.model.predict(player.timeseries)
            else:
                future_features = self.feature_engineer.create_future_features(
                    player.timeseries, self.horizon
                )
                if not future_features.empty:
                    future_features_numeric = future_features.select_dtypes(include=np.number)
                    predictions = self.model.predict(future_features_numeric)
                else:
                    predictions = np.array([0.0] * self.horizon)

            news_sig = self.news_signal.generate_signal(player.news.get("summary", "") if player.news else "")
            final_ep = np.sum(predictions) * news_sig["adjustment_factor"]
            all_expected_points[player.id] = final_ep

        self.expected_points = all_expected_points

    def select_best_11(self) -> Squad:
        """
        Select the optimal 11-player squad.

        Returns
        -------
        Squad
            The optimized squad.
        """
        if not self.expected_points:
            raise RuntimeError("Model not fitted. Call fit() first.")

        logger.info("Optimizing squad with %s optimizer...", self.config.get('optimizer', 'greedy'))

        squad = self.optimizer.solve(
            players=self.players,
            expected_points=self.expected_points,
            formation=self.formation,
        )

        logger.info("Squad optimization complete.")
        return squad
