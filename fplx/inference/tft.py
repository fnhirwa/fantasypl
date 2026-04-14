"""Temporal Fusion Transformer (TFT) inference adapter.

This module provides optional deep-learning inference for FPLX using
`pytorch-forecasting`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fplx.data.tft_dataset import make_tft_datasets


@dataclass
class TFTQuantilePredictions:
    """Container for TFT quantile outputs for a single gameweek."""

    p10: dict[int, float]
    p50: dict[int, float]
    p90: dict[int, float]

    def to_optimizer_inputs(self) -> tuple[dict[int, float], dict[int, float]]:
        """Map quantiles to objective mean and downside risk.

        Returns
        -------
        expected_points : dict[int, float]
            Uses q50 as robust expected value proxy.
        downside_risk : dict[int, float]
            Uses q50 - q10 as downside spread.
        """
        expected_points = {pid: float(v) for pid, v in self.p50.items()}
        downside_risk = {
            pid: max(0.0, float(self.p50.get(pid, 0.0) - self.p10.get(pid, 0.0))) for pid in expected_points
        }
        return expected_points, downside_risk


class TFTForecaster:
    """Wrapper around PyTorch Forecasting's TemporalFusionTransformer."""

    def __init__(
        self,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        encoder_length: int = 15,
        prediction_length: int = 1,
    ):
        self.quantiles = quantiles
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.model = None
        self._trainer = None

    @staticmethod
    def _imports():
        try:
            import lightning.pytorch as pl
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_forecasting.metrics import QuantileLoss
        except ImportError as e:
            raise ImportError(
                "TFT support requires optional dependencies: pip install pytorch-forecasting lightning torch"
            ) from e
        return pl, TemporalFusionTransformer, QuantileLoss

    def fit(
        self,
        panel_df: pd.DataFrame,
        training_cutoff: int,
        max_epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        hidden_size: int = 32,
        attention_head_size: int = 4,
        dropout: float = 0.1,
    ):
        """Train TFT on panel data."""
        pl, TemporalFusionTransformer, QuantileLoss = self._imports()

        training, validation = make_tft_datasets(
            panel_df,
            training_cutoff=training_cutoff,
            encoder_length=self.encoder_length,
            prediction_length=self.prediction_length,
        )

        train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        self.model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            loss=QuantileLoss(self.quantiles),
            output_size=len(self.quantiles),
            reduce_on_plateau_patience=4,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(self.model, train_loader, val_loader)
        self._trainer = trainer
        return self

    def save(self, checkpoint_path: str | Path):
        if self.model is None:
            raise RuntimeError("Model is not trained/loaded.")
        if self._trainer is None:
            raise RuntimeError("No trainer available for checkpoint save. Fit the model first.")
        self._trainer.save_checkpoint(str(checkpoint_path))

    def load(self, checkpoint_path: str | Path):
        """Load a trained TFT checkpoint."""
        _, TemporalFusionTransformer, _ = self._imports()
        self.model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
        return self

    def predict_gameweek(
        self,
        panel_df: pd.DataFrame,
        target_gw: int,
        batch_size: int = 256,
    ) -> TFTQuantilePredictions:
        """Predict quantiles for one target gameweek across all players."""
        if self.model is None:
            raise RuntimeError("Model is not trained/loaded.")

        training, prediction = make_tft_datasets(
            panel_df[panel_df["time_idx"] <= target_gw].copy(),
            training_cutoff=target_gw - 1,
            encoder_length=self.encoder_length,
            prediction_length=self.prediction_length,
        )

        _ = training  # required for consistent schema creation in from_dataset
        pred_loader = prediction.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        # Quantile output shape: [n_samples, prediction_length, n_quantiles]
        pred_out = self.model.predict(
            pred_loader,
            mode="quantiles",
            return_x=True,
            return_index=True,
        )

        preds = None
        x = None
        index_df = None

        if hasattr(pred_out, "output"):
            preds = pred_out.output
            x = getattr(pred_out, "x", None)
            index_df = getattr(pred_out, "index", None)
        elif isinstance(pred_out, tuple):
            if len(pred_out) >= 1:
                preds = pred_out[0]
            if len(pred_out) >= 2:
                x = pred_out[1]
            if len(pred_out) >= 3:
                index_df = pred_out[2]
        else:
            preds = pred_out

        if preds is None:
            raise RuntimeError("TFT prediction output is empty.")

        q = preds.detach().cpu().numpy()
        q = q[:, 0, :]  # one-step forecast

        # Recover sample player ids from prediction index when available.
        if index_df is not None and "group_id" in index_df.columns:
            player_ids = index_df["group_id"].astype(int).to_numpy()
        elif x is not None and "groups" in x:
            groups = x["groups"].detach().cpu().numpy()
            player_ids = groups[:, 0].astype(int)
        else:
            raise RuntimeError("Unable to recover TFT sample player IDs from prediction output.")

        # Deduplicate by keeping last sample for each player in case of overlap.
        p10, p50, p90 = {}, {}, {}
        for pid, row in zip(player_ids, q, strict=False):
            p10[pid] = float(row[0])
            p50[pid] = float(row[1])
            p90[pid] = float(row[2])

        return TFTQuantilePredictions(p10=p10, p50=p50, p90=p90)


__all__ = ["TFTForecaster", "TFTQuantilePredictions"]
