"""Train a Temporal Fusion Transformer (TFT) on vaastav panel data.

Example:
    python scripts/train_tft.py \
        --data-dir Fantasy \
        --season 2024-25 \
        --train-end-gw 30 \
        --checkpoint artifacts/tft_2024_25.ckpt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplx.data.tft_dataset import build_tft_panel
from fplx.data.vaastav_loader import VaastavLoader
from fplx.inference.tft import TFTForecaster

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train TFT forecaster for FPLX")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--train-end-gw", type=int, default=30)
    parser.add_argument("--encoder-length", type=int, default=15)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--attention-head-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint", default="artifacts/tft.ckpt")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    loader = VaastavLoader(season=args.season, data_dir=args.data_dir)
    merged = loader.load_merged_gw()
    panel = build_tft_panel(merged)

    logger.info("Panel rows=%d players=%d", len(panel), panel["group_id"].nunique())

    forecaster = TFTForecaster(
        encoder_length=args.encoder_length,
        prediction_length=args.prediction_length,
    )
    forecaster.fit(
        panel_df=panel,
        training_cutoff=args.train_end_gw,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
    )

    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(ckpt)
    logger.info("Saved TFT checkpoint to %s", ckpt)


if __name__ == "__main__":
    main()
