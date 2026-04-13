"""Full-season backtest: inference + optimization over 38 gameweeks.

Usage:
    python scripts/backtest_season.py \
        --data-dir /path/to/Fantasy-Premier-League \
        --season 2023-24 \
        --output results/backtest_2023_24.json

This script:
  1. Loads the vaastav dataset.
  2. For each gameweek t (walk-forward):
     a. Build player objects with history up to GW t-1.
     b. Run inference pipeline -> E[P_i], Var[P_i].
     c. Run baselines (rolling avg, EWMA) -> E[P_i] only.
     d. Solve ILP under risk-neutral and risk-averse strategies.
     e. Compute actual points from GW t.
     f. Compute oracle (hindsight-optimal squad).
  3. Aggregate InferenceMetrics and OptimizationMetrics.
  4. Save results to JSON.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplx.core.player import Player
from fplx.data.vaastav_loader import VaastavLoader
from fplx.evaluation.metrics import InferenceMetrics, OptimizationMetrics
from fplx.inference.pipeline import PlayerInferencePipeline
from fplx.selection.optimizer import GreedyOptimizer, TwoLevelILPOptimizer

logger = logging.getLogger(__name__)

# Minimum gameweeks of history before we start predicting
MIN_HISTORY = 5


# Baselines
def rolling_mean_predict(points_history: np.ndarray, window: int = 5) -> float:
    """Rolling average baseline prediction."""
    if len(points_history) == 0:
        return 0.0
    if len(points_history) < window:
        return float(np.mean(points_history))
    return float(np.mean(points_history[-window:]))


def ewma_predict(points_history: np.ndarray, alpha: float = 0.3) -> float:
    """EWMA baseline prediction."""
    if len(points_history) == 0:
        return 0.0
    ewma = points_history[0]
    for pt in points_history[1:]:
        ewma = alpha * pt + (1 - alpha) * ewma
    return float(ewma)


def compute_oracle_squad(
    players: list[Player],
    actual_points: dict[int, float],
    budget: float = 100.0,
    max_from_team: int = 3,
) -> float:
    """
    Compute the oracle (hindsight-optimal) total points.

    Uses the ILP with actual points as objective coefficients.
    """
    try:
        optimizer = TwoLevelILPOptimizer(budget=budget, max_from_team=max_from_team)
        full_squad = optimizer.optimize(players, actual_points)
        return full_squad.lineup.expected_points
    except Exception as e:
        logger.warning("Oracle computation failed: %s", e)
        # Fallback: sum of top 11 actual points (ignoring constraints)
        top = sorted(actual_points.values(), reverse=True)[:11]
        return sum(top)


def run_backtest(
    season: str = "2023-24",
    data_dir: str | None = None,
    start_gw: int = 6,
    end_gw: int = 38,
    risk_lambdas: list[float] | None = None,
) -> dict:
    """
    Run the full-season backtest.

    Parameters
    ----------
    season : str
        Season string.
    data_dir : str, optional
        Path to local Fantasy-Premier-League clone. If None, fetches from GitHub.
    start_gw : int
        First gameweek to predict (needs MIN_HISTORY prior GWs).
    end_gw : int
        Last gameweek.
    risk_lambdas : list[float]
        Risk aversion values to test.

    Returns
    -------
    dict
        Combined inference and optimization metrics.
    """
    if risk_lambdas is None:
        risk_lambdas = [0.0, 0.5, 1.0]

    loader = VaastavLoader(season=season, data_dir=data_dir)
    inf_metrics = InferenceMetrics()
    opt_metrics = OptimizationMetrics()

    logger.info("Starting backtest: GW%d to GW%d", start_gw, end_gw)

    for target_gw in range(start_gw, end_gw + 1):
        t0 = time.perf_counter()
        logger.info("=== GW%d ===", target_gw)

        # Build player objects with history up to GW t-1
        players = loader.build_player_objects(up_to_gw=target_gw - 1)

        # Get actual points for GW t
        try:
            actual_points_gw = loader.get_actual_points(target_gw)
        except FileNotFoundError:
            logger.warning("GW%d data not found, skipping.", target_gw)
            continue

        # Filter to players who have enough history and appear in target GW
        valid_players = [
            p
            for p in players
            if len(p.timeseries) >= MIN_HISTORY
            and "points" in p.timeseries.columns
            and p.id in actual_points_gw
        ]

        if len(valid_players) < 50:
            logger.warning("Only %d valid players for GW%d, skipping.", len(valid_players), target_gw)
            continue

        # Run inference for each player
        ep_fused = {}  # E[P_i] from fused pipeline
        var_fused = {}  # Var[P_i] from fused pipeline
        ep_rolling = {}  # baseline: rolling mean
        ep_ewma = {}  # baseline: EWMA

        for p in valid_players:
            pts_history = p.timeseries["points"].values.astype(float)

            # Baselines
            ep_rolling[p.id] = rolling_mean_predict(pts_history)
            ep_ewma[p.id] = ewma_predict(pts_history)

            # Inference pipeline
            try:
                pipeline = PlayerInferencePipeline()
                pipeline.ingest_observations(pts_history)
                pipeline.run()
                mu, var = pipeline.predict_next()
                ep_fused[p.id] = mu
                var_fused[p.id] = var
            except Exception as e:
                logger.debug("Inference failed for %s: %s", p.name, e)
                ep_fused[p.id] = ep_rolling[p.id]
                var_fused[p.id] = 4.0  # default variance

            # Collect inference metrics
            actual = actual_points_gw.get(p.id, 0.0)
            inf_metrics.add(
                predicted_mean=ep_fused[p.id],
                predicted_var=var_fused[p.id],
                actual=actual,
                model_preds={
                    "rolling_avg": ep_rolling[p.id],
                    "ewma": ep_ewma[p.id],
                    "hmm_kf_fused": ep_fused[p.id],
                },
            )

        # Optimization: solve under multiple strategies
        strategy_actual_pts = {}

        # Strategy 1: Greedy baseline with rolling avg
        try:
            greedy = GreedyOptimizer(budget=100.0)
            greedy_squad = greedy.optimize(valid_players, ep_rolling)
            greedy_actual = sum(actual_points_gw.get(p.id, 0.0) for p in greedy_squad.lineup.players)
            strategy_actual_pts["greedy_rolling"] = greedy_actual
        except Exception as e:
            logger.warning("Greedy failed: %s", e)

        # Strategy 2+: ILP with different risk aversions
        for lam in risk_lambdas:
            label = f"ilp_lambda_{lam:.1f}"
            try:
                opt = TwoLevelILPOptimizer(budget=100.0, risk_aversion=lam)
                full_squad = opt.optimize(valid_players, ep_fused, expected_variance=var_fused)
                actual_total = sum(actual_points_gw.get(p.id, 0.0) for p in full_squad.lineup.players)
                strategy_actual_pts[label] = actual_total
            except Exception as e:
                logger.warning("ILP (lambda=%.1f) failed: %s", lam, e)

        # Oracle
        oracle_pts = compute_oracle_squad(valid_players, actual_points_gw)

        opt_metrics.add_gameweek(target_gw, strategy_actual_pts, oracle_pts)

        elapsed = time.perf_counter() - t0
        logger.info(
            "GW%d done in %.1fs. Oracle=%.0f. Strategies: %s",
            target_gw,
            elapsed,
            oracle_pts,
            {k: f"{v:.0f}" for k, v in strategy_actual_pts.items()},
        )

    return {
        "season": season,
        "start_gw": start_gw,
        "end_gw": end_gw,
        "inference": inf_metrics.compute(),
        "optimization": opt_metrics.compute(),
    }


def main():
    parser = argparse.ArgumentParser(description="FPLX full-season backtest")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to local Fantasy-Premier-League clone (optional; fetches from GitHub if omitted)",
    )
    parser.add_argument("--season", default="2023-24")
    parser.add_argument("--start-gw", type=int, default=6)
    parser.add_argument("--end-gw", type=int, default=38)
    parser.add_argument("--output", default="results/backtest.json")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    results = run_backtest(
        season=args.season,
        data_dir=args.data_dir,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    inf = results["inference"]
    print(f"\n--- Inference (n={inf.get('n_predictions', 0)}) ---")
    print(f"  Fused MSE:          {inf.get('mse', 0):.3f}")
    print(f"  Fused MAE:          {inf.get('mae', 0):.3f}")
    print(f"  95% CI Coverage:    {inf.get('calibration_95', 0):.1%}")
    print(f"  Mean Log-Lik:       {inf.get('mean_log_likelihood', 0):.3f}")

    if "ablation" in inf:
        print("\n  Ablation (MSE):")
        for name, m in inf["ablation"].items():
            print(f"    {name:20s} {m['mse']:.3f}")

    opt = results["optimization"]
    print(f"\n--- Optimization ({opt.get('n_gameweeks', 0)} GWs) ---")
    print(f"  Oracle Total:       {opt.get('oracle_total', 0):.0f}")
    for name, s in opt.get("strategies", {}).items():
        print(f"\n  [{name}]")
        print(f"    Total Points:     {s['total_points']:.0f}")
        print(f"    Mean/GW:          {s['mean_per_gw']:.1f} +/- {s['std_per_gw']:.1f}")
        print(f"    Worst GW:         {s['worst_gw_points']:.0f}")
        print(f"    Optimality Gap:   {s['mean_optimality_gap']:.1%}")
        print(f"    %% of Oracle:      {s['pct_of_oracle']:.1f}%%")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
