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
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplx.core.player import Player
from fplx.data.vaastav_loader import VaastavLoader
from fplx.evaluation.metrics import InferenceMetrics, OptimizationMetrics
from fplx.inference.enriched import compute_xpoints, enriched_predict
from fplx.inference.pipeline import PlayerInferencePipeline
from fplx.selection.lagrangian import LagrangianOptimizer
from fplx.selection.optimizer import GreedyOptimizer, TwoLevelILPOptimizer
from fplx.utils.config import Config

logger = logging.getLogger(__name__)

MIN_HISTORY = 5


def estimate_availability(timeseries, lookback: int = 3) -> float:
    """Estimate P(plays next GW) from recent minutes."""
    if "minutes" not in timeseries.columns:
        return 1.0
    recent = timeseries["minutes"].values[-lookback:]
    if len(recent) == 0:
        return 1.0
    played = (recent > 0).mean()
    return 0.05 if played == 0.0 else float(played)


def filter_played(series: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    """Keep only values from GWs where player actually played."""
    mask = minutes > 0
    filtered = series[mask]
    return filtered if len(filtered) >= 2 else series


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
    enable_hmm_learning: bool = False,
    fusion_mode: str = "calibrated_alpha",
    fusion_params: dict | None = None,
    hmm_variance_floor: float = 1.0,
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
    enable_hmm_learning : bool
        If True, run per-player HMM Baum-Welch before inference. Disabled by
        default to avoid overfitting/unstable per-player EM on short histories.
    fusion_mode : str
        Fusion mode for combining HMM and Kalman outputs.
        Options: 'precision' or 'calibrated_alpha'.
    fusion_params : dict, optional
        Optional parameters for fusion mode (e.g., alpha grid settings).
    hmm_variance_floor : float
        Floor applied to HMM variance before fusion.

    Returns
    -------
    dict
        Combined inference and optimization metrics.
    """
    if risk_lambdas is None:
        risk_lambdas = [0.0, 0.5, 1.0]
    fusion_params = fusion_params or {}

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
        ep_enriched = {}  # enriched (xG/xA/BPS decomposition)
        var_enriched = {}
        ep_xfused = {}  # HMM+KF on xPoints (denoised signal)
        var_xfused = {}
        ep_fused = {}  # HMM+KF on raw points (baseline)
        var_fused = {}
        ep_rolling = {}
        ep_ewma = {}

        for p in valid_players:
            pts_raw = p.timeseries["points"].values.astype(float)
            mins = (
                p.timeseries["minutes"].values.astype(float)
                if "minutes" in p.timeseries.columns
                else np.ones(len(pts_raw)) * 90
            )
            avail = estimate_availability(p.timeseries)

            # Baselines (availability-adjusted)
            ep_rolling[p.id] = rolling_mean_predict(pts_raw) * avail
            ep_ewma[p.id] = ewma_predict(pts_raw) * avail

            # Enriched prediction (structural decomposition)
            enr_mu, enr_var = enriched_predict(p.timeseries, p.position)
            ep_enriched[p.id] = enr_mu
            var_enriched[p.id] = enr_var

            # Compute xPoints: per-GW expected points from underlying rates
            xpts_full = compute_xpoints(p.timeseries, p.position)
            xpts_played = filter_played(xpts_full, mins)
            pts_played = filter_played(pts_raw, mins)

            # HMM+KF on xPoints (denoised signal)
            hmm_xmu, kf_xmu = enr_mu, enr_mu  # fallbacks
            try:
                if len(xpts_played) >= MIN_HISTORY:
                    pipe_x = PlayerInferencePipeline(
                        hmm_variance_floor=hmm_variance_floor,
                        fusion_mode=fusion_mode,
                        fusion_params=fusion_params if fusion_params else None,
                    )
                    pipe_x.ingest_observations(xpts_played)
                    if enable_hmm_learning and len(xpts_played) >= 10:
                        pipe_x.learn_parameters(n_iter=10)
                    result_x = pipe_x.run()
                    mu_x, var_x = pipe_x.predict_next()
                    ep_xfused[p.id] = mu_x * avail
                    var_xfused[p.id] = avail * var_x + avail * (1 - avail) * mu_x**2
                    hmm_xmu = result_x.hmm_predicted_mean * avail
                    kf_xmu = result_x.kf_predicted_mean * avail
                else:
                    ep_xfused[p.id] = enr_mu
                    var_xfused[p.id] = enr_var
            except Exception as e:
                logger.debug("xPoints inference failed for %s: %s", p.name, e)
                ep_xfused[p.id] = enr_mu
                var_xfused[p.id] = enr_var

            # HMM+KF on raw points (for ablation comparison)
            hmm_mu, kf_mu = ep_rolling[p.id], ep_rolling[p.id]
            try:
                if len(pts_played) >= MIN_HISTORY:
                    pipe_raw = PlayerInferencePipeline(
                        hmm_variance_floor=hmm_variance_floor,
                        fusion_mode=fusion_mode,
                        fusion_params=fusion_params if fusion_params else None,
                    )
                    pipe_raw.ingest_observations(pts_played)
                    result_raw = pipe_raw.run()
                    mu_raw, var_raw = pipe_raw.predict_next()
                    ep_fused[p.id] = mu_raw * avail
                    var_fused[p.id] = avail * var_raw + avail * (1 - avail) * mu_raw**2
                    hmm_mu = result_raw.hmm_predicted_mean * avail
                    kf_mu = result_raw.kf_predicted_mean * avail
                else:
                    ep_fused[p.id] = ep_rolling[p.id]
                    var_fused[p.id] = 4.0
            except Exception as e:
                logger.debug("Raw inference failed for %s: %s", p.name, e)
                ep_fused[p.id] = ep_rolling[p.id]
                var_fused[p.id] = 4.0

            # Metrics: track enriched as primary (best predictor)
            actual = actual_points_gw.get(p.id, 0.0)
            inf_metrics.add(
                predicted_mean=ep_enriched[p.id],
                predicted_var=var_enriched[p.id],
                actual=actual,
                model_preds={
                    "rolling_avg": ep_rolling[p.id],
                    "ewma": ep_ewma[p.id],
                    "hmm_raw": hmm_mu,
                    "kf_raw": kf_mu,
                    "fused_raw": ep_fused[p.id],
                    "hmm_xpts": hmm_xmu,
                    "kf_xpts": kf_xmu,
                    "fused_xpts": ep_xfused[p.id],
                    "enriched": ep_enriched[p.id],
                },
            )

        # ========== Optimization strategies ==========
        strategy_actual_pts = {}

        # Greedy baseline
        try:
            greedy = GreedyOptimizer(budget=100.0)
            sq = greedy.optimize(valid_players, ep_rolling)
            strategy_actual_pts["greedy_rolling"] = sum(
                actual_points_gw.get(p.id, 0.0) for p in sq.lineup.players
            )
        except Exception as e:
            logger.warning("Greedy failed: %s", e)

        # ILP + enriched (best predictor) at different lambdas
        for lam in risk_lambdas:
            try:
                opt = TwoLevelILPOptimizer(budget=100.0, risk_aversion=lam)
                sq = opt.optimize(valid_players, ep_enriched, expected_variance=var_enriched)
                strategy_actual_pts[f"ilp_enriched_{lam:.1f}"] = sum(
                    actual_points_gw.get(p.id, 0.0) for p in sq.lineup.players
                )
            except Exception as e:
                logger.warning("ILP enriched (lam=%.1f) failed: %s", lam, e)

        # ILP + xPoints-fused (HMM+KF on denoised signal)
        for lam in risk_lambdas:
            try:
                opt = TwoLevelILPOptimizer(budget=100.0, risk_aversion=lam)
                sq = opt.optimize(valid_players, ep_xfused, expected_variance=var_xfused)
                strategy_actual_pts[f"ilp_xfused_{lam:.1f}"] = sum(
                    actual_points_gw.get(p.id, 0.0) for p in sq.lineup.players
                )
            except Exception as e:
                logger.warning("ILP xfused (lam=%.1f) failed: %s", lam, e)

        # ILP + EWMA baseline
        try:
            opt = TwoLevelILPOptimizer(budget=100.0, risk_aversion=0.0)
            sq = opt.optimize(valid_players, ep_ewma)
            strategy_actual_pts["ilp_ewma"] = sum(actual_points_gw.get(p.id, 0.0) for p in sq.lineup.players)
        except Exception as e:
            logger.warning("ILP+EWMA failed: %s", e)

        # Lagrangian + enriched
        try:
            lagr = LagrangianOptimizer(budget=100.0, max_iter=100)
            lr = lagr.solve(valid_players, ep_enriched, expected_variance=var_enriched)
            if lr.full_squad:
                strategy_actual_pts["lagr_enriched"] = sum(
                    actual_points_gw.get(p.id, 0.0) for p in lr.full_squad.lineup.players
                )
        except Exception as e:
            logger.warning("Lagrangian failed: %s", e)

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
    def resolve(cli_value: Any, cfg_value: Any, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if cfg_value is not None:
            return cfg_value
        return default

    parser = argparse.ArgumentParser(description="FPLX full-season backtest")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file (CLI flags override config values)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to local Fantasy-Premier-League clone (optional; fetches from GitHub if omitted)",
    )
    parser.add_argument("--season", default=None)
    parser.add_argument("--start-gw", type=int, default=None)
    parser.add_argument("--end-gw", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--enable-hmm-learning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable per-player HMM Baum-Welch learning",
    )
    parser.add_argument(
        "--fusion-mode",
        default=None,
        choices=["precision", "calibrated_alpha"],
        help="Fusion strategy for combining HMM and Kalman predictions",
    )
    parser.add_argument(
        "--fusion-alpha-grid-step",
        type=float,
        default=None,
        help="Grid step for calibrated alpha search",
    )
    parser.add_argument(
        "--fusion-alpha-default",
        type=float,
        default=None,
        help="Fallback alpha when there is insufficient history",
    )
    parser.add_argument(
        "--fusion-alpha-min-history",
        type=int,
        default=None,
        help="Minimum history length before alpha calibration",
    )
    parser.add_argument(
        "--hmm-variance-floor",
        type=float,
        default=None,
        help="Floor applied to HMM variance before fusion",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    cfg_dict = {}
    if args.config:
        cfg = Config()
        cfg.load_from_file(Path(args.config))
        cfg_dict = cfg.to_dict()

    inference_cfg = cfg_dict.get("inference", {})
    fusion_cfg = inference_cfg.get("fusion_params", {})

    season = resolve(args.season, cfg_dict.get("season"), "2023-24")
    data_dir = resolve(args.data_dir, cfg_dict.get("data_dir"), None)
    start_gw = int(resolve(args.start_gw, cfg_dict.get("start_gw"), 6))
    end_gw = int(resolve(args.end_gw, cfg_dict.get("end_gw"), 38))
    output = resolve(args.output, cfg_dict.get("output"), "results/backtest.json")
    enable_hmm_learning = bool(
        resolve(args.enable_hmm_learning, inference_cfg.get("enable_hmm_learning"), False)
    )
    fusion_mode = resolve(args.fusion_mode, inference_cfg.get("fusion_mode"), "calibrated_alpha")
    hmm_variance_floor = float(resolve(args.hmm_variance_floor, inference_cfg.get("hmm_variance_floor"), 1.0))
    fusion_params = {
        "grid_step": float(resolve(args.fusion_alpha_grid_step, fusion_cfg.get("grid_step"), 0.05)),
        "default_alpha": float(resolve(args.fusion_alpha_default, fusion_cfg.get("default_alpha"), 0.7)),
        "min_history": int(resolve(args.fusion_alpha_min_history, fusion_cfg.get("min_history"), 8)),
    }

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    results = run_backtest(
        season=season,
        data_dir=data_dir,
        start_gw=start_gw,
        end_gw=end_gw,
        enable_hmm_learning=enable_hmm_learning,
        fusion_mode=fusion_mode,
        fusion_params=fusion_params,
        hmm_variance_floor=hmm_variance_floor,
    )

    # Save
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
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

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
