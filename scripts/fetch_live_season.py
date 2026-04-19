"""Fetch current-season FPL data and run FPLX for live gameweek deployment.

Usage:
    python scripts/fetch_live_season.py \
        --config config/fplx.config.json \
        --data-dir Fantasy \
        --output results/live_gw_2025_26.json

    # With risk penalty:
    python scripts/fetch_live_season.py --risk-lambda 0.5 --use-semivar

    # Test on a smaller pool:
    python scripts/fetch_live_season.py --max-players 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplx.core.player import Player
from fplx.data.double_gameweek import (
    aggregate_dgw_timeseries,
    get_fixture_counts_from_bootstrap,
    scale_predictions_for_dgw,
)
from fplx.inference.enriched import enriched_predict
from fplx.inference.multivariate_hmm import MultivariateHMM, build_feature_matrix
from fplx.selection.optimizer import TwoLevelILPOptimizer
from fplx.utils.config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── FPL API ───────────────────────────────────────────────────────────────────
BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
ELEMENT_URL = f"{BASE_URL}/element-summary/{{player_id}}/"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
STAT_RENAME = {
    "round": "gameweek",
    "total_points": "points",
    "minutes": "minutes",
    "goals_scored": "goals",
    "assists": "assists",
    "expected_goals": "xG",
    "expected_assists": "xA",
    "bonus": "bonus",
    "bps": "bps",
    "clean_sheets": "clean_sheets",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "saves": "saves",
    "goals_conceded": "goals_conceded",
    "own_goals": "own_goals",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
}


# ── Data fetching ─────────────────────────────────────────────────────────────


def fetch_bootstrap(cache_dir: Path, force_refresh: bool = False) -> dict:
    cache_file = cache_dir / "bootstrap_live.json"
    if not force_refresh and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < 6:
            logger.info("Bootstrap cache %.1fh old — using cached.", age_h)
            with open(cache_file) as f:
                return json.load(f)
    logger.info("Fetching bootstrap-static from FPL API...")
    resp = requests.get(BOOTSTRAP_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data


def fetch_player_history(
    player_id: int,
    cache_dir: Path,
    force_refresh: bool = False,
) -> list[dict]:
    cache_file = cache_dir / f"player_{player_id}.json"
    if not force_refresh and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < 12:
            with open(cache_file) as f:
                return json.load(f)
    url = ELEMENT_URL.format(player_id=player_id)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        history = resp.json().get("history", [])
        with open(cache_file, "w") as f:
            json.dump(history, f)
        return history
    except requests.RequestException as e:
        logger.warning("Failed to fetch history for player %d: %s", player_id, e)
        return []


def build_timeseries(history: list[dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df = df.rename(columns={k: v for k, v in STAT_RENAME.items() if k in df.columns})
    for col in [
        "points",
        "minutes",
        "goals",
        "assists",
        "xG",
        "xA",
        "bonus",
        "bps",
        "clean_sheets",
        "yellow_cards",
        "red_cards",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "gameweek" in df.columns:
        df = df.sort_values("gameweek").reset_index(drop=True)
    # Collapse DGW rows to one-per-GW with per-fixture normalised scores.
    # This ensures inference components always see single-game-equivalent data.
    df = aggregate_dgw_timeseries(df)
    return df


def get_current_gw(events: list[dict]) -> int:
    for ev in events:
        if ev.get("is_current"):
            return ev["id"]
    for ev in events:
        if ev.get("is_next"):
            return ev["id"]
    finished = [ev["id"] for ev in events if ev.get("finished")]
    return max(finished) + 1 if finished else 1


# ── Player construction ───────────────────────────────────────────────────────


def build_players(
    bootstrap: dict,
    cache_dir: Path,
    target_gw: int,
    min_history: int = 5,
    max_players: int | None = None,
    force_refresh: bool = False,
) -> list[Player]:
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements = bootstrap["elements"]

    if max_players is not None:
        elements = sorted(elements, key=lambda e: -e.get("total_points", 0))
        elements = elements[:max_players]

    players: list[Player] = []
    for i, elem in enumerate(elements):
        if i % 100 == 0:
            logger.info("Fetching history: %d / %d players", i, len(elements))

        player_id = elem["id"]
        ts = build_timeseries(fetch_player_history(player_id, cache_dir, force_refresh))

        # Keep only GWs before the target gameweek
        if not ts.empty and "gameweek" in ts.columns:
            ts = ts[ts["gameweek"] < target_gw].copy()

        if ts.empty:
            ts = pd.DataFrame({
                "gameweek": [0],
                "points": [0.0],
                "minutes": [0.0],
                "xG": [0.0],
                "xA": [0.0],
                "bonus": [0.0],
                "bps": [0.0],
                "clean_sheets": [0.0],
                "yellow_cards": [0.0],
                "red_cards": [0.0],
            })

        chance = elem.get("chance_of_playing_next_round")
        avail = 1.0 if chance is None else chance / 100.0
        news_txt = elem.get("news", "") or ""

        players.append(
            Player(
                id=player_id,
                name=elem["web_name"],
                team=teams.get(elem["team"], "Unknown"),
                position=POSITION_MAP.get(elem["element_type"], "MID"),
                price=elem["now_cost"] / 10.0,
                timeseries=ts,
                news={"text": news_txt, "availability": avail, "summary": news_txt},
            )
        )

    logger.info("Built %d players with history up to GW%d", len(players), target_gw - 1)
    return players


# ── Inference ─────────────────────────────────────────────────────────────────


def run_inference(
    players: list[Player],
    target_gw: int,
    config: Config,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Run enriched + MV-HMM blend inference.

    Returns
    -------
    expected_points : dict[int, float]
    variances       : dict[int, float]
    downside_risks  : dict[int, float]  (semi-deviation, for semivar ILP)
    """
    expected_points: dict[int, float] = {}
    variances: dict[int, float] = {}
    downside_risks: dict[int, float] = {}

    for player in players:
        ts = player.timeseries

        # ── Availability ──────────────────────────────────────────────────────
        if ts.empty or len(ts) < 3:
            expected_points[player.id] = 0.0
            variances[player.id] = 4.0
            downside_risks[player.id] = 0.0
            continue

        recent_mins = ts["minutes"].values[-3:] if "minutes" in ts.columns else [1, 1, 1]
        avail = max(0.05, float((np.array(recent_mins) > 0).mean()))
        avail = min(avail, float(player.news.get("availability", 1.0)))

        try:
            # ── Enriched prediction ───────────────────────────────────────────
            # Signature: enriched_predict(timeseries, position, ...)
            # Returns:   (expected_points, variance, downside_risk)  — 3 values
            enr_mu, enr_var, enr_dr = enriched_predict(ts, player.position)

            # ── MV-HMM prediction ─────────────────────────────────────────────
            # Constructor: MultivariateHMM(position=str)  — no n_states/n_iter
            # fit:         mvhmm.fit(obs, n_iter=int, prior_weight=float)
            # predict:     mvhmm.predict_next_points(obs) -> (float, float)
            feat = build_feature_matrix(ts, position=player.position)  # always ndarray

            if feat.shape[0] >= 3:
                mvhmm = MultivariateHMM(position=player.position)
                mvhmm.fit(feat, n_iter=15, prior_weight=0.85)
                mvhmm_mu, mvhmm_var = mvhmm.predict_next_points(feat)
            else:
                mvhmm_mu, mvhmm_var = enr_mu, enr_var

            # ── Calibrated blend (default alpha=0.8 enriched) ─────────────────
            alpha = 0.8
            mu = alpha * enr_mu + (1.0 - alpha) * mvhmm_mu
            var = alpha**2 * enr_var + (1.0 - alpha) ** 2 * mvhmm_var
            # Semi-deviation: blend enriched DR with MV-HMM equivalent
            dr = alpha * enr_dr + (1.0 - alpha) * (max(mvhmm_var, 0.0) ** 0.5 / 2**0.5)

        except Exception as e:
            logger.debug("Inference failed for %s (%d): %s", player.name, player.id, e)
            pts = ts["points"] if "points" in ts.columns else pd.Series([0.0])
            mu = float(pts.tail(5).mean())
            var = float(pts.tail(5).var(ddof=0)) + 1.0
            dr = var**0.5 / 2**0.5

        expected_points[player.id] = max(0.0, float(mu) * avail)
        variances[player.id] = max(0.1, float(var) * avail)
        downside_risks[player.id] = max(0.0, float(dr) * avail)

    return expected_points, variances, downside_risks


# ── Output formatting ─────────────────────────────────────────────────────────


def format_squad_output(
    squad,  # FullSquad
    players: list[Player],
    expected_points: dict[int, float],
    variances: dict[int, float],
    target_gw: int,
    season: str = "2025-26",
) -> dict:
    """Format FullSquad as a structured dict with player names."""
    lineup_ids = {p.id for p in squad.lineup.players}

    def row(p: Player, role: str) -> dict:
        ep = expected_points.get(p.id, 0.0)
        var = variances.get(p.id, 0.0)
        return {
            "id": p.id,
            "name": p.name,
            "team": p.team,
            "position": p.position,
            "price": p.price,
            "predicted_points": round(ep, 2),
            "predicted_std": round(var**0.5, 2),
            "role": role,
            "is_captain": (squad.lineup.captain is not None and p.id == squad.lineup.captain.id),
        }

    lineup_rows = sorted(
        [row(p, "starter") for p in squad.lineup.players],
        key=lambda r: r["position"],
    )
    # bench = squad members not in lineup, accessed via squad.squad_players
    bench_players = [p for p in squad.squad_players if p.id not in lineup_ids]
    bench_rows = [row(p, "bench") for p in bench_players]

    # Vice-captain = highest EP among starters excluding captain
    vc_name = None
    non_cap = [r for r in lineup_rows if not r["is_captain"]]
    if non_cap:
        vc_name = max(non_cap, key=lambda r: r["predicted_points"])["name"]

    return {
        "season": season,
        "gameweek": target_gw,
        "formation": squad.lineup.formation,
        "total_squad_cost": round(sum(p.price for p in squad.squad_players), 1),
        "predicted_lineup_points": round(sum(r["predicted_points"] for r in lineup_rows), 2),
        "captain": squad.lineup.captain.name if squad.lineup.captain else None,
        "vice_captain": vc_name,
        "lineup": lineup_rows,
        "bench": bench_rows,
    }


def print_squad(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"FPLX SQUAD — {result['season']} GW{result['gameweek']}")
    print("=" * 70)
    print(f"Formation:     {result['formation']}")
    print(f"Squad cost:    £{result['total_squad_cost']}m")
    print(f"Predicted pts: {result['predicted_lineup_points']:.1f}")
    print(f"Captain:       {result['captain']}  |  VC: {result['vice_captain']}")
    print()
    print(f"{'Pos':<5} {'Name':<25} {'Club':<20} {'£':>5} {'EP':>5} {'σ':>5}  Role")
    print("-" * 70)
    for p in result["lineup"]:
        cap = " [C]" if p["is_captain"] else ""
        print(
            f"{p['position']:<5} {p['name']:<25} {p['team']:<20} "
            f"£{p['price']:<4.1f} {p['predicted_points']:>5.1f} "
            f"{p['predicted_std']:>5.2f}{cap}"
        )
    print()
    print("BENCH:")
    for p in result["bench"]:
        print(
            f"{p['position']:<5} {p['name']:<25} {p['team']:<20} "
            f"£{p['price']:<4.1f} {p['predicted_points']:>5.1f} "
            f"{p['predicted_std']:>5.2f}"
        )
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FPLX live GW deployment")
    p.add_argument("--config", default="config/fplx.config.json")
    p.add_argument("--data-dir", default="Fantasy")
    p.add_argument("--target-gw", type=int, default=None)
    p.add_argument("--budget", type=float, default=100.0)
    p.add_argument("--risk-lambda", type=float, default=0.0)
    p.add_argument("--use-semivar", action="store_true")
    p.add_argument("--min-history", type=int, default=5)
    p.add_argument("--max-players", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--season", default="2025-26")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = Config()
    if Path(args.config).exists():
        config.load_from_file(Path(args.config))

    cache_dir = Path(args.data_dir) / "live_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    bootstrap = fetch_bootstrap(cache_dir, force_refresh=args.force_refresh)
    target_gw = args.target_gw or get_current_gw(bootstrap["events"])
    logger.info("Target gameweek: GW%d", target_gw)

    # ── Players ───────────────────────────────────────────────────────────────
    players = build_players(
        bootstrap,
        cache_dir,
        target_gw=target_gw,
        min_history=args.min_history,
        max_players=args.max_players,
        force_refresh=args.force_refresh,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running inference pipeline...")
    expected_points, variances, downside_risks = run_inference(players, target_gw, config)

    # ── DGW / BGW scaling ─────────────────────────────────────────────────────
    # Scale single-game predictions for players with multiple fixtures (DGW)
    # or zero fixtures (BGW) in the target gameweek.
    fixture_counts = get_fixture_counts_from_bootstrap(bootstrap, target_gw)
    n_dgw = sum(1 for n in fixture_counts.values() if n > 1)
    n_bgw = sum(1 for n in fixture_counts.values() if n == 0)
    if n_dgw:
        logger.info("GW%d: %d DGW players — scaling E[P] and Var[P].", target_gw, n_dgw)
    if n_bgw:
        logger.info("GW%d: %d BGW players — zeroing predictions.", target_gw, n_bgw)
    expected_points, variances, downside_risks = scale_predictions_for_dgw(
        expected_points, variances, downside_risks, fixture_counts
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    logger.info("Running two-level ILP (lambda=%.1f, semivar=%s)...", args.risk_lambda, args.use_semivar)

    optimizer = TwoLevelILPOptimizer(
        budget=args.budget,
        max_from_team=3,
        risk_aversion=args.risk_lambda,
    )
    squad = optimizer.optimize(
        players=players,
        expected_points=expected_points,
        expected_variance=variances,
        downside_risk=downside_risks if args.use_semivar else None,
    )

    # ── Output ────────────────────────────────────────────────────────────────
    result = format_squad_output(
        squad,
        players,
        expected_points,
        variances,
        target_gw=target_gw,
        season=args.season,
    )
    print_squad(result)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
