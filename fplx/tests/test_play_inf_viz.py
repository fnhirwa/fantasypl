"""
Visualize Inference for a Single Player

Picks the highest-profile injured player from the current GW,
fetches their full history, runs inference with and without
news, and generates a diagnostic plot.

Usage:
    python tests/test_play_inf_viz.py
    python tests/test_play_inf_viz.py --player-id 301
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplx.data.news_collector import NewsCollector
from fplx.inference.hmm import N_STATES, STATE_NAMES
from fplx.inference.pipeline import PlayerInferencePipeline
from fplx.signals.news import NewsSignal

FPL_BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_bootstrap():
    resp = requests.get(f"{FPL_BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    return resp.json()


def fetch_player_history(player_id):
    url = f"{FPL_BASE_URL}/element-summary/{player_id}/"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()["history"]


def find_current_gameweek(bootstrap):
    for event in bootstrap["events"]:
        if event.get("is_current"):
            return event["id"]
    return 1


def pick_target_player(bootstrap, player_id=None):
    """Pick target: explicit ID, or highest-scoring injured player."""
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    elements_by_id = {el["id"]: el for el in bootstrap["elements"]}

    if player_id and player_id in elements_by_id:
        el = elements_by_id[player_id]
    else:
        # Find highest-scoring player with news
        candidates = [
            el
            for el in bootstrap["elements"]
            if el.get("status") in ("i", "d", "s", "u") or (el.get("news", "") or "").strip()
        ]
        if not candidates:
            # Fallback: just pick top scorer
            candidates = bootstrap["elements"]
        el = max(candidates, key=lambda x: x.get("total_points", 0))

    return {
        "id": el["id"],
        "name": el["web_name"],
        "full_name": f"{el.get('first_name', '')} {el.get('second_name', '')}".strip(),
        "team": teams.get(el["team"], "?"),
        "position": positions.get(el["element_type"], "?"),
        "status": el.get("status", "a"),
        "news": el.get("news", ""),
        "chance_next": el.get("chance_of_playing_next_round"),
        "price": el["now_cost"] / 10.0,
        "total_points": el.get("total_points", 0),
    }


def run_visualization(player_id=None):
    bootstrap = fetch_bootstrap()
    current_gw = find_current_gameweek(bootstrap)

    # Collect news
    collector = NewsCollector()
    collector.collect_from_bootstrap(bootstrap, current_gw)

    # Pick player
    player = pick_target_player(bootstrap, player_id)
    print(f"Target: {player['full_name']} ({player['name']})")
    print(f"  Team: {player['team']}, Position: {player['position']}")
    print(f"  Status: {player['status']}, News: {player['news']}")
    print(f"  Price: £{player['price']}m, Total points: {player['total_points']}")

    # Fetch history
    print(f"\nFetching gameweek history...")
    history = fetch_player_history(player["id"])
    if len(history) < 3:
        print(f"  Only {len(history)} gameweeks. Need at least 3.")
        return

    gameweeks = [gw["round"] for gw in history]
    points = np.array([gw["total_points"] for gw in history], dtype=float)
    minutes = np.array([gw["minutes"] for gw in history], dtype=float)

    print(f"  {len(history)} gameweeks loaded (GW{gameweeks[0]}-GW{gameweeks[-1]})")
    print(f"  Points: {points.tolist()}")

    # ---- Run inference: without news ----
    pipe_a = PlayerInferencePipeline()
    pipe_a.ingest_observations(points)
    result_a = pipe_a.run()
    ep_a, var_a = pipe_a.predict_next()

    # ---- Run inference: with news ----
    pipe_b = PlayerInferencePipeline()
    pipe_b.ingest_observations(points)

    news_signal = NewsSignal()
    snapshot = collector.get_player_news(player["id"], current_gw)
    injected = False
    signal_info = {}
    if snapshot:
        enriched_text = snapshot.to_news_signal_input()
        if enriched_text:
            signal_info = news_signal.generate_signal(enriched_text)
            pipe_b.inject_news(signal_info, timestep=len(points) - 1)
            injected = True
            print(f"\n  News signal injected at GW{gameweeks[-1]}:")
            print(f"    Text: '{enriched_text}'")
            print(f"    Availability: {signal_info['availability']:.2f}")
            print(f"    Minutes risk: {signal_info['minutes_risk']:.2f}")
            print(f"    Confidence:   {signal_info['confidence']:.2f}")

    result_b = pipe_b.run()
    ep_b, var_b = pipe_b.predict_next()

    # print comparison
    print(f"\n{'':=<70}")
    print(f"INFERENCE COMPARISON: {player['name']}")
    print(f"{'':=<70}")
    print(f"{'Metric':<30} {'No News':>12} {'With News':>12} {'Delta':>10}")
    print(f"{'':-<70}")
    print(f"{'E[P] next GW':<30} {ep_a:>12.2f} {ep_b:>12.2f} {ep_b - ep_a:>+10.2f}")
    print(
        f"{'Std next GW':<30} {np.sqrt(var_a):>12.2f} {np.sqrt(var_b):>12.2f} {np.sqrt(var_b) - np.sqrt(var_a):>+10.2f}"
    )
    print(
        f"{'Viterbi state (last GW)':<30} {STATE_NAMES[result_a.viterbi_path[-1]]:>12} {STATE_NAMES[result_b.viterbi_path[-1]]:>12}"
    )
    print(
        f"{'P(Injured) last GW':<30} {result_a.smoothed_beliefs[-1, 0]:>12.4f} {result_b.smoothed_beliefs[-1, 0]:>12.4f} {result_b.smoothed_beliefs[-1, 0] - result_a.smoothed_beliefs[-1, 0]:>+10.4f}"
    )
    print(
        f"{'P(Star) last GW':<30} {result_a.smoothed_beliefs[-1, 4]:>12.4f} {result_b.smoothed_beliefs[-1, 4]:>12.4f} {result_b.smoothed_beliefs[-1, 4] - result_a.smoothed_beliefs[-1, 4]:>+10.4f}"
    )

    # Generate plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gw_arr = np.array(gameweeks)
        T = len(points)

        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        fig.suptitle(
            f"{player['full_name']} ({player['team']}) — GW{current_gw} "
            f"[{player['status'].upper()}] {player['news'][:50]}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        # Panel 1: Points + predictions
        ax1 = axes[0]
        ax1.bar(gw_arr, points, color="#ddd", edgecolor="#999", width=0.6, label="Actual points")
        ax1.plot(gw_arr, result_a.fused_mean, color="#2980b9", lw=1.5, label="Fused (no news)")
        ax1.fill_between(
            gw_arr,
            result_a.fused_mean - 1.96 * np.sqrt(result_a.fused_var),
            result_a.fused_mean + 1.96 * np.sqrt(result_a.fused_var),
            alpha=0.1,
            color="#2980b9",
        )
        ax1.plot(gw_arr, result_b.fused_mean, color="#e74c3c", lw=2, label="Fused (with news)")
        ax1.fill_between(
            gw_arr,
            result_b.fused_mean - 1.96 * np.sqrt(result_b.fused_var),
            result_b.fused_mean + 1.96 * np.sqrt(result_b.fused_var),
            alpha=0.1,
            color="#e74c3c",
        )
        ax1.set_ylabel("Points")
        ax1.set_title("Point Estimates & 95% CI")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Viterbi states
        ax2 = axes[1]
        ax2.step(
            gw_arr, result_a.viterbi_path, where="mid", color="#2980b9", lw=1.5, label="Viterbi (no news)"
        )
        ax2.step(
            gw_arr,
            result_b.viterbi_path,
            where="mid",
            color="#e74c3c",
            lw=2,
            ls="--",
            label="Viterbi (with news)",
        )
        ax2.set_ylabel("State")
        ax2.set_yticks(range(N_STATES))
        ax2.set_yticklabels(STATE_NAMES)
        ax2.set_title("HMM Viterbi Path")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Smoothed posteriors (with news)
        ax3 = axes[2]
        im = ax3.imshow(
            result_b.smoothed_beliefs.T,
            aspect="auto",
            cmap="YlOrRd",
            extent=[gw_arr[0] - 0.5, gw_arr[-1] + 0.5, -0.5, N_STATES - 0.5],
            origin="lower",
            interpolation="nearest",
        )
        ax3.set_ylabel("State")
        ax3.set_yticks(range(N_STATES))
        ax3.set_yticklabels(STATE_NAMES)
        ax3.set_title("Smoothed Posteriors (with news injection)")
        plt.colorbar(im, ax=ax3, label="P(state)", shrink=0.8)

        # Panel 4: Uncertainty comparison
        ax4 = axes[3]
        ax4.plot(
            gw_arr, np.sqrt(result_a.kalman_uncertainty), color="#2980b9", lw=1, label="KF std (no news)"
        )
        ax4.plot(
            gw_arr, np.sqrt(result_b.kalman_uncertainty), color="#e74c3c", lw=1.5, label="KF std (with news)"
        )
        ax4.plot(
            gw_arr, np.sqrt(result_a.fused_var), color="#2980b9", lw=1.5, ls="--", label="Fused std (no news)"
        )
        ax4.plot(
            gw_arr, np.sqrt(result_b.fused_var), color="#e74c3c", lw=2, ls="--", label="Fused std (with news)"
        )
        ax4.set_xlabel("Gameweek")
        ax4.set_ylabel("Uncertainty (std)")
        ax4.set_title("Uncertainty Comparison")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_dir = Path(__file__).parent.parent / "output" / "inference_viz"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"inference_{player['name'].lower()}_gw{current_gw}.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.show(block=False)
        plt.close()
        print(f"\nPlot saved to: {filepath}")

    except ImportError:
        print("\nMatplotlib not available. Skipping plot.")


def main():
    parser = argparse.ArgumentParser(description="Visualize inference for a player")
    parser.add_argument(
        "--player-id",
        type=int,
        default=None,
        help="FPL player ID (default: auto-pick highest-scoring flagged player)",
    )
    args = parser.parse_args()
    run_visualization(args.player_id)


if __name__ == "__main__":
    main()
