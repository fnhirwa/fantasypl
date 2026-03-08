"""
Example 4: Batch Inference — Full Squad Ranking
=================================================
Runs inference for ALL players (or a filtered subset),
compares E[P] with and without news injection, and shows
how news shifts the squad ranking.

This directly demonstrates the pipeline that feeds into
the ILP optimizer.

Usage:
    python tests/test_batch_inference.py
    python tests/test_batch_inference.py --top 50    # only top 50 by total points
    python tests/test_batch_inference.py --position MID
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplx.data.news_collector import NewsCollector
from fplx.inference.hmm import STATE_NAMES
from fplx.inference.pipeline import PlayerInferencePipeline
from fplx.signals.news import NewsSignal

FPL_BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_bootstrap():
    print("Fetching bootstrap-static...")
    resp = requests.get(f"{FPL_BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    return resp.json()


def fetch_player_history(player_id):
    url = f"{FPL_BASE_URL}/element-summary/{player_id}/"
    resp = requests.get(url)
    resp.raise_for_status()
    return [gw["total_points"] for gw in resp.json().get("history", [])]


def run_batch(top_n=30, position_filter=None):
    bootstrap = fetch_bootstrap()
    current_gw = 1
    for event in bootstrap["events"]:
        if event.get("is_current"):
            current_gw = event["id"]
            break

    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    # Collect news
    collector = NewsCollector()
    collector.collect_from_bootstrap(bootstrap, current_gw)

    # Filter and sort players
    elements = bootstrap["elements"]
    if position_filter:
        pos_id = {v: k for k, v in positions.items()}.get(position_filter.upper())
        if pos_id:
            elements = [el for el in elements if el["element_type"] == pos_id]

    elements = sorted(elements, key=lambda x: -x.get("total_points", 0))[:top_n]

    print(f"GW{current_gw}: Running inference for {len(elements)} players")
    if position_filter:
        print(f"  Position filter: {position_filter.upper()}")
    print()

    news_signal = NewsSignal()
    results = []

    for i, el in enumerate(elements):
        pid = el["id"]
        name = el["web_name"]

        try:
            points = fetch_player_history(pid)
        except Exception:
            continue

        if len(points) < 3:
            continue

        points_arr = np.array(points, dtype=float)

        # Without news
        pipe_a = PlayerInferencePipeline()
        pipe_a.ingest_observations(points_arr)
        pipe_a.run()
        ep_a, var_a = pipe_a.predict_next()

        # With news
        pipe_b = PlayerInferencePipeline()
        pipe_b.ingest_observations(points_arr)

        snapshot = collector.get_player_news(pid, current_gw)
        injected = False
        if snapshot:
            enriched = snapshot.to_news_signal_input()
            if enriched:
                sig = news_signal.generate_signal(enriched)
                pipe_b.inject_news(sig, timestep=len(points_arr) - 1)
                injected = True

        result_b = pipe_b.run()
        ep_b, var_b = pipe_b.predict_next()

        results.append({
            "id": pid,
            "name": name,
            "team": teams.get(el["team"], "?"),
            "pos": positions.get(el["element_type"], "?"),
            "price": el["now_cost"] / 10.0,
            "status": el.get("status", "a"),
            "news": (el.get("news", "") or "")[:30],
            "ep_no_news": ep_a,
            "ep_with_news": ep_b,
            "std_with_news": np.sqrt(var_b),
            "state": STATE_NAMES[result_b.viterbi_path[-1]],
            "injected": injected,
        })

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(elements)}...")

        time.sleep(0.25)

    # Rankings
    print(f"\n{'':=<95}")
    print(f"SQUAD RANKING — GW{current_gw} (by E[P] with news)")
    print(f"{'':=<95}")

    rank_with_news = sorted(results, key=lambda x: -x["ep_with_news"])

    print(
        f"{'#':<4} {'Player':<16} {'Team':<12} {'Pos':<5} {'£':>5} "
        f"{'E[P]':>6} {'Std':>5} {'State':<8} {'Δ':>6} {'News'}"
    )
    print("-" * 95)

    for rank, r in enumerate(rank_with_news, 1):
        delta = r["ep_with_news"] - r["ep_no_news"]
        delta_str = f"{delta:+.2f}" if abs(delta) > 0.005 else "  ---"
        news_str = r["news"] if r["injected"] else ""

        print(
            f"{rank:<4} {r['name']:<16} {r['team']:<12} {r['pos']:<5} {r['price']:>5.1f} "
            f"{r['ep_with_news']:>6.2f} {r['std_with_news']:>5.2f} "
            f"{r['state']:<8} {delta_str:>6} {news_str}"
        )

    # Rank shift analysis
    rank_no_news = sorted(results, key=lambda x: -x["ep_no_news"])
    rank_map_no_news = {r["id"]: i for i, r in enumerate(rank_no_news)}
    rank_map_with_news = {r["id"]: i for i, r in enumerate(rank_with_news)}

    shifted = []
    for r in results:
        old_rank = rank_map_no_news[r["id"]]
        new_rank = rank_map_with_news[r["id"]]
        shift = old_rank - new_rank  # positive = moved up
        if shift != 0:
            shifted.append({**r, "old_rank": old_rank + 1, "new_rank": new_rank + 1, "shift": shift})

    if shifted:
        shifted_sorted = sorted(shifted, key=lambda x: -abs(x["shift"]))
        print(f"\n--- Rank Shifts from News Injection ---")
        print(f"{'Player':<16} {'Old #':>6} {'New #':>6} {'Shift':>6} {'News'}")
        print("-" * 60)
        for s in shifted_sorted[:15]:
            direction = "↑" if s["shift"] > 0 else "↓"
            print(
                f"{s['name']:<16} {s['old_rank']:>6} {s['new_rank']:>6} "
                f"{direction}{abs(s['shift']):>5} {s['news']}"
            )
    else:
        print("\n  No rank shifts from news injection in this batch.")

    # Risk-adjusted ranking
    risk_aversion = 0.5
    print(f"\n--- Risk-Adjusted Ranking (λ={risk_aversion}) ---")
    print(f"{'#':<4} {'Player':<16} {'E[P]':>6} {'Std':>5} {'Adj Score':>10}")
    print("-" * 50)

    for rank, r in enumerate(
        sorted(results, key=lambda x: -(x["ep_with_news"] - risk_aversion * x["std_with_news"])), 1
    ):
        adj = r["ep_with_news"] - risk_aversion * r["std_with_news"]
        print(f"{rank:<4} {r['name']:<16} {r['ep_with_news']:>6.2f} {r['std_with_news']:>5.2f} {adj:>10.2f}")
        if rank >= 20:
            break


def main():
    parser = argparse.ArgumentParser(description="Batch inference for squad ranking")
    parser.add_argument("--top", type=int, default=30, help="Number of top players to evaluate")
    parser.add_argument("--position", type=str, default=None, help="Filter by position: GK, DEF, MID, FWD")
    args = parser.parse_args()
    run_batch(top_n=args.top, position_filter=args.position)


if __name__ == "__main__":
    main()
