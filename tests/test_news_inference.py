"""News Signal → Inference Pipeline (Live API)

Hits the FPL API, collects news, runs each flagged player
through the inference pipeline, and compares predictions
with vs. without news injection.

This is the key test: does news injection measurably shift
HMM beliefs and final predictions?

Usage:
    python examples/test_news_inference.py
"""

import logging
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

logging.basicConfig(level=logging.WARNING)

FPL_BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_bootstrap():
    """Fetch main FPL data."""
    print("Fetching bootstrap-static...")
    resp = requests.get(f"{FPL_BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    return resp.json()


def fetch_player_history(player_id):
    """Fetch detailed gameweek history for one player."""
    url = f"{FPL_BASE_URL}/element-summary/{player_id}/"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    history = data.get("history", [])
    return [gw.get("total_points", 0) for gw in history]


def find_current_gameweek(bootstrap):
    for event in bootstrap["events"]:
        if event.get("is_current"):
            return event["id"]
    return 1


def run_test():
    bootstrap = fetch_bootstrap()
    current_gw = find_current_gameweek(bootstrap)
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    print(f"Current Gameweek: {current_gw}")

    # Step 1: Collect news
    collector = NewsCollector()
    collector.collect_from_bootstrap(bootstrap, current_gw)
    flagged = collector.get_players_with_news(current_gw)

    print(f"Players with active news/flags: {len(flagged)}")

    # Step 2: Pick test players
    # Select a mix: some injured, some doubtful, some returning
    # Also include a few fully available high-scorers as controls
    status_labels = {
        "i": "Injured",
        "d": "Doubtful",
        "s": "Suspended",
        "u": "Unavailable",
        "n": "Not in squad",
        "a": "Available",
    }

    # Get top scorers for controls
    elements_by_id = {el["id"]: el for el in bootstrap["elements"]}
    top_available = sorted(
        [el for el in bootstrap["elements"] if el.get("status") == "a" and not el.get("news", "").strip()],
        key=lambda x: -x.get("total_points", 0),
    )[:3]

    # Flagged players sorted by total points (most impactful first)
    flagged_sorted = sorted(
        flagged, key=lambda s: -elements_by_id.get(s.player_id, {}).get("total_points", 0)
    )

    # Take top 10 flagged + 3 controls
    test_players = []
    for snap in flagged_sorted[:10]:
        el = elements_by_id[snap.player_id]
        test_players.append({
            "id": snap.player_id,
            "name": el["web_name"],
            "team": teams.get(el["team"], "?"),
            "position": positions.get(el["element_type"], "?"),
            "status": el.get("status", "a"),
            "news_text": snap.news_text,
            "has_news": True,
        })

    for el in top_available:
        test_players.append({
            "id": el["id"],
            "name": el["web_name"],
            "team": teams.get(el["team"], "?"),
            "position": positions.get(el["element_type"], "?"),
            "status": "a",
            "news_text": "",
            "has_news": False,
        })

    # tep 3: Run inference for each test player
    news_signal = NewsSignal()
    results = []

    print(f"\nFetching history and running inference for {len(test_players)} players...")
    print("-" * 90)

    for i, tp in enumerate(test_players):
        pid = tp["id"]
        name = tp["name"]

        # Fetch gameweek-by-gameweek points
        try:
            points = fetch_player_history(pid)
        except Exception as e:
            print(f"  [{i + 1}/{len(test_players)}] {name}: failed to fetch history ({e})")
            continue

        if len(points) < 3:
            print(f"  [{i + 1}/{len(test_players)}] {name}: insufficient history ({len(points)} GWs)")
            continue

        points_arr = np.array(points, dtype=float)

        # Pipeline A: WITHOUT news injection
        pipe_a = PlayerInferencePipeline()
        pipe_a.ingest_observations(points_arr)
        result_a = pipe_a.run()
        ep_a, var_a = pipe_a.predict_next()

        # Pipeline B: WITH news injection
        pipe_b = PlayerInferencePipeline()
        pipe_b.ingest_observations(points_arr)

        snapshot = collector.get_player_news(pid, current_gw)
        injected = False
        if snapshot:
            enriched_text = snapshot.to_news_signal_input()
            if enriched_text:
                signal = news_signal.generate_signal(enriched_text)
                pipe_b.inject_news(signal, timestep=len(points_arr) - 1)
                injected = True

        result_b = pipe_b.run()
        ep_b, var_b = pipe_b.predict_next()

        # Current inferred state
        state_a = STATE_NAMES[result_a.viterbi_path[-1]]
        state_b = STATE_NAMES[result_b.viterbi_path[-1]]

        # P(Injured) shift
        p_inj_a = result_a.smoothed_beliefs[-1, 0]
        p_inj_b = result_b.smoothed_beliefs[-1, 0]

        results.append({
            **tp,
            "n_gws": len(points),
            "last_5_avg": np.mean(points_arr[-5:]) if len(points_arr) >= 5 else np.mean(points_arr),
            "ep_no_news": ep_a,
            "ep_with_news": ep_b,
            "var_no_news": var_a,
            "var_with_news": var_b,
            "state_no_news": state_a,
            "state_with_news": state_b,
            "p_injured_no_news": p_inj_a,
            "p_injured_with_news": p_inj_b,
            "injected": injected,
        })

        tag = "NEWS" if injected else "CTRL"
        delta_ep = ep_b - ep_a
        delta_sign = "+" if delta_ep >= 0 else ""
        print(
            f"  [{i + 1}/{len(test_players)}] [{tag}] {name:<16} "
            f"E[P]: {ep_a:.2f} → {ep_b:.2f} ({delta_sign}{delta_ep:.2f})  "
            f"State: {state_a} → {state_b}  "
            f"P(Inj): {p_inj_a:.3f} → {p_inj_b:.3f}"
        )

        # Rate limit: be kind to the FPL API
        time.sleep(0.3)

    # Step 4: Summary
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    print(
        f"\n{'Player':<16} {'Status':<6} {'News':<30} {'E[P]':>6} {'ΔE[P]':>7} "
        f"{'State':>8} {'P(Inj)':>7} {'Injected'}"
    )
    print("-" * 90)

    for r in results:
        news_short = r["news_text"][:27] + "..." if len(r["news_text"]) > 27 else r["news_text"]
        if not news_short:
            news_short = "(no news)"
        delta = r["ep_with_news"] - r["ep_no_news"]
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.2f}"

        print(
            f"{r['name']:<16} {r['status']:<6} {news_short:<30} "
            f"{r['ep_with_news']:>6.2f} {delta_str:>7} "
            f"{r['state_with_news']:>8} {r['p_injured_with_news']:>7.3f} "
            f"{'  YES' if r['injected'] else '   NO'}"
        )

    # Step 5: Validate the key claim
    news_players = [r for r in results if r["injected"]]
    ctrl_players = [r for r in results if not r["injected"]]

    if news_players:
        print("\n--- Validation ---")

        # Injured/unavailable players should have lower E[P] after injection
        injured = [r for r in news_players if r["status"] in ("i", "u", "s")]
        if injured:
            avg_delta = np.mean([r["ep_with_news"] - r["ep_no_news"] for r in injured])
            avg_p_inj_shift = np.mean([r["p_injured_with_news"] - r["p_injured_no_news"] for r in injured])
            print(f"  Injured/suspended/unavailable players ({len(injured)}):")
            print(f"    Avg E[P] shift from news:    {avg_delta:+.3f} (should be negative)")
            print(f"    Avg P(Injured) shift:         {avg_p_inj_shift:+.3f} (should be positive)")

        # Doubtful players
        doubtful = [r for r in news_players if r["status"] == "d"]
        if doubtful:
            avg_delta = np.mean([r["ep_with_news"] - r["ep_no_news"] for r in doubtful])
            print(f"  Doubtful players ({len(doubtful)}):")
            print(f"    Avg E[P] shift from news:    {avg_delta:+.3f} (should be slightly negative)")
        # Controls should have zero shift
        if ctrl_players:
            avg_delta_ctrl = np.mean([r["ep_with_news"] - r["ep_no_news"] for r in ctrl_players])
            print(f"  Control players ({len(ctrl_players)}):")
            print(f"    Avg E[P] shift (no news):    {avg_delta_ctrl:+.3f} (should be ~0.000)")


def main():
    run_test()


if __name__ == "__main__":
    main()
