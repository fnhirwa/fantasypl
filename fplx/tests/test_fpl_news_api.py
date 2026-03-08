"""Inspect FPL API News Data.
Usage:
    python examples/inspect_news_data.py
"""

import json
from collections import Counter
from pathlib import Path

import requests

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_bootstrap():
    """Fetch bootstrap-static data from FPL API."""
    print("Fetching bootstrap-static from FPL API...")
    resp = requests.get(FPL_BOOTSTRAP_URL)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Total players: {len(data['elements'])}")
    print(f"  Total teams: {len(data['teams'])}")
    print(f"  Total gameweeks: {len(data['events'])}")
    return data


def find_current_gameweek(data):
    """Extract current gameweek from events."""
    for event in data["events"]:
        if event.get("is_current"):
            return event["id"]
    return 1


def inspect_news(data):
    """Show all players with active news, grouped by status."""
    teams = {t["id"]: t["name"] for t in data["teams"]}
    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    current_gw = find_current_gameweek(data)

    print(f"\nCurrent Gameweek: {current_gw}")
    print("=" * 80)

    # Categorize players by status
    status_labels = {
        "a": "Available",
        "d": "Doubtful",
        "i": "Injured",
        "s": "Suspended",
        "u": "Unavailable",
        "n": "Not in squad",
    }

    status_counts = Counter()
    flagged_players = []

    for el in data["elements"]:
        status = el.get("status", "a")
        status_counts[status] += 1

        # Collect players with non-trivial news
        news = el.get("news", "") or ""
        if news.strip() or status not in ("a",):
            flagged_players.append({
                "id": el["id"],
                "name": el["web_name"],
                "team": teams.get(el["team"], "?"),
                "position": positions.get(el["element_type"], "?"),
                "status": status,
                "news": news,
                "chance_this": el.get("chance_of_playing_this_round"),
                "chance_next": el.get("chance_of_playing_next_round"),
                "news_added": el.get("news_added", ""),
                "price": el["now_cost"] / 10.0,
                "total_points": el.get("total_points", 0),
            })

    # Print status distribution
    print("\nPlayer Status Distribution:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        label = status_labels.get(status, status)
        print(f"  {label:<15} ({status}): {count}")

    print(f"\nPlayers with active news/flags: {len(flagged_players)}")
    print("=" * 80)

    # Print flagged players grouped by status
    for status_code in ["i", "s", "u", "d", "n"]:
        group = [p for p in flagged_players if p["status"] == status_code]
        if not group:
            continue

        label = status_labels.get(status_code, status_code)
        print(f"\n--- {label.upper()} ({len(group)} players) ---")
        print(f"{'Name':<18} {'Team':<14} {'Pos':<5} {'Chance':>7} {'News'}")
        print("-" * 80)

        for p in sorted(group, key=lambda x: -x["total_points"]):
            chance = p["chance_next"]
            chance_str = f"{chance}%" if chance is not None else "  N/A"
            news_short = p["news"][:42] + "..." if len(p["news"]) > 42 else p["news"]
            print(f"{p['name']:<18} {p['team']:<14} {p['position']:<5} {chance_str:>7} {news_short}")
    # Also show a few "available" players with news text (e.g., returning)
    available_with_news = [p for p in flagged_players if p["status"] == "a" and p["news"].strip()]
    if available_with_news:
        print(f"\n--- AVAILABLE WITH NEWS ({len(available_with_news)} players) ---")
        print(f"{'Name':<18} {'Team':<14} {'Pos':<5} {'Chance':>7} {'News'}")
        print("-" * 80)
        for p in sorted(available_with_news, key=lambda x: -x["total_points"])[:15]:
            chance = p["chance_next"]
            chance_str = f"{chance}%" if chance is not None else "  N/A"
            news_short = p["news"][:42] + "..." if len(p["news"]) > 42 else p["news"]
            print(f"{p['name']:<18} {p['team']:<14} {p['position']:<5} {chance_str:>7} {news_short}")

    return flagged_players


def save_snapshot(data, flagged_players):
    """Save raw data for offline testing."""
    current_gw = find_current_gameweek(data)
    cache_dir = Path.home() / ".fplx" / "snapshots"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filepath = cache_dir / f"bootstrap_gw{current_gw:02d}.json"
    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"\nSaved full bootstrap to: {filepath}")

    # Also save just the flagged players for quick inspection
    flagged_path = cache_dir / f"flagged_gw{current_gw:02d}.json"
    with open(flagged_path, "w") as f:
        json.dump(flagged_players, f, indent=2)
    print(f"Saved flagged players to: {flagged_path}")


def main():
    data = fetch_bootstrap()
    flagged = inspect_news(data)
    save_snapshot(data, flagged)


if __name__ == "__main__":
    main()
