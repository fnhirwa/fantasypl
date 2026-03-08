"""News collection and per-gameweek persistence."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NewsSnapshot:
    """
    A single player's news state at a specific gameweek.

    Attributes
    ----------
    player_id : int
    gameweek : int
    news_text : str
        Raw news string from FPL API.
    status : str
        FPL status code: "a", "d", "i", "s", "u", "n".
    chance_this_round : float or None
        Probability of playing this round (0-100 scale from API, stored as 0-1).
    chance_next_round : float or None
        Probability of playing next round (0-1).
    timestamp : str
        When the news was added (ISO format from API).
    """

    def __init__(
        self,
        player_id: int,
        gameweek: int,
        news_text: str = "",
        status: str = "a",
        chance_this_round: Optional[float] = None,
        chance_next_round: Optional[float] = None,
        timestamp: str = "",
    ):
        self.player_id = player_id
        self.gameweek = gameweek
        self.news_text = news_text
        self.status = status
        self.chance_this_round = chance_this_round
        self.chance_next_round = chance_next_round
        self.timestamp = timestamp

    def to_news_signal_input(self) -> str:
        """
        Convert to the text format that NewsSignal.generate_signal() expects.

        Combines the raw news text with status information to give the
        existing NewsParser richer input.
        """
        parts = []

        if self.news_text and self.news_text.strip():
            parts.append(self.news_text.strip())

        # Augment with status if not already implied by text
        status_text = {
            "i": "injured",
            "s": "suspended",
            "u": "unavailable",
            "d": "doubtful",
            "n": "not in squad",
        }
        if self.status in status_text and status_text[self.status] not in " ".join(parts).lower():
            parts.append(f"Status: {status_text[self.status]}")

        # Augment with chance percentage
        if self.chance_next_round is not None and self.chance_next_round < 1.0:
            pct = int(self.chance_next_round * 100)
            parts.append(f"{pct}% chance of playing")

        return ". ".join(parts) if parts else ""

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "gameweek": self.gameweek,
            "news_text": self.news_text,
            "status": self.status,
            "chance_this_round": self.chance_this_round,
            "chance_next_round": self.chance_next_round,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NewsSnapshot":
        return cls(**d)


class NewsCollector:
    """
    Collects and persists player news snapshots per gameweek.

    Usage (live):
        collector = NewsCollector(cache_dir="~/.fplx/news")
        collector.collect_from_bootstrap(bootstrap_data, gameweek=25)
        # Later, feed into inference:
        snapshots = collector.get_player_history(player_id=123)

    Usage (backtest):
        collector = NewsCollector(cache_dir="~/.fplx/news")
        # Load all pre-collected snapshots
        for gw in range(1, 39):
            snapshots = collector.get_gameweek(gw)
            # inject into pipeline per player

    Parameters
    ----------
    cache_dir : Path or str, optional
        Directory to persist snapshots as JSON.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".fplx" / "news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory store: {gameweek: {player_id: NewsSnapshot}}
        self._store: dict[int, dict[int, NewsSnapshot]] = {}

    # Collection from FPL API data
    def collect_from_bootstrap(self, bootstrap_data: dict, gameweek: int) -> int:
        """
        Extract news from a bootstrap-static API response.

        This is the key method. Call it each gameweek with fresh API data.

        Parameters
        ----------
        bootstrap_data : dict
            Response from https://fantasy.premierleague.com/api/bootstrap-static/
        gameweek : int
            Current gameweek number.

        Returns
        -------
        int
            Number of players with active news.
        """
        elements = bootstrap_data.get("elements", [])
        gw_snapshots = {}
        news_count = 0

        for el in elements:
            player_id = el["id"]

            news_text = el.get("news", "") or ""
            status = el.get("status", "a") or "a"

            # Convert API percentages (0-100 or None) to 0-1
            chance_this = el.get("chance_of_playing_this_round")
            chance_next = el.get("chance_of_playing_next_round")

            if chance_this is not None:
                chance_this = chance_this / 100.0
            if chance_next is not None:
                chance_next = chance_next / 100.0

            snapshot = NewsSnapshot(
                player_id=player_id,
                gameweek=gameweek,
                news_text=news_text,
                status=status,
                chance_this_round=chance_this,
                chance_next_round=chance_next,
                timestamp=el.get("news_added", ""),
            )

            gw_snapshots[player_id] = snapshot

            if news_text.strip() or status not in ("a",):
                news_count += 1

        self._store[gameweek] = gw_snapshots
        self._persist_gameweek(gameweek)

        logger.info(
            "GW %s: collected news for %d players (%d with active news)",
            gameweek,
            len(gw_snapshots),
            news_count,
        )
        return news_count

    def get_player_news(self, player_id: int, gameweek: int) -> Optional[NewsSnapshot]:
        """Get a specific player's news at a specific gameweek."""
        self._ensure_loaded(gameweek)
        gw_data = self._store.get(gameweek, {})
        return gw_data.get(player_id)

    def get_player_history(self, player_id: int) -> list[NewsSnapshot]:
        """
        Get all news snapshots for a player across all collected gameweeks.

        Returns list sorted by gameweek.
        """
        self._load_all()
        history = []
        for gw in sorted(self._store.keys()):
            snapshot = self._store[gw].get(player_id)
            if snapshot is not None:
                history.append(snapshot)
        return history

    def get_gameweek(self, gameweek: int) -> dict[int, NewsSnapshot]:
        """Get all player news for a specific gameweek."""
        self._ensure_loaded(gameweek)
        return self._store.get(gameweek, {})

    def get_players_with_news(self, gameweek: int) -> list[NewsSnapshot]:
        """Get only players with non-trivial news at a gameweek."""
        gw_data = self.get_gameweek(gameweek)
        return [snap for snap in gw_data.values() if snap.news_text.strip() or snap.status not in ("a",)]

    # Persistence (JSON per gameweek)
    def _persist_gameweek(self, gameweek: int):
        filepath = self.cache_dir / f"gw{gameweek:02d}.json"
        gw_data = self._store.get(gameweek, {})
        serialized = {str(pid): snap.to_dict() for pid, snap in gw_data.items()}
        with open(filepath, "w") as f:
            json.dump(serialized, f, indent=2)

    def _ensure_loaded(self, gameweek: int):
        if gameweek in self._store:
            return
        filepath = self.cache_dir / f"gw{gameweek:02d}.json"
        if filepath.exists():
            with open(filepath) as f:
                raw = json.load(f)
            self._store[gameweek] = {int(pid): NewsSnapshot.from_dict(data) for pid, data in raw.items()}

    def _load_all(self):
        for filepath in sorted(self.cache_dir.glob("gw*.json")):
            gw_str = filepath.stem.replace("gw", "")
            try:
                gw = int(gw_str)
                self._ensure_loaded(gw)
            except ValueError:
                continue

    # Bulk collection for backtesting
    def collect_season_from_api(self, data_loader) -> int:
        """
        Collect news for all gameweeks in a season.

        Requires calling the FPL API once per gameweek (the bootstrap-static
        endpoint only gives current-week news). For backtesting, you'd need
        to have cached the bootstrap data weekly during the season.

        For a single-shot collection (current state only), just call
        collect_from_bootstrap() once with the current bootstrap data and
        the current gameweek number.

        Parameters
        ----------
        data_loader : FPLDataLoader
            Your existing data loader.

        Returns
        -------
        int
            Number of gameweeks collected.
        """
        bootstrap = data_loader.fetch_bootstrap_data(force_refresh=True)

        # Determine current gameweek
        events = bootstrap.get("events", [])
        current_gw = 1
        for event in events:
            if event.get("is_current"):
                current_gw = event["id"]
                break

        self.collect_from_bootstrap(bootstrap, current_gw)
        return 1  # Only current GW available from a single API call


__all__ = ["NewsCollector", "NewsSnapshot"]
