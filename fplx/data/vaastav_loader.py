"""Loader for the vaastav/Fantasy-Premier-League dataset.

Supports two modes:
  1. Remote: fetch CSVs directly from GitHub (no clone needed).
  2. Local: read from a cloned repo directory.

Usage (remote):
    loader = VaastavLoader(season="2023-24")
    players = loader.build_player_objects(up_to_gw=20)

Usage (local):
    loader = VaastavLoader(season="2023-24", data_dir="./Fantasy-Premier-League")
    players = loader.build_player_objects(up_to_gw=20)

Dataset: https://github.com/vaastav/Fantasy-Premier-League
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from fplx.core.player import Player

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

COLUMN_MAP = {
    "total_points": "points",
    "goals_scored": "goals",
    "expected_goals": "xG",
    "expected_assists": "xA",
    "GW": "gameweek",
    "round": "gameweek",
}

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
    "GK": "GK",
    "GKP": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}


class VaastavLoader:
    """
    Load historical FPL data from the vaastav dataset.

    Parameters
    ----------
    season : str
        Season string, e.g. "2023-24".
    data_dir : str or Path, optional
        Path to a local clone. If None, fetches from GitHub.
    cache_dir : str or Path, optional
        Where to cache downloaded CSVs. Defaults to ~/.fplx/vaastav/.
    """

    def __init__(
        self,
        season: str = "2023-24",
        data_dir: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
    ):
        self.season = season
        self.data_dir = Path(data_dir) if data_dir else None
        # Default cache is project-local to keep artifacts within the workspace.
        project_root = Path(__file__).resolve().parents[2]
        self.cache_dir = Path(cache_dir) if cache_dir else project_root / ".fplx" / "vaastav"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._merged_gw: Optional[pd.DataFrame] = None
        self._player_raw: Optional[pd.DataFrame] = None

    @staticmethod
    def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Coalesce duplicate-named columns into a single column.

        This handles schema variants where multiple source columns map to the
        same canonical name (e.g., ``GW`` and ``round`` -> ``gameweek``).
        """
        out = df.copy()
        duplicate_names = out.columns[out.columns.duplicated()].unique()

        for name in duplicate_names:
            cols = out.loc[:, out.columns == name]
            merged = cols.iloc[:, 0]
            for i in range(1, cols.shape[1]):
                merged = merged.combine_first(cols.iloc[:, i])

            out = out.loc[:, out.columns != name]
            out[name] = merged

        return out

    def _read_csv(self, relative_path: str) -> pd.DataFrame:
        """
        Read a CSV from local clone or GitHub, with caching.

        Parameters
        ----------
        relative_path : str
            Path relative to data/{season}/, e.g. "gws/merged_gw.csv".
        """
        cache_file = self.cache_dir / self.season / relative_path
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # 1. Try local clone
        if self.data_dir:
            local = self.data_dir / "data" / self.season / relative_path
            if local.exists():
                return pd.read_csv(local, encoding="utf-8-sig")

        # 2. Try cache
        if cache_file.exists():
            return pd.read_csv(cache_file, encoding="utf-8-sig")

        # 3. Fetch from GitHub
        url = f"{BASE_URL}/{self.season}/{relative_path}"
        logger.info("Fetching %s", url)
        df = pd.read_csv(url, encoding="utf-8-sig")

        # Cache for next time
        df.to_csv(cache_file, index=False)
        logger.info("Cached to %s", cache_file)
        return df

    def load_merged_gw(self) -> pd.DataFrame:
        """
        Load the merged gameweek file (all GWs, all players, one CSV).

        Returns
        -------
        pd.DataFrame
            One row per player-gameweek appearance.
        """
        if self._merged_gw is not None:
            return self._merged_gw

        df = self._read_csv("gws/merged_gw.csv")
        df = df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})
        df = self._coalesce_duplicate_columns(df)

        if "gameweek" in df.columns:
            df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")

        self._merged_gw = df
        logger.info(
            "Loaded merged_gw: %d rows, %d players, GW %d-%d",
            len(df),
            df["element"].nunique(),
            df["gameweek"].min(),
            df["gameweek"].max(),
        )
        return df

    def load_player_raw(self) -> pd.DataFrame:
        """Load season-level player metadata."""
        if self._player_raw is not None:
            return self._player_raw
        self._player_raw = self._read_csv("players_raw.csv")
        return self._player_raw

    def load_gameweek(self, gw: int) -> pd.DataFrame:
        """Load a single gameweek from merged data."""
        df = self.load_merged_gw()
        return df[df["gameweek"] == gw].copy()

    def build_player_objects(
        self,
        up_to_gw: Optional[int] = None,
    ) -> list[Player]:
        """
        Build Player objects with timeseries up to a given gameweek.

        Parameters
        ----------
        up_to_gw : int, optional
            Only include gameweeks 1..up_to_gw. If None, include all.

        Returns
        -------
        list[Player]
        """
        all_gw = self.load_merged_gw()

        if up_to_gw is not None:
            all_gw = all_gw[all_gw["gameweek"] <= up_to_gw]

        if all_gw.empty:
            return []

        players = []
        grouped = all_gw.groupby("element")

        for pid, grp in grouped:
            pid = int(pid)
            grp = grp.sort_values("gameweek").reset_index(drop=True)

            # Player metadata from the row itself
            name = str(grp["name"].iloc[0]) if "name" in grp.columns else f"Player_{pid}"
            team = str(grp["team"].iloc[0]) if "team" in grp.columns else "Unknown"
            pos_raw = grp["position"].iloc[0] if "position" in grp.columns else "MID"
            price = grp["value"].iloc[-1] / 10.0 if "value" in grp.columns else 5.0

            position = POSITION_MAP.get(pos_raw, POSITION_MAP.get(str(pos_raw), "MID"))

            # Build timeseries with available columns
            keep = [
                c
                for c in [
                    "gameweek",
                    "points",
                    "minutes",
                    "goals",
                    "assists",
                    "xG",
                    "xA",
                    "bonus",
                    "clean_sheets",
                    "goals_conceded",
                    "saves",
                    "bps",
                    "influence",
                    "creativity",
                    "threat",
                    "ict_index",
                    "value",
                    "selected",
                    "transfers_in",
                    "transfers_out",
                ]
                if c in grp.columns
            ]
            timeseries = grp[keep].copy()
            for col in timeseries.columns:
                timeseries[col] = pd.to_numeric(timeseries[col], errors="coerce")

            player = Player(
                id=pid,
                name=name,
                team=team,
                position=position,
                price=float(price),
                timeseries=timeseries,
            )
            players.append(player)

        logger.info("Built %d Player objects (up_to_gw=%s).", len(players), up_to_gw)
        return players

    def get_actual_points(self, gw: int) -> dict[int, float]:
        """
        Get actual points scored by each player in a specific gameweek.

        Returns
        -------
        dict[int, float]
            {player_id: actual_points}
        """
        df = self.load_gameweek(gw)
        pts_col = "points" if "points" in df.columns else "total_points"
        return dict(zip(df["element"].astype(int), df[pts_col].astype(float)))


__all__ = ["VaastavLoader"]
