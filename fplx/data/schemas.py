"""Data validation schemas for FPL data sources."""

from pydantic import BaseModel, Field


class BootstrapStatic(BaseModel):
    """Schema for the main FPL bootstrap-static endpoint."""

    events: list[dict] = Field(..., description="List of gameweek events.")
    teams: list[dict] = Field(..., description="List of all teams.")
    elements: list[dict] = Field(..., description="List of all players.")
    element_types: list[dict] = Field(
        ..., description="Mapping of element type IDs to positions."
    )


class Fixture(BaseModel):
    """Schema for a single fixture."""

    id: int
    kickoff_time: str
    team_h: int
    team_a: int
    team_h_difficulty: int
    team_a_difficulty: int


class PlayerHistory(BaseModel):
    """Schema for a player's historical performance data."""

    past: list[dict]
    history: list[dict]


class PlayerSummary(BaseModel):
    """Schema for a player's summary data."""

    fixtures: list[dict]
    history: list[dict]
    history_past: list[dict]
