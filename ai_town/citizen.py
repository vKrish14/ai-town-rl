"""Citizen domain state and bounded per-citizen history."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citizen:
    id: int
    name: str
    home: tuple[int, int]
    position: tuple[int, int]
    health: float = 80
    energy: float = 75
    hunger: float = 20
    happiness: float = 65
    stress: float = 20
    social: float = 45
    money: float = 35
    employed: bool = False
    occupation: str = "Unemployed"
    salary: float = 12
    food: int = 1
    personality: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    activity: str = "Wandering"
    pending_action: int | None = None
    target: tuple[int, int] | None = None
    path: list[tuple[int, int]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    financial_history: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    age: int = 20

    @property
    def alive(self) -> bool:
        return self.health > 0

    def record(self, event: dict[str, Any], limit: int = 2048) -> None:
        self.history.append(event)
        del self.history[:-limit]
