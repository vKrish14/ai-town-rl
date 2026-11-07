"""Derived, renderer-independent town analytics snapshots."""

from __future__ import annotations

from typing import Any


def daily_town_snapshot(town: Any, day: int) -> dict[str, float | int]:
    """Capture an end-of-day summary from actual citizens and recorded events."""
    citizens = town.citizens
    living = [citizen for citizen in citizens if citizen.alive]
    wealth = [max(0.0, float(citizen.money)) for citizen in citizens]
    wealth_total = sum(wealth)
    count = len(wealth)
    relevant = [
        event for event in town.events
        if event.get("timestamp", {}).get("day") == day
    ]
    return {
        "day": int(day),
        "population": count,
        "living_population": len(living),
        "average_wellbeing": (
            sum(town._wellbeing(citizen) for citizen in living) / len(living)
            if living else 0.0
        ),
        "employment_rate": (
            sum(bool(citizen.employed) for citizen in living) / len(living)
            if living else 0.0
        ),
        "wealth_total": wealth_total,
        "wealth_average": wealth_total / count if count else 0.0,
        "social_satisfaction_average": (
            sum(float(citizen.social) for citizen in living) / len(living)
            if living else 0.0
        ),
        "social_interactions": sum(event.get("event") == "social" for event in relevant),
        "salary_payments": sum(
            float(event.get("data", {}).get("amount", 0.0))
            for event in relevant if event.get("event") == "earning"
        ),
        "shopping_spend": sum(
            -float(event.get("data", {}).get("amount", 0.0))
            for event in relevant if event.get("event") == "purchase"
        ),
        "food_stock": sum(town.food_stock),
    }
