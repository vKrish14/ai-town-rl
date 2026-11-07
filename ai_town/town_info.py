"""Source-of-truth town and establishment views derived from simulation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_town.buildings import BUILDINGS, presentation_buildings


@dataclass(frozen=True)
class BuildingIdentity:
    id: str
    name: str
    kind: str
    position: tuple[int, int]
    simulation_position: tuple[int, int]
    index: int


_BUILDING_LABELS = {
    "homes": "House",
    "offices": "Office",
    "groceries": "Grocery",
    "restaurant": "Restaurant",
    "clinic": "Clinic",
    "parks": "Park",
    "town": "Town centre",
    "police": "Police station",
}


def building_catalog(config: dict[str, Any] | None = None) -> tuple[BuildingIdentity, ...]:
    """Build stable named identities from the shared simulation/render layout."""
    cfg = config or {}
    configured = cfg.get("town", {}).get("establishments", {})
    visual = presentation_buildings()
    records = []
    for kind, display_positions in visual.items():
        # Civic safety lots are visual/observability landmarks and have no
        # simulation footprint or navigation effect.
        sim_positions = BUILDINGS.get(kind, display_positions)
        for index, position in enumerate(display_positions):
            building_id = f"{kind[:-1] if kind.endswith('s') else kind}_{index + 1:02d}"
            # Keep stable singular IDs even for irregular category names.
            if kind == "groceries":
                building_id = f"grocery_{index + 1:02d}"
            elif kind == "offices":
                building_id = f"office_{index + 1:02d}"
            elif kind == "homes":
                building_id = f"home_{index + 1:02d}"
            elif kind == "parks":
                building_id = f"park_{index + 1:02d}"
            elif kind == "town":
                building_id = "town_centre_01"
            elif kind == "police":
                building_id = "police_station_01"
            elif kind == "clinic":
                building_id = f"clinic_{index + 1:02d}"
            elif kind == "restaurant":
                building_id = f"restaurant_{index + 1:02d}"
            fallback = f"{_BUILDING_LABELS[kind]} {index + 1:02d}"
            name = str(configured.get(building_id, fallback)).strip() or fallback
            records.append(
                BuildingIdentity(
                    building_id,
                    name,
                    kind,
                    tuple(position),
                    tuple(sim_positions[index]),
                    index,
                )
            )
    return tuple(records)


def building_visual_bounds(building: BuildingIdentity) -> tuple[float, float, float, float]:
    """Return the visible building silhouette bounds in map cells.

    The decorative garden apron extends well in front of the walls. Selection
    should hug the architecture, rather than outlining that apron as empty space.
    Keep these dimensions in sync with the renderer's 32px source tile.
    """
    x, y = building.position
    if building.kind == "parks":
        return x, y, x + 6, y + 5
    if building.kind == "offices":
        return x - 4 / 32, y - 52 / 32, x + 84 / 32, y + 24 / 32
    if building.kind == "homes":
        roof_top = -52 if building.index % 3 == 1 else -46
        return x - 5 / 32, y + roof_top / 32, x + 79 / 32, y + 30 / 32
    # Service buildings have the shared gabled roof, with walls ending above
    # the garden apron at row 24 (y-relative source pixel).
    return x - 5 / 32, y - 52 / 32, x + 79 / 32, y + 24 / 32


def building_at_world_position(
    catalog: tuple[BuildingIdentity, ...], x: float, y: float
) -> BuildingIdentity | None:
    """Hit-test the same visible footprint used by the selected outline."""
    for building in catalog:
        left, top, right, bottom = building_visual_bounds(building)
        if left <= x <= right and top <= y <= bottom:
            return building
    return None


def town_identity(config: dict[str, Any] | None, town: Any) -> dict[str, Any]:
    """Return configured identity plus clearly marked unsupported metadata."""
    identity = (config or {}).get("town", {})
    return {
        "name": str(identity.get("name", "AI Town")),
        "description": str(identity.get("description", "A simulated town.")),
        "founding_day": int(identity.get("founding_day", 1)),
        "simulated_age_days": max(1, int(town.day) + 1),
        "season": "Not simulated",
        "weather": "Not simulated",
        "treasury": "Not modeled",
    }


def town_statistics(town: Any) -> dict[str, float | int]:
    """Return derived town statistics using current citizens and recorded events."""
    citizens = town.citizens
    living = [citizen for citizen in citizens if citizen.alive]
    wealth = [max(0.0, float(c.money)) for c in citizens]
    wealth_total = sum(wealth)
    n = len(wealth)
    gini = (
        sum(abs(a - b) for a in wealth for b in wealth) / (2 * n * wealth_total)
        if n and wealth_total > 0
        else 0.0
    )
    social_events = sum(event.get("event") == "social" for event in town.events)
    salaries = sum(
        float(event.get("data", {}).get("amount", 0.0))
        for event in town.events
        if event.get("event") == "earning"
    )
    shopping = sum(
        -float(event.get("data", {}).get("amount", 0.0))
        for event in town.events
        if event.get("event") == "purchase"
    )
    window_ticks = int(town.ticks_per_day) * 30
    recent_reports = sum(
        event.get("category") == "safety"
        and int(event.get("tick", -window_ticks)) > int(town.tick) - window_ticks
        for event in town.events
    )
    resident_days = sum(
        int(snapshot.get("living_population", 0))
        for snapshot in town.daily_summaries[-30:]
    )
    resident_days += sum(citizen.alive for citizen in citizens) * (
        (town.tick % max(1, town.ticks_per_day)) / max(1, town.ticks_per_day)
    )
    crime_rate = 1000.0 * recent_reports / resident_days if resident_days else 0.0
    calendar_date = town.calendar_date.isoformat()
    return {
        "population": n,
        "living_population": len(living),
        "average_wellbeing": (
            sum(town._wellbeing(c) for c in living) / len(living) if living else 0.0
        ),
        "employment_rate": (
            sum(bool(c.employed) for c in living) / len(living) if living else 0.0
        ),
        "wealth_total": wealth_total,
        "wealth_average": wealth_total / n if n else 0.0,
        "wealth_gini": gini,
        "social_satisfaction_average": (
            sum(float(c.social) for c in living) / len(living) if living else 0.0
        ),
        "social_interactions_recorded": social_events,
        "salary_payments_total": salaries,
        "shopping_spend_total": shopping,
        "food_stock_total": sum(town.food_stock),
        "food_price": float(town.food_price),
        "day": int(town.day + 1),
        "hour": int(town.tick % max(1, town.ticks_per_day)),
        "reported_crimes_30d": recent_reports,
        "crime_rate_per_1000_resident_days": crime_rate,
        "calendar_date": calendar_date,
    }


def inspect_building(town: Any, identity: BuildingIdentity) -> dict[str, Any]:
    """Build a type-aware inspector from real simulation and event records."""
    people = town.citizens
    if identity.kind == "homes":
        residents = [c for c in people if tuple(c.home) == identity.simulation_position]
        visitors = [c for c in people if c.position == identity.simulation_position]
    elif identity.kind == "police":
        residents = []
        visitors = []
    else:
        access_points = set(town._access_points(identity.kind))
        residents = []
        visitors = [c for c in people if tuple(c.position) in access_points]
    events = [e for e in town.events if e.get("building_id") == identity.id]
    data: dict[str, Any] = {
        "id": identity.id,
        "name": identity.name,
        "kind": identity.kind,
        "position": identity.position,
        "simulation_position": identity.simulation_position,
        "status": "Status tracking unavailable",
        "capacity": None,
        "occupancy": len(visitors),
        "occupancy_basis": "Resident currently at home" if identity.kind == "homes" else "Citizens currently at the entrance area",
        "residents": [c.name for c in residents],
        "visitors": [c.name for c in visitors],
        "recent_events": events[-8:][::-1],
    }
    if identity.kind == "homes":
        data["house_number"] = identity.index + 1
        data["household"] = [
            {
                "name": c.name,
                "wealth": float(c.money),
                "health": float(c.health),
                "wellbeing": float(town._wellbeing(c)),
                "employment": c.occupation if c.employed else "Unemployed",
            }
            for c in residents
        ]
    elif identity.kind == "offices":
        data["employees"] = [c.name for c in people if c.employed]
        data["vacancies"] = int(town.vacancies)
        data["employee_assignment_status"] = "Town employment is tracked; assignment to a specific office is not tracked"
        data["vacancy_scope"] = "Town-wide vacancies"
        data["salary_values"] = sorted({float(c.salary) for c in people if c.employed})
    elif identity.kind == "groceries":
        data.update(
            food_stock=int(town.food_stock[identity.index]),
            food_capacity=int(town.shared_economy.food_capacity),
            food_price=float(town.food_price),
            recorded_purchases=sum(e.get("event") == "purchase" for e in events),
        )
    elif identity.kind == "restaurant":
        meals = [e for e in events if e.get("event") == "purchase"]
        data.update(
            food_inventory=None,
            inventory_status="Not tracked by the simulation",
            recorded_meals=len(meals),
            recorded_revenue=sum(-float(e.get("data", {}).get("amount", 0)) for e in meals),
        )
    elif identity.kind == "clinic":
        data.update(
            recorded_visits=sum(e.get("event") == "clinic" for e in events),
            capacity_status="Not modeled",
        )
    elif identity.kind in {"parks", "town"}:
        data.update(
            social_interactions=sum(e.get("event") == "social" for e in events),
            capacity_status="Not modeled",
        )
    elif identity.kind == "police":
        officers = [c.name for c in people if c.occupation == "Police Officer"]
        data.update(
            status=f"{len(officers)} named officer(s); incident dispatch is not modeled",
            officers=officers,
            recorded_reports=sum(e.get("category") == "safety" for e in events),
            service_scope="Officers follow the shared RL policy; reports remain ambient telemetry",
        )
    return data
