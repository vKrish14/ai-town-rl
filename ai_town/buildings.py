"""Static town layout data."""

TOWN_WIDTH = 50
TOWN_HEIGHT = 32
BUILDINGS = {
    "homes": [
        (3, 3), (9, 3), (19, 3), (35, 3), (46, 3), (46, 13),
          (3, 19), (9, 20), (19, 19), (22, 19), (19, 25), (22, 25),
        (39, 19), (45, 19), (39, 24), (45, 24), (12, 3),
    ],
    "offices": [(27, 13), (32, 3), (40, 3), (27, 3)],
    "groceries": [(3, 13), (21, 13), (43, 13)],
    "restaurant": [(25, 17), (37, 13)],
    "parks": [(7, 13), (27, 20), (3, 25)],
    "clinic": [(22, 3), (43, 3)],
    "town": [(18, 13)],
    "police": [(28, 27)],
}


def presentation_buildings() -> dict[str, list[tuple[int, int]]]:
    """Return renderer lists copied from the shared, road-cleared site plan."""
    return {
        # Presentation lots are arranged as three legible neighborhoods around
        # a civic centre. Wider gaps leave room for yards, paths, and sightlines.
        # The renderer and transition grid use the same road-cleared parcels.
        # This prevents visual facilities from drifting away from their RL sites.
        **{kind: list(points) for kind, points in BUILDINGS.items()},
    }
