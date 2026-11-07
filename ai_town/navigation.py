"""Deterministic A* pathfinding on a four-neighbour grid."""

from __future__ import annotations
import heapq

Position = tuple[int, int]


def astar(
    start: Position,
    goal: Position,
    width: int,
    height: int,
    obstacles: set[Position],
    occupied: set[Position] | None = None,
) -> list[Position]:
    blocked = obstacles | set(occupied or ())
    blocked.discard(start)
    blocked.discard(goal)
    frontier = [(0, start)]
    came: dict[Position, Position | None] = {start: None}
    cost = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in (
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1),
        ):
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height) or nxt in blocked:
                continue
            new_cost = cost[current] + 1
            if new_cost < cost.get(nxt, 1e9):
                cost[nxt] = new_cost
                came[nxt] = current
                h = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + h, nxt))
    if goal not in came:
        return []
    path = []
    current = goal
    while current != start:
        path.append(current)
        previous = came[current]
        if previous is None:
            return []
        current = previous
    return list(reversed(path))
