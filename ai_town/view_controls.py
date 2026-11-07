"""Small, renderer-only mappings for keyboard and camera pointer controls."""

from __future__ import annotations

import pygame


_KEY_CONTROLS = {
    pygame.K_SPACE: "pause",
    pygame.K_r: "reset",
    pygame.K_f: "follow",
    pygame.K_a: "analytics",
    pygame.K_b: "all_bars",
    pygame.K_c: "compare",
    pygame.K_TAB: "next_compare_citizen",
    pygame.K_1: "random_policy",
    pygame.K_2: "rule_policy",
    pygame.K_3: "learned_policy",
    pygame.K_4: "ppo_policy",
    pygame.K_5: "mappo_policy",
    pygame.K_m: "toggle_sound",
    pygame.K_EQUALS: "speed_up",
    pygame.K_PLUS: "speed_up",
    pygame.K_MINUS: "speed_down",
    pygame.K_LEFT: "pan_left",
    pygame.K_RIGHT: "pan_right",
    pygame.K_UP: "pan_up",
    pygame.K_DOWN: "pan_down",
}


def keyboard_control(key: int) -> str | None:
    """Return the viewer action assigned to a Pygame key, if any."""
    return _KEY_CONTROLS.get(key)


def camera_key_delta(key: int, pixels: int = 40) -> tuple[int, int] | None:
    """Return the camera-target delta for an arrow key."""
    return {
        pygame.K_LEFT: (pixels, 0),
        pygame.K_RIGHT: (-pixels, 0),
        pygame.K_UP: (0, pixels),
        pygame.K_DOWN: (0, -pixels),
    }.get(key)


def mouse_pan_event(
    event_type: int,
    button: int | None,
    position: tuple[int, int],
    anchor: tuple[int, int] | None,
) -> tuple[tuple[int, int] | None, tuple[int, int]]:
    """Update right-drag anchor and return the incremental camera delta."""
    if event_type == pygame.MOUSEBUTTONDOWN and button == 3:
        return position, (0, 0)
    if event_type == pygame.MOUSEBUTTONUP and button == 3:
        return None, (0, 0)
    if event_type == pygame.MOUSEMOTION and anchor is not None:
        return position, (position[0] - anchor[0], position[1] - anchor[1])
    return anchor, (0, 0)
