import pygame

from ai_town.view_controls import camera_key_delta, keyboard_control, mouse_pan_event


def test_keyboard_shortcuts_map_to_viewer_actions():
    expected = {
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
        pygame.K_EQUALS: "speed_up",
        pygame.K_PLUS: "speed_up",
        pygame.K_MINUS: "speed_down",
    }
    assert {key: keyboard_control(key) for key in expected} == expected
    assert keyboard_control(pygame.K_q) is None


def test_arrow_keys_pan_camera_in_expected_directions():
    assert camera_key_delta(pygame.K_LEFT) == (40, 0)
    assert camera_key_delta(pygame.K_RIGHT) == (-40, 0)
    assert camera_key_delta(pygame.K_UP) == (0, 40)
    assert camera_key_delta(pygame.K_DOWN) == (0, -40)
    assert camera_key_delta(pygame.K_q) is None


def test_right_drag_uses_incremental_deltas_and_ignores_left_click():
    anchor, delta = mouse_pan_event(pygame.MOUSEBUTTONDOWN, 3, (100, 80), None)
    assert (anchor, delta) == ((100, 80), (0, 0))
    anchor, delta = mouse_pan_event(pygame.MOUSEMOTION, None, (113, 74), anchor)
    assert (anchor, delta) == ((113, 74), (13, -6))
    anchor, delta = mouse_pan_event(pygame.MOUSEBUTTONUP, 3, (113, 74), anchor)
    assert (anchor, delta) == (None, (0, 0))
    assert mouse_pan_event(pygame.MOUSEBUTTONDOWN, 1, (20, 20), None) == (None, (0, 0))
