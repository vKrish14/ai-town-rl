import pytest
import pygame

from ai_town.app import format_comparison_value, next_policy_mode, simulation_steps_due, wrap_ui_text


def test_playback_clock_advances_and_preserves_fractional_time():
    due, remainder = simulation_steps_due(0.0, 1.0, 1)
    assert due == 1
    assert remainder == pytest.approx(0.0)


def test_policy_mode_cycle_is_labeled_and_wraps():
    assert next_policy_mode("random") == "rule"
    assert next_policy_mode("rule") == "learned"
    assert next_policy_mode("learned") == "random"
    with pytest.raises(ValueError):
        next_policy_mode("demo")


def test_comparison_values_hide_binary_float_artifacts():
    assert format_comparison_value(16.199999999999996) == "16.2"
    assert format_comparison_value(25) == "25.0"
    assert format_comparison_value("0.3/0.2") == "0.3/0.2"

    due, remainder = simulation_steps_due(0.1, 0.9, 1)
    assert due == 1
    assert remainder == pytest.approx(0.0)


def test_playback_clock_scales_speed_and_bounds_catch_up():
    assert simulation_steps_due(0.0, 1.0, 5)[0] == 5
    assert simulation_steps_due(0.0, 1.0, 2)[0] == 2
    due, remainder = simulation_steps_due(0.0, 10.0, 10)
    assert due == 10
    assert remainder == pytest.approx(9.0)


def test_building_inspector_wraps_long_truthful_status_values():
    pygame.font.init()
    font = pygame.font.Font(None, 12)
    lines = wrap_ui_text("Town employment is tracked; assignment to a specific office is not tracked", font, 110)
    assert len(lines) > 1
    assert all(font.size(line)[0] <= 110 for line in lines)


@pytest.mark.parametrize(
    "accumulator, elapsed, speed",
    [(-0.1, 0.2, 1), (0.0, -0.1, 1), (0.0, 0.2, 3)],
)
def test_playback_clock_rejects_invalid_inputs(accumulator, elapsed, speed):
    with pytest.raises(ValueError):
        simulation_steps_due(accumulator, elapsed, speed)
