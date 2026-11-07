import pygame
from types import SimpleNamespace

import pytest

from ai_town.visuals import (
    citizen_sprite,
    citizen_is_hovered,
    citizen_wellbeing,
    draw_wellbeing_bar,
    town_depth_layers,
    town_surface,
    wellbeing_color,
    wellbeing_bar_visible,
    wellbeing_tier,
)
from ai_town.simulation import Town
from ai_town.buildings import TOWN_HEIGHT, TOWN_WIDTH, presentation_buildings
from ai_town.town_info import BuildingIdentity, building_visual_bounds


def test_town_art_is_cached_and_matches_existing_grid():
    first = town_surface(24)
    second = town_surface(24)
    assert first is second
    assert first.get_size() == (TOWN_WIDTH * 24, TOWN_HEIGHT * 24)
    assert first.get_at((2, 2)).a == 255


def test_foreground_props_are_cached_and_have_world_depth_coordinates():
    first = town_depth_layers(24)
    second = town_depth_layers(24)
    assert first is second
    assert len(first) > 25
    assert all(image.get_width() > 0 and image.get_height() > 0 for image, *_ in first)
    assert all(0 <= x < TOWN_WIDTH * 24 and 0 <= y < TOWN_HEIGHT * 24 for _, x, y, _ in first)


def test_configured_establishment_name_is_baked_into_its_sign_layer():
    unnamed = town_depth_layers(24)
    named = town_depth_layers(24, {"home_01": "House 01"})
    assert unnamed is not named
    assert bytes(unnamed[0][0].get_buffer()) != bytes(named[0][0].get_buffer())


def test_presentation_building_footprints_clear_roads_and_each_other():
    lots = presentation_buildings()
    footprints = []
    roads = (
        (0, 9 - 3 / 32, TOWN_WIDTH, 11 + 3 / 32),
        (15 - 3 / 32, 0, 17 + 3 / 32, TOWN_HEIGHT),
        (34 - 3 / 32, 9 - 3 / 32, 36 + 3 / 32, 28 + 3 / 32),
        (34 - 3 / 32, 27 - 3 / 32, TOWN_WIDTH, 29 + 3 / 32),
    )
    lake = (
        30 + 4 / 32, 11 + 5 / 32,
        30 + 4 / 32 + 3 - 8 / 32,
        11 + 5 / 32 + 6 + 8 / 32,
    )
    for kind, positions in lots.items():
        for index, (x, y) in enumerate(positions):
            identity = BuildingIdentity(f"{kind}_{index}", "Test", kind, (x, y), (x, y), index)
            box = building_visual_bounds(identity)
            assert box[0] >= 0 and box[1] >= 0
            assert box[2] <= TOWN_WIDTH and box[3] <= TOWN_HEIGHT
            for road in roads:
                assert not (box[0] < road[2] and road[0] < box[2] and box[1] < road[3] and road[1] < box[3]), (kind, x, y)
            assert not (box[0] < lake[2] and lake[0] < box[2] and box[1] < lake[3] and lake[1] < box[3]), (kind, x, y, "lake")
            footprints.append((kind, x, y, box))
    for index, (kind_a, x_a, y_a, a) in enumerate(footprints):
        for kind_b, x_b, y_b, b in footprints[index + 1:]:
            assert not (a[0] + .24 < b[2] and b[0] + .24 < a[2] and a[1] + .24 < b[3] and b[1] + .24 < a[3]), (
                (kind_a, x_a, y_a), (kind_b, x_b, y_b)
            )


def test_hover_bar_hit_region_and_display_modes():
    assert citizen_is_hovered(106, 140, 100, 150, 24)
    assert not citizen_is_hovered(116, 140, 100, 150, 24)
    assert not citizen_is_hovered(100, 154, 100, 150, 24)
    assert wellbeing_bar_visible(False, citizen_id=2, selected_id=1, hovered=True)
    assert wellbeing_bar_visible(False, citizen_id=1, selected_id=1, hovered=False)
    assert not wellbeing_bar_visible(False, citizen_id=2, selected_id=1, hovered=False)
    assert wellbeing_bar_visible(True, citizen_id=2, selected_id=1, hovered=False)


def test_citizen_sprites_are_cached_and_have_four_directions():
    color = (170, 95, 80)
    sprites = [citizen_sprite(0, direction, 1, color, 24) for direction in range(4)]
    assert all(sprite.get_size() == (27, 48) for sprite in sprites)
    assert citizen_sprite(0, 0, 1, color, 24) is sprites[0]
    assert any(bytes(sprites[0].get_buffer()) != bytes(sprite.get_buffer()) for sprite in sprites[1:])


def test_existing_personality_traits_create_distinct_sprite_styles():
    color = (170, 95, 80)
    quiet_style = citizen_sprite(4, 0, 0, color, 24, (0.05, 0.05, 0.05, 0.05))
    outgoing_style = citizen_sprite(4, 0, 0, color, 24, (0.95, 0.95, 0.95, 0.95))
    assert bytes(quiet_style.get_buffer()) != bytes(outgoing_style.get_buffer())


def test_police_officer_has_a_distinct_sprite_and_cache_entry():
    citizen = citizen_sprite(16, 0, 0, (120, 140, 150), 24, occupation="Police Officer")
    resident = citizen_sprite(16, 0, 0, (120, 140, 150), 24, occupation="Unemployed")
    assert citizen is not resident
    assert bytes(citizen.get_buffer()) != bytes(resident.get_buffer())


def test_walk_cycle_changes_pose_and_idle_is_stable():
    color = (170, 95, 80)
    idle_a = citizen_sprite(3, 0, 0, color, 24)
    idle_b = citizen_sprite(3, 0, 4, color, 24)
    walk_a = citizen_sprite(3, 0, 1, color, 24)
    walk_b = citizen_sprite(3, 0, 2, color, 24)
    assert bytes(idle_a.get_buffer()) == bytes(idle_b.get_buffer())
    assert bytes(walk_a.get_buffer()) != bytes(walk_b.get_buffer())


def test_wellbeing_uses_all_six_attributes_and_inverts_hunger_and_stress():
    citizen = SimpleNamespace(health=80, energy=70, hunger=20, happiness=60, stress=30, social=90)
    assert citizen_wellbeing(citizen) == pytest.approx(75)
    citizen.hunger = 80
    hungry_score = citizen_wellbeing(citizen)
    citizen.hunger = 20
    citizen.stress = 90
    stressed_score = citizen_wellbeing(citizen)
    assert hungry_score < 75
    assert stressed_score < 75


def test_display_wellbeing_matches_the_simulation_wellbeing_formula():
    citizen = Town(population=1, seed=13).citizens[0]
    citizen.health, citizen.energy = 83, 47
    citizen.hunger, citizen.happiness = 39, 71
    citizen.stress, citizen.social = 28, 64
    assert citizen_wellbeing(citizen) == pytest.approx(Town._wellbeing(citizen))


def test_wellbeing_weighting_clamps_values_and_validates_weights():
    citizen = SimpleNamespace(health=120, energy=-5, hunger=110, happiness=50, stress=-10, social=50)
    assert citizen_wellbeing(citizen) == pytest.approx(50)
    assert citizen_wellbeing(citizen, {"health": 5}) == pytest.approx(70)
    with pytest.raises(ValueError):
        citizen_wellbeing(citizen, {"health": -1})
    with pytest.raises(ValueError):
        citizen_wellbeing(citizen, {"unknown": 1})
    with pytest.raises(ValueError):
        citizen_wellbeing(citizen, {"health": 0, "energy": 0, "hunger": 0, "happiness": 0, "stress": 0, "social": 0})


@pytest.mark.parametrize(
    ("score", "tier"),
    [(0, "critical"), (24.99, "critical"), (25, "low"), (49.99, "low"), (50, "moderate"), (74.99, "moderate"), (75, "good"), (100, "good")],
)
def test_wellbeing_color_tiers_have_explicit_thresholds(score, tier):
    assert wellbeing_tier(score) == tier
    assert all(0 <= channel <= 255 for channel in wellbeing_color(score))


def test_wellbeing_indicator_is_one_slim_combined_bar():
    canvas = pygame.Surface((100, 50), pygame.SRCALPHA)
    draw_wellbeing_bar(canvas, (50, 20), 50, width=40, height=5)
    assert canvas.get_at((40, 22))[:3] == wellbeing_color(50)
    assert canvas.get_at((65, 22))[:3] == (93, 94, 75)
    assert canvas.get_at((40, 19)).a == 0
    assert canvas.get_at((40, 25)).a == 0
