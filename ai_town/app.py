"""Cozy procedural town viewer. Simulation mechanics stay in TownParallelEnv."""

from __future__ import annotations
import argparse, math, sys, json, webbrowser
import time
from pathlib import Path
import pygame
import numpy as np
from ai_town.environment import TownParallelEnv
from ai_town.buildings import TOWN_HEIGHT, TOWN_WIDTH
from ai_town.baselines import RandomPolicy, RulePolicy
from ai_town.rl import load_policy_checkpoint
from ai_town.simulation import Action, ACTION_NAMES
from ai_town.config import load_config
from ai_town.view_controls import camera_key_delta, keyboard_control, mouse_pan_event
from ai_town.town_info import (
    building_at_world_position,
    building_visual_bounds,
    building_catalog,
    inspect_building,
    town_identity,
    town_statistics,
)
from ai_town.audio import UiSoundEffects
from experiments.research_dashboard import build_dashboard

CREAM = (246, 239, 218)
INK = (54, 67, 58)
GREEN = (94, 143, 102)
PALE = (221, 231, 200)
PATH = (212, 191, 153)
BLUE = (99, 150, 179)
COLORS = [
    (219, 111, 83),
    (83, 143, 172),
    (229, 186, 99),
    (155, 114, 173),
    (116, 168, 130),
    (211, 133, 161),
    (97, 116, 166),
    (208, 151, 96),
    (89, 157, 151),
    (184, 105, 91),
    (120, 170, 105),
    (214, 127, 79),
    (103, 136, 184),
    (191, 153, 194),
    (106, 164, 149),
    (188, 121, 137),
]
SECONDS_PER_SIMULATION_HOUR = 1.0


def simulation_steps_due(
    accumulator: float, elapsed: float, speed: int
) -> tuple[int, float]:
    """Return bounded simulation steps and the unconsumed frame-time remainder."""
    if accumulator < 0 or elapsed < 0:
        raise ValueError("elapsed times must be nonnegative")
    if speed not in (1, 2, 5, 10):
        raise ValueError("speed must be one of 1, 2, 5, or 10")
    accumulated = accumulator + elapsed
    interval = SECONDS_PER_SIMULATION_HOUR / speed
    due = min(10, int((accumulated + 1e-9) / interval))
    return due, max(0.0, accumulated - due * interval)


def next_policy_mode(mode: str) -> str:
    """Cycle through the clearly labelled baseline and learned modes."""
    modes = ("random", "rule", "learned")
    if mode not in modes:
        raise ValueError(f"unknown policy mode: {mode}")
    return modes[(modes.index(mode) + 1) % len(modes)]


def next_available_policy(mode: str, algorithm: str = "") -> str:
    """Cycle among installed baselines and the project's saved PPO/MAPPO actors."""
    order = ("random", "rule", "ppo", "mappo")
    current = mode
    if mode == "learned":
        current = algorithm.casefold() if algorithm.casefold() in ("ppo", "mappo") else "mappo"
    if current not in order:
        raise ValueError(f"unknown policy mode: {mode}")
    return order[(order.index(current) + 1) % len(order)]


def format_comparison_value(value) -> str:
    """Format comparison values without exposing binary floating-point noise."""
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):.1f}"
    return str(value)


def draw_building_inspector(screen, font, small, vieww, height, data):
    """Render a contextual establishment inspector from a town_info snapshot."""
    x, y, width = vieww + 12, 180, 316
    rect = pygame.Rect(x, y, width, height - y - 18)
    pygame.draw.rect(screen, (255, 252, 240), rect, border_radius=13)
    pygame.draw.rect(screen, (224, 217, 197), rect, 1, border_radius=13)
    screen.blit(font.render(data["name"], True, INK), (x + 14, y + 12))
    screen.blit(
        small.render(f"{data['kind'].replace('_', ' ').title()}  ·  {data['id']}", True, (82, 98, 84)),
        (x + 14, y + 38),
    )
    status_lines = wrap_ui_text(f"Map {data['position']}  ·  {data['status']}", small, width - 28)[:2]
    for index, line in enumerate(status_lines):
        screen.blit(small.render(line, True, (82, 98, 84)), (x + 14, y + 57 + index * 14))
    capacity = "Not modeled" if data["capacity"] is None else str(data["capacity"])
    rows = [(data["occupancy_basis"], f"{data['occupancy']}  ·  capacity {capacity}")]
    if data["kind"] == "homes":
        for resident in data["household"]:
            rows.extend([
                ("Resident", resident["name"]),
                ("Wealth / health", f"${resident['wealth']:.0f}  ·  {resident['health']:.0f}"),
                ("Wellbeing / work", f"{resident['wellbeing']:.0f}  ·  {resident['employment']}"),
            ])
    elif data["kind"] == "offices":
        rows.extend([
            ("Town employed", ", ".join(data["employees"]) or "None"),
            ("Town vacancies", str(data["vacancies"])),
            ("Assignment", data["employee_assignment_status"]),
            ("Configured salary", ", ".join(f"${value:g}" for value in data["salary_values"]) or "No employees"),
        ])
    elif data["kind"] == "groceries":
        rows.extend([
            ("Food stock", f"{data['food_stock']} / {data['food_capacity']}"),
            ("Food price", f"${data['food_price']:.2f}"),
            ("Recorded purchases", str(data["recorded_purchases"])),
        ])
    elif data["kind"] == "restaurant":
        rows.extend([
            ("Recorded meals", str(data["recorded_meals"])),
            ("Recorded revenue", f"${data['recorded_revenue']:.2f}"),
            ("Inventory", data["inventory_status"]),
        ])
    elif data["kind"] == "clinic":
        rows.extend([
            ("Recorded visits", str(data["recorded_visits"])),
            ("Capacity", data["capacity_status"]),
        ])
    elif data["kind"] == "police":
        rows.extend([
                ("On duty", ", ".join(data["officers"]) or "No officer assigned"),
            ("Recorded reports", str(data["recorded_reports"])),
            ("Authority service", data["service_scope"]),
        ])
    else:
        rows.extend([
            ("Current visitors", ", ".join(data["visitors"]) or "None at the entrance"),
            ("Social interactions", str(data["social_interactions"])),
            ("Capacity", data["capacity_status"]),
        ])
    cursor_y = y + 88
    content_bottom = height - 100
    for label, value in rows:
        lines = wrap_ui_text(value, small, width - 28)[:3]
        row_height = 15 + 14 * len(lines) + 8
        if cursor_y + row_height > content_bottom:
            break
        screen.blit(small.render(label, True, (104, 111, 91)), (x + 14, cursor_y))
        for index, line in enumerate(lines):
            screen.blit(small.render(line, True, INK), (x + 14, cursor_y + 15 + index * 14))
        cursor_y += row_height
    cursor_y += 6
    if cursor_y + 25 > content_bottom:
        return
    screen.blit(font.render("Recent activity", True, INK), (x + 14, cursor_y))
    cursor_y += 28
    events = data["recent_events"]
    if not events:
        screen.blit(small.render("No recorded events for this establishment yet.", True, (104, 111, 91)), (x + 14, cursor_y))
    visible_events = min(8, max(0, (content_bottom - cursor_y) // 20))
    for event in events[:visible_events]:
        time = event.get("timestamp", {})
        stamp = f"D{time.get('day', 1):02d} {time.get('hour', 0):02d}:00"
        text = f"{stamp}  {event.get('description', event.get('detail', 'Event'))}"
        screen.blit(small.render(text[:43], True, (82, 98, 84)), (x + 14, cursor_y))
        cursor_y += 20


def draw_section_shell(screen, font, small, vieww, height, title_text, subtitle):
    """Draw the shared contextual panel surface and return its content origin."""
    x, y, width = vieww + 12, 180, 316
    rect = pygame.Rect(x, y, width, height - y - 18)
    pygame.draw.rect(screen, (255, 252, 240), rect, border_radius=13)
    pygame.draw.rect(screen, (224, 217, 197), rect, 1, border_radius=13)
    screen.blit(font.render(title_text, True, INK), (x + 14, y + 11))
    subtitle_lines = wrap_ui_text(subtitle, small, width - 28)
    for index, line in enumerate(subtitle_lines[:2]):
        screen.blit(small.render(line, True, (82, 98, 84)), (x + 14, y + 38 + index * 13))
    return x + 14, y + 68 + max(0, len(subtitle_lines[:2]) - 1) * 13, width - 28, rect.bottom


def wrap_ui_text(value, font, width):
    """Wrap inspector text to a pixel width without clipping long status labels."""
    lines, current = [], ""
    for word in str(value).split():
        candidate = f"{current} {word}".strip()
        if current and font.size(candidate)[0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def run(mode="rule", checkpoint=None, max_frames=None, screenshot_path=None, profile=False, preview_analytics=False, preview_all_bars=False, preview_section="People", preview_data="Economy", preview_building=None):
    pygame.init()
    screen = pygame.display.set_mode((1280, 800), pygame.RESIZABLE)
    pygame.display.set_caption("AI Town | Living Society")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui", 16)
    small = pygame.font.SysFont("segoeui", 12)
    micro = pygame.font.SysFont("segoeui", 10, bold=True)
    title = pygame.font.SysFont("segoeui", 25, bold=True)
    config = load_config()
    viewer_seed = int(config.get("visualization", {}).get("demo_seed", 7))
    pygame.display.set_caption(f"{config.get('town', {}).get('name', 'AI Town')} | Living Society")
    town_metadata = None
    env = TownParallelEnv(
        config["simulation"]["population"],
        config["simulation"]["episode_days"],
        config["reward"],
        config["simulation"]["ticks_per_day"],
        config["economy"],
        config["simulation"].get("movement_steps_per_tick", 3),
        continue_forever=True,
        town_events=config.get("town_events"),
    )
    obs, _ = env.reset(seed=viewer_seed)
    town_metadata = town_identity(config, env.town)
    building_records = building_catalog(config)
    building_by_id = {building.id: building for building in building_records}
    building_name_map = {building.id: building.name for building in building_records}
    previous_positions = [c.position for c in env.town.citizens]
    movement_elapsed = 0.0
    simulation_accumulator = 0.0
    policy = None

    def select(m):
        if m == "random":
            return RandomPolicy()
        if m == "rule":
            return RulePolicy()
        checkpoint_path = Path("models") / f"{m}.pt" if m in ("ppo", "mappo") else checkpoint
        p, _ = load_policy_checkpoint(checkpoint_path, config["simulation"]["population"])
        return p

    policy = select(mode)
    selected_algorithm = getattr(policy, "algorithm", "")
    evaluation_summary = None

    def read_evaluation_summary(algorithm):
        if not algorithm:
            return None
        result_path = Path("results") / f"{algorithm.lower()}.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        rows = result.get("episodes", [])
        if not rows:
            return None
        evaluated_population = result.get("config", {}).get("simulation", {}).get("population", "?")
        return (
            f"Eval n{evaluated_population} · {len(rows)}×{result.get('days', '?')}d  "
            f"R {result.get('mean_reward', 0):+.1f}  "
            f"W {np.mean([r['wellbeing'] for r in rows]):.0f}  "
            f"Alive {np.mean([r['survival_rate'] for r in rows]) * 100:.0f}%"
        )

    evaluation_summary = read_evaluation_summary(selected_algorithm)

    def change_policy_mode(new_mode):
        nonlocal mode, policy, selected_algorithm, evaluation_summary, train_metrics
        if new_mode not in ("random", "rule", "learned", "ppo", "mappo"):
            raise ValueError(f"unknown policy mode: {new_mode}")
        try:
            policy = select(new_mode)
            mode = new_mode
            selected_algorithm = getattr(policy, "algorithm", "")
            evaluation_summary = read_evaluation_summary(selected_algorithm)
            train_metrics = read_training_metrics(selected_algorithm)
        except (FileNotFoundError, RuntimeError):
            mode = "rule"
            policy = RulePolicy()
            selected_algorithm = ""
            evaluation_summary = None
    speed = 1
    speed_levels = (1, 2, 5, 10)
    paused = True
    selected = len(env.town.citizens) - 1
    selected_building_id = None
    show_building_labels = False
    active_view = preview_section if preview_section in ("Town", "People", "Places", "Data") else "People"
    data_view = preview_data if preview_data in ("Economy", "Social", "Events", "RL Lab", "Reports") else "Economy"
    show_building_directory = active_view == "Places"
    place_directory_offset = 0
    if preview_building in building_by_id:
        selected_building_id = preview_building
        active_view = "Places"
        show_building_directory = False
    show_people_directory = False
    citizen_directory_offset = 0
    search_query = ""
    search_mode = None
    event_filter = "All"
    stats_cache = None
    stats_cache_key = None
    timeline_offset = 0
    compare = False
    compare_id = 1
    follow = False
    zoom = 1.12
    offset = np.array([0.0, 0.0])
    camera_target = offset.copy()
    pan_anchor = None
    ticks = 0
    running = True
    show_analytics = preview_analytics
    analytics_collapsed = False
    show_all_wellbeing = preview_all_bars
    wellbeing_alpha = {c.id: (255.0 if preview_all_bars else 0.0) for c in env.town.citizens}
    wellbeing_display = {c.id: 0.0 for c in env.town.citizens}
    wellbeing_weights = config.get("visualization", {}).get("wellbeing_weights")
    def read_training_metrics(algorithm):
        if algorithm not in ("PPO", "MAPPO"):
            return []
        metric_file = Path("logs") / f"train_{algorithm.lower()}.jsonl"
        try:
            return [json.loads(line) for line in metric_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            return []

    train_metrics = read_training_metrics(selected_algorithm)
    sound = UiSoundEffects(enabled=True)
    sound.start_music()

    def open_research_dashboard():
        report_path = build_dashboard()
        webbrowser.open(report_path.as_uri())
    frames = 0
    frame_times = []
    while running:
        frame_started = time.perf_counter() if profile else 0.0
        dt = clock.tick(60) / 1000
        movement_elapsed += dt
        w, h = screen.get_size()
        panel = 340
        vieww = w - panel
        # Leave a two-cell side margin so labels over outermost east-quarter
        # buildings remain inside the map viewport rather than under the HUD.
        # All map artwork is cached at an integer pixel scale. Keep the world
        # transform on that same scale so sprites, clicks, and outlines share
        # one origin at every zoom level.
        tile = max(12, int(min(vieww / (TOWN_WIDTH + 4), h / (TOWN_HEIGHT + 2)) * zoom))
        ox = (vieww - TOWN_WIDTH * tile) / 2 + offset[0]
        oy = (h - TOWN_HEIGHT * tile) / 2 + offset[1]
        for ev in pygame.event.get():
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                sound.play("tap")
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and pygame.Rect(vieww + panel - 43, 6, 26, 22).collidepoint(ev.pos):
                sound.toggle()
                sound.play("open")
            elif ev.type == pygame.KEYDOWN and search_mode is not None:
                if ev.key == pygame.K_BACKSPACE:
                    search_query = search_query[:-1]
                elif ev.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    search_mode = None
                    pygame.key.stop_text_input()
            elif ev.type == pygame.TEXTINPUT and search_mode is not None:
                search_query += ev.text
            elif ev.type == pygame.KEYDOWN:
                control = keyboard_control(ev.key)
                if control == "pause":
                    paused = not paused
                elif control == "reset":
                    obs, _ = env.reset(seed=viewer_seed)
                    ticks = 0
                    simulation_accumulator = 0.0
                    previous_positions = [c.position for c in env.town.citizens]
                elif control == "follow":
                    follow = not follow
                elif control == "analytics":
                    show_analytics = not show_analytics
                elif control == "all_bars":
                    show_all_wellbeing = not show_all_wellbeing
                elif control == "compare":
                    compare = not compare
                elif control == "toggle_sound":
                    enabled = sound.toggle()
                    if enabled:
                        sound.play("open")
                elif control == "next_compare_citizen":
                    compare_id = (compare_id + 1) % len(env.town.citizens)
                elif control in ("random_policy", "rule_policy", "learned_policy", "ppo_policy", "mappo_policy"):
                    change_policy_mode({
                        "random_policy": "random",
                        "rule_policy": "rule",
                        "learned_policy": "learned",
                        "ppo_policy": "ppo",
                        "mappo_policy": "mappo",
                    }[control])
                elif control == "speed_up":
                    speed = next((value for value in speed_levels if value > speed), 10)
                elif control == "speed_down":
                    speed = next(
                        (value for value in reversed(speed_levels) if value < speed), 1
                    )
                elif (pan_delta := camera_key_delta(ev.key)) is not None:
                    camera_target += np.asarray(pan_delta)
            elif ev.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if active_view == "People" and show_people_directory and mouse_x >= vieww and mouse_y >= 249:
                    citizen_directory_offset = max(0, citizen_directory_offset - ev.y * 3)
                elif active_view == "Places" and show_building_directory and mouse_x >= vieww and mouse_y >= 249:
                    place_directory_offset = max(0, place_directory_offset - ev.y * 4)
                elif mouse_x >= vieww and mouse_y >= 532:
                    timeline_offset = max(0, timeline_offset - ev.y)
                else:
                    world_x = (mouse_x - ox) / tile
                    world_y = (mouse_y - oy) / tile
                    old_tile = tile
                    zoom = float(np.clip(zoom * (1.1 if ev.y > 0 else 0.9), 0.7, 2.5))
                    tile = max(12, int(min(vieww / (TOWN_WIDTH + 4), h / (TOWN_HEIGHT + 2)) * zoom))
                    new_ox = (vieww - TOWN_WIDTH * tile) / 2
                    new_oy = (h - TOWN_HEIGHT * tile) / 2
                    camera_target[0] += mouse_x - (new_ox + world_x * tile)
                    camera_target[1] += mouse_y - (new_oy + world_y * tile)
            elif (
                (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3)
                or (ev.type == pygame.MOUSEBUTTONUP and ev.button == 3)
                or (ev.type == pygame.MOUSEMOTION and pan_anchor is not None)
            ):
                pan_anchor, pan_delta = mouse_pan_event(
                    ev.type, getattr(ev, "button", None), ev.pos, pan_anchor
                )
                camera_target += np.asarray(pan_delta)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and show_analytics and pygame.Rect(25, 22, 270, 34).collidepoint(ev.pos):
                analytics_collapsed = not analytics_collapsed
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and ev.pos[0] < vieww
            ):
                pos = np.array([(ev.pos[0] - ox) / tile, (ev.pos[1] - oy) / tile])
                d = [
                    np.linalg.norm(
                        np.array((
                            c.position[0] + .5,
                            c.position[1] + .5 + (1 if c.position == c.home else 0),
                        ))
                        - pos
                    )
                    for c in env.town.citizens
                ]
                hit_building = building_at_world_position(
                    building_records, float(pos[0]), float(pos[1])
                )
                if hit_building is not None and min(d, default=999) > 0.55:
                    selected_building_id = hit_building.id
                    active_view = "Places"
                    show_building_directory = False
                    search_mode = None
                    pygame.key.stop_text_input()
                    follow = False
                    left, top, right, bottom = building_visual_bounds(hit_building)
                    camera_target[0] = vieww / 2 - (left + right) * 0.5 * tile
                    camera_target[1] = h / 2 - (top + bottom) * 0.5 * tile
                else:
                    selected = int(np.argmin(d))
                    selected_building_id = None
                    active_view = "People"
                    show_building_directory = False
                    show_people_directory = False
                    search_mode = None
                    pygame.key.stop_text_input()
                    person = env.town.citizens[selected]
                    render_y = person.position[1] + (1 if person.position == person.home else 0)
                    camera_target[0] = vieww / 2 - (person.position[0] + .5) * tile
                    camera_target[1] = h / 2 - (render_y + .5) * tile
                    follow = False
                    timeline_offset = 0
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and ev.pos[0] >= vieww
                and 110 <= ev.pos[1] <= 144
            ):
                ix = (ev.pos[0] - vieww - 12) // 53
                if ix == 0:
                    paused = not paused
                    if not paused:
                        sound.play("start")
                elif ix == 1:
                    obs, _ = env.reset(seed=viewer_seed)
                    ticks = 0
                    simulation_accumulator = 0.0
                    previous_positions = [c.position for c in env.town.citizens]
                elif ix == 2:
                    follow = not follow
                elif ix == 3:
                    show_analytics = not show_analytics
                elif ix == 4:
                    open_research_dashboard()
                elif ix == 5:
                    compare = not compare
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and ev.pos[0] >= vieww + 12
                and vieww + 12 <= ev.pos[0] <= vieww + 328
                and 146 <= ev.pos[1] <= 168
            ):
                tab_index = min(3, max(0, (ev.pos[0] - (vieww + 12)) // 79))
                active_view = ("Town", "People", "Places", "Data")[tab_index]
                search_mode = None
                pygame.key.stop_text_input()
                if active_view == "Places":
                    show_building_directory = selected_building_id is None
                elif active_view == "People":
                    show_people_directory = False
                elif active_view != "Data":
                    selected_building_id = None
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and active_view == "People" and vieww + 12 <= ev.pos[0] <= vieww + 130
                and (178 <= ev.pos[1] <= 205 if show_people_directory else 298 <= ev.pos[1] <= 323)
            ):
                show_people_directory = not show_people_directory
                citizen_directory_offset = 0
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and active_view == "People" and vieww + 205 <= ev.pos[0] <= vieww + 325
                and (178 <= ev.pos[1] <= 205 if show_people_directory else 298 <= ev.pos[1] <= 323)
            ):
                show_all_wellbeing = not show_all_wellbeing
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and active_view == "Places" and vieww + 12 <= ev.pos[0] <= vieww + 130
                and 178 <= ev.pos[1] <= 205
            ):
                show_building_labels = not show_building_labels
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and active_view == "Places" and vieww + 135 <= ev.pos[0] <= vieww + 195
                and 178 <= ev.pos[1] <= 205
            ):
                show_building_directory = not show_building_directory
                place_directory_offset = 0
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and active_view == "Data" and vieww + 12 <= ev.pos[0] <= vieww + 328
                and 178 <= ev.pos[1] <= 207
            ):
                page = min(4, max(0, (ev.pos[0] - (vieww + 12)) // 63))
                data_view = ("Economy", "Social", "Events", "RL Lab", "Reports")[page]
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and active_view == "Data" and data_view == "Events"
                and vieww + 12 <= ev.pos[0] <= vieww + 328
                and 218 <= ev.pos[1] <= 243
            ):
                event_filter = ("All", "Economy", "Social", "Health", "Employment", "Safety", "Calendar")[min(6, max(0, (ev.pos[0] - (vieww + 26)) // 41))]
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and search_mode is None and (
                    (active_view == "People" and show_people_directory)
                    or (active_view == "Places" and show_building_directory)
                    or (active_view == "Data" and data_view == "Events")
                )
                and ev.pos[0] >= vieww + 12 and 218 <= ev.pos[1] <= 243
                and not (active_view == "Data" and data_view == "Events")
            ):
                search_mode = active_view
                search_query = ""
                pygame.key.start_text_input()
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and search_mode is None and active_view == "Data" and data_view == "Events"
                and ev.pos[0] >= vieww + 12 and 248 <= ev.pos[1] <= 273
            ):
                search_mode = "Data"
                search_query = ""
                pygame.key.start_text_input()
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and active_view == "Places" and show_building_directory
                and ev.pos[0] >= vieww + 12
                and ev.pos[1] >= 249
            ):
                row = (ev.pos[1] - 249) // 24 + place_directory_offset
                matches = [
                    building for building in building_records
                    if search_query.casefold() in building.name.casefold()
                    or search_query.casefold() in building.kind.casefold()
                    or search_query.casefold() in building.id.casefold()
                ]
                if 0 <= row < len(matches):
                    building = matches[row]
                    selected_building_id = building.id
                    active_view = "Places"
                    show_building_directory = False
                    search_mode = None
                    pygame.key.stop_text_input()
                    follow = False
                    camera_target[0] = vieww / 2 - (building.position[0] + 1.0) * tile
                    camera_target[1] = h / 2 - (building.position[1] + 0.5) * tile
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and active_view == "People" and show_people_directory
                and ev.pos[0] >= vieww + 12 and ev.pos[1] >= 249
            ):
                row = (ev.pos[1] - 249) // 34
                matches = [
                    citizen for citizen in env.town.citizens
                    if search_query.casefold() in citizen.name.casefold()
                    or search_query.casefold() in citizen.occupation.casefold()
                    or search_query.casefold() in citizen.activity.casefold()
                ]
                row += citizen_directory_offset
                if 0 <= row < len(matches):
                    selected = matches[row].id
                    selected_building_id = None
                    show_people_directory = False
                    search_mode = None
                    pygame.key.stop_text_input()
                    person = env.town.citizens[selected]
                    render_y = person.position[1] + (1 if person.position == person.home else 0)
                    camera_target[0] = vieww / 2 - (person.position[0] + .5) * tile
                    camera_target[1] = h / 2 - (render_y + .5) * tile
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and ev.pos[0] >= vieww + 120
                and ev.pos[0] <= vieww + 240
                and 72 <= ev.pos[1] <= 103
            ):
                change_policy_mode(next_available_policy(mode, selected_algorithm))
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and ev.pos[0] >= vieww + panel - 88
                and 72 <= ev.pos[1] <= 103
            ):
                speed = speed_levels[(speed_levels.index(speed) + 1) % len(speed_levels)]
        if follow:
            c = env.town.citizens[selected]
            camera_target = np.array(
                [
                    vieww / 2 - (c.position[0] + 0.5) * tile,
                    h / 2 - (c.position[1] + 0.5) * tile,
                ]
            )
        camera_target[0] = np.clip(camera_target[0], min(0, vieww-TOWN_WIDTH*tile)-(vieww-TOWN_WIDTH*tile)/2, max(0, vieww-TOWN_WIDTH*tile)-(vieww-TOWN_WIDTH*tile)/2)
        camera_target[1] = np.clip(camera_target[1], min(0, h-TOWN_HEIGHT*tile)-(h-TOWN_HEIGHT*tile)/2, max(0, h-TOWN_HEIGHT*tile)-(h-TOWN_HEIGHT*tile)/2)
        offset += (camera_target - offset) * min(1.0, dt * 9)
        ox = (vieww - TOWN_WIDTH * tile) / 2 + offset[0]
        oy = (h - TOWN_HEIGHT * tile) / 2 + offset[1]
        if not paused:
            due_steps, simulation_accumulator = simulation_steps_due(
                simulation_accumulator, dt, speed
            )
            for _ in range(due_steps):
                if not env.agents:
                    paused = True
                    break
                previous_positions = [c.position for c in env.town.citizens]
                x = np.stack([obs[a] for a in env.possible_agents])
                if mode in ("learned", "ppo", "mappo"):
                    actions, _ = policy.act(
                        x,
                        action_masks=np.stack(
                            [env.action_mask(a) for a in env.possible_agents]
                        ),
                    )
                else:
                    actions, _ = policy.act(x)
                obs, *_ = env.step(
                    {
                        a: int(actions[i])
                        for i, a in enumerate(env.possible_agents)
                    }
                )
                # Keep the live deployment rolling without unbounded UI histories.
                town = env.town
                if len(town.events) > 10_000:
                    del town.events[:-10_000]
                if len(town.daily_summaries) > 365:
                    del town.daily_summaries[:-365]
                for citizen in town.citizens:
                    if len(citizen.rewards) > 4_096:
                        del citizen.rewards[:-4_096]
                ticks += 1
                movement_elapsed = 0.0
                if not env.agents:
                    paused = True
                    simulation_accumulator = 0.0
                    break
        # Cached original pixel-inspired world art; simulation grid remains unchanged.
        from ai_town.visuals import (
            citizen_wellbeing,
            citizen_is_hovered,
            citizen_sprite,
            draw_wellbeing_bar,
            town_surface,
            town_depth_layers,
            wellbeing_tier,
            wellbeing_bar_visible,
        )
        screen.fill((111, 148, 94))
        world_art = town_surface(int(tile))
        screen.blit(world_art, (int(ox), int(oy)))
        # Time-based water shimmer is presentation-only and costs a few arc draws.
        water_x, water_y = ox + 31.5 * tile, oy + 14 * tile
        ripple_phase = (frames * 0.045) % 1.0
        for ripple_index in range(3):
            radius = (0.16 + ((ripple_phase + ripple_index / 3) % 1.0) * 0.5) * tile
            ripple_rect = pygame.Rect(water_x-radius, water_y-radius*.36, radius*2, radius*.72)
            pygame.draw.arc(screen, (160, 214, 210), ripple_rect, 0.12, math.pi-0.12, max(1, int(tile*.035)))
        # Citizens use directional four-way sprites and movement-driven walk cycles.
        wellbeing_bars = []
        depth_items = [
            (depth_y, 0, image, ox + x, oy + y, None)
            for image, x, y, depth_y in town_depth_layers(int(tile), building_name_map)
        ]
        citizen_screen_positions = []
        citizen_name_tags = []
        for c in env.town.citizens:
            old_x, old_y = previous_positions[c.id]
            current_x, current_y = c.position
            old_y += 1 if (old_x, old_y) == c.home else 0
            current_y += 1 if c.position == c.home else 0
            # Use most of the current simulated-hour interval for movement so
            # 1× reads as a deliberate, watchable pace instead of a burst.
            move_alpha = min(1.0, movement_elapsed / (SECONDS_PER_SIMULATION_HOUR * .85 / speed))
            draw_x = old_x + (current_x - old_x) * move_alpha
            draw_y = old_y + (current_y - old_y) * move_alpha
            walking = (old_x, old_y) != (current_x, current_y) or bool(c.path)
            if abs(current_x-old_x) >= abs(current_y-old_y):
                direction = 3 if current_x >= old_x else 2
            else:
                direction = 1 if current_y < old_y else 0
            walk_frame = (int(movement_elapsed * 11) % 4 + 1) if walking else 0
            citizen_color = COLORS[c.id % len(COLORS)]
            sprite = citizen_sprite(c.id, direction, walk_frame, citizen_color, max(12, int(tile)), c.personality, c.occupation)
            px = int(ox + (draw_x + .5) * tile)
            py = int(oy + (draw_y + .5) * tile)
            citizen_screen_positions.append((px, py))
            if not walking:
                py += math.sin(frames * 0.08 + c.id * 0.7) * tile * 0.025
            if c.id == selected:
                pygame.draw.ellipse(screen, (255, 237, 171), (px-tile*.48, py-tile*.2, tile*.96, tile*.36), max(2,int(tile*.045)))
                pygame.draw.ellipse(screen, (255, 250, 220), (px-tile*.43, py-tile*.17, tile*.86, tile*.29), 1)
            badge = small.render(ACTION_NAMES[c.pending_action][0] if c.pending_action is not None else "•", True, (255,255,255))
            badge_x, badge_y = px+tile*.28, py-tile*.78
            avatar = (px-sprite.get_width()//2, py-sprite.get_height()+int(tile*.1))
            depth_items.append((py - oy, 1, sprite, avatar[0], avatar[1], (badge, badge_x, badge_y)))
            mouse_x, mouse_y = pygame.mouse.get_pos()
            hovered = citizen_is_hovered(mouse_x, mouse_y, px, py, tile)
            if tile >= 17 or hovered or c.id == selected:
                name_label = small.render(c.name.split()[0], True, (250, 246, 230))
                tag_x = px - tile * .55 if c.position == c.home else px
                name_rect = name_label.get_rect(center=(tag_x, int(py - tile * 2.32)))
                citizen_name_tags.append((name_label, name_rect, c.id == selected))
            visible_target = wellbeing_bar_visible(show_all_wellbeing, c.id, selected, hovered)
            alpha_target = 255.0 if visible_target else 0.0
            wellbeing_alpha[c.id] += (alpha_target-wellbeing_alpha[c.id]) * min(1.0, dt*10)
            target_score = citizen_wellbeing(c, wellbeing_weights)
            wellbeing_display[c.id] += (target_score-wellbeing_display[c.id]) * min(1.0, dt*8)
            if wellbeing_alpha[c.id] > 2:
                wellbeing_bars.append((c.id, px, int(py-1.84*tile), wellbeing_display[c.id], round(wellbeing_alpha[c.id]), c.position))
        # Painter's ordering lets citizens walk behind foreground trees/houses.
        for _, item_type, image, item_x, item_y, badge_data in sorted(depth_items, key=lambda item: (item[0], item[1])):
            screen.blit(image, (int(item_x), int(item_y)))
            if badge_data is not None:
                badge, badge_x, badge_y = badge_data
                pygame.draw.circle(screen, (57,76,62), (int(badge_x),int(badge_y)), max(6,int(tile*.17)))
                pygame.draw.circle(screen, (213,226,184), (int(badge_x),int(badge_y)), max(6,int(tile*.17)), 1)
                screen.blit(badge,(badge_x-badge.get_width()/2,badge_y-badge.get_height()/2))
        for name_label, name_rect, is_selected in citizen_name_tags:
            pill = name_rect.inflate(10, 4)
            pygame.draw.rect(screen, (49, 67, 55), pill, border_radius=5)
            pygame.draw.rect(screen, (227, 199, 125) if is_selected else (185, 202, 167), pill, 1, border_radius=5)
            screen.blit(name_label, name_rect)
        # Renderer-only community patrol prop: a softly pulsing beacon adds a
        # little town life without creating a citizen or affecting RL state.
        patrol_x, patrol_y = ox + 29.4 * tile, oy + 29.4 * tile
        pygame.draw.ellipse(screen, (73, 85, 65), (patrol_x - tile*.48, patrol_y + tile*.07, tile*.98, tile*.25))
        pygame.draw.rect(screen, (225, 230, 220), (patrol_x - tile*.45, patrol_y - tile*.22, tile*.92, tile*.34), border_radius=max(3, int(tile*.09)))
        pygame.draw.rect(screen, (45, 89, 127), (patrol_x - tile*.43, patrol_y - tile*.01, tile*.88, tile*.085), border_radius=max(2, int(tile*.025)))
        pygame.draw.rect(screen, (70, 105, 121), (patrol_x - tile*.18, patrol_y - tile*.20, tile*.38, tile*.14), border_radius=max(2, int(tile*.04)))
        pygame.draw.circle(screen, (55, 66, 67), (int(patrol_x - tile*.28), int(patrol_y + tile*.09)), max(2, int(tile*.085)))
        pygame.draw.circle(screen, (55, 66, 67), (int(patrol_x + tile*.30), int(patrol_y + tile*.09)), max(2, int(tile*.085)))
        beacon_alpha = int(110 + 100 * (0.5 + 0.5 * math.sin(frames * .18)))
        pygame.draw.circle(screen, (77, 145, 217), (int(patrol_x - tile*.07), int(patrol_y - tile*.29)), max(2, int(tile*.065)))
        pygame.draw.circle(screen, (218, 91, 83 if beacon_alpha > 180 else 115), (int(patrol_x + tile*.08), int(patrol_y - tile*.29)), max(2, int(tile*.065)))
        # Day/night overlay.
        hour = (env.town.tick % 24) + (simulation_accumulator / (SECONDS_PER_SIMULATION_HOUR / speed) if not paused else 0.0)
        lamp_positions = [(5, 9), (13, 9), (22, 9), (30, 9), (17, 5), (17, 18)]
        night = max(0, math.cos((hour - 2) / 24 * math.tau))
        if night > 0.08:
            overlay = pygame.Surface((vieww, h), pygame.SRCALPHA)
            overlay.fill((24, 41, 76, int(105 * night)))
            screen.blit(overlay, (0, 0))
            lamp_glow = pygame.Surface((vieww, h), pygame.SRCALPHA)
            for x, y in lamp_positions:
                lx, ly = ox + (x + 0.5) * tile, oy + (y + 0.5) * tile
                pygame.draw.circle(
                    lamp_glow,
                    (255, 220, 145, int(24 * night)),
                    (int(lx), int(ly - tile * 0.2)),
                    max(8, int(tile * 1.2)),
                )
            screen.blit(lamp_glow, (0, 0))
            for x, y in lamp_positions:
                lx, ly = ox + (x + 0.5) * tile, oy + (y + 0.5) * tile
                pygame.draw.circle(
                    screen,
                    (255, 224, 151),
                    (int(lx), int(ly - tile * 0.29)),
                    max(3, int(tile * 0.09)),
                )
        # Establishment names remain readable at normal zoom; numbered homes
        # keep the map compact. Hover and selection reveal each configured name.
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_world = ((mouse_x - ox) / tile, (mouse_y - oy) / tile)
        hovered_building = building_at_world_position(
            building_records, mouse_world[0], mouse_world[1]
        )
        if show_building_labels or selected_building_id:
            label_font = pygame.font.SysFont("segoeui", max(9, min(12, int(tile * .38))))
            label_rects = [name_rect.inflate(8, 4) for _, name_rect, _ in citizen_name_tags]
            ordered_records = sorted(
                building_records,
                key=lambda item: item.id == selected_building_id or (
                    hovered_building is not None and item.id == hovered_building.id
                ),
                reverse=True,
            )
            for building in ordered_records:
                chosen = building.id == selected_building_id
                hovered = hovered_building is not None and building.id == hovered_building.id
                # Resident name tags already identify homes; avoid a second
                # floating label over every roof unless its inspector is active.
                if building.kind == "homes" and not (chosen or hovered):
                    continue
                if not show_building_labels and not (chosen or hovered):
                    continue
                compact = building.name if (chosen or hovered) else (
                    building.name.split()[0] if building.kind == "homes" else building.name
                )
                if zoom < .9 and not (chosen or hovered):
                    compact = building.name.split()[0] if building.kind == "homes" else {
                        "offices": "OFF", "groceries": "SHOP", "restaurant": "EAT",
                        "clinic": "MED", "parks": "PARK", "town": "TOWN",
                    }.get(building.kind, compact)
                label = label_font.render(compact, True, INK)
                if building.kind == "parks":
                    center_x = ox + (building.position[0] + 3) * tile
                    top_y = oy + (building.position[1] + .05) * tile
                else:
                    center_x = ox + (building.position[0] + 1.25) * tile
                    top_y = oy + (building.position[1] - 1.65) * tile
                rect = label.get_rect(midtop=(round(center_x), round(top_y)))
                rect.inflate_ip(10, 5)
                for _ in range(10):
                    collides_citizen = any(
                        rect.colliderect(pygame.Rect(cx - tile*.35, cy - tile*1.8, tile*.7, tile*1.9))
                        for cx, cy in citizen_screen_positions
                    )
                    collides_label = any(rect.colliderect(other) for other in label_rects)
                    if not collides_citizen and not collides_label:
                        break
                    rect.y -= rect.height + 2
                label_rects.append(rect)
                fill = (255, 250, 226) if not chosen else (255, 232, 157)
                pygame.draw.rect(screen, fill, rect, border_radius=5)
                pygame.draw.rect(screen, (131, 108, 67) if chosen else (208, 198, 169), rect, 1, border_radius=5)
                screen.blit(label, label.get_rect(center=rect.center))
                if chosen:
                    left, top, right, bottom = building_visual_bounds(building)
                    bounds = pygame.Rect(
                        round(ox + left * tile), round(oy + top * tile),
                        round((right - left) * tile), round((bottom - top) * tile),
                    )
                    pygame.draw.rect(screen, (255, 228, 127), bounds, max(2, int(tile*.05)), border_radius=5)
        # UI-layer bars stay crisp above lighting, foliage, and decorative flowers.
        pileup_rank = {}
        for citizen_id, px, bar_y, score, alpha, position in sorted(wellbeing_bars, key=lambda row: (row[5][1], row[5][0], row[0])):
            rank = pileup_rank.get(position, 0)
            pileup_rank[position] = rank + 1
            draw_wellbeing_bar(screen, (px, bar_y - rank * max(6, int(tile*.2))), score,
                               max(26, int(tile*1.1)), max(6, int(tile*.2)), alpha)
        # Side panel.
        pygame.draw.rect(screen, (242, 238, 222), (vieww, 0, panel, h))
        pygame.draw.rect(screen, (45, 75, 61), (vieww, 0, panel, 100))
        pygame.draw.rect(screen, (213, 177, 112), (vieww, 98, panel, 2))
        pygame.draw.line(screen, (220, 212, 193), (vieww, 0), (vieww, h), 2)
        cache_key = (env.town.tick, len(env.town.events))
        if cache_key != stats_cache_key:
            stats_cache = town_statistics(env.town)
            stats_cache_key = cache_key
        town_metadata = town_identity(config, env.town)
        screen.blit(micro.render(f"{town_metadata['name'].upper()}  /  COMMUNITY LIFE", True, (195, 211, 183)), (vieww + 22, 11))
        screen.blit(title.render(town_metadata["name"], True, (250, 246, 230)), (vieww + 20, 25))
        header_state = "PAUSED" if paused else "LIVE"
        status_rect = pygame.Rect(vieww + panel - 91, 30, 69, 22)
        pygame.draw.rect(screen, (65, 101, 78) if paused else (183, 133, 76), status_rect, border_radius=11)
        pygame.draw.circle(screen, (225, 215, 164) if paused else (247, 231, 188), (status_rect.x + 11, status_rect.centery), 3)
        screen.blit(micro.render(header_state, True, (250, 246, 230)), (status_rect.x + 20, status_rect.y + 5))
        sound_rect = pygame.Rect(vieww + panel - 43, 6, 26, 22)
        pygame.draw.rect(screen, (83, 116, 89) if sound.enabled else (76, 82, 71), sound_rect, border_radius=7)
        icon_color = (250, 246, 230) if sound.enabled and sound.available else (186, 193, 177)
        pygame.draw.circle(screen, icon_color, (sound_rect.x + 8, sound_rect.y + 15), 3)
        pygame.draw.line(screen, icon_color, (sound_rect.x + 11, sound_rect.y + 15), (sound_rect.x + 11, sound_rect.y + 6), 2)
        pygame.draw.line(screen, icon_color, (sound_rect.x + 11, sound_rect.y + 6), (sound_rect.x + 19, sound_rect.y + 4), 2)
        if not sound.enabled or not sound.available:
            pygame.draw.line(screen, (234, 193, 143), (sound_rect.x + 5, sound_rect.y + 4), (sound_rect.x + 21, sound_rect.y + 19), 2)
        screen.blit(small.render(f"Day {env.town.day+1:02d} · {int(hour)%24:02d}:00  /  {stats_cache['population']} residents", True, (214, 222, 197)), (vieww + 22, 54))
        policy_name = {"random": "Random", "rule": "Rule", "learned": selected_algorithm or "Learned", "ppo": "PPO", "mappo": "MAPPO"}[mode]
        policy_rect = pygame.Rect(vieww + 120, 73, 120, 27)
        policy_fill = (83, 116, 89) if not policy_rect.collidepoint(pygame.mouse.get_pos()) else (99, 135, 102)
        pygame.draw.rect(screen, policy_fill, policy_rect, border_radius=8)
        pygame.draw.rect(screen, (139, 164, 128), policy_rect, 1, border_radius=8)
        policy_label = small.render(f"Policy · {policy_name}  ›", True, (250, 246, 230))
        screen.blit(policy_label, (policy_rect.centerx-policy_label.get_width()/2, policy_rect.y+7))
        speed_rect = pygame.Rect(vieww + panel - 84, 73, 64, 27)
        pygame.draw.rect(screen, (83, 116, 89) if not speed_rect.collidepoint(pygame.mouse.get_pos()) else (99, 135, 102), speed_rect, border_radius=8)
        pygame.draw.rect(screen, (139, 164, 128), speed_rect, 1, border_radius=8)
        speed_label = small.render(f"{speed}×  +", True, (250, 246, 230))
        screen.blit(speed_label, (speed_rect.centerx-speed_label.get_width()/2, speed_rect.y+7))

        def button(txt, x, y, width=56, active=False):
            rect = pygame.Rect(x, y, width, 34)
            fill = (79, 112, 83) if active else (229, 224, 205)
            if rect.collidepoint(pygame.mouse.get_pos()):
                fill = (93, 129, 94) if active else (220, 228, 199)
            pygame.draw.rect(
                screen, fill, rect, border_radius=9
            )
            label = small.render(txt, True, (250, 246, 230) if active else INK)
            screen.blit(label, (x + (width - label.get_width()) / 2, y + 10))

        button("Play" if paused else "Pause", vieww + 12, 110, width=49)
        button("Reset", vieww + 65, 110, width=49)
        button("Follow", vieww + 118, 110, width=49, active=follow)
        button("Pulse", vieww + 171, 110, width=49, active=show_analytics)
        button("Graphs", vieww + 224, 110, width=49)
        button("Compare", vieww + 277, 110, width=49, active=compare)
        c = env.town.citizens[selected]
        pygame.draw.rect(screen, (250, 248, 235), (vieww + 12, 172, panel - 24, 120), border_radius=13)
        pygame.draw.rect(screen, (219, 213, 188), (vieww + 12, 172, panel - 24, 120), 1, border_radius=13)
        pygame.draw.rect(screen, (100, 137, 91), (vieww + 12, 184, 4, 96), border_radius=2)
        for index, tab in enumerate(("Town", "People", "Places", "Data")):
            rect = pygame.Rect(vieww + 12 + index * 79, 146, 75, 24)
            pygame.draw.rect(
                screen,
                (73, 104, 77) if active_view == tab else (229, 224, 205),
                rect,
                border_radius=7,
            )
            label = small.render(tab, True, (250, 246, 230) if active_view == tab else INK)
            screen.blit(label, label.get_rect(center=rect.center))
        pygame.draw.circle(screen, (226, 231, 206), (vieww + 40, 203), 19)
        portrait = citizen_sprite(c.id, 0, 0, COLORS[c.id % len(COLORS)], 31, c.personality, c.occupation)
        screen.blit(portrait, (vieww + 24, 185))
        screen.blit(
            font.render(f"{c.name}", True, INK),
            (vieww + 62, 175),
        )
        screen.blit(
            small.render(
                f"{c.activity}  ·  Age {c.age}  ·  {c.occupation}",
                True,
                (94, 111, 87),
            ),
            (vieww + 62, 200),
        )
        screen.blit(
            font.render(
                f"${c.money:.0f}   Food {c.food}   Salary ${c.salary:.0f} · Loc {c.position}",
                True,
                INK,
            ),
            (vieww + 62, 224),
        )
        active = (
            ACTION_NAMES[c.pending_action]
            if c.pending_action is not None
            else "Choosing activity"
        )
        traits = "  ".join(f"{v:.1f}" for v in c.personality)
        last_reward = c.rewards[-1] if c.rewards else 0.0
        screen.blit(
            small.render(
                f"{active}  ·  reward {last_reward:+.2f}", True, INK
            ),
            (vieww + 62, 243),
        )
        screen.blit(
            small.render(f"Personality A/S/R/L: {traits}", True, (82, 98, 84)),
            (vieww + 62, 260),
        )
        wellbeing_score = citizen_wellbeing(c, wellbeing_weights)
        screen.blit(
            small.render(f"Wellbeing  {wellbeing_score:.0f}  ·  {wellbeing_tier(wellbeing_score)}", True, INK),
            (vieww + 62, 276),
        )
        vals = [
            ("Health", c.health),
            ("Energy", c.energy),
            ("Hunger", c.hunger),
            ("Happiness", c.happiness),
            ("Stress", c.stress),
            ("Social", c.social),
        ]
        for j, (name, val) in enumerate(vals):
            yy = (330 + j * 32) if active_view == "People" else (300 + j * 36)
            screen.blit(small.render(name, True, INK), (vieww + 22, yy))
            screen.blit(small.render(f"{val:.0f}", True, INK), (vieww + panel - 43, yy))
            pygame.draw.rect(
                screen,
                (222, 216, 198),
                (vieww + 22, yy + 17, panel - 44, 9),
                border_radius=5,
            )
            color = (
                (79, 146, 104) if name not in ("Hunger", "Stress") else (213, 156, 83)
            )
            pygame.draw.rect(
                screen,
                color,
                (vieww + 22, yy + 17, (panel - 44) * val / 100, 9),
                border_radius=5,
            )
        timeline_offset = min(timeline_offset, max(0, len(c.history) - 1))
        history_end = max(0, len(c.history) - timeline_offset)
        timeline = c.history[max(0, history_end - 4) : history_end]
        history_range = (
            f"{timeline_offset + 1}-{min(len(c.history), timeline_offset + 4)} / {len(c.history)}"
            if c.history
            else "No events yet"
        )
        screen.blit(
            font.render(f"Life timeline  {history_range}", True, INK),
            (vieww + 22, 532),
        )
        for j, event in enumerate(reversed(timeline)):
            y = 562 + j * 34
            pygame.draw.circle(screen, GREEN, (vieww + 27, y + 6), 4)
            screen.blit(
                small.render(
                    f"Hr {event['tick']:03d} {event['event'][:7]} {event['detail'][:19]} {event['reward']:+.2f}",
                    True,
                    (79, 91, 79),
                ),
                (vieww + 40, y),
            )
        screen.blit(small.render("1 Random · 2 Rule · 3 Custom · 4 PPO · 5 MAPPO", True, (93, 105, 91)), (vieww + 20, h - 40))
        screen.blit(small.render("M sound · Arrows pan · Wheel zoom · R reset", True, (93, 105, 91)), (vieww + 20, h - 22))
        if mode in ("learned", "ppo", "mappo") and train_metrics:
            metric = train_metrics[-1]
            screen.blit(
                small.render(
                    f"Train ep {metric['episode']} | policy {metric['mean_reward']:.2f} | actor {metric['actor_loss']:.3f} | critic {metric['critic_loss']:.3f}",
                    True,
                    (91, 104, 91),
                ),
                (vieww + 20, h - 58),
            )
        if mode in ("learned", "ppo", "mappo"):
            screen.blit(
                small.render(
                    evaluation_summary or "No saved evaluation for this policy",
                    True,
                    (91, 104, 91),
                ),
                (vieww + 20, h - 80),
            )
        if show_analytics:
            pulse_rect = pygame.Rect(25, 22, 270, 34 if analytics_collapsed else 245)
            pygame.draw.rect(screen, (255, 251, 237), pulse_rect, border_radius=12)
            pygame.draw.rect(screen, (192, 183, 154), pulse_rect, 2, border_radius=12)
            screen.blit(font.render(f"Town pulse  {'+' if analytics_collapsed else '−'}", True, INK), (42, 28))
            if not analytics_collapsed:
                cs = env.town.citizens
                wealth = np.asarray([z.money for z in cs])
                gini = float(
                    np.abs(wealth[:, None] - wealth[None, :]).mean()
                    / (2 * wealth.mean() + 1e-6)
                )
                analytics = [
                    ("Happiness", np.mean([z.happiness for z in cs]), .01*np.mean([z.happiness for z in cs]), GREEN),
                    ("Health", np.mean([z.health for z in cs]), .01*np.mean([z.health for z in cs]), GREEN),
                    ("Energy", np.mean([z.energy for z in cs]), .01*np.mean([z.energy for z in cs]), GREEN),
                    ("Stress", np.mean([z.stress for z in cs]), .01*np.mean([z.stress for z in cs]), (211,151,78)),
                    ("Social", np.mean([z.social for z in cs]), .01*np.mean([z.social for z in cs]), GREEN),
                    ("Employment", 100*np.mean([z.employed for z in cs]), np.mean([z.employed for z in cs]), GREEN),
                    ("Survival", 100*np.mean([z.health > 0 for z in cs]), np.mean([z.health > 0 for z in cs]), GREEN),
                    ("Avg. wealth", wealth.mean(), min(1.0, wealth.mean()/100), (184,145,81)),
                    ("Wealth Gini", gini, gini, (184,145,81)),
                    ("Hunger alerts", sum(z.hunger > 80 for z in cs), min(1.0, sum(z.hunger > 80 for z in cs)/max(1,len(cs))), (211,151,78)),
                ]
                for j, (label, value, fraction, color) in enumerate(analytics):
                    col, row = j % 2, j // 2
                    card = pygame.Rect(36 + col*119, 64 + row*34, 111, 30)
                    pygame.draw.rect(screen, (249,246,231), card, border_radius=7)
                    pygame.draw.rect(screen, (230,224,203), card, 1, border_radius=7)
                    screen.blit(small.render(label, True, (84,97,81)), (card.x+6, card.y+4))
                    metric_text = small.render(f"{value:.0f}" if label not in ("Avg. wealth", "Wealth Gini") else (f"${value:.0f}" if label == "Avg. wealth" else f"{value:.2f}"), True, INK)
                    screen.blit(metric_text, (card.right-metric_text.get_width()-6, card.y+4))
                    pygame.draw.rect(screen, (226,222,207), (card.x+6, card.y+20, card.width-12, 5), border_radius=3)
                    pygame.draw.rect(screen, color, (card.x+6, card.y+20, max(0,int((card.width-12)*np.clip(fraction,0,1))), 5), border_radius=3)
        if compare:
            other = env.town.citizens[compare_id]
            pygame.draw.rect(
                screen,
                (255, 251, 237),
                (vieww + 14, 590, panel - 28, 126),
                border_radius=12,
            )
            screen.blit(
                font.render(f"Compare: {other.name} (Tab changes)", True, INK),
                (vieww + 26, 598),
            )
            for j, (label, v1, v2) in enumerate(
                [
                    ("Health", c.health, other.health),
                    ("Hunger", c.hunger, other.hunger),
                    ("Happiness", c.happiness, other.happiness),
                    ("Money", c.money, other.money),
                    (
                        "A/S/R/L",
                        "/".join(f"{v:.1f}" for v in c.personality),
                        "/".join(f"{v:.1f}" for v in other.personality),
                    ),
                ]
            ):
                left_value = format_comparison_value(v1)
                right_value = format_comparison_value(v2)
                screen.blit(
                    small.render(f"{label:10} {left_value:>5} | {right_value:>5}", True, INK),
                    (vieww + 28, 625 + j * 16),
                )
        if active_view == "Town":
            pygame.draw.rect(screen, CREAM, (vieww + 10, 172, panel - 20, h - 182), border_radius=13)
            x, cy, width, bottom = draw_section_shell(
                screen, font, small, vieww, h, town_metadata["name"], town_metadata["description"]
            )
            summary_rows = [
                ("Population · living", f"{stats_cache['population']} · {stats_cache['living_population']}"),
                ("Average wellbeing", f"{stats_cache['average_wellbeing']:.1f}"),
                ("Employment", f"{stats_cache['employment_rate'] * 100:.0f}%"),
                ("Average wealth", f"${stats_cache['wealth_average']:.2f}"),
                ("Wealth inequality (Gini)", f"{stats_cache['wealth_gini']:.3f}"),
                ("Day in simulation", f"{stats_cache['day']}  ·  Age {town_metadata['simulated_age_days']} {'day' if town_metadata['simulated_age_days'] == 1 else 'days'}"),
                ("Weather / season", f"{town_metadata['weather']}  ·  {town_metadata['season']}"),
                ("Town treasury", town_metadata["treasury"]),
            ]
            for index, (label, value) in enumerate(summary_rows):
                row_y = cy + index * 43
                screen.blit(small.render(label, True, (104, 111, 91)), (x, row_y))
                screen.blit(font.render(value, True, INK), (x, row_y + 15))
            feed_y = cy + len(summary_rows) * 43 + 4
            screen.blit(font.render("Recent town events", True, INK), (x, feed_y))
            meaningful = [event for event in env.town.events if event["category"] != "decision"]
            for index, event in enumerate(reversed(meaningful[-3:])):
                y = feed_y + 27 + index * 25
                stamp = event.get("timestamp", {})
                line = f"D{stamp.get('day', 1):02d} {stamp.get('hour', 0):02d}:00  {event['description']}"
                screen.blit(small.render(line[:44], True, (82, 98, 84)), (x, y))
        elif active_view == "People":
            controls_y = 180 if show_people_directory else 298
            directory_fill = (73, 104, 77) if show_people_directory else (229, 224, 205)
            bars_fill = (73, 104, 77) if show_all_wellbeing else (229, 224, 205)
            pygame.draw.rect(screen, directory_fill, (vieww + 12, controls_y, 118, 24), border_radius=7)
            screen.blit(small.render("Citizen directory", True, (250, 246, 230) if show_people_directory else INK), (vieww + 21, controls_y + 5))
            pygame.draw.rect(screen, bars_fill, (vieww + 205, controls_y, 120, 24), border_radius=7)
            screen.blit(small.render(f"Bars: {'All' if show_all_wellbeing else 'Hover + selected'}", True, (250, 246, 230) if show_all_wellbeing else INK), (vieww + 211, controls_y + 5))
            if show_people_directory:
                pygame.draw.rect(screen, CREAM, (vieww + 10, 172, panel - 20, h - 182), border_radius=13)
                pygame.draw.rect(screen, directory_fill, (vieww + 12, 180, 118, 24), border_radius=7)
                screen.blit(small.render("Citizen directory", True, (250, 246, 230) if show_people_directory else INK), (vieww + 21, 185))
                pygame.draw.rect(screen, bars_fill, (vieww + 205, 180, 120, 24), border_radius=7)
                screen.blit(small.render(f"Bars: {'All' if show_all_wellbeing else 'Hover + selected'}", True, (250, 246, 230) if show_all_wellbeing else INK), (vieww + 211, 185))
                x, cy, width, bottom = draw_section_shell(screen, font, small, vieww, h, "Citizens", "Search by name, job, activity, or status")
                search_rect = pygame.Rect(vieww + 26, 218, 284, 25)
                pygame.draw.rect(screen, (249, 246, 231), search_rect, border_radius=6)
                pygame.draw.rect(screen, (121, 151, 109) if search_mode == "People" else (218, 211, 190), search_rect, 1, border_radius=6)
                query = search_query or "Type to filter citizens…"
                screen.blit(small.render(query[:39], True, (82, 98, 84)), (search_rect.x + 8, search_rect.y + 6))
                matches = [
                    citizen for citizen in env.town.citizens
                    if search_query.casefold() in citizen.name.casefold()
                    or search_query.casefold() in citizen.occupation.casefold()
                    or search_query.casefold() in citizen.activity.casefold()
                ]
                visible_count = max(1, (bottom - 270) // 34)
                citizen_directory_offset = min(citizen_directory_offset, max(0, len(matches) - visible_count))
                visible_matches = matches[citizen_directory_offset:citizen_directory_offset + visible_count]
                for index, citizen in enumerate(visible_matches):
                    row_y = 249 + index * 34
                    if row_y + 30 > bottom - 18:
                        break
                    row_rect = pygame.Rect(vieww + 24, row_y, 292, 30)
                    pygame.draw.rect(screen, (230, 237, 216) if citizen.id == selected else (249, 246, 231), row_rect, border_radius=6)
                    screen.blit(small.render(f"{citizen.name}  ·  {citizen.occupation}", True, INK), (row_rect.x + 8, row_rect.y + 4))
                    detail = small.render(f"{citizen.activity}  ·  W {citizen_wellbeing(citizen, wellbeing_weights):.0f}", True, (104, 111, 91))
                    screen.blit(detail, (row_rect.x + 8, row_rect.y + 17))
                if len(matches) > visible_count:
                    screen.blit(small.render(f"Showing {citizen_directory_offset + 1}–{min(len(matches), citizen_directory_offset + visible_count)} of {len(matches)} · scroll for more", True, (104, 111, 91)), (x, bottom - 17))
        elif active_view == "Places":
            pygame.draw.rect(screen, (250, 248, 235), (vieww + 10, 172, panel - 20, h - 182), border_radius=13)
            labels_fill = (73, 104, 77) if show_building_labels else (229, 224, 205)
            directory_fill = (73, 104, 77) if show_building_directory else (229, 224, 205)
            pygame.draw.rect(screen, labels_fill, (vieww + 12, 180, 118, 24), border_radius=7)
            screen.blit(small.render(f"Labels: {'On' if show_building_labels else 'Off'}", True, (250, 246, 230) if show_building_labels else INK), (vieww + 21, 185))
            pygame.draw.rect(screen, directory_fill, (vieww + 135, 180, 90, 24), border_radius=7)
            screen.blit(small.render("Directory", True, (250, 246, 230) if show_building_directory else INK), (vieww + 155, 185))
            if selected_building_id in building_by_id and not show_building_directory:
                building_data = inspect_building(env.town, building_by_id[selected_building_id])
                draw_building_inspector(screen, font, small, vieww, h, building_data)
            elif show_building_directory:
                x, cy, width, bottom = draw_section_shell(screen, font, small, vieww, h, "Establishments", "Search by name or type; click to inspect")
                search_rect = pygame.Rect(vieww + 26, 218, 284, 25)
                pygame.draw.rect(screen, (249, 246, 231), search_rect, border_radius=6)
                pygame.draw.rect(screen, (121, 151, 109) if search_mode == "Places" else (218, 211, 190), search_rect, 1, border_radius=6)
                query = search_query or "Type to filter places…"
                screen.blit(small.render(query[:39], True, (82, 98, 84)), (search_rect.x + 8, search_rect.y + 6))
                matches = [
                    building for building in building_records
                    if search_query.casefold() in building.name.casefold()
                    or search_query.casefold() in building.kind.casefold()
                    or search_query.casefold() in building.id.casefold()
                ]
                visible_count = max(1, (bottom - 258) // 24)
                place_directory_offset = min(place_directory_offset, max(0, len(matches) - visible_count))
                visible_matches = matches[place_directory_offset:place_directory_offset + visible_count]
                for index, building in enumerate(visible_matches):
                    row_y = 249 + index * 24
                    if row_y + 25 >= bottom - 16:
                        break
                    row_rect = pygame.Rect(vieww + 22, row_y, 296, 24)
                    pygame.draw.rect(screen, (230, 237, 216) if building.id == selected_building_id else (249, 246, 231), row_rect, border_radius=6)
                    screen.blit(small.render(building.name, True, INK), (row_rect.x + 8, row_rect.y + 5))
                    kind_label = small.render(building.kind.title(), True, (104, 111, 91))
                    screen.blit(kind_label, (row_rect.right - kind_label.get_width() - 8, row_rect.y + 5))
                if len(matches) > visible_count:
                    screen.blit(small.render(f"Showing {place_directory_offset + 1}–{min(len(matches), place_directory_offset + visible_count)} of {len(matches)} · scroll for more", True, (104, 111, 91)), (x, bottom - 17))
            else:
                x, cy, width, bottom = draw_section_shell(screen, font, small, vieww, h, "Places", "Select a building on the map or open the directory")
        elif active_view == "Data":
            pygame.draw.rect(screen, (250, 248, 235), (vieww + 10, 172, panel - 20, h - 182), border_radius=13)
            pages = ("Economy", "Social", "Events", "RL Lab", "Reports")
            for index, page in enumerate(pages):
                rect = pygame.Rect(vieww + 12 + index * 63, 180, 60, 25)
                pygame.draw.rect(screen, (73, 104, 77) if data_view == page else (229, 224, 205), rect, border_radius=6)
                label = small.render(page, True, (250, 246, 230) if data_view == page else INK)
                screen.blit(label, label.get_rect(center=rect.center))
            shell_y = 212
            shell_height = h - shell_y - 18
            shell = pygame.Rect(vieww + 12, shell_y, 316, shell_height)
            pygame.draw.rect(screen, (255, 252, 240), shell, border_radius=13)
            pygame.draw.rect(screen, (224, 217, 197), shell, 1, border_radius=13)
            x, cy = vieww + 26, shell_y + 16
            screen.blit(font.render(data_view, True, INK), (x, cy))
            cy += 34
            if data_view == "Economy":
                rows = [
                    ("Total citizen wealth", f"${stats_cache['wealth_total']:.2f}"),
                    ("Average wealth", f"${stats_cache['wealth_average']:.2f}"),
                    ("Wealth Gini", f"{stats_cache['wealth_gini']:.3f}"),
                    ("Employment", f"{stats_cache['employment_rate'] * 100:.0f}%"),
                    ("Recorded salary payments", f"${stats_cache['salary_payments_total']:.2f}"),
                    ("Recorded shopping spend", f"${stats_cache['shopping_spend_total']:.2f}"),
                    ("Food stock / price", f"{stats_cache['food_stock_total']} units  ·  ${stats_cache['food_price']:.2f}"),
                ]
            elif data_view == "Social":
                rows = [
                    ("Average social satisfaction", f"{stats_cache['social_satisfaction_average']:.1f}"),
                    ("Recorded peer interactions", str(stats_cache["social_interactions_recorded"])),
                    ("Reported incidents · last 30d", str(stats_cache["reported_crimes_30d"])),
                    ("Crime reports / 1k resident-days", f"{stats_cache['crime_rate_per_1000_resident_days']:.1f}"),
                    ("Safety services", "Civic landmark; response not modeled"),
                ]
                rows.extend((f"Low social: {citizen.name}", f"{citizen.social:.0f}  ·  {citizen.activity}") for citizen in sorted(env.town.citizens, key=lambda person: person.social)[:5])
            elif data_view == "Events":
                filters = ("All", "Economy", "Social", "Health", "Employment", "Safety", "Calendar")
                for index, label in enumerate(filters):
                    rect = pygame.Rect(vieww + 26 + index * 41, 249, 39, 22)
                    pygame.draw.rect(screen, (213, 230, 200) if event_filter == label else (235, 229, 211), rect, border_radius=5)
                    visible_label = {
                        "Economy": "Econ", "Social": "Soc", "Health": "Hlth",
                        "Employment": "Jobs", "Safety": "Safe", "Calendar": "Dates",
                    }.get(label, label)
                    screen.blit(small.render(visible_label, True, (250, 246, 230) if event_filter == label else INK), (rect.x + 5, rect.y + 5))
                search_rect = pygame.Rect(vieww + 26, 278, 284, 24)
                pygame.draw.rect(screen, (249, 246, 231), search_rect, border_radius=6)
                pygame.draw.rect(screen, (121, 151, 109) if search_mode == "Data" else (218, 211, 190), search_rect, 1, border_radius=6)
                screen.blit(small.render(search_query or "Filter event text / citizen / place…", True, (82, 98, 84)), (search_rect.x + 8, search_rect.y + 6))
                screen.blit(small.render("Red safety cards are story reports only.", True, (137, 67, 58)), (x, 306))
                filtered = []
                query = search_query.casefold()
                for event in reversed(env.town.events):
                    if event.get("category") == "decision":
                        continue
                    if event_filter != "All" and event.get("category") != event_filter.casefold():
                        continue
                    citizen = next((person for person in env.town.citizens if person.id == event.get("citizen_id")), None)
                    building = building_by_id.get(event.get("building_id"))
                    search_text = " ".join((event.get("description", ""), citizen.name if citizen else "", building.name if building else "", event.get("category", ""))).casefold()
                    if query and query not in search_text:
                        continue
                    filtered.append(event)
                    if len(filtered) >= 8:
                        break
                for index, event in enumerate(filtered):
                    row_y = 330 + index * 44
                    stamp = event.get("timestamp", {})
                    person = next((p.name for p in env.town.citizens if p.id == event.get("citizen_id")), "Town")
                    is_safety_alert = event.get("category") == "safety"
                    if is_safety_alert:
                        card = pygame.Rect(vieww + 20, row_y - 2, 300, 42)
                        pygame.draw.rect(screen, (255, 239, 235), card, border_radius=6)
                        pygame.draw.rect(screen, (206, 78, 69), card, 1, border_radius=6)
                        screen.blit(small.render("STORY", True, (183, 55, 48)), (card.x + 6, row_y + 1))
                        line1 = f"D{stamp.get('day', 1):02d} {stamp.get('hour', 0):02d}:00 · {person}"
                        line_color = (158, 48, 43)
                        line_x = card.x + 44
                    else:
                        line1 = f"D{stamp.get('day', 1):02d} {stamp.get('hour', 0):02d}:00  {event['category'].title()} · {person}"
                        line_color = INK
                        line_x = x
                    line2 = event.get("description", "")
                    screen.blit(small.render(line1[:43], True, line_color), (line_x, row_y))
                    for text_index, text_line in enumerate(wrap_ui_text(line2, small, 278)[:2]):
                        screen.blit(small.render(text_line, True, (105, 76, 73) if is_safety_alert else (82, 98, 84)), (x, row_y + 15 + text_index * 12))
            elif data_view == "RL Lab":
                rows = [
                    ("Active selection", "Learned" if mode in ("learned", "ppo", "mappo") else mode.title()),
                    ("Live town residents", str(len(env.town.citizens))),
                    ("Checkpoint population", str(getattr(policy, "population", "Baseline"))),
                    ("Policy / algorithm", selected_algorithm or ("Rule baseline" if mode == "rule" else "Random baseline")),
                    ("Checkpoint", Path(checkpoint).name if checkpoint else "None"),
                    ("Training diagnostics", "Separate from town wellbeing"),
                    ("Evaluation", evaluation_summary or "No saved evaluation for this policy"),
                ]
                if mode in ("learned", "ppo", "mappo") and train_metrics:
                    metric = train_metrics[-1]
                    rows.extend([
                        ("Training episode", str(metric.get("episode", "Unavailable"))),
                        ("Mean reward", f"{metric.get('mean_reward', 0):.3f}"),
                        ("Actor / critic loss", f"{metric.get('actor_loss', 0):.4f} / {metric.get('critic_loss', 0):.4f}"),
                        ("Entropy", f"{float(metric.get('entropy', 0)):.3f}"),
                    ])
            else:
                rows = [
                    ("Daily snapshots stored", str(len(env.town.daily_summaries))),
                    ("Campaign calendar", stats_cache["calendar_date"]),
                    ("Day / hour", f"{stats_cache['day']} / {stats_cache['hour']:02d}:00"),
                    ("Population wellbeing", f"{stats_cache['average_wellbeing']:.1f}"),
                    ("Employment", f"{stats_cache['employment_rate'] * 100:.0f}%"),
                    ("Economy", f"Wealth ${stats_cache['wealth_total']:.2f}  ·  Gini {stats_cache['wealth_gini']:.3f}"),
                    ("Social activity", str(stats_cache["social_interactions_recorded"])),
                    ("RL result", evaluation_summary or "No saved evaluation for this policy"),
                ]
            if data_view != "Events":
                row_step = 32 if data_view == "RL Lab" else 42
                for index, (label, value) in enumerate(rows):
                    row_y = cy + index * row_step
                    screen.blit(small.render(label, True, (104, 111, 91)), (x, row_y))
                    screen.blit(small.render(str(value)[:45], True, INK), (x, row_y + 16))
            if data_view == "Economy":
                trend_y = cy + len(rows) * 42 + 2
                screen.blit(font.render("Recent daily activity", True, INK), (x, trend_y))
                snapshots = env.town.daily_summaries[-5:]
                if not snapshots:
                    screen.blit(small.render("Daily trends appear after the first simulated day.", True, (104, 111, 91)), (x, trend_y + 26))
                for index, snapshot in enumerate(reversed(snapshots)):
                    line_y = trend_y + 27 + index * 32
                    line = (
                        f"D{snapshot['day']:02d}  W {snapshot['average_wellbeing']:.0f}  "
                        f"Salary ${snapshot['salary_payments']:.0f}  Spend ${snapshot['shopping_spend']:.0f}"
                    )
                    screen.blit(small.render(line, True, (82, 98, 84)), (x, line_y))
            if data_view == "Reports":
                trend_y = cy + len(rows) * 42 + 2
                screen.blit(font.render("Daily summary history", True, INK), (x, trend_y))
                snapshots = env.town.daily_summaries[-5:]
                for index, snapshot in enumerate(reversed(snapshots)):
                    line_y = trend_y + 27 + index * 32
                    line = (
                        f"D{snapshot['day']:02d}  alive {snapshot['living_population']}/"
                        f"{snapshot['population']}  W {snapshot['average_wellbeing']:.0f}  "
                        f"jobs {snapshot['employment_rate'] * 100:.0f}%"
                    )
                    screen.blit(small.render(line, True, (82, 98, 84)), (x, line_y))
            if data_view == "RL Lab":
                comparison_y = cy + len(rows) * row_step + 2
                screen.blit(font.render("Saved evaluation comparison", True, INK), (x, comparison_y))
                for index, label in enumerate(("Rule", "PPO", "MAPPO")):
                    result_path = Path("results") / f"{label.lower()}.json"
                    try:
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        episodes = result.get("episodes", [])
                        wellbeing = float(np.mean([row["wellbeing"] for row in episodes])) if episodes else float("nan")
                        survival = float(np.mean([row["survival_rate"] for row in episodes])) if episodes else float("nan")
                        line = f"{label}: W {wellbeing:.1f}  Alive {survival * 100:.0f}%  n={len(episodes)}"
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        line = f"{label}: no saved evaluation"
                    screen.blit(small.render(line, True, (82, 98, 84)), (x, comparison_y + 27 + index * 25))
        # Keep research and control hints legible above every section panel.
        footer_top = max(600, h - 84)
        footer_rect = pygame.Rect(vieww + 10, footer_top, panel - 20, h - footer_top - 8)
        pygame.draw.rect(screen, (236, 233, 215), footer_rect, border_radius=12)
        pygame.draw.line(screen, (210, 204, 180), (footer_rect.x + 12, footer_rect.y), (footer_rect.right - 12, footer_rect.y), 1)
        eval_line = evaluation_summary if mode in ("learned", "ppo", "mappo") and evaluation_summary else "Policy outcomes update as citizens act"
        screen.blit(micro.render(str(eval_line), True, (67, 83, 68)), (vieww + 20, footer_top + 7))
        if mode in ("learned", "ppo", "mappo") and train_metrics:
            metric = train_metrics[-1]
            training_line = f"Train ep {metric['episode']} · R {metric['mean_reward']:.2f} · W {metric.get('mean_wellbeing', 0):.1f}"
        else:
            training_line = "Saved metrics and learning curves · Graphs"
        screen.blit(micro.render(training_line, True, (87, 99, 82)), (vieww + 20, footer_top + 24))
        screen.blit(micro.render("1 Random · 2 Rule · 3 Custom · 4 PPO · 5 MAPPO", True, (87, 99, 82)), (vieww + 20, footer_top + 41))
        screen.blit(micro.render("M Sound · Arrows pan · Wheel zoom · R reset", True, (87, 99, 82)), (vieww + 20, footer_top + 58))
        pygame.display.flip()
        if profile:
            frame_times.append(time.perf_counter() - frame_started)
        if screenshot_path:
            pygame.image.save(screen, screenshot_path)
            screenshot_path = None
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False
    pygame.quit()
    if profile and frame_times:
        ordered = sorted(frame_times)
        mean_seconds = sum(ordered) / len(ordered)
        p95_seconds = ordered[min(len(ordered) - 1, round(.95 * (len(ordered) - 1)))]
        print(json.dumps({"profile_frames": len(ordered), "mean_frame_ms": mean_seconds * 1000, "p95_frame_ms": p95_seconds * 1000, "mean_fps": 1 / mean_seconds}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random", "rule", "learned", "ppo", "mappo"], default="rule")
    p.add_argument("--checkpoint", default="models/mappo.pt")
    p.add_argument("--smoke-frames", type=int)
    p.add_argument("--screenshot")
    p.add_argument("--profile", action="store_true", help="report render-loop frame timing")
    p.add_argument("--preview-analytics", action="store_true", help="open the town pulse overlay for a screenshot")
    p.add_argument("--preview-all-bars", action="store_true", help="show all citizen bars for a screenshot")
    p.add_argument("--preview-section", choices=["Town", "People", "Places", "Data"], default="People", help="open a UI section for screenshot smoke")
    p.add_argument("--preview-data", choices=["Economy", "Social", "Events", "RL Lab", "Reports"], default="Economy", help="open a Data subsection for screenshot smoke")
    p.add_argument("--preview-building", help="open a named building inspector for screenshot smoke")
    a = p.parse_args()
    run(a.mode, a.checkpoint, a.smoke_frames, a.screenshot, a.profile, a.preview_analytics, a.preview_all_bars, a.preview_section, a.preview_data, a.preview_building)




