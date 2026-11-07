"""Cached, renderer-only pixel-inspired town art and citizen sprites.

All coordinates are presentation coordinates on the town grid. This module
never changes walkability or simulation state.
"""

from __future__ import annotations

import math
import random

import pygame

from ai_town.buildings import BUILDINGS, TOWN_HEIGHT, TOWN_WIDTH, presentation_buildings

BASE_TILE = 32
_MAP_CACHE: dict[int, pygame.Surface] = {}
_DEPTH_CACHE: dict[tuple[int, tuple[tuple[str, str], ...]], list[tuple[pygame.Surface, int, int, int]]] = {}
_SPRITE_CACHE: dict[tuple[int, int, int, int, int, int], pygame.Surface] = {}
WELLBEING_THRESHOLDS = (25.0, 50.0, 75.0)
_WELLBEING_COLORS = (
    (198, 70, 62),
    (232, 143, 63),
    (232, 202, 78),
    (157, 190, 91),
    (70, 154, 92),
)
PALETTE = {
    "grass": (118, 158, 93), "grass_light": (137, 174, 105),
    "grass_dark": (91, 133, 79), "soil": (166, 135, 91),
    "path": (218, 194, 151), "path_edge": (157, 139, 105),
    "water": (77, 151, 171), "ink": (54, 65, 56),
}


def citizen_is_hovered(mouse_x: float, mouse_y: float, center_x: float, foot_y: float, tile_px: float) -> bool:
    """Return whether the pointer is inside the renderer's citizen hit region."""
    return abs(mouse_x - center_x) <= tile_px * 0.62 and foot_y - 2 * tile_px <= mouse_y <= foot_y + tile_px * 0.15


def wellbeing_bar_visible(show_all: bool, citizen_id: int, selected_id: int, hovered: bool) -> bool:
    """Apply the viewer's selected/hovered versus all-citizens bar mode."""
    return show_all or citizen_id == selected_id or hovered


def _building(surface: pygame.Surface, x: int, y: int, kind: str, index: int, name: str | None = None) -> None:
    t = BASE_TILE
    px, py = x * t, y * t
    colors = {
        "homes": ((232, 203, 157), (151, 91, 70), (116, 160, 171)),
        "offices": ((177, 194, 175), (90, 126, 121), (139, 190, 191)),
        "groceries": ((236, 216, 157), (179, 112, 69), (147, 192, 184)),
        "restaurant": ((222, 171, 139), (142, 78, 70), (165, 192, 188)),
        "clinic": ((230, 228, 204), (127, 157, 147), (169, 195, 194)),
        "town": ((232, 208, 151), (135, 104, 69), (149, 184, 177)),
        "police": ((190, 207, 198), (68, 95, 112), (144, 190, 197)),
    }
    wall, roof, window = colors.get(kind, colors["homes"])
    if kind == "homes":
        walls = ((232, 203, 157), (219, 188, 142), (235, 214, 170), (210, 190, 159))
        roofs = ((151, 91, 70), (129, 83, 67), (157, 100, 72), (119, 81, 70), (146, 90, 82))
        wall, roof = walls[index % len(walls)], roofs[(index * 2 + index // 3) % len(roofs)]
        window = ((116, 160, 171), (141, 177, 180), (104, 145, 161))[index % 3]
    width, height = 76, 54 if kind == "homes" else 60
    # Matching garden pads give every civic and residential lot a consistent
    # ground footprint, making selection bounds and street setbacks predictable.
    pygame.draw.ellipse(surface, (68, 98, 61), (px - 2, py + 40, width + 7, 21))
    garden_y, garden_height = 48, 22
    pygame.draw.rect(surface, (113, 151, 86), (px - 5, py + garden_y, width + 8, garden_height))
    flower_y = garden_y + garden_height // 2
    for fx in range(px, px + width, 9):
        pygame.draw.circle(surface, ((237, 197, 118), (214, 142, 153), (237, 230, 196))[(fx // 9 + index) % 3], (fx, py + flower_y), 2)
    # Wall with timber corners, sill and shadowed foundation.
    body_y = py - height + 22
    pygame.draw.rect(surface, (83, 86, 66), (px + 4, body_y + height - 5, width - 1, 8))
    # A narrow shaded return and bright foundation lip make the facade read
    # as a little 2.5D building when the depth layers pass in front of it.
    side_shade = tuple(max(0, channel - 30) for channel in wall)
    pygame.draw.polygon(surface, side_shade, [
        (px + width - 3, body_y + 13),
        (px + width + 6, body_y + 18),
        (px + width + 6, body_y + height - 5),
        (px + width - 3, body_y + height - 4),
    ])
    pygame.draw.rect(surface, wall, (px + 3, body_y + 12, width - 3, height - 15))
    pygame.draw.rect(surface, tuple(max(0, c - 24) for c in wall), (px + 3, body_y + height - 8, width - 3, 4))
    pygame.draw.line(surface, (244, 226, 188), (px + 4, body_y + height - 4), (px + width - 4, body_y + height - 4), 2)
    # Vary the roofline by use: homes stay pitched, while offices read as a
    # modern civic/workplace block with a stepped flat roof.
    roof_y = body_y - 5
    if kind == "offices":
        pygame.draw.rect(surface, (68, 84, 78), (px - 4, roof_y + 8, width + 8, 17))
        pygame.draw.rect(surface, roof, (px - 3, roof_y + 5, width + 6, 16))
        pygame.draw.rect(surface, tuple(min(255, c + 23) for c in roof), (px + 3, roof_y + 2, width - 6, 5))
        # A small rooftop lantern reaches the shared roofline without turning
        # the office silhouette back into a house-shaped gable.
        pygame.draw.rect(surface, (76, 91, 84), (px + width//2 - 9, roof_y - 9, 18, 9))
        pygame.draw.rect(surface, (146, 176, 164), (px + width//2 - 6, roof_y - 7, 12, 4))
        pygame.draw.line(surface, (213, 226, 207), (px - 3, roof_y + 20), (px + width + 3, roof_y + 20), 2)
        for wx in range(px + 8, px + width - 4, 13):
            pygame.draw.rect(surface, (91, 125, 125), (wx, body_y + 18, 9, 7))
            pygame.draw.line(surface, (192, 214, 203), (wx + 1, body_y + 19), (wx + 7, body_y + 19), 1)
    else:
        pygame.draw.polygon(surface, (74, 68, 58), [(px - 5, roof_y + 23), (px + width // 2, roof_y - 5), (px + width + 4, roof_y + 23)])
        pygame.draw.polygon(surface, roof, [(px - 4, roof_y + 19), (px + width // 2, roof_y - 9), (px + width + 3, roof_y + 19)])
        for row in range(3):
            yy = roof_y + 7 + row * 6
            inset = row * 8
            pygame.draw.line(surface, tuple(min(255, c + 22) for c in roof), (px + inset, yy), (px + width - inset, yy), 2)
        pygame.draw.line(surface, (244, 218, 168), (px - 4, roof_y + 19), (px + width + 3, roof_y + 19), 3)
    if kind == "homes" and index % 3 == 1:
        chimney_x = px + width - 16
        pygame.draw.rect(surface, (102, 76, 62), (chimney_x, roof_y - 14, 8, 18))
        pygame.draw.rect(surface, (155, 119, 91), (chimney_x - 2, roof_y - 15, 12, 3))
    if kind == "homes" and index % 4 == 2:
        dormer_x = px + width // 2 - 8
        pygame.draw.polygon(surface, (82, 68, 58), [(dormer_x - 3, roof_y + 4), (dormer_x + 8, roof_y - 7), (dormer_x + 19, roof_y + 4)])
        pygame.draw.rect(surface, wall, (dormer_x, roof_y + 2, 16, 11))
        pygame.draw.rect(surface, (79, 91, 81), (dormer_x + 3, roof_y + 4, 10, 7))
        pygame.draw.rect(surface, window, (dormer_x + 4, roof_y + 5, 8, 5))
    # Window frames and warm panes.
    for wx in (px + 14, px + 49):
        pygame.draw.rect(surface, (92, 80, 64), (wx - 2, body_y + 28, 15, 15))
        pygame.draw.rect(surface, window, (wx, body_y + 30, 11, 11))
        pygame.draw.line(surface, (232, 222, 187), (wx + 5, body_y + 30), (wx + 5, body_y + 41), 1)
        pygame.draw.line(surface, (232, 222, 187), (wx, body_y + 35), (wx + 11, body_y + 35), 1)
    pygame.draw.rect(surface, (105, 73, 55), (px + 31, body_y + 40, 13, 22))
    pygame.draw.circle(surface, (229, 195, 116), (px + 41, body_y + 51), 1)
    if kind == "clinic":
        pygame.draw.rect(surface, (180, 88, 82), (px + 30, body_y - 3, 18, 5))
        pygame.draw.rect(surface, (250, 238, 217), (px + 36, body_y - 9, 6, 17))
        pygame.draw.rect(surface, (250, 238, 217), (px + 30, body_y - 3, 18, 5))
    if kind == "police":
        pygame.draw.rect(surface, (240, 235, 215), (px + 27, body_y - 5, 25, 8))
        pygame.draw.circle(surface, (66, 112, 160), (px + 33, body_y - 1), 3)
        pygame.draw.circle(surface, (185, 74, 72), (px + 46, body_y - 1), 3)
    if name:
        if not pygame.font.get_init():
            pygame.font.init()
        if kind == "homes":
            sign_rect = pygame.Rect(px + 43, body_y + 42, 34, 11)
            sign_color = (238, 222, 180)
            sign_text = name.split()[0][:6] if name else "Home"
        else:
            sign_rect = pygame.Rect(px + 10, body_y + 14, width - 20, 12)
            sign_color = (239, 226, 193)
            sign_text = name[:16]
        pygame.draw.rect(surface, (93, 76, 56), sign_rect.inflate(2, 2))
        pygame.draw.rect(surface, sign_color, sign_rect)
        sign_font = pygame.font.Font(None, 9 if kind == "homes" else 10)
        sign_image = sign_font.render(sign_text, True, (60, 69, 57))
        surface.blit(sign_image, sign_image.get_rect(center=sign_rect.center))
    if kind in ("groceries", "restaurant"):
        awning = (164, 82, 73) if kind == "restaurant" else (132, 154, 91)
        pygame.draw.rect(surface, (245, 232, 197), (px + 6, body_y + 47, 62, 9))
        for stripe in range(7):
            pygame.draw.rect(surface, awning if stripe % 2 == 0 else (239, 218, 176), (px + 7 + stripe * 9, body_y + 48, 8, 8))
    elif kind == "town":
        pygame.draw.circle(surface, (246, 224, 169), (px + 39, body_y + 16), 8)
        pygame.draw.circle(surface, (101, 94, 72), (px + 39, body_y + 16), 7, 1)
        pygame.draw.line(surface, (91, 82, 64), (px + 39, body_y + 16), (px + 39, body_y + 11), 1)


def _tree(surface: pygame.Surface, x: int, y: int, variant: int) -> None:
    t = BASE_TILE
    px, py = int((x + .5) * t), int((y + .7) * t)
    pygame.draw.ellipse(surface, (68, 99, 64), (px - 18, py + 8, 38, 11))
    pygame.draw.rect(surface, (111, 79, 52), (px - 4, py - 4, 9, 22))
    shades = [(67, 119, 75), (77, 133, 77), (89, 143, 84)]
    col = shades[variant % len(shades)]
    for ox, oy, radius in [(-9, -14, 14), (7, -18, 15), (0, -27, 13), (-1, -7, 14)]:
        pygame.draw.circle(surface, tuple(max(0, c - 17) for c in col), (px + ox + 2, py + oy + 3), radius)
        pygame.draw.circle(surface, col, (px + ox, py + oy), radius)
    pygame.draw.circle(surface, (124, 167, 91), (px - 6, py - 24), 5)


def town_surface(tile_px: int) -> pygame.Surface:
    """Return a cached, complete town illustration scaled with nearest-neighbour."""
    tile_px = max(12, int(tile_px))
    if tile_px in _MAP_CACHE:
        return _MAP_CACHE[tile_px]
    t = BASE_TILE
    world = pygame.Surface((TOWN_WIDTH * t, TOWN_HEIGHT * t))
    world.fill(PALETTE["grass"])
    rng = random.Random(73)
    # Low-contrast, overlapping organic patches replace per-cell colour blocks.
    for _ in range(210):
        cx, cy = rng.randrange(world.get_width()), rng.randrange(world.get_height())
        rx, ry = rng.randrange(22, 100), rng.randrange(18, 76)
        shade = rng.choice(((115, 155, 91), (120, 160, 95), (123, 161, 97), (114, 154, 90)))
        pygame.draw.ellipse(world, shade, (cx-rx, cy-ry, rx*2, ry*2))
    area_scale = (TOWN_WIDTH * TOWN_HEIGHT) / (36 * 24)
    for _ in range(int(1800 * area_scale)):
        x, y = rng.randrange(world.get_width()), rng.randrange(world.get_height())
        c = rng.choice(((93, 137, 79), (143, 174, 105), (154, 176, 111)))
        if rng.random() < .72:
            pygame.draw.line(world, c, (x, y+3), (x+2, y), 1)
            pygame.draw.line(world, c, (x+3, y+3), (x+2, y), 1)
        else:
            world.set_at((x, y), c)
    # Fine, clustered blades add a hand-painted surface texture at close zoom.
    # A fixed seed keeps the authored terrain identical across runs.
    grass_shades = ((96, 139, 82), (108, 149, 88), (145, 174, 108), (158, 181, 116))
    for _ in range(int(4200 * area_scale)):
        x, y = rng.randrange(world.get_width()), rng.randrange(world.get_height())
        shade = rng.choice(grass_shades)
        height = rng.choice((2, 3, 4))
        pygame.draw.line(world, shade, (x, y + height), (x + rng.choice((-1, 0, 1)), y), 1)
        if rng.random() < .42:
            pygame.draw.line(world, shade, (x + 2, y + height), (x + 3, y + 1), 1)
    # A broad east-west boulevard and north-south market street.
    pygame.draw.rect(world, (112, 111, 94), (0, 9*t-3, TOWN_WIDTH*t, 2*t+6))
    pygame.draw.rect(world, (202, 181, 143), (0, 9*t+3, TOWN_WIDTH*t, 2*t-6))
    pygame.draw.rect(world, (112, 111, 94), (15*t-3, 0, 2*t+6, TOWN_HEIGHT*t))
    pygame.draw.rect(world, (202, 181, 143), (15*t+3, 0, 2*t-6, TOWN_HEIGHT*t))
    # Beveled curb edges catch light on their upper lip and shade the pavement below.
    pygame.draw.line(world, (91, 103, 82), (0, 9*t+2), (TOWN_WIDTH*t, 9*t+2), 2)
    pygame.draw.line(world, (232, 211, 169), (0, 9*t+5), (TOWN_WIDTH*t, 9*t+5), 2)
    pygame.draw.line(world, (91, 103, 82), (15*t+2, 0), (15*t+2, TOWN_HEIGHT*t), 2)
    pygame.draw.line(world, (232, 211, 169), (15*t+5, 0), (15*t+5, TOWN_HEIGHT*t), 2)
    # New east quarter road and southern connector serve the expanded housing.
    pygame.draw.rect(world, (112, 111, 94), (34*t-3, 9*t-3, 2*t+6, 19*t+6))
    pygame.draw.rect(world, (202, 181, 143), (34*t+3, 9*t+3, 2*t-6, 19*t-6))
    pygame.draw.rect(world, (112, 111, 94), (34*t-3, 27*t-3, (TOWN_WIDTH-34)*t, 2*t+6))
    pygame.draw.rect(world, (202, 181, 143), (34*t+3, 27*t+3, (TOWN_WIDTH-34)*t, 2*t-6))
    pygame.draw.line(world, (91, 103, 82), (34*t+2, 9*t), (34*t+2, 28*t), 2)
    pygame.draw.line(world, (232, 211, 169), (34*t+5, 9*t), (34*t+5, 28*t), 2)
    pygame.draw.line(world, (91, 103, 82), (34*t, 27*t+2), (TOWN_WIDTH*t, 27*t+2), 2)
    pygame.draw.line(world, (232, 211, 169), (34*t, 27*t+5), (TOWN_WIDTH*t, 27*t+5), 2)
    # Continuous pedestrian paths connect the civic square, park and districts;
    # their thin edging keeps foot traffic readable without dotted visual noise.
    routes = [
        [(16, 11), (16, 13), (18, 14), (20, 15)],
        [(20, 15), (20, 17), (21, 19), (23, 21)],
        [(34, 18), (36, 18), (38, 18)],
        [(11, 27), (13, 27), (15, 27)],
    ]
    for route in routes:
        points = [(ax*t + t//2, ay*t + t//2) for ax, ay in route]
        pygame.draw.lines(world, (157, 139, 105), False, points, 13)
        pygame.draw.lines(world, (218, 194, 151), False, points, 9)
    # Water with an irregular natural edge and animated-looking fixed highlights.
    pygame.draw.ellipse(world, (195, 177, 132), (30*t+4, 11*t+5, 3*t-8, 6*t+8))
    pygame.draw.ellipse(world, (80, 151, 169), (30*t+12, 11*t+11, 3*t-20, 6*t-4))
    pygame.draw.ellipse(world, (106, 177, 186), (31*t, 12*t, 2*t, 4*t))
    for i in range(9):
        x, y = 31*t + 8 + (i*13)%44, 12*t + 16 + (i*13)%106
        pygame.draw.line(world, (169, 218, 211), (x, y), (x+9+(i%3)*3, y), 2)
    # Landscaped parks with paths, flower beds, trees, and a central feature.
    for index, (x, y) in enumerate(presentation_buildings()["parks"]):
        w, h = 6, 5
        pygame.draw.rect(world, (78, 126, 79), (x*t, y*t, w*t, h*t), border_radius=18)
        pygame.draw.rect(world, (139, 178, 105), (x*t+5, y*t+5, w*t-10, h*t-10), border_radius=16)
        pygame.draw.rect(world, (211, 190, 147), (x*t+12, y*t+h*t//2-3, w*t-24, 6), border_radius=3)
        for j in range(8):
            fx=x*t+12+(j*31)%(w*t-24); fy=y*t+13+(j*23)%(h*t-26)
            pygame.draw.circle(world, [(237,207,126),(223,148,163),(240,231,194)][j%3], (fx,fy), 3)
        # compact benches
        pygame.draw.rect(world, (126, 82, 52), (x*t+w*t//2-14, y*t+h*t//2-10, 28, 5))
        pygame.draw.line(world, (81, 78, 62), (x*t+w*t//2-10, y*t+h*t//2-5), (x*t+w*t//2-10, y*t+h*t//2), 2)
        if index == 2:
            # A tiny reflective pond and stepping stones make the new south-west
            # commons feel like a designed destination rather than spare lawn.
            pygame.draw.ellipse(world, (96, 117, 77), (x*t+22, y*t+28, 54, 31))
            pygame.draw.ellipse(world, (105, 174, 183), (x*t+25, y*t+30, 48, 26))
            pygame.draw.arc(world, (195, 221, 195), (x*t+31, y*t+34, 24, 9), 0.2, 2.7, 2)
            for stone_x in range(x*t+77, x*t+w*t-9, 15):
                pygame.draw.ellipse(world, (204, 189, 156), (stone_x, y*t+h*t//2-3, 9, 6))
    # Presentation lots move road-anchored services onto nearby clear parcels.
    # Agent destinations and navigation remain the original simulation coordinates.
    # Lamps with little flower-ring bases.
    for x,y in [(5,9),(13,9),(22,9),(30,9),(17,5),(17,18)]:
        px,py=(x+.5)*t,(y+.5)*t
        pygame.draw.circle(world,(100,130,79),(int(px),int(py+6)),8)
        pygame.draw.line(world,(68,76,66),(px,py+7),(px,py-13),3)
        pygame.draw.line(world,(68,76,66),(px,py-13),(px+5,py-15),2)
        pygame.draw.circle(world,(248,222,150),(int(px+5),int(py-15)),4)
    scaled = pygame.transform.scale(world, (TOWN_WIDTH*tile_px, TOWN_HEIGHT*tile_px))
    _MAP_CACHE[tile_px] = scaled
    return scaled


def town_depth_layers(tile_px: int, names: dict[str, str] | None = None) -> list[tuple[pygame.Surface, int, int, int]]:
    """Return scaled foreground props as (image, x, y, depth_y) world pixels.

    Keeping props separate lets citizens pass behind trees and building fronts
    according to their foot position, while the base terrain remains cached.
    """
    tile_px = max(12, int(tile_px))
    names = names or {}
    name_key = tuple(sorted(names.items()))
    cache_key = (tile_px, name_key)
    if cache_key in _DEPTH_CACHE:
        return _DEPTH_CACHE[cache_key]
    t = BASE_TILE
    lots = presentation_buildings()
    layers: list[tuple[pygame.Surface, int, int, int]] = []
    for kind, positions in lots.items():
        if kind == "parks":
            continue  # parks are terrain features drawn into the cached ground
        for index, (x, y) in enumerate(positions):
            canvas = pygame.Surface((TOWN_WIDTH * t, TOWN_HEIGHT * t), pygame.SRCALPHA)
            prefix = {"homes": "home", "offices": "office", "groceries": "grocery", "restaurant": "restaurant", "clinic": "clinic", "town": "town_centre", "police": "police_station"}[kind]
            _building(canvas, x, y, kind, index, names.get(f"{prefix}_{index + 1:02d}"))
            left, top = max(0, x*t-8), max(0, y*t-64)
            bounds = pygame.Rect(left, top, min(100, canvas.get_width()-left), min(140, canvas.get_height()-top))
            image = canvas.subsurface(bounds).copy()
            scaled = pygame.transform.scale(image, (max(1, round(image.get_width()*tile_px/t)), max(1, round(image.get_height()*tile_px/t))))
            # Sort at the front wall: porch residents use a renderer-only
            # one-cell offset and should remain in front of their own house.
            depth_offset = 1.35 if kind != "police" else 1.42
            layers.append((scaled, round(left*tile_px/t), round(top*tile_px/t), round((y+depth_offset)*tile_px)))
    tree_coords = [(1,2),(11,7),(37,5),(1,15),(12,22),(40,30),(6,16),(48,6),(11,17),(2,6),(25,12),(7,6),(5,17),(12,30),(1,20),(30,5),(48,22),(48,30),(32,29),(22,30)]
    for index, (x, y) in enumerate(tree_coords):
        canvas = pygame.Surface((TOWN_WIDTH*t, TOWN_HEIGHT*t), pygame.SRCALPHA)
        _tree(canvas, x, y, index)
        left, top = max(0, x*t-20), max(0, y*t-32)
        bounds = pygame.Rect(left, top, min(48, canvas.get_width()-left), min(72, canvas.get_height()-top))
        image = canvas.subsurface(bounds).copy()
        scaled = pygame.transform.scale(image, (max(1, round(image.get_width()*tile_px/t)), max(1, round(image.get_height()*tile_px/t))))
        layers.append((scaled, round(left*tile_px/t), round(top*tile_px/t), round((y+1.4)*tile_px)))
    _DEPTH_CACHE[cache_key] = layers
    return layers


def citizen_sprite(
    citizen_id: int,
    direction: int,
    frame: int,
    color: tuple[int, int, int],
    tile_px: int,
    personality: tuple[float, ...] | None = None,
    occupation: str | None = None,
) -> pygame.Surface:
    """Get cached four-way pixel person sprite; direction order: down/up/left/right."""
    tile_px = max(12, int(tile_px))
    traits = personality or (0.0, 0.0, 0.0, 0.0)
    style_signal = sum((index + 1) * min(1.0, max(0.0, float(value))) for index, value in enumerate(traits[:4])) / 10.0
    hair_style = min(3, max(0, int(style_signal * 4)))
    is_officer = occupation == "Police Officer"
    key = (citizen_id, direction, frame % 4, tile_px, hair_style, int(is_officer))
    if key in _SPRITE_CACHE:
        return _SPRITE_CACHE[key]
    t = BASE_TILE
    image = pygame.Surface((t, t*2), pygame.SRCALPHA)
    skin_palette = ((222, 176, 133), (196, 139, 102), (239, 199, 158), (169, 112, 84), (211, 158, 128))
    skin = skin_palette[citizen_id % len(skin_palette)]
    # bob and feet alternate only while walking; frame 0 is the idle pose.
    bob = (1 if frame % 2 else 0) if frame else 0
    stride = (-2, 2, 1, -1)[frame % 4] if frame else 0
    pygame.draw.ellipse(image, (54,75,55,110), (6, 49, 21, 7))
    # Legs, shoes, coat silhouette.
    if direction in (2,3):
        arm_stride = -stride // 2
        pygame.draw.rect(image, skin, (5+arm_stride, 27+bob, 5, 12), border_radius=2)
        pygame.draw.rect(image, skin, (24-arm_stride, 27+bob, 4, 12), border_radius=2)
        pygame.draw.rect(image, (69,67,58), (12+stride, 38+bob, 5, 11))
        pygame.draw.rect(image, (69,67,58), (18-stride, 38+bob, 5, 11))
        pygame.draw.rect(image, (56,53,48), (10+stride, 47+bob, 7, 3))
        pygame.draw.rect(image, (56,53,48), (18-stride, 47+bob, 7, 3))
        pygame.draw.rect(image, color, (8, 23+bob, 18, 19), border_radius=4)
        pygame.draw.rect(image, (242,218,173), (11, 27+bob, 4, 10))
        pygame.draw.circle(image, skin, (17, 16+bob), 9)
        hair = [(81,56,48),(61,73,75),(111,74,49),(80,64,83),(52,69,58)][(citizen_id+hair_style)%5]
        pygame.draw.arc(image, hair, (8,7+bob,19,18), math.pi, math.tau, 5)
        pygame.draw.rect(image, hair, (8, 15+bob, 4, 9))
        if hair_style == 2:
            pygame.draw.circle(image, hair, (25, 13+bob), 4)
        elif hair_style == 3:
            pygame.draw.line(image, (229,190,109), (9,12+bob), (25,12+bob), 2)
        if direction == 2:
            pygame.draw.circle(image,(56,48,41),(13,17+bob),1)
        else:
            pygame.draw.circle(image,(56,48,41),(22,17+bob),1)
    else:
        arm_stride = -stride // 2
        pygame.draw.rect(image, skin, (5+arm_stride, 27+bob, 5, 12), border_radius=2)
        pygame.draw.rect(image, skin, (23-arm_stride, 27+bob, 4, 12), border_radius=2)
        pygame.draw.rect(image,(67,65,57),(12+stride,38+bob,5,11))
        pygame.draw.rect(image,(67,65,57),(18-stride,38+bob,5,11))
        pygame.draw.rect(image,(57,53,48),(10+stride,47+bob,7,3))
        pygame.draw.rect(image,(57,53,48),(18-stride,47+bob,7,3))
        pygame.draw.rect(image,color,(8,23+bob,18,19),border_radius=4)
        pygame.draw.rect(image,tuple(min(255,c+38) for c in color),(8,31+bob,18,3))
        pygame.draw.circle(image,skin,(17,16+bob),9)
        hair = [(81,56,48),(61,73,75),(111,74,49),(80,64,83),(52,69,58)][(citizen_id+hair_style)%5]
        pygame.draw.ellipse(image,hair,(8,7+bob,19,12))
        if hair_style == 1:
            pygame.draw.rect(image, tuple(min(255,c+34) for c in color), (9, 21+bob, 17, 3))
        elif hair_style == 2:
            pygame.draw.circle(image, hair, (25, 14+bob), 4)
        elif hair_style == 3:
            pygame.draw.rect(image, (229,190,109), (9,12+bob,17,2))
        if direction == 0:
            pygame.draw.circle(image,(66,52,44),(12,16+bob),2)
            pygame.draw.circle(image,(66,52,44),(22,16+bob),2)
    if is_officer:
        # A restrained navy uniform, brass badge, shoulder marks and cap make
        # the officer legible at game scale without changing policy movement.
        pygame.draw.rect(image, (40, 66, 82), (8, 23+bob, 18, 20), border_radius=4)
        pygame.draw.line(image, (100, 133, 142), (10, 25+bob), (24, 25+bob), 2)
        pygame.draw.rect(image, (202, 190, 139), (10, 25+bob, 4, 2))
        pygame.draw.rect(image, (202, 190, 139), (21, 25+bob, 4, 2))
        pygame.draw.circle(image, (227, 191, 99), (13, 32+bob), 3)
        pygame.draw.circle(image, (247, 221, 146), (13, 32+bob), 1)
        pygame.draw.line(image, (29, 48, 60), (9, 41+bob), (25, 41+bob), 2)
        pygame.draw.rect(image, (38, 58, 72), (8, 9+bob, 18, 5), border_radius=2)
        pygame.draw.rect(image, (54, 78, 90), (11, 6+bob, 12, 5), border_radius=2)
        pygame.draw.circle(image, (227, 191, 99), (17, 9+bob), 2)
    # A little wider than one tile keeps faces and clothing legible at town scale.
    result = pygame.transform.scale(image,(max(12, int(tile_px*1.16)),2*tile_px))
    _SPRITE_CACHE[key] = result
    return result


def citizen_wellbeing(citizen, weights: dict[str, float] | None = None) -> float:
    """Return a normalized 0–100 wellbeing score from current citizen state.

    Hunger and stress are inverted; all six inputs are clamped to [0, 100].
    Optional nonnegative weights are normalized by their sum.
    """
    values = {
        "health": citizen.health,
        "energy": citizen.energy,
        "hunger": 100.0 - citizen.hunger,
        "happiness": citizen.happiness,
        "stress": 100.0 - citizen.stress,
        "social": citizen.social,
    }
    normalized = {name: min(100.0, max(0.0, float(value))) for name, value in values.items()}
    selected = {name: 1.0 for name in normalized}
    if weights is not None:
        unknown = set(weights) - set(normalized)
        if unknown:
            raise ValueError(f"unknown wellbeing weight(s): {', '.join(sorted(unknown))}")
        for name, weight in weights.items():
            numeric = float(weight)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("wellbeing weights must be finite and nonnegative")
            selected[name] = numeric
    total = sum(selected.values())
    if total <= 0:
        raise ValueError("at least one wellbeing weight must be positive")
    return sum(normalized[name] * selected[name] for name in normalized) / total


def wellbeing_tier(score: float) -> str:
    """Map a score to its accessible low-to-high colour band."""
    bounded = min(100.0, max(0.0, float(score)))
    if bounded < WELLBEING_THRESHOLDS[0]:
        return "critical"
    if bounded < WELLBEING_THRESHOLDS[1]:
        return "low"
    if bounded < WELLBEING_THRESHOLDS[2]:
        return "moderate"
    return "good"


def wellbeing_color(score: float) -> tuple[int, int, int]:
    """Interpolate the red→orange→yellow→green wellbeing ramp."""
    bounded = min(100.0, max(0.0, float(score)))
    stops = (0.0, *WELLBEING_THRESHOLDS, 100.0)
    for index in range(len(stops) - 1):
        if bounded <= stops[index + 1]:
            fraction = (bounded - stops[index]) / (stops[index + 1] - stops[index])
            a, b = _WELLBEING_COLORS[index], _WELLBEING_COLORS[index + 1]
            return tuple(round(x + (y - x) * fraction) for x, y in zip(a, b))
    return _WELLBEING_COLORS[-1]


def draw_wellbeing_bar(
    target: pygame.Surface,
    center: tuple[int, int],
    score: float,
    width: int,
    height: int = 5,
    alpha: int = 255,
) -> None:
    """Draw one compact combined-health bar at its center point."""
    width, height = max(12, int(width)), max(3, int(height))
    bar = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(bar, (42, 52, 44, 238), (0, 0, width, height), border_radius=height // 2)
    inner = pygame.Rect(1, 1, width - 2, height - 2)
    pygame.draw.rect(bar, (93, 94, 75, 235), inner, border_radius=max(1, (height - 2) // 2))
    fill_width = round((width - 4) * min(100.0, max(0.0, float(score))) / 100.0)
    if fill_width > 0:
        pygame.draw.rect(bar, wellbeing_color(score), (2, 2, fill_width, height - 4), border_radius=max(1, (height - 4) // 2))
    pygame.draw.rect(bar, (255, 245, 213, 245), (0, 0, width, height), 1, border_radius=height // 2)
    bar.set_alpha(max(0, min(255, int(alpha))))
    target.blit(bar, (int(center[0] - width / 2), int(center[1])))
