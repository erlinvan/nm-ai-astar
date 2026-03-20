import numpy as np
from collections import deque
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_EMPTY, TERRAIN_SETTLEMENT,
                    TERRAIN_PORT, TERRAIN_RUIN, TERRAIN_FOREST, TERRAIN_MOUNTAIN)
from simulator.world import World
from simulator.settlement import Settlement


def _is_land(terrain: int) -> bool:
    return terrain not in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN)


def _is_ocean_adjacent(world: World, x: int, y: int) -> bool:
    for nx, ny in world.get_neighbors(x, y, radius=1):
        if world.get_terrain(nx, ny) == TERRAIN_OCEAN:
            return True
    return False


def _place_ocean_border(world: World, rng: np.random.Generator) -> None:
    thickness = int(rng.integers(1, 3))
    for y in range(world.height):
        for x in range(world.width):
            if x < thickness or x >= world.width - thickness:
                world.set_terrain(x, y, TERRAIN_OCEAN)
            elif y < thickness or y >= world.height - thickness:
                world.set_terrain(x, y, TERRAIN_OCEAN)


def _place_fjords(world: World, rng: np.random.Generator) -> None:
    num_fjords = int(rng.integers(3, 7))
    edges = ['N', 'S', 'E', 'W']
    for _ in range(num_fjords):
        edge = edges[rng.integers(0, 4)]
        length = int(rng.integers(5, 16))
        if edge == 'N':
            x = int(rng.integers(2, world.width - 2))
            y = 0
            dx_choices, dy_choices = [0, -1, 1], [1]
        elif edge == 'S':
            x = int(rng.integers(2, world.width - 2))
            y = world.height - 1
            dx_choices, dy_choices = [0, -1, 1], [-1]
        elif edge == 'W':
            x = 0
            y = int(rng.integers(2, world.height - 2))
            dx_choices, dy_choices = [1], [0, -1, 1]
        else:
            x = world.width - 1
            y = int(rng.integers(2, world.height - 2))
            dx_choices, dy_choices = [-1], [0, -1, 1]

        for _ in range(length):
            if 0 <= x < world.width and 0 <= y < world.height:
                world.set_terrain(x, y, TERRAIN_OCEAN)
            if edge in ('N', 'S'):
                dx = int(rng.choice(dx_choices))
                dy = dy_choices[0]
            else:
                dx = dx_choices[0]
                dy = int(rng.choice(dy_choices))
            x += dx
            y += dy
            if not (0 <= x < world.width and 0 <= y < world.height):
                break


def _place_mountains(world: World, rng: np.random.Generator) -> None:
    num_chains = int(rng.integers(2, 5))
    interior_margin = 3
    for _ in range(num_chains):
        length = int(rng.integers(8, 21))
        x = int(rng.integers(interior_margin, world.width - interior_margin))
        y = int(rng.integers(interior_margin, world.height - interior_margin))
        dx = int(rng.integers(-1, 2))
        dy = int(rng.integers(-1, 2))
        if dx == 0 and dy == 0:
            dx = 1
        for _ in range(length):
            if 0 <= x < world.width and 0 <= y < world.height:
                if world.get_terrain(x, y) not in (TERRAIN_OCEAN,):
                    world.set_terrain(x, y, TERRAIN_MOUNTAIN)
                    for nx, ny in world.get_neighbors(x, y, radius=1):
                        if world.get_terrain(nx, ny) not in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                            if rng.random() < 0.4:
                                world.set_terrain(nx, ny, TERRAIN_MOUNTAIN)
            if rng.random() < 0.3:
                dx = int(rng.integers(-1, 2))
                dy = int(rng.integers(-1, 2))
                if dx == 0 and dy == 0:
                    dx = 1
            x += dx
            y += dy
            if not (0 <= x < world.width and 0 <= y < world.height):
                break


def _place_forests(world: World, rng: np.random.Generator) -> None:
    num_clusters = int(rng.integers(10, 21))
    empty_cells = [
        (x, y)
        for y in range(world.height)
        for x in range(world.width)
        if world.get_terrain(x, y) in (TERRAIN_PLAINS, TERRAIN_EMPTY)
    ]
    if not empty_cells:
        return

    for _ in range(num_clusters):
        if not empty_cells:
            break
        idx = int(rng.integers(0, len(empty_cells)))
        cx, cy = empty_cells[idx]
        cluster_size = int(rng.integers(3, 9))

        queue = deque([(cx, cy)])
        placed = 0
        visited = {(cx, cy)}
        while queue and placed < cluster_size:
            fx, fy = queue.popleft()
            if world.get_terrain(fx, fy) in (TERRAIN_PLAINS, TERRAIN_EMPTY):
                world.set_terrain(fx, fy, TERRAIN_FOREST)
                placed += 1
                for nx, ny in world.get_neighbors(fx, fy, radius=1):
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        if world.get_terrain(nx, ny) in (TERRAIN_PLAINS, TERRAIN_EMPTY):
                            queue.append((nx, ny))

        empty_cells = [c for c in empty_cells if c not in visited]


def _place_settlements(world: World, rng: np.random.Generator) -> list:
    num_settlements = int(rng.integers(5, 13))
    min_dist = 4
    candidates = [
        (x, y)
        for y in range(1, world.height - 1)
        for x in range(1, world.width - 1)
        if world.get_terrain(x, y) in (TERRAIN_PLAINS, TERRAIN_EMPTY, TERRAIN_FOREST)
    ]

    placed_positions: list[tuple[int, int]] = []
    rng.shuffle(candidates)  # type: ignore[arg-type]

    for cx, cy in candidates:
        if len(placed_positions) >= num_settlements:
            break
        too_close = False
        for px, py in placed_positions:
            if abs(cx - px) + abs(cy - py) < min_dist:
                too_close = True
                break
        if not too_close:
            placed_positions.append((cx, cy))

    settlements = []
    for i, (sx, sy) in enumerate(placed_positions):
        is_coastal = _is_ocean_adjacent(world, sx, sy)
        s = Settlement.create(sx, sy, owner_id=i, rng=rng, has_port=is_coastal)
        terrain = TERRAIN_PORT if is_coastal else TERRAIN_SETTLEMENT
        world.set_terrain(sx, sy, terrain)
        world.add_settlement(s)
        settlements.append(s)

    return settlements


def generate_map(seed: int, width: int = 40, height: int = 40) -> World:
    rng = np.random.default_rng(seed)
    world = World(width=width, height=height)

    _place_ocean_border(world, rng)
    _place_fjords(world, rng)
    _place_mountains(world, rng)
    _place_forests(world, rng)
    _place_settlements(world, rng)

    return world
