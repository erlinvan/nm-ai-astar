from __future__ import annotations
from typing import Optional

import numpy as np
from config import (
    MAX_VIEWPORT_SIZE, DEFAULT_MAP_WIDTH, DEFAULT_MAP_HEIGHT,
    TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_SETTLEMENT, TERRAIN_PORT,
    TERRAIN_FOREST, TERRAIN_RUIN,
)


def compute_tile_starts(map_size: int, viewport_size: int = MAX_VIEWPORT_SIZE) -> list[int]:
    starts = []
    pos = 0
    while pos < map_size:
        starts.append(pos)
        if pos + viewport_size >= map_size:
            break
        pos = min(pos + viewport_size, map_size - viewport_size)
    return starts


def generate_tiling(
    width: int = DEFAULT_MAP_WIDTH,
    height: int = DEFAULT_MAP_HEIGHT,
    viewport_w: int = MAX_VIEWPORT_SIZE,
    viewport_h: int = MAX_VIEWPORT_SIZE,
) -> list[dict]:
    x_starts = compute_tile_starts(width, viewport_w)
    y_starts = compute_tile_starts(height, viewport_h)

    tiles = []
    for y in y_starts:
        for x in x_starts:
            vw = min(viewport_w, width - x)
            vh = min(viewport_h, height - y)
            tiles.append({"viewport_x": x, "viewport_y": y, "viewport_w": vw, "viewport_h": vh})
    return tiles


def _compute_interest_map(initial_grid: np.ndarray) -> np.ndarray:
    h, w = initial_grid.shape
    interest = np.zeros((h, w), dtype=np.float64)

    settlement_positions = []
    for y in range(h):
        for x in range(w):
            t = initial_grid[y, x]
            if t in (TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN):
                settlement_positions.append((x, y))

    for y in range(h):
        for x in range(w):
            t = initial_grid[y, x]
            if t in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            if not settlement_positions:
                interest[y, x] = 0.3
                continue

            min_dist = min(
                max(abs(x - sx), abs(y - sy))
                for sx, sy in settlement_positions
            )

            if min_dist == 0:
                interest[y, x] = 10.0
            elif min_dist <= 2:
                interest[y, x] = 5.0
            elif min_dist <= 5:
                interest[y, x] = 2.0
            elif min_dist <= 8:
                interest[y, x] = 1.0
            else:
                interest[y, x] = 0.3

            if t == TERRAIN_FOREST and min_dist <= 3:
                interest[y, x] *= 1.5

    return interest


def generate_overlapping_viewports(
    width: int,
    height: int,
    initial_grid: np.ndarray,
    num_queries: int,
    viewport_w: int = MAX_VIEWPORT_SIZE,
    viewport_h: int = MAX_VIEWPORT_SIZE,
) -> list[dict]:
    interest = _compute_interest_map(initial_grid)

    base_tiles = generate_tiling(width, height, viewport_w, viewport_h)

    if num_queries <= len(base_tiles):
        return prioritize_tiles(base_tiles, initial_grid, num_queries)

    remaining = num_queries - len(base_tiles)
    overlap_tiles = _generate_interest_viewports(
        interest, width, height, remaining, viewport_w, viewport_h,
    )

    return base_tiles + overlap_tiles


def _generate_interest_viewports(
    interest: np.ndarray,
    width: int,
    height: int,
    num_viewports: int,
    viewport_w: int = MAX_VIEWPORT_SIZE,
    viewport_h: int = MAX_VIEWPORT_SIZE,
) -> list[dict]:
    h, w = interest.shape
    working_interest = interest.copy()
    tiles = []

    for _ in range(num_viewports):
        best_score = -1.0
        best_x, best_y = 0, 0

        step_x = max(1, viewport_w // 3)
        step_y = max(1, viewport_h // 3)

        for vy in range(0, h - viewport_h + 1, step_y):
            for vx in range(0, w - viewport_w + 1, step_x):
                region = working_interest[vy:vy + viewport_h, vx:vx + viewport_w]
                score = float(region.sum())
                if score > best_score:
                    best_score = score
                    best_x, best_y = vx, vy

        for vy in [0, max(0, h - viewport_h)]:
            for vx in [0, max(0, w - viewport_w)]:
                region = working_interest[vy:vy + viewport_h, vx:vx + viewport_w]
                score = float(region.sum())
                if score > best_score:
                    best_score = score
                    best_x, best_y = vx, vy

        vw = min(viewport_w, width - best_x)
        vh = min(viewport_h, height - best_y)
        tiles.append({
            "viewport_x": best_x,
            "viewport_y": best_y,
            "viewport_w": vw,
            "viewport_h": vh,
        })

        working_interest[best_y:best_y + vh, best_x:best_x + vw] *= 0.5

    return tiles


def allocate_queries(
    num_seeds: int,
    total_budget: int,
    width: int = DEFAULT_MAP_WIDTH,
    height: int = DEFAULT_MAP_HEIGHT,
    initial_grids: Optional[list[np.ndarray]] = None,
) -> dict:
    base_tiles = generate_tiling(width, height)
    queries_per_seed_base = len(base_tiles)
    total_base = queries_per_seed_base * num_seeds

    if initial_grids is not None and len(initial_grids) == num_seeds:
        dynamism = []
        for grid in initial_grids:
            static = np.sum((grid == TERRAIN_OCEAN) | (grid == TERRAIN_MOUNTAIN))
            total_cells = grid.size
            dyn_ratio = 1.0 - (static / max(total_cells, 1))

            n_settlements = np.sum(
                (grid == TERRAIN_SETTLEMENT) | (grid == TERRAIN_PORT)
            )
            score = dyn_ratio * (1.0 + 0.1 * n_settlements)
            dynamism.append(score)

        total_dynamism = sum(dynamism) or 1.0
        per_seed_queries = [queries_per_seed_base] * num_seeds
        extras = total_budget - queries_per_seed_base * num_seeds

        if extras > 0:
            weights = [d / total_dynamism for d in dynamism]
            for _ in range(extras):
                best_idx = max(range(num_seeds), key=lambda i: weights[i])
                per_seed_queries[best_idx] += 1
                weights[best_idx] *= 0.5  # decay to spread extras

        allocation = {
            "strategy": "adaptive_overlap",
            "per_seed_queries": per_seed_queries,
            "tiles_per_seed": queries_per_seed_base,
            "total_queries": sum(per_seed_queries),
            "remaining": 0,
        }
    elif total_base <= total_budget:
        remaining = total_budget - total_base
        extra_per_seed = remaining // num_seeds
        per_seed = queries_per_seed_base + extra_per_seed

        allocation = {
            "strategy": "full_coverage_overlap",
            "per_seed_queries": [per_seed] * num_seeds,
            "tiles_per_seed": queries_per_seed_base,
            "total_queries": per_seed * num_seeds,
            "remaining": total_budget - per_seed * num_seeds,
        }
    else:
        per_seed = total_budget // num_seeds
        allocation = {
            "strategy": "partial_coverage",
            "per_seed_queries": [per_seed] * num_seeds,
            "tiles_per_seed": per_seed,
            "total_queries": per_seed * num_seeds,
            "remaining": total_budget - per_seed * num_seeds,
        }

    return allocation


def prioritize_tiles(
    tiles: list[dict],
    initial_grid: np.ndarray,
    max_tiles: int,
) -> list[dict]:
    scored = []
    for tile in tiles:
        x, y = tile["viewport_x"], tile["viewport_y"]
        w, h = tile["viewport_w"], tile["viewport_h"]
        region = initial_grid[y : y + h, x : x + w]
        static_count = np.sum((region == TERRAIN_OCEAN) | (region == TERRAIN_MOUNTAIN))
        dynamic_ratio = 1.0 - (static_count / max(region.size, 1))

        settlement_count = np.sum(
            (region == TERRAIN_SETTLEMENT) | (region == TERRAIN_PORT)
        )
        score = dynamic_ratio * (1.0 + 0.2 * settlement_count)
        scored.append((score, tile))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [t[1] for t in scored[:max_tiles]]
