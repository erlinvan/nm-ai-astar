import numpy as np
from config import (
    TERRAIN_TO_CLASS,
    NUM_CLASSES,
    PROBABILITY_FLOOR,
    HIGH_CONFIDENCE,
    STATIC_CONFIDENCE,
    TERRAIN_OCEAN,
    TERRAIN_MOUNTAIN,
    TERRAIN_FOREST,
    TERRAIN_SETTLEMENT,
    TERRAIN_PORT,
    TERRAIN_RUIN,
    CLASS_EMPTY,
    CLASS_SETTLEMENT,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
)
from typing import Optional
from observation_store import ObservationStore


def _compute_settlement_distance_map(initial_grid: np.ndarray) -> np.ndarray:
    h, w = initial_grid.shape
    settlement_positions = []
    for y in range(h):
        for x in range(w):
            if initial_grid[y, x] in (TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN):
                settlement_positions.append((x, y))

    dist_map = np.full((h, w), 999.0)
    if not settlement_positions:
        return dist_map

    for y in range(h):
        for x in range(w):
            min_d = min(
                max(abs(x - sx), abs(y - sy))
                for sx, sy in settlement_positions
            )
            dist_map[y, x] = float(min_d)

    return dist_map


def build_prediction(
    seed_index: int,
    store: ObservationStore,
    initial_grid: np.ndarray,
    floor: float = PROBABILITY_FLOOR,
) -> np.ndarray:
    h, w = store.height, store.width
    prediction = np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES)

    _fill_static_cells(prediction, initial_grid, floor)
    _fill_observed_cells(prediction, seed_index, store, initial_grid, floor)
    _fill_unobserved_dynamic_cells(prediction, seed_index, store, initial_grid, floor)
    _apply_floor_and_normalize(prediction, floor)

    return prediction


def _fill_static_cells(prediction: np.ndarray, initial_grid: np.ndarray, floor: float):
    h, w, _ = prediction.shape
    ocean_mask = initial_grid == TERRAIN_OCEAN
    mountain_mask = initial_grid == TERRAIN_MOUNTAIN

    ocean_dist = _confident_distribution(CLASS_EMPTY, STATIC_CONFIDENCE, floor)
    mountain_dist = _confident_distribution(CLASS_MOUNTAIN, STATIC_CONFIDENCE, floor)

    prediction[ocean_mask] = ocean_dist
    prediction[mountain_mask] = mountain_dist


def _fill_observed_cells(
    prediction: np.ndarray,
    seed_index: int,
    store: ObservationStore,
    initial_grid: np.ndarray,
    floor: float,
):
    h, w, _ = prediction.shape
    dist_map = _compute_settlement_distance_map(initial_grid)

    for y in range(h):
        for x in range(w):
            obs_count = store.get_observation_count(seed_index, x, y)
            if obs_count == 0:
                continue

            terrain = initial_grid[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            counts = store.class_counts[seed_index, y, x]

            settlement_dist = dist_map[y, x]
            prior_strength = _adaptive_prior_strength(settlement_dist, obs_count)
            alpha_prior = _terrain_aware_prior(terrain, settlement_dist) * prior_strength

            alpha_posterior = alpha_prior + counts
            prediction[y, x] = alpha_posterior / alpha_posterior.sum()


def _adaptive_prior_strength(settlement_dist: float, obs_count: int) -> float:
    if settlement_dist <= 2:
        return 0.3
    elif settlement_dist <= 5:
        return 0.5
    elif settlement_dist <= 8:
        return 1.0
    else:
        return 2.0


def _terrain_aware_prior(terrain: int, settlement_dist: float) -> np.ndarray:
    prior = np.full(NUM_CLASSES, 1.0)

    if terrain == TERRAIN_FOREST:
        if settlement_dist <= 3:
            prior[CLASS_FOREST] = 5.0
            prior[CLASS_SETTLEMENT] = 2.0
            prior[CLASS_RUIN] = 1.5
            prior[CLASS_EMPTY] = 2.0
        else:
            prior[CLASS_FOREST] = 10.0
            prior[CLASS_EMPTY] = 1.5
    elif terrain in (TERRAIN_SETTLEMENT, TERRAIN_PORT):
        prior[CLASS_SETTLEMENT] = 3.0
        prior[CLASS_PORT] = 2.0
        prior[CLASS_RUIN] = 2.0
        prior[CLASS_EMPTY] = 1.0
    elif terrain == TERRAIN_RUIN:
        prior[CLASS_RUIN] = 2.0
        prior[CLASS_FOREST] = 2.0
        prior[CLASS_SETTLEMENT] = 1.5
        prior[CLASS_EMPTY] = 2.0
    else:
        if settlement_dist <= 3:
            prior[CLASS_EMPTY] = 3.0
            prior[CLASS_SETTLEMENT] = 2.0
            prior[CLASS_RUIN] = 1.0
        else:
            prior[CLASS_EMPTY] = 8.0

    prior /= prior.sum()
    return prior


def _fill_unobserved_dynamic_cells(
    prediction: np.ndarray,
    seed_index: int,
    store: ObservationStore,
    initial_grid: np.ndarray,
    floor: float,
):
    h, w, _ = prediction.shape
    coverage = store.get_coverage_mask(seed_index)
    dist_map = _compute_settlement_distance_map(initial_grid)

    for y in range(h):
        for x in range(w):
            if coverage[y, x]:
                continue
            terrain = initial_grid[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            neighbor_dist = _gather_neighbor_distributions(
                prediction, coverage, x, y, h, w, radius=5,
            )
            if neighbor_dist is not None:
                settlement_d = dist_map[y, x]
                terrain_prior = _prior_from_initial_terrain(terrain, settlement_d, floor)
                blend_weight = min(1.0, settlement_d / 10.0)
                prediction[y, x] = (1.0 - blend_weight) * neighbor_dist + blend_weight * terrain_prior
            else:
                prediction[y, x] = _prior_from_initial_terrain(
                    terrain, dist_map[y, x], floor,
                )


def _gather_neighbor_distributions(
    prediction: np.ndarray,
    coverage: np.ndarray,
    cx: int,
    cy: int,
    h: int,
    w: int,
    radius: int = 5,
) -> Optional[np.ndarray]:
    weighted_sum = np.zeros(NUM_CLASSES)
    total_weight = 0.0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = cy + dy, cx + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not coverage[ny, nx]:
                continue
            dist = max(abs(dx), abs(dy))
            if dist == 0:
                continue
            weight = 1.0 / (dist * dist)
            weighted_sum += prediction[ny, nx] * weight
            total_weight += weight

    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def _prior_from_initial_terrain(
    terrain: int, settlement_dist: float, floor: float,
) -> np.ndarray:
    cls = TERRAIN_TO_CLASS.get(terrain, 0)

    if terrain == TERRAIN_FOREST:
        dist = np.full(NUM_CLASSES, floor)
        if settlement_dist <= 3:
            dist[CLASS_FOREST] = 0.60
            dist[CLASS_EMPTY] = 0.12
            dist[CLASS_SETTLEMENT] = 0.10
            dist[CLASS_RUIN] = 0.08
            dist[CLASS_PORT] = 0.05
        elif settlement_dist <= 6:
            dist[CLASS_FOREST] = 0.75
            dist[CLASS_EMPTY] = 0.10
            dist[CLASS_SETTLEMENT] = 0.05
            dist[CLASS_RUIN] = 0.04
        else:
            dist[CLASS_FOREST] = 0.88
            dist[CLASS_EMPTY] = 0.05
        return dist / dist.sum()

    dist = np.full(NUM_CLASSES, floor)
    if settlement_dist <= 3:
        dist[cls] = 0.30
        dist[CLASS_EMPTY] = 0.20
        dist[CLASS_SETTLEMENT] = 0.15
        dist[CLASS_RUIN] = 0.12
        dist[CLASS_PORT] = 0.08
        dist[CLASS_FOREST] = 0.08
    elif settlement_dist <= 6:
        dist[cls] = 0.45
        dist[CLASS_EMPTY] = 0.20
        dist[CLASS_SETTLEMENT] = 0.08
        dist[CLASS_RUIN] = 0.06
    else:
        dist[cls] = 0.60
        dist[CLASS_EMPTY] = 0.20
    return dist / dist.sum()


def _confident_distribution(target_class: int, confidence: float, floor: float) -> np.ndarray:
    dist = np.full(NUM_CLASSES, floor)
    dist[target_class] = confidence
    return dist / dist.sum()


def build_prediction_with_mc(
    seed_index: int,
    store: Optional[ObservationStore],
    initial_grid: np.ndarray,
    mc_prediction: np.ndarray,
    mc_pseudo_count: float = 3.0,
    floor: float = PROBABILITY_FLOOR,
) -> np.ndarray:
    h, w = initial_grid.shape
    prediction = mc_prediction.copy()

    _fill_static_cells(prediction, initial_grid, floor)

    if store is not None:
        dist_map = _compute_settlement_distance_map(initial_grid)

        for y in range(h):
            for x in range(w):
                terrain = initial_grid[y, x]
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue

                obs_count = store.get_observation_count(seed_index, x, y)
                if obs_count == 0:
                    continue

                settlement_d = dist_map[y, x]
                adaptive_pseudo = mc_pseudo_count * min(2.0, max(0.3, settlement_d / 5.0))

                mc_alpha = prediction[y, x] * adaptive_pseudo
                obs_counts = store.class_counts[seed_index, y, x]
                posterior = mc_alpha + obs_counts
                prediction[y, x] = posterior / posterior.sum()

    _apply_floor_and_normalize(prediction, floor)
    return prediction


def _apply_floor_and_normalize(prediction: np.ndarray, floor: float):
    max_entropy = np.log(NUM_CLASSES)
    eps = 1e-10
    clipped = np.clip(prediction, eps, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    entropy_ratio = entropy / max_entropy

    adaptive_floor = floor * (1.0 + 0.5 * entropy_ratio)
    floor_expanded = np.expand_dims(adaptive_floor, axis=-1)

    prediction[:] = np.maximum(prediction, floor_expanded)
    sums = prediction.sum(axis=-1, keepdims=True)
    prediction[:] = prediction / sums
