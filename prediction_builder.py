import numpy as np
from config import (
    NUM_CLASSES,
    PROBABILITY_FLOOR,
    STATIC_CONFIDENCE,
    TERRAIN_OCEAN,
    TERRAIN_MOUNTAIN,
    TERRAIN_FOREST,
    TERRAIN_SETTLEMENT,
    TERRAIN_PORT,
    TERRAIN_RUIN,
    CLASS_EMPTY,
    CLASS_MOUNTAIN,
)
from typing import Optional
from observation_store import ObservationStore


# Maximum probability allowed for any non-static cell class.
# Even with overwhelming evidence, we cap predictions because ground truth
# distributions rarely exceed ~75% for any single class in dynamic areas.
MAX_DYNAMIC_PROB = 0.95

# Minimum probability floor for dynamic cells (higher than PROBABILITY_FLOOR
# to account for the stochastic nature of the simulation)
DYNAMIC_FLOOR = 0.02


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
    """Fill predictions for observed cells using Dirichlet posterior.

    Key insight: each observation is ONE stochastic simulation result. The ground
    truth is a probability distribution from hundreds of simulations. With few
    observations, we need strong priors to avoid overconfidence.

    Prior strength is HIGH when observations are few (spreading the prediction),
    and LOW when observations are many (trusting the empirical distribution).
    """
    h, w, _ = prediction.shape
    dist_map = _compute_settlement_distance_map(initial_grid)

    agg_counts = store.aggregated_counts
    agg_obs = store.aggregated_obs_count

    for y in range(h):
        for x in range(w):
            seed_obs_count = store.get_observation_count(seed_index, x, y)

            if agg_counts is not None and agg_obs is not None:
                total_obs = int(agg_obs[y, x])
                counts = agg_counts[y, x].copy()
            else:
                total_obs = seed_obs_count
                counts = store.class_counts[seed_index, y, x].copy()

            if total_obs == 0:
                continue

            terrain = initial_grid[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            settlement_dist = dist_map[y, x]

            prior_strength = _adaptive_prior_strength(settlement_dist, total_obs)
            alpha_prior = _terrain_aware_prior(terrain, settlement_dist) * prior_strength

            alpha_posterior = alpha_prior + counts
            prediction[y, x] = alpha_posterior / alpha_posterior.sum()


def _adaptive_prior_strength(settlement_dist: float, obs_count: int) -> float:
    return max(100.0, 1000.0 / (obs_count + 0.5))


def _terrain_aware_prior(terrain: int, settlement_dist: float) -> np.ndarray:
    prior = _gt_calibrated_prior(terrain, settlement_dist)
    prior /= prior.sum()
    return prior


def _gt_calibrated_prior(terrain: int, settlement_dist: float) -> np.ndarray:
    """GT-calibrated priors from R13+R15 empirical distributions (10 seeds).

    Class order: [Empty, Settlement, Port, Ruin, Forest, Mountain]
    """
    if terrain == TERRAIN_FOREST:
        if settlement_dist <= 2:
            return np.array([0.119, 0.166, 0.007, 0.019, 0.689, 0.001])
        elif settlement_dist <= 5:
            return np.array([0.064, 0.107, 0.009, 0.013, 0.807, 0.001])
        else:
            return np.array([0.016, 0.039, 0.008, 0.006, 0.932, 0.001])
    elif terrain in (TERRAIN_SETTLEMENT, TERRAIN_PORT):
        return np.array([0.472, 0.253, 0.020, 0.025, 0.235, 0.001])
    elif terrain == TERRAIN_RUIN:
        return np.array([0.472, 0.253, 0.020, 0.025, 0.235, 0.001])
    else:
        if settlement_dist <= 2:
            return np.array([0.754, 0.162, 0.007, 0.019, 0.058, 0.001])
        elif settlement_dist <= 5:
            return np.array([0.841, 0.106, 0.008, 0.013, 0.032, 0.001])
        else:
            return np.array([0.945, 0.037, 0.006, 0.004, 0.007, 0.001])


def _fill_unobserved_dynamic_cells(
    prediction: np.ndarray,
    seed_index: int,
    store: ObservationStore,
    initial_grid: np.ndarray,
    floor: float,
):
    h, w, _ = prediction.shape

    agg_obs_count = store.aggregated_obs_count
    if agg_obs_count is not None:
        coverage = agg_obs_count > 0
    else:
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
    dist = _gt_calibrated_prior(terrain, settlement_dist)
    dist = np.maximum(dist, floor)
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
    mc_weight: float = 0.08,
    floor: float = PROBABILITY_FLOOR,
) -> np.ndarray:
    h, w = initial_grid.shape
    prediction = build_prediction(seed_index, store, initial_grid, floor) if store is not None else np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES)

    dist_map = _compute_settlement_distance_map(initial_grid)

    for y in range(h):
        for x in range(w):
            terrain = initial_grid[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            settlement_d = dist_map[y, x]
            w_mc = mc_weight * min(1.0, settlement_d / 8.0 + 0.3)
            prediction[y, x] = (1.0 - w_mc) * prediction[y, x] + w_mc * mc_prediction[y, x]

    _apply_floor_and_normalize(prediction, floor)
    return prediction


def _apply_floor_and_normalize(prediction: np.ndarray, floor: float):
    h, w, c = prediction.shape

    prediction[:] = np.maximum(prediction, floor)

    sums = prediction.sum(axis=-1, keepdims=True)
    prediction[:] = prediction / sums

    max_probs = prediction.max(axis=-1)
    needs_cap = max_probs > MAX_DYNAMIC_PROB

    if np.any(needs_cap):
        for y in range(h):
            for x in range(w):
                if not needs_cap[y, x]:
                    continue
                _cap_cell_probability(prediction[y, x], MAX_DYNAMIC_PROB, DYNAMIC_FLOOR)

    sums = prediction.sum(axis=-1, keepdims=True)
    prediction[:] = prediction / sums


def _cap_cell_probability(cell: np.ndarray, max_prob: float, min_prob: float):
    max_idx = np.argmax(cell)
    if cell[max_idx] <= max_prob:
        return

    excess = cell[max_idx] - max_prob
    cell[max_idx] = max_prob

    other_mask = np.ones(len(cell), dtype=bool)
    other_mask[max_idx] = False
    other_sum = cell[other_mask].sum()

    if other_sum > 0:
        cell[other_mask] += excess * (cell[other_mask] / other_sum)
    else:
        cell[other_mask] = excess / (len(cell) - 1)

    cell[:] = np.maximum(cell, min_prob)
