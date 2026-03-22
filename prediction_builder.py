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
    CLASS_PORT,
    CLASS_MOUNTAIN,
)
from typing import Optional
from observation_store import ObservationStore

# Type alias for observation-derived priors: (terrain, dist_bucket) -> probability array
RoundPriors = dict[tuple[int, str, str], np.ndarray]


MAX_DYNAMIC_PROB = 0.99
DYNAMIC_FLOOR = 0.001


def _dist_bucket(settlement_dist: float) -> str:
    if settlement_dist <= 2:
        return "d0-2"
    elif settlement_dist <= 3:
        return "d3"
    elif settlement_dist <= 5:
        return "d4-5"
    return "d6+"


def _suppress_mountain(p: np.ndarray) -> np.ndarray:
    """Set Mountain class to near-zero for dynamic cells and redistribute mass."""
    p = p.copy()
    mountain_excess = p[CLASS_MOUNTAIN] - 0.001
    if mountain_excess > 0:
        p[CLASS_MOUNTAIN] = 0.001
        other_mask = np.ones(NUM_CLASSES, dtype=bool)
        other_mask[CLASS_MOUNTAIN] = False
        other_sum = p[other_mask].sum()
        if other_sum > 0:
            p[other_mask] += mountain_excess * (p[other_mask] / other_sum)
        else:
            p[other_mask] = mountain_excess / (NUM_CLASSES - 1)
    return p


def _suppress_port(p: np.ndarray) -> np.ndarray:
    p = p.copy()
    port_excess = p[CLASS_PORT] - 0.001
    if port_excess > 0:
        p[CLASS_PORT] = 0.001
        other_mask = np.ones(NUM_CLASSES, dtype=bool)
        other_mask[CLASS_PORT] = False
        other_mask[CLASS_MOUNTAIN] = False
        other_sum = p[other_mask].sum()
        if other_sum > 0:
            p[other_mask] += port_excess * (p[other_mask] / other_sum)
        else:
            p[other_mask] = port_excess / other_mask.sum()
    return p


_SETTLEMENT_AREA_TERRAINS = {TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN}
_MIN_OBS_FOR_RELIABLE_PRIOR = 30
_PRIOR_REGULARIZATION_STRENGTH = 60  # Blend round_priors toward GT-calibrated to reduce sampling noise

# Per-class floor as fraction of GT-calibrated value. Observations underestimate rare classes
# due to sparse sampling (~1.3 obs/cell). Floors only activate when obs prior falls below
# the threshold, so rounds with good observation coverage are unaffected.
_CLASS_FLOOR_FRACTIONS = {
    1: 0.5,
    2: 0.5,
    3: 0.7,
}


def _apply_class_floors(p: np.ndarray, calibrated: np.ndarray) -> np.ndarray:
    """Enforce minimum per-class probabilities as a fraction of GT-calibrated values.
    Redistributes mass from the largest class to avoid changing the overall shape."""
    p = p.copy()
    for class_idx, floor_frac in _CLASS_FLOOR_FRACTIONS.items():
        min_val = calibrated[class_idx] * floor_frac
        if p[class_idx] < min_val:
            deficit = min_val - p[class_idx]
            p[class_idx] = min_val
            largest = np.argmax(p)
            p[largest] = max(p[largest] - deficit, 0.001)
    return p


def compute_round_priors(
    store: ObservationStore,
    initial_grids: list[np.ndarray],
    num_seeds: int,
) -> RoundPriors:
    obs_by_key: dict[tuple[int, str, str], np.ndarray] = {}
    pooled_by_bucket: dict[str, np.ndarray] = {}

    for seed_idx in range(num_seeds):
        ig = initial_grids[seed_idx]
        dist_map = _compute_settlement_distance_map(ig)
        ocean_dist_map = _compute_ocean_distance_map(ig)
        for y in range(store.height):
            for x in range(store.width):
                terrain = ig[y, x]
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                if store.observation_count[seed_idx, y, x] == 0:
                    continue
                bucket = _dist_bucket(dist_map[y, x])
                coastal_tag = "coastal" if ocean_dist_map[y, x] == 1 else "inland"
                key = (terrain, bucket, coastal_tag)
                if key not in obs_by_key:
                    obs_by_key[key] = np.zeros(NUM_CLASSES)
                obs_by_key[key] += store.class_counts[seed_idx, y, x]

                if terrain in _SETTLEMENT_AREA_TERRAINS:
                    pool_key = bucket + ":" + coastal_tag
                    if pool_key not in pooled_by_bucket:
                        pooled_by_bucket[pool_key] = np.zeros(NUM_CLASSES)
                    pooled_by_bucket[pool_key] += store.class_counts[seed_idx, y, x]

    priors: RoundPriors = {}
    for key, counts in obs_by_key.items():
        terrain, bucket, coastal_tag = key
        total = counts.sum()
        if total <= 0:
            continue

        is_coastal = coastal_tag == "coastal"
        calibrated = _gt_calibrated_prior(
            terrain,
            1.5 if bucket == "d0-2" else 3.0 if bucket == "d3" else 4.5 if bucket == "d4-5" else 8.0,
            is_coastal,
        )
        calibrated = calibrated / calibrated.sum()

        if terrain in _SETTLEMENT_AREA_TERRAINS and total < _MIN_OBS_FOR_RELIABLE_PRIOR:
            pool_key = bucket + ":" + coastal_tag
            pooled = pooled_by_bucket.get(pool_key)
            if pooled is not None and pooled.sum() >= _MIN_OBS_FOR_RELIABLE_PRIOR:
                obs_weight = total / _MIN_OBS_FOR_RELIABLE_PRIOR
                pooled_p = pooled / pooled.sum()
                obs_p = counts / total
                p = obs_weight * obs_p + (1.0 - obs_weight) * pooled_p
            else:
                obs_weight = total / _MIN_OBS_FOR_RELIABLE_PRIOR
                obs_p = counts / total
                p = obs_weight * obs_p + (1.0 - obs_weight) * calibrated
        else:
            p = counts / total

        p = (total * p + _PRIOR_REGULARIZATION_STRENGTH * calibrated) / (total + _PRIOR_REGULARIZATION_STRENGTH)

        p = _apply_class_floors(p, calibrated)
        p = _suppress_mountain(p)
        if coastal_tag == "inland":
            p = _suppress_port(p)
        p = np.maximum(p, 0.001)
        priors[key] = p / p.sum()
    return priors


def _compute_chebyshev_distance_map(initial_grid: np.ndarray, targets: set) -> np.ndarray:
    h, w = initial_grid.shape
    positions = []
    for y in range(h):
        for x in range(w):
            if initial_grid[y, x] in targets:
                positions.append((x, y))

    dist_map = np.full((h, w), 999.0)
    if not positions:
        return dist_map

    for y in range(h):
        for x in range(w):
            min_d = min(
                max(abs(x - px), abs(y - py))
                for px, py in positions
            )
            dist_map[y, x] = float(min_d)

    return dist_map


def _compute_settlement_distance_map(initial_grid: np.ndarray) -> np.ndarray:
    return _compute_chebyshev_distance_map(initial_grid, {TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN})


def _compute_ocean_distance_map(initial_grid: np.ndarray) -> np.ndarray:
    return _compute_chebyshev_distance_map(initial_grid, {TERRAIN_OCEAN})


def build_prediction(
    seed_index: int,
    store: ObservationStore,
    initial_grid: np.ndarray,
    floor: float = PROBABILITY_FLOOR,
    round_priors: Optional[RoundPriors] = None,
) -> np.ndarray:
    h, w = store.height, store.width
    prediction = np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES)

    _fill_static_cells(prediction, initial_grid, floor)
    _fill_observed_cells(prediction, seed_index, store, initial_grid, floor, round_priors)
    _fill_unobserved_dynamic_cells(prediction, seed_index, store, initial_grid, floor, round_priors)
    _apply_floor_and_normalize(prediction, floor, initial_grid)

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
    round_priors: Optional[RoundPriors] = None,
):
    h, w, _ = prediction.shape
    dist_map = _compute_settlement_distance_map(initial_grid)
    ocean_dist_map = _compute_ocean_distance_map(initial_grid)

    use_per_seed = round_priors is not None

    for y in range(h):
        for x in range(w):
            if use_per_seed:
                total_obs = int(store.observation_count[seed_index, y, x])
                counts = store.class_counts[seed_index, y, x].copy()
            else:
                agg_counts = store.aggregated_counts
                agg_obs = store.aggregated_obs_count
                if agg_counts is not None and agg_obs is not None:
                    total_obs = int(agg_obs[y, x])
                    counts = agg_counts[y, x].copy()
                else:
                    total_obs = int(store.observation_count[seed_index, y, x])
                    counts = store.class_counts[seed_index, y, x].copy()

            if total_obs == 0:
                continue

            terrain = initial_grid[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            settlement_dist = dist_map[y, x]
            is_coastal = ocean_dist_map[y, x] == 1

            prior_strength = _adaptive_prior_strength(settlement_dist, total_obs)

            if round_priors is not None:
                coastal_tag = "coastal" if is_coastal else "inland"
                key = (terrain, _dist_bucket(settlement_dist), coastal_tag)
                if key in round_priors:
                    alpha_prior = round_priors[key].copy() * prior_strength
                else:
                    alpha_prior = _terrain_aware_prior(terrain, settlement_dist, is_coastal) * prior_strength
            else:
                alpha_prior = _terrain_aware_prior(terrain, settlement_dist, is_coastal) * prior_strength

            alpha_posterior = alpha_prior + counts
            p = alpha_posterior / alpha_posterior.sum()

            # Port only appears on coastal cells (ocean_dist == 1)
            if not is_coastal and p[CLASS_PORT] > 0.001:
                port_excess = p[CLASS_PORT] - 0.001
                p[CLASS_PORT] = 0.001
                other_mask = np.ones(NUM_CLASSES, dtype=bool)
                other_mask[CLASS_PORT] = False
                other_mask[CLASS_MOUNTAIN] = False
                other_sum = p[other_mask].sum()
                if other_sum > 0:
                    p[other_mask] += port_excess * (p[other_mask] / other_sum)
                else:
                    p[other_mask] = port_excess / other_mask.sum()

            prediction[y, x] = p


def _adaptive_prior_strength(settlement_dist: float, obs_count: int) -> float:
    return 100.0


def _terrain_aware_prior(terrain: int, settlement_dist: float, is_coastal: bool = False) -> np.ndarray:
    prior = _gt_calibrated_prior(terrain, settlement_dist, is_coastal)
    prior /= prior.sum()
    return prior


def _gt_calibrated_prior(terrain: int, settlement_dist: float, is_coastal: bool = False) -> np.ndarray:
    """GT-calibrated priors from R11+R13+R14+R15+R17 empirical distributions (25 seeds).

    Class order: [Empty, Settlement, Port, Ruin, Forest, Mountain]
    Port appears ONLY on coastal cells (ocean Chebyshev dist == 1). Inland Port is 0.
    """
    if terrain == TERRAIN_FOREST:
        if settlement_dist <= 1:
            if is_coastal:
                return np.array([0.125, 0.185, 0.116, 0.026, 0.547, 0.001])
            return np.array([0.131, 0.313, 0.001, 0.027, 0.529, 0.001])
        elif settlement_dist <= 2:
            if is_coastal:
                return np.array([0.105, 0.151, 0.101, 0.021, 0.623, 0.001])
            return np.array([0.113, 0.267, 0.001, 0.023, 0.597, 0.001])
        elif settlement_dist <= 3:
            if is_coastal:
                return np.array([0.071, 0.118, 0.081, 0.016, 0.715, 0.001])
            return np.array([0.077, 0.206, 0.001, 0.017, 0.700, 0.001])
        elif settlement_dist <= 5:
            if is_coastal:
                return np.array([0.035, 0.064, 0.041, 0.007, 0.852, 0.001])
            return np.array([0.039, 0.125, 0.001, 0.010, 0.826, 0.001])
        else:
            if is_coastal:
                return np.array([0.009, 0.020, 0.012, 0.003, 0.956, 0.001])
            return np.array([0.012, 0.044, 0.001, 0.004, 0.941, 0.001])
    elif terrain == TERRAIN_PORT:
        if settlement_dist <= 1:
            return np.array([0.425, 0.132, 0.205, 0.028, 0.210, 0.001])
        elif settlement_dist <= 2:
            return np.array([0.425, 0.132, 0.205, 0.028, 0.210, 0.001])
        elif settlement_dist <= 3:
            return np.array([0.450, 0.110, 0.180, 0.024, 0.236, 0.001])
        elif settlement_dist <= 5:
            return np.array([0.500, 0.075, 0.145, 0.017, 0.263, 0.001])
        else:
            return np.array([0.530, 0.060, 0.120, 0.015, 0.275, 0.001])
    elif terrain == TERRAIN_SETTLEMENT:
        if settlement_dist <= 1:
            if is_coastal:
                return np.array([0.379, 0.310, 0.086, 0.034, 0.191, 0.001])
            return np.array([0.377, 0.403, 0.001, 0.034, 0.187, 0.001])
        elif settlement_dist <= 2:
            if is_coastal:
                return np.array([0.379, 0.310, 0.086, 0.034, 0.191, 0.001])
            return np.array([0.377, 0.403, 0.001, 0.034, 0.187, 0.001])
        elif settlement_dist <= 3:
            return np.array([0.480, 0.290, 0.003, 0.025, 0.202, 0.001])
        elif settlement_dist <= 5:
            return np.array([0.500, 0.240, 0.003, 0.018, 0.239, 0.001])
        else:
            return np.array([0.550, 0.180, 0.003, 0.015, 0.252, 0.001])
    elif terrain == TERRAIN_RUIN:
        if settlement_dist <= 1:
            return np.array([0.430, 0.230, 0.015, 0.070, 0.255, 0.001])
        elif settlement_dist <= 2:
            return np.array([0.430, 0.230, 0.015, 0.070, 0.255, 0.001])
        elif settlement_dist <= 3:
            return np.array([0.460, 0.210, 0.015, 0.060, 0.255, 0.001])
        elif settlement_dist <= 5:
            return np.array([0.500, 0.170, 0.012, 0.048, 0.270, 0.001])
        else:
            return np.array([0.540, 0.120, 0.010, 0.045, 0.285, 0.001])
    else:
        if settlement_dist <= 1:
            if is_coastal:
                return np.array([0.636, 0.170, 0.111, 0.024, 0.060, 0.001])
            return np.array([0.605, 0.306, 0.001, 0.026, 0.064, 0.001])
        elif settlement_dist <= 2:
            if is_coastal:
                return np.array([0.678, 0.146, 0.106, 0.021, 0.051, 0.001])
            return np.array([0.658, 0.264, 0.001, 0.023, 0.056, 0.001])
        elif settlement_dist <= 3:
            if is_coastal:
                return np.array([0.770, 0.107, 0.075, 0.015, 0.034, 0.001])
            return np.array([0.742, 0.202, 0.001, 0.017, 0.038, 0.001])
        elif settlement_dist <= 5:
            if is_coastal:
                return np.array([0.875, 0.064, 0.039, 0.007, 0.015, 0.001])
            return np.array([0.845, 0.125, 0.001, 0.010, 0.020, 0.001])
        else:
            if is_coastal:
                return np.array([0.961, 0.020, 0.012, 0.003, 0.004, 0.001])
            return np.array([0.953, 0.038, 0.001, 0.003, 0.005, 0.001])


def _fill_unobserved_dynamic_cells(
    prediction: np.ndarray,
    seed_index: int,
    store: ObservationStore,
    initial_grid: np.ndarray,
    floor: float,
    round_priors: Optional[RoundPriors] = None,
):
    h, w, _ = prediction.shape

    if round_priors is not None:
        coverage = store.get_coverage_mask(seed_index)
    else:
        agg_obs_count = store.aggregated_obs_count
        if agg_obs_count is not None:
            coverage = agg_obs_count > 0
        else:
            coverage = store.get_coverage_mask(seed_index)

    dist_map = _compute_settlement_distance_map(initial_grid)
    ocean_dist_map = _compute_ocean_distance_map(initial_grid)

    for y in range(h):
        for x in range(w):
            if coverage[y, x]:
                continue
            terrain = initial_grid[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            is_coastal = ocean_dist_map[y, x] == 1
            neighbor_dist = _gather_neighbor_distributions(
                prediction, coverage, x, y, h, w, radius=5,
            )
            if neighbor_dist is not None:
                settlement_d = dist_map[y, x]
                terrain_prior = _prior_from_initial_terrain(terrain, settlement_d, floor, round_priors, is_coastal)
                blend_weight = min(1.0, settlement_d / 10.0)
                prediction[y, x] = (1.0 - blend_weight) * neighbor_dist + blend_weight * terrain_prior
            else:
                prediction[y, x] = _prior_from_initial_terrain(
                    terrain, dist_map[y, x], floor, round_priors, is_coastal,
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
    round_priors: Optional[RoundPriors] = None,
    is_coastal: bool = False,
) -> np.ndarray:
    if round_priors is not None:
        coastal_tag = "coastal" if is_coastal else "inland"
        key = (terrain, _dist_bucket(settlement_dist), coastal_tag)
        if key in round_priors:
            dist = round_priors[key].copy()
            dist = np.maximum(dist, floor)
            return dist / dist.sum()
    dist = _gt_calibrated_prior(terrain, settlement_dist, is_coastal)
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
    round_priors: Optional[RoundPriors] = None,
) -> np.ndarray:
    h, w = initial_grid.shape
    prediction = build_prediction(seed_index, store, initial_grid, floor, round_priors) if store is not None else np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES)

    dist_map = _compute_settlement_distance_map(initial_grid)

    for y in range(h):
        for x in range(w):
            terrain = initial_grid[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            settlement_d = dist_map[y, x]
            w_mc = mc_weight * min(1.0, settlement_d / 8.0 + 0.3)
            prediction[y, x] = (1.0 - w_mc) * prediction[y, x] + w_mc * mc_prediction[y, x]

    _apply_floor_and_normalize(prediction, floor, initial_grid)
    return prediction


def _apply_floor_and_normalize(
    prediction: np.ndarray, floor: float,
    initial_grid: Optional[np.ndarray] = None,
):
    h, w, c = prediction.shape

    if initial_grid is not None:
        ocean_dist_map = _compute_ocean_distance_map(initial_grid)
        for y in range(h):
            for x in range(w):
                terrain = initial_grid[y, x]
                if terrain not in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    prediction[y, x] = _suppress_mountain(prediction[y, x])
                    if ocean_dist_map[y, x] != 1:
                        prediction[y, x] = _suppress_port(prediction[y, x])

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
