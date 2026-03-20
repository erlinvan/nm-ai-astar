"""CMA-ES parameter fitting against observed simulation data."""

import time
import numpy as np
import cma

from simulator.params import SimParams
from simulator.world import World
from monte_carlo import run_monte_carlo, clone_world
from config import (
    NUM_CLASSES, PROBABILITY_FLOOR, TERRAIN_OCEAN, TERRAIN_MOUNTAIN,
    TERRAIN_TO_CLASS,
)
from observation_store import ObservationStore

TUNABLE_PARAMS = [
    ("forest_food_yield",      0.1,  2.0),
    ("plains_food_yield",      0.05, 1.0),
    ("port_food_multiplier",   1.0,  2.5),
    ("growth_rate",            0.02, 0.60),
    ("food_consumption",       0.1,  1.5),
    ("expansion_threshold",    1.5,  8.0),
    ("expansion_prob",         0.01, 0.99),
    ("port_threshold",         1.0,  6.0),
    ("longship_threshold",     0.5,  5.0),
    ("raid_range",             1.0,  6.0),
    ("longship_range_bonus",   1.0,  8.0),
    ("base_raid_prob",         0.01, 0.60),
    ("raid_threshold",         0.3,  2.0),
    ("conquest_prob",          0.01, 0.99),
    ("loot_fraction",          0.05, 0.80),
    ("defense_damage",         0.01, 0.60),
    ("pop_damage_fraction",    0.01, 0.60),
    ("trade_range",            2.0,  15.0),
    ("trade_efficiency",       0.01, 0.50),
    ("tech_diffusion_rate",    0.01, 0.30),
    ("base_winter_severity",   0.05, 0.90),
    ("winter_variance",        0.01, 0.50),
    ("winter_food_loss",       0.05, 0.80),
    ("min_population",         0.05, 1.0),
    ("reclaim_prob",           0.01, 0.99),
    ("forest_regrowth_prob",   0.01, 0.99),
    ("plains_regrowth_prob",   0.01, 0.99),
]

INT_PARAMS = {"raid_range", "longship_range_bonus"}

PARAM_NAMES = [p[0] for p in TUNABLE_PARAMS]
PARAM_BOUNDS = [(p[1], p[2]) for p in TUNABLE_PARAMS]

CORE_TUNABLE_PARAMS = [
    ("growth_rate",            0.02, 0.60),
    ("expansion_prob",         0.01, 0.99),
    ("conquest_prob",          0.01, 0.99),
    ("base_winter_severity",   0.05, 0.90),
    ("winter_food_loss",       0.05, 0.80),
    ("reclaim_prob",           0.01, 0.99),
    ("forest_regrowth_prob",   0.01, 0.99),
    ("plains_regrowth_prob",   0.01, 0.99),
    ("forest_food_yield",      0.1,  2.0),
    ("food_consumption",       0.1,  1.5),
    ("expansion_threshold",    1.5,  8.0),
    ("base_raid_prob",         0.01, 0.60),
]

CORE_PARAM_NAMES = [p[0] for p in CORE_TUNABLE_PARAMS]
CORE_PARAM_BOUNDS = [(p[1], p[2]) for p in CORE_TUNABLE_PARAMS]


def _make_params(x: np.ndarray, param_names=None, param_bounds=None) -> SimParams:
    if param_names is None:
        param_names = PARAM_NAMES
    if param_bounds is None:
        param_bounds = PARAM_BOUNDS
    d = SimParams().to_dict()
    for i, name in enumerate(param_names):
        lo, hi = param_bounds[i]
        val = float(np.clip(x[i], lo, hi))
        if name in INT_PARAMS:
            val = int(round(val))
        d[name] = val
    return SimParams.from_dict(d)


def _observation_fitness_single_seed(
    x: np.ndarray,
    world: World,
    store: ObservationStore,
    seed_index: int,
    initial_grid: np.ndarray,
    mc_runs: int = 20,
    param_names=None,
    param_bounds=None,
) -> tuple[float, float]:
    """Return (total_kl, total_weight) for one seed."""
    params = _make_params(x, param_names, param_bounds)
    mc_pred = run_monte_carlo(
        world, params, num_runs=mc_runs, years=50,
        base_seed=seed_index * 10000 + int(abs(x.sum()) * 1000) % 100000, workers=1,
    )

    h, w = initial_grid.shape
    total_kl = 0.0
    total_weight = 0.0

    for y in range(h):
        for x_coord in range(w):
            terrain = initial_grid[y, x_coord]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            obs_count = store.get_observation_count(seed_index, x_coord, y)
            if obs_count == 0:
                continue

            obs_dist = store.class_counts[seed_index, y, x_coord] / obs_count
            pred_dist = mc_pred[y, x_coord]

            obs_clipped = np.clip(obs_dist, 1e-10, 1.0)
            pred_clipped = np.clip(pred_dist, 1e-10, 1.0)

            kl = float(np.sum(obs_clipped * np.log(obs_clipped / pred_clipped)))
            cell_entropy = float(-np.sum(obs_clipped * np.log(obs_clipped)))

            if cell_entropy > 1e-8:
                total_kl += cell_entropy * kl
                total_weight += cell_entropy

    return total_kl, total_weight


def _observation_fitness(
    x: np.ndarray,
    worlds: list,
    store: ObservationStore,
    seed_indices: list[int],
    initial_grids: list[np.ndarray],
    mc_runs: int = 20,
    param_names=None,
    param_bounds=None,
) -> float:
    """Negative score: how well candidate params match observed terrain across all seeds."""
    total_kl = 0.0
    total_weight = 0.0

    for world, seed_idx, grid in zip(worlds, seed_indices, initial_grids):
        kl, weight = _observation_fitness_single_seed(
            x, world, store, seed_idx, grid, mc_runs, param_names, param_bounds,
        )
        total_kl += kl
        total_weight += weight

    if total_weight < 1e-8:
        return 0.0

    weighted_kl = total_kl / total_weight
    score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
    return -score


def _ground_truth_fitness(
    x: np.ndarray,
    world: World,
    ground_truth: np.ndarray,
    initial_grid: np.ndarray,
    mc_runs: int = 20,
) -> float:
    """For local testing: score candidate params against known ground truth."""
    params = _make_params(x)
    mc_pred = run_monte_carlo(
        world, params, num_runs=mc_runs, years=50,
        base_seed=int(x.sum() * 1000) % 100000, workers=1,
    )

    mc_pred = np.maximum(mc_pred, PROBABILITY_FLOOR)
    mc_pred = mc_pred / mc_pred.sum(axis=-1, keepdims=True)

    from utils import score_prediction
    score = score_prediction(ground_truth, mc_pred)
    return -score


def fit_params_from_observations(
    worlds: list,
    store: ObservationStore,
    seed_indices: list[int],
    initial_grids: list[np.ndarray],
    mc_runs_per_eval: int = 20,
    max_generations: int = 30,
    popsize: int = 8,
    verbose: bool = True,
    use_core_params: bool = True,
) -> SimParams:
    param_names = CORE_PARAM_NAMES if use_core_params else PARAM_NAMES
    param_bounds = CORE_PARAM_BOUNDS if use_core_params else PARAM_BOUNDS

    x0 = [SimParams().to_dict()[name] for name in param_names]
    sigma0 = 0.3
    bounds = list(zip(*param_bounds))

    opts = {
        "popsize": popsize,
        "maxiter": max_generations,
        "bounds": bounds,
        "verbose": -9,
        "seed": 42,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_score = float("inf")
    start = time.time()

    if verbose:
        print(f"  Fitting {len(param_names)} params across {len(seed_indices)} seeds")

    while not es.stop():
        candidates = es.ask()
        fitnesses = [
            _observation_fitness(
                c, worlds, store, seed_indices, initial_grids,
                mc_runs_per_eval, param_names, param_bounds,
            )
            for c in candidates
        ]
        es.tell(candidates, fitnesses)

        gen_best = min(fitnesses)
        if gen_best < best_score:
            best_score = gen_best

        if verbose:
            elapsed = time.time() - start
            print(f"  Gen {es.countiter:3d}: best={-best_score:.2f}  gen_best={-gen_best:.2f}  ({elapsed:.0f}s)")

    raw = es.result.xbest
    xbest = np.array(x0 if raw is None else list(raw))
    result = _make_params(xbest, param_names, param_bounds)
    if verbose:
        elapsed = time.time() - start
        print(f"  Fitting done: score={-best_score:.2f} in {elapsed:.1f}s")
        for name, val in zip(param_names, list(xbest)):
            default = SimParams().to_dict()[name]
            print(f"    {name:25s}: {default:.3f} -> {val:.3f}")

    return result


def fit_params_from_ground_truth(
    world: World,
    ground_truth: np.ndarray,
    initial_grid: np.ndarray,
    mc_runs_per_eval: int = 20,
    max_generations: int = 30,
    popsize: int = 8,
    verbose: bool = True,
) -> SimParams:
    """For local testing: fit params against known ground truth distribution."""
    x0 = [SimParams().to_dict()[name] for name in PARAM_NAMES]
    sigma0 = 0.3
    bounds = list(zip(*PARAM_BOUNDS))

    opts = {
        "popsize": popsize,
        "maxiter": max_generations,
        "bounds": bounds,
        "verbose": -9,
        "seed": 42,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_score = float("inf")
    start = time.time()

    while not es.stop():
        candidates = es.ask()
        fitnesses = [
            _ground_truth_fitness(c, world, ground_truth, initial_grid, mc_runs_per_eval)
            for c in candidates
        ]
        es.tell(candidates, fitnesses)

        gen_best = min(fitnesses)
        if gen_best < best_score:
            best_score = gen_best

        if verbose:
            elapsed = time.time() - start
            print(f"  Gen {es.countiter:3d}: best={-best_score:.2f}  gen_best={-gen_best:.2f}  ({elapsed:.0f}s)")

    raw2 = es.result.xbest
    xbest2 = np.array(x0 if raw2 is None else list(raw2))
    result = _make_params(xbest2)
    if verbose:
        elapsed = time.time() - start
        print(f"  Fitting done: score={-best_score:.2f} in {elapsed:.1f}s")
        for name, val in zip(PARAM_NAMES, list(xbest2)):
            default = SimParams().to_dict()[name]
            print(f"    {name:25s}: {default:.3f} -> {val:.3f}")

    return result
