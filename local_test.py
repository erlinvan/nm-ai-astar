"""
Local test harness that simulates the full competition flow.

Usage:
    python local_test.py                          # Default: seed=42, 400 MC runs
    python local_test.py --map-seed 99 --quick    # Fast mode (50 runs)
    python local_test.py --gt-runs 500            # Higher-quality ground truth
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import (
    NUM_CLASSES, PROBABILITY_FLOOR, CLASS_NAMES,
    TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN,
)
from simulator.map_gen import generate_map
from simulator.params import SimParams
from simulator.engine import SimulationEngine
from monte_carlo import run_monte_carlo, clone_world
from prediction_builder import (
    build_prediction, build_prediction_with_mc, _apply_floor_and_normalize,
    _compute_settlement_distance_map,
)
from observation_store import ObservationStore
from query_strategy import generate_overlapping_viewports, allocate_queries, generate_tiling, prioritize_tiles
from world_builder import calibrate_settlements_from_observations
from utils import score_prediction, compute_kl_divergence, compute_entropy


def generate_ground_truth(
    world, params, num_runs=500, years=50, base_seed=0,
):
    start = time.time()
    gt = run_monte_carlo(world, params, num_runs=num_runs, years=years, base_seed=base_seed)
    _apply_floor_and_normalize(gt, PROBABILITY_FLOOR)
    elapsed = time.time() - start
    print(f"  Ground truth: {num_runs} runs in {elapsed:.1f}s")
    return gt


def simulate_observations_tiling(world, params, num_queries, years=50, base_seed=1000):
    store = ObservationStore(1, world.width, world.height)
    tiles = generate_tiling(world.width, world.height)
    selected = prioritize_tiles(tiles, world.grid, max_tiles=num_queries)

    for i, tile in enumerate(selected):
        _simulate_single_viewport(world, params, store, tile, i, years, base_seed)

    return store


def simulate_observations_overlapping(world, params, num_queries, years=50, base_seed=2000):
    store = ObservationStore(1, world.width, world.height)
    viewports = generate_overlapping_viewports(
        world.width, world.height, world.grid, num_queries,
    )

    for i, tile in enumerate(viewports):
        _simulate_single_viewport(world, params, store, tile, i, years, base_seed)

    return store


def _simulate_single_viewport(world, params, store, tile, index, years, base_seed):
    sim_world = clone_world(world)
    sim_seed = base_seed + index * 13337
    engine = SimulationEngine(world=sim_world, params=params, seed=sim_seed)
    engine.run(years=years)

    vx, vy = tile["viewport_x"], tile["viewport_y"]
    vw, vh = tile["viewport_w"], tile["viewport_h"]
    viewport_grid = engine.get_grid()[vy:vy + vh, vx:vx + vw].tolist()

    settlements = []
    for s in sim_world.alive_settlements():
        if vx <= s.x < vx + vw and vy <= s.y < vy + vh:
            settlements.append({
                "x": s.x, "y": s.y,
                "population": s.population, "food": s.food,
                "wealth": s.wealth, "defense": s.defense,
                "has_port": s.has_port, "alive": s.alive,
                "owner_id": s.owner_id,
            })

    viewport = {"x": vx, "y": vy, "w": vw, "h": vh}
    store.add_observation(0, viewport, viewport_grid, settlements)


def run_strategy(name, prediction, ground_truth, elapsed=None):
    score = score_prediction(ground_truth, prediction)
    time_str = f" ({elapsed:.1f}s)" if elapsed else ""
    print(f"  {name:30s}  {score:6.2f}{time_str}")
    return score


def print_detailed_breakdown(name, prediction, ground_truth, initial_grid):
    score = score_prediction(ground_truth, prediction)
    print(f"\n  === {name} (score={score:.2f}) ===")

    h, w, c = ground_truth.shape

    class_kl = np.zeros(c)
    class_entropy = np.zeros(c)
    class_count = np.zeros(c)

    for y in range(h):
        for x in range(w):
            cell_entropy = compute_entropy(ground_truth[y, x])
            if cell_entropy < 1e-8:
                continue
            dominant_class = int(np.argmax(ground_truth[y, x]))
            cell_kl = compute_kl_divergence(ground_truth[y, x], prediction[y, x])
            class_kl[dominant_class] += cell_entropy * cell_kl
            class_entropy[dominant_class] += cell_entropy
            class_count[dominant_class] += 1

    print(f"    Per-class:")
    for cls_idx in range(c):
        name_cls = CLASS_NAMES.get(cls_idx, f"Class{cls_idx}")
        if class_entropy[cls_idx] < 1e-8:
            continue
        weighted_kl = class_kl[cls_idx] / class_entropy[cls_idx]
        cls_score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
        print(f"      {name_cls:12s}: score={cls_score:6.2f}  cells={int(class_count[cls_idx]):4d}")

    dist_map = _compute_settlement_distance_map(initial_grid)
    buckets = [(0, 2, "dist 0-2"), (3, 5, "dist 3-5"), (6, 10, "dist 6-10"), (11, 999, "dist 11+")]
    print(f"    Per-distance:")

    for lo, hi, label in buckets:
        total_kl = 0.0
        total_ent = 0.0
        count = 0

        for y in range(h):
            for x in range(w):
                d = dist_map[y, x]
                if d < lo or d > hi:
                    continue
                cell_entropy = compute_entropy(ground_truth[y, x])
                if cell_entropy < 1e-8:
                    continue
                cell_kl = compute_kl_divergence(ground_truth[y, x], prediction[y, x])
                total_kl += cell_entropy * cell_kl
                total_ent += cell_entropy
                count += 1

        if total_ent < 1e-8:
            continue
        weighted_kl = total_kl / total_ent
        bucket_score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
        print(f"      {label:12s}: score={bucket_score:6.2f}  cells={count:4d}")


def main():
    parser = argparse.ArgumentParser(description="Local test harness for Astar Island")
    parser.add_argument("--map-seed", type=int, default=42)
    parser.add_argument("--gt-runs", type=int, default=500)
    parser.add_argument("--mc-runs", type=int, default=400)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--param-scale", type=float, default=0.3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fit", action="store_true", help="Run CMA-ES parameter fitting")
    parser.add_argument("--fit-gens", type=int, default=25, help="CMA-ES generations")
    parser.add_argument("--fit-mc", type=int, default=20, help="MC runs per fitness eval")
    parser.add_argument("--detailed", action="store_true", help="Show per-class and per-distance breakdown")
    args = parser.parse_args()

    if args.quick:
        args.gt_runs = min(args.gt_runs, 100)
        args.mc_runs = min(args.mc_runs, 50)
        args.fit_gens = min(args.fit_gens, 10)
        args.fit_mc = min(args.fit_mc, 10)

    print(f"=== Astar Island Local Test ===")
    print(f"Map seed: {args.map_seed}  |  GT runs: {args.gt_runs}  |  MC runs: {args.mc_runs}")
    print(f"Queries: {args.queries}  |  Param randomization: {args.param_scale}")

    world = generate_map(seed=args.map_seed, width=40, height=40)
    print(f"\nMap: {world.width}x{world.height}  |  Settlements: {len(world.alive_settlements())}")

    rng = np.random.default_rng(args.map_seed + 100)
    hidden_params = SimParams().randomize(rng, scale=args.param_scale)

    print(f"\nGenerating ground truth...")
    ground_truth = generate_ground_truth(
        world, hidden_params, num_runs=args.gt_runs, base_seed=0,
    )

    h, w = world.height, world.width
    initial_grid = world.copy_grid()

    print(f"\n{'Strategy':30s}  {'Score':>6s}")
    print("-" * 42)

    uniform = np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES)
    _apply_floor_and_normalize(uniform, PROBABILITY_FLOOR)
    results = {}
    results["Uniform"] = run_strategy("Uniform", uniform, ground_truth)

    from main import build_baseline_predictions
    baseline = build_baseline_predictions([initial_grid])[0]
    results["Initial terrain"] = run_strategy("Initial terrain", baseline, ground_truth)

    start = time.time()
    mc_default = run_monte_carlo(world, SimParams(), num_runs=args.mc_runs, years=50, base_seed=5000)
    _apply_floor_and_normalize(mc_default, PROBABILITY_FLOOR)
    results["MC (default params)"] = run_strategy(
        "MC (default params)", mc_default, ground_truth, time.time() - start,
    )

    start = time.time()
    mc_noisy = run_monte_carlo(
        world, SimParams(), num_runs=args.mc_runs, years=50,
        base_seed=6000, param_noise=0.15,
    )
    _apply_floor_and_normalize(mc_noisy, PROBABILITY_FLOOR)
    results["MC (noisy 0.15)"] = run_strategy(
        "MC (noisy 0.15)", mc_noisy, ground_truth, time.time() - start,
    )

    store_tiling = simulate_observations_tiling(world, hidden_params, args.queries)
    obs_pred_tiling = build_prediction(0, store_tiling, initial_grid)
    results["Obs tiling"] = run_strategy(
        f"Obs tiling ({args.queries}q)", obs_pred_tiling, ground_truth,
    )

    store_overlap = simulate_observations_overlapping(world, hidden_params, args.queries)
    obs_pred_overlap = build_prediction(0, store_overlap, initial_grid)
    results["Obs overlapping"] = run_strategy(
        f"Obs overlapping ({args.queries}q)", obs_pred_overlap, ground_truth,
    )

    blended_tiling = build_prediction_with_mc(0, store_tiling, initial_grid, mc_default)
    results["MC + Obs tiling"] = run_strategy(
        "MC + Obs tiling", blended_tiling, ground_truth,
    )

    blended_overlap = build_prediction_with_mc(0, store_overlap, initial_grid, mc_default)
    results["MC + Obs overlap"] = run_strategy(
        "MC + Obs overlap", blended_overlap, ground_truth,
    )

    calibrate_settlements_from_observations(world, store_overlap, 0)
    start = time.time()
    mc_calibrated = run_monte_carlo(world, SimParams(), num_runs=args.mc_runs, years=50, base_seed=7000)
    _apply_floor_and_normalize(mc_calibrated, PROBABILITY_FLOOR)
    results["MC (calibrated)"] = run_strategy(
        "MC (calibrated)", mc_calibrated, ground_truth, time.time() - start,
    )

    blended_calibrated = build_prediction_with_mc(0, store_overlap, initial_grid, mc_calibrated)
    results["Calibrated + Obs overlap"] = run_strategy(
        "Calibrated + Obs overlap", blended_calibrated, ground_truth,
    )

    if args.fit:
        from param_fitter import fit_params_from_ground_truth, fit_params_from_observations

        print(f"\n  Fitting params via CMA-ES (gens={args.fit_gens}, mc={args.fit_mc}/eval)...")
        start = time.time()
        fitted_params = fit_params_from_ground_truth(
            world, ground_truth, initial_grid,
            mc_runs_per_eval=args.fit_mc,
            max_generations=args.fit_gens,
            popsize=8,
        )
        fit_time = time.time() - start

        start = time.time()
        mc_fitted = run_monte_carlo(
            world, fitted_params, num_runs=args.mc_runs, years=50, base_seed=9000,
        )
        _apply_floor_and_normalize(mc_fitted, PROBABILITY_FLOOR)
        results["MC (CMA-ES fitted)"] = run_strategy(
            "MC (CMA-ES fitted)", mc_fitted, ground_truth, fit_time + time.time() - start,
        )

        fitted_blend = build_prediction_with_mc(0, store_overlap, initial_grid, mc_fitted)
        results["Fitted + Obs overlap"] = run_strategy(
            "Fitted + Obs overlap", fitted_blend, ground_truth,
        )

        print(f"\n  Fitting from observations only (gens={args.fit_gens})...")
        start = time.time()
        obs_fitted_params = fit_params_from_observations(
            [world], store_overlap, [0], [initial_grid],
            mc_runs_per_eval=args.fit_mc,
            max_generations=args.fit_gens,
            popsize=8,
        )
        obs_fit_time = time.time() - start

        start = time.time()
        mc_obs_fitted = run_monte_carlo(
            world, obs_fitted_params, num_runs=args.mc_runs, years=50, base_seed=11000,
        )
        _apply_floor_and_normalize(mc_obs_fitted, PROBABILITY_FLOOR)
        results["MC (obs-fitted)"] = run_strategy(
            "MC (obs-fitted)", mc_obs_fitted, ground_truth, obs_fit_time + time.time() - start,
        )

        obs_fitted_blend = build_prediction_with_mc(0, store_overlap, initial_grid, mc_obs_fitted)
        results["Obs-fitted + Obs"] = run_strategy(
            "Obs-fitted + Obs", obs_fitted_blend, ground_truth,
        )

    start = time.time()
    mc_oracle = run_monte_carlo(
        world, hidden_params, num_runs=args.mc_runs, years=50, base_seed=8000,
    )
    _apply_floor_and_normalize(mc_oracle, PROBABILITY_FLOOR)
    results["Oracle (true params)"] = run_strategy(
        "Oracle (true params)", mc_oracle, ground_truth, time.time() - start,
    )

    print(f"\n=== Ranking ===")
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(ranked, 1):
        bar = "\u2588" * int(score / 2) + "\u2591" * (50 - int(score / 2))
        print(f"  {i}. {name:30s} {score:6.2f}  {bar}")

    if args.detailed:
        print(f"\n=== Detailed Breakdown ===")
        best_name = ranked[0][0]
        for name, score in ranked[:3]:
            if name == "Uniform":
                continue
            pred_map = {
                "MC + Obs overlap": blended_overlap,
                "Calibrated + Obs overlap": blended_calibrated,
                "MC (calibrated)": mc_calibrated,
                "MC (default params)": mc_default,
                "MC (noisy 0.15)": mc_noisy,
                "Oracle (true params)": mc_oracle,
                "Initial terrain": baseline,
                "Obs overlapping": obs_pred_overlap,
                "Obs tiling": obs_pred_tiling,
                "MC + Obs tiling": blended_tiling,
            }
            if name in pred_map:
                print_detailed_breakdown(name, pred_map[name], ground_truth, initial_grid)


if __name__ == "__main__":
    main()
