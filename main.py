import sys
import time
import argparse
from typing import Optional
import numpy as np

from config import (
    TERRAIN_TO_CLASS,
    NUM_CLASSES,
    TERRAIN_OCEAN,
    TERRAIN_MOUNTAIN,
    TERRAIN_FOREST,
    MAX_QUERIES,
    PROBABILITY_FLOOR,
    CLASS_NAMES,
)
from api_client import AstarIslandClient
from query_strategy import generate_overlapping_viewports, allocate_queries
from observation_store import ObservationStore
from prediction_builder import (
    build_prediction, build_prediction_with_mc, _apply_floor_and_normalize,
)
from submission import submit_all_predictions, _validate_prediction
from monte_carlo import run_monte_carlo
from world_builder import build_world_from_state, calibrate_settlements_from_observations
from simulator.params import SimParams


def parse_initial_grids(detail: dict) -> list[np.ndarray]:
    grids = []
    for state in detail["initial_states"]:
        grids.append(np.array(state["grid"], dtype=np.int32))
    return grids


def run_observation_phase(
    client: AstarIslandClient,
    round_id: str,
    num_seeds: int,
    initial_grids: list[np.ndarray],
    width: int,
    height: int,
) -> ObservationStore:
    store = ObservationStore(num_seeds, width, height)
    allocation = allocate_queries(num_seeds, MAX_QUERIES, width, height, initial_grids=initial_grids)

    strategy = allocation["strategy"]
    per_seed = allocation["per_seed_queries"]
    print(f"Strategy: {strategy}")
    print(f"Per-seed queries: {per_seed}, Total: {allocation['total_queries']}")

    query_count = 0

    for seed_idx in range(num_seeds):
        num_queries_for_seed = per_seed[seed_idx]
        viewports = generate_overlapping_viewports(
            width, height, initial_grids[seed_idx], num_queries_for_seed,
        )

        for tile in viewports:
            try:
                result = client.simulate(
                    round_id=round_id,
                    seed_index=seed_idx,
                    viewport_x=tile["viewport_x"],
                    viewport_y=tile["viewport_y"],
                    viewport_w=tile["viewport_w"],
                    viewport_h=tile["viewport_h"],
                )
                store.add_observation(
                    seed_idx,
                    result["viewport"],
                    result["grid"],
                    result.get("settlements", []),
                )
                query_count += 1
                queries_used = result.get("queries_used", query_count)
                queries_max = result.get("queries_max", MAX_QUERIES)
                print(f"  Seed {seed_idx} tile ({tile['viewport_x']},{tile['viewport_y']}): "
                      f"OK [{queries_used}/{queries_max}]")

            except Exception as e:
                print(f"  Seed {seed_idx} tile ({tile['viewport_x']},{tile['viewport_y']}): "
                      f"FAILED - {e}")
                if "429" in str(e) or "budget" in str(e).lower():
                    print("Query budget exhausted. Moving to prediction phase.")
                    store.save(round_id)
                    return store

        coverage = store.coverage_ratio(seed_idx)
        print(f"Seed {seed_idx}: {coverage:.1%} coverage")

        # Save incrementally after each seed to avoid losing data on timeout/interrupt
        store.save(round_id)

    return store


def run_mc_phase(
    detail: dict,
    num_seeds: int,
    width: int,
    height: int,
    mc_runs: int = 200,
    param_noise: float = 0.0,
    params: SimParams = SimParams(),
    store: Optional[ObservationStore] = None,
) -> list[np.ndarray]:

    mc_predictions = []
    for seed_idx in range(num_seeds):
        world = build_world_from_state(
            detail["initial_states"][seed_idx], width, height, rng_seed=seed_idx,
        )

        if store is not None:
            calibrate_settlements_from_observations(world, store, seed_idx)

        print(f"  Seed {seed_idx}: running {mc_runs} simulations...")
        start = time.time()
        mc_pred = run_monte_carlo(
            world, params, num_runs=mc_runs, years=50,
            base_seed=seed_idx * 10000, param_noise=param_noise,
        )
        _apply_floor_and_normalize(mc_pred, PROBABILITY_FLOOR)
        elapsed = time.time() - start
        print(f"  Seed {seed_idx}: done ({elapsed:.1f}s)")
        mc_predictions.append(mc_pred)
    return mc_predictions


def run_prediction_phase(
    store: ObservationStore,
    initial_grids: list[np.ndarray],
    num_seeds: int,
    mc_predictions=None,
    mc_pseudo_count: float = 3.0,
) -> list[np.ndarray]:
    predictions = []
    for seed_idx in range(num_seeds):
        if mc_predictions is not None:
            pred = build_prediction_with_mc(
                seed_idx, store, initial_grids[seed_idx], mc_predictions[seed_idx],
                mc_pseudo_count=mc_pseudo_count,
            )
        else:
            pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
        _validate_prediction(pred)
        predictions.append(pred)
        coverage = store.coverage_ratio(seed_idx)
        print(f"Seed {seed_idx}: prediction built ({coverage:.1%} observed)")
    return predictions


def run_submission_phase(
    client: AstarIslandClient,
    round_id: str,
    predictions: list[np.ndarray],
) -> list[dict]:
    results = []
    for seed_idx, pred in enumerate(predictions):
        resp = client.submit(round_id, seed_idx, pred.tolist())
        status = resp.get("status", "unknown")
        print(f"Seed {seed_idx}: {status}")
        results.append(resp)
    return results


def run_analysis_phase(
    client: AstarIslandClient,
    round_id: str,
    num_seeds: int,
    predictions: list[np.ndarray],
) -> None:
    print("\n--- Analysis Phase ---")
    for seed_idx in range(num_seeds):
        try:
            analysis = client.get_analysis(round_id, seed_idx)
        except Exception as e:
            print(f"  Seed {seed_idx}: analysis not available ({e})")
            continue

        gt_grid = analysis.get("ground_truth")
        if gt_grid is None:
            print(f"  Seed {seed_idx}: no ground truth in response")
            continue

        gt = np.array(gt_grid, dtype=np.float64)
        pred = predictions[seed_idx]

        if gt.shape != pred.shape:
            print(f"  Seed {seed_idx}: shape mismatch gt={gt.shape} pred={pred.shape}")
            continue

        from utils import score_prediction, compute_kl_divergence, compute_entropy
        overall_score = score_prediction(gt, pred)
        print(f"  Seed {seed_idx}: score = {overall_score:.2f}")

        _print_per_class_analysis(gt, pred, seed_idx)
        _print_per_distance_analysis(gt, pred, seed_idx)


def _print_per_class_analysis(gt: np.ndarray, pred: np.ndarray, seed_idx: int):
    from utils import compute_kl_divergence, compute_entropy
    h, w, c = gt.shape

    class_kl = np.zeros(c)
    class_entropy = np.zeros(c)
    class_count = np.zeros(c)

    for y in range(h):
        for x in range(w):
            cell_entropy = compute_entropy(gt[y, x])
            if cell_entropy < 1e-8:
                continue

            dominant_class = int(np.argmax(gt[y, x]))
            cell_kl = compute_kl_divergence(gt[y, x], pred[y, x])

            class_kl[dominant_class] += cell_entropy * cell_kl
            class_entropy[dominant_class] += cell_entropy
            class_count[dominant_class] += 1

    print(f"    Per-class breakdown (seed {seed_idx}):")
    for cls_idx in range(c):
        name = CLASS_NAMES.get(cls_idx, f"Class{cls_idx}")
        if class_entropy[cls_idx] < 1e-8:
            continue
        weighted_kl = class_kl[cls_idx] / class_entropy[cls_idx]
        cls_score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
        print(f"      {name:12s}: score={cls_score:6.2f}  cells={int(class_count[cls_idx]):4d}  "
              f"w_kl={weighted_kl:.4f}")


def _print_per_distance_analysis(gt: np.ndarray, pred: np.ndarray, seed_idx: int):
    from utils import compute_kl_divergence, compute_entropy
    from prediction_builder import _compute_settlement_distance_map

    h, w, c = gt.shape

    settlement_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            dominant = int(np.argmax(gt[y, x]))
            if dominant in (1, 2, 3):
                settlement_mask[y, x] = True

    dist_map = np.full((h, w), 999.0)
    positions = list(zip(*np.where(settlement_mask)))
    if positions:
        for y in range(h):
            for x in range(w):
                min_d = min(
                    max(abs(x - sx), abs(y - sy))
                    for sy, sx in positions
                )
                dist_map[y, x] = float(min_d)

    buckets = [(0, 2, "dist 0-2"), (3, 5, "dist 3-5"), (6, 10, "dist 6-10"), (11, 999, "dist 11+")]
    print(f"    Per-distance breakdown (seed {seed_idx}):")

    for lo, hi, label in buckets:
        total_kl = 0.0
        total_ent = 0.0
        count = 0

        for y in range(h):
            for x in range(w):
                d = dist_map[y, x]
                if d < lo or d > hi:
                    continue
                cell_entropy = compute_entropy(gt[y, x])
                if cell_entropy < 1e-8:
                    continue
                cell_kl = compute_kl_divergence(gt[y, x], pred[y, x])
                total_kl += cell_entropy * cell_kl
                total_ent += cell_entropy
                count += 1

        if total_ent < 1e-8:
            continue
        weighted_kl = total_kl / total_ent
        bucket_score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
        print(f"      {label:12s}: score={bucket_score:6.2f}  cells={count:4d}  w_kl={weighted_kl:.4f}")


def build_baseline_predictions(initial_grids: list[np.ndarray]) -> list[np.ndarray]:
    predictions = []
    for grid in initial_grids:
        h, w = grid.shape
        pred = np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES)

        for y in range(h):
            for x in range(w):
                terrain = grid[y, x]
                if terrain == TERRAIN_OCEAN:
                    pred[y, x] = np.array([0.94, 0.012, 0.012, 0.012, 0.012, 0.012])
                elif terrain == TERRAIN_MOUNTAIN:
                    pred[y, x] = np.array([0.012, 0.012, 0.012, 0.012, 0.012, 0.94])
                elif terrain == TERRAIN_FOREST:
                    pred[y, x] = np.array([0.05, 0.02, 0.01, 0.05, 0.85, 0.02])

        sums = pred.sum(axis=-1, keepdims=True)
        pred = pred / sums
        predictions.append(pred)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Astar Island Competition Pipeline")
    parser.add_argument("--token", type=str, help="API auth token")
    parser.add_argument("--round-id", type=str, help="Round ID (auto-detects active round if omitted)")
    parser.add_argument("--baseline-only", action="store_true", help="Submit baseline predictions without querying")
    parser.add_argument("--dry-run", action="store_true", help="Build predictions but don't submit")
    parser.add_argument("--observe-only", action="store_true", help="Only observe, don't submit")
    parser.add_argument("--mc-runs", type=int, default=400, help="Monte Carlo runs per seed (0 to disable)")
    parser.add_argument("--mc-pseudo-count", type=float, default=3.0, help="MC pseudo count for observation blending")
    parser.add_argument("--param-noise", type=float, default=0.15, help="Parameter randomization noise scale")
    parser.add_argument("--fit-params", action="store_true", help="Run CMA-ES parameter fitting after observations")
    parser.add_argument("--fit-gens", type=int, default=25, help="CMA-ES generations")
    parser.add_argument("--fit-mc", type=int, default=20, help="MC runs per fitness eval during fitting")
    parser.add_argument("--analyze", action="store_true", help="Run post-round analysis on completed rounds")
    parser.add_argument("--analyze-round", type=str, help="Analyze a specific completed round")
    args = parser.parse_args()

    client = AstarIslandClient(token=args.token)

    if args.analyze_round:
        _run_standalone_analysis(client, args)
        return

    if args.round_id:
        round_id = args.round_id
        detail = client.get_round_detail(round_id)
    else:
        active = client.get_active_round()
        if not active:
            print("No active round found.")
            sys.exit(1)
        round_id = active["id"]
        detail = client.get_round_detail(round_id)

    width = detail["map_width"]
    height = detail["map_height"]
    num_seeds = detail["seeds_count"]
    initial_grids = parse_initial_grids(detail)

    print(f"Round: {round_id}")
    print(f"Map: {width}x{height}, Seeds: {num_seeds}")
    print(f"Settlements per seed: {[len(s['settlements']) for s in detail['initial_states']]}")

    if args.baseline_only:
        print("\n--- Baseline Predictions ---")
        predictions = build_baseline_predictions(initial_grids)
        if not args.dry_run:
            print("\n--- Submitting ---")
            run_submission_phase(client, round_id, predictions)
        else:
            print("Dry run — skipping submission")
        return

    mc_predictions = None
    store = None
    fitted_params = None

    # Phase 1: Observe (fast, uses API budget — do this FIRST)
    print("\n--- Observation Phase ---")

    store = None
    try:
        store = ObservationStore.load(round_id)
        print(f"Loaded cached observations ({store.total_observations()} total)")
        for s in range(num_seeds):
            print(f"  Seed {s}: {store.coverage_ratio(s):.1%} coverage")
    except FileNotFoundError:
        pass

    budget = client.get_budget()
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} used")

    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining > 0:
        new_store = run_observation_phase(client, round_id, num_seeds, initial_grids, width, height)

        if store is not None:
            store.merge(new_store)
        else:
            store = new_store

        save_path = store.save(round_id)
        print(f"\nObservations saved to {save_path}")
        print(f"Total observations: {store.total_observations()}")
        for s in range(num_seeds):
            print(f"  Seed {s}: {store.coverage_ratio(s):.1%} coverage")

        if args.observe_only:
            print("Observe-only mode — skipping prediction and submission")
            return
    elif store is not None:
        print("No new queries available — using cached observations.")
    else:
        print("No queries remaining and no cached observations.")

    if store is not None:
        store.aggregate_across_seeds()
        agg = store.aggregated_obs_count
        if agg is not None:
            print(f"Cross-seed aggregation: {int(np.sum(agg > 0))} cells with data")

    # Phase 2: Fit params from observations (if available)
    if args.fit_params and store is not None:
        from param_fitter import fit_params_from_observations
        print(f"\n--- Parameter Fitting Phase (CMA-ES, {args.fit_gens} gens, all seeds) ---")

        fit_worlds = []
        fit_grids = []
        fit_seeds = []
        for seed_idx in range(num_seeds):
            if store.coverage_ratio(seed_idx) < 0.01:
                continue
            w = build_world_from_state(
                detail["initial_states"][seed_idx], width, height, rng_seed=seed_idx,
            )
            calibrate_settlements_from_observations(w, store, seed_idx)
            fit_worlds.append(w)
            fit_grids.append(initial_grids[seed_idx])
            fit_seeds.append(seed_idx)

        if fit_worlds:
            fitted_params = fit_params_from_observations(
                fit_worlds, store, fit_seeds, fit_grids,
                mc_runs_per_eval=args.fit_mc,
                max_generations=args.fit_gens,
            )
        else:
            print("  No seeds with sufficient coverage for fitting")

    # Phase 3: Monte Carlo simulation
    if args.mc_runs > 0:
        label_parts = []
        if fitted_params is not None:
            label_parts.append("fitted params")
        if store is not None:
            label_parts.append("calibrated")
        label = " + ".join(label_parts) if label_parts else "default params"
        noise = args.param_noise if fitted_params is None else 0.0
        print(f"\n--- Monte Carlo Phase ({label}, {args.mc_runs} runs/seed, noise={noise}) ---")
        mc_predictions = run_mc_phase(
            detail, num_seeds, width, height,
            mc_runs=args.mc_runs,
            param_noise=noise,
            params=fitted_params if fitted_params is not None else SimParams.gt_tuned(),
            store=store,
        )

    # Phase 4: Build final predictions
    if store is not None and mc_predictions is not None:
        print("\n--- Prediction Phase (blending MC + observations) ---")
        effective_pseudo_count = args.mc_pseudo_count
        if fitted_params is not None:
            effective_pseudo_count = max(args.mc_pseudo_count, 10.0)
            print(f"  Using elevated mc_pseudo_count={effective_pseudo_count} (fitted params)")
        predictions = run_prediction_phase(
            store, initial_grids, num_seeds, mc_predictions,
            mc_pseudo_count=effective_pseudo_count,
        )
    elif mc_predictions is not None:
        predictions = mc_predictions
    elif store is not None:
        print("\n--- Prediction Phase (observations only, no MC) ---")
        predictions = run_prediction_phase(store, initial_grids, num_seeds)
    else:
        print("\n--- Baseline Predictions (no MC, no observations) ---")
        predictions = build_baseline_predictions(initial_grids)

    # Phase 5: Submit
    if not args.dry_run:
        print("\n--- Submission Phase ---")
        run_submission_phase(client, round_id, predictions)
    else:
        print("\nDry run — skipping submission")

    if args.analyze:
        run_analysis_phase(client, round_id, num_seeds, predictions)

    print("\nDone.")


def _run_standalone_analysis(client: AstarIslandClient, args):
    round_id = args.analyze_round
    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]

    print(f"Analyzing round: {round_id}")

    try:
        my_preds = client.get_my_predictions(round_id)
    except Exception as e:
        print(f"Could not fetch predictions: {e}")
        return

    predictions = []
    for seed_idx in range(num_seeds):
        matching = [p for p in my_preds if p.get("seed_index") == seed_idx]
        if not matching:
            print(f"  Seed {seed_idx}: no prediction found")
            predictions.append(None)
            continue
        pred_data = matching[-1].get("prediction")
        if pred_data is None:
            predictions.append(None)
            continue
        predictions.append(np.array(pred_data, dtype=np.float64))

    for seed_idx in range(num_seeds):
        if predictions[seed_idx] is None:
            continue
        try:
            analysis = client.get_analysis(round_id, seed_idx)
        except Exception as e:
            print(f"  Seed {seed_idx}: analysis not available ({e})")
            continue

        gt_data = analysis.get("ground_truth")
        if gt_data is None:
            print(f"  Seed {seed_idx}: no ground truth in response")
            continue

        gt = np.array(gt_data, dtype=np.float64)
        pred = predictions[seed_idx]

        if gt.shape != pred.shape:
            print(f"  Seed {seed_idx}: shape mismatch")
            continue

        from utils import score_prediction
        score = score_prediction(gt, pred)
        print(f"\n  Seed {seed_idx}: overall score = {score:.2f}")
        _print_per_class_analysis(gt, pred, seed_idx)
        _print_per_distance_analysis(gt, pred, seed_idx)


if __name__ == "__main__":
    main()
