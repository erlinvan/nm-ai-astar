"""Extract empirical class distributions from ground truth data.

Produces data-driven priors to replace hand-tuned values in prediction_builder.py.
Run: ASTAR_TOKEN=... python3 calibration.py --round 15
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from api_client import AstarIslandClient
from config import (
    NUM_CLASSES,
    CLASS_NAMES,
    TERRAIN_OCEAN,
    TERRAIN_MOUNTAIN,
    TERRAIN_PLAINS,
    TERRAIN_EMPTY,
    TERRAIN_SETTLEMENT,
    TERRAIN_PORT,
    TERRAIN_RUIN,
    TERRAIN_FOREST,
    TERRAIN_NAMES,
)
from backtest import (
    KNOWN_ROUNDS,
    load_gt_cache,
    save_gt_cache,
    resolve_round_id,
)

GT_CACHE_DIR = Path(__file__).parent / ".gt_cache"

DISTANCE_BUCKETS = [
    (0, 3, "dist_0_3"),
    (4, 6, "dist_4_6"),
    (7, 999, "dist_7_plus"),
]

DYNAMIC_TERRAINS = {
    TERRAIN_PLAINS: "Plains",
    TERRAIN_EMPTY: "Empty",
    TERRAIN_SETTLEMENT: "Settlement",
    TERRAIN_PORT: "Port",
    TERRAIN_RUIN: "Ruin",
    TERRAIN_FOREST: "Forest",
}


def compute_settlement_distance_map(initial_grid: np.ndarray) -> np.ndarray:
    """Chebyshev distance to nearest settlement/port/ruin."""
    h, w = initial_grid.shape
    positions = []
    for y in range(h):
        for x in range(w):
            if initial_grid[y, x] in (TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN):
                positions.append((x, y))

    dist_map = np.full((h, w), 999.0)
    if not positions:
        return dist_map

    for y in range(h):
        for x in range(w):
            min_d = min(max(abs(x - sx), abs(y - sy)) for sx, sy in positions)
            dist_map[y, x] = float(min_d)

    return dist_map


def load_all_gt(
    client: AstarIslandClient,
    round_id: str,
    num_seeds: int,
    initial_grids: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load GT + initial_grid for all seeds, fetching and caching as needed."""
    results = []
    for seed_idx in range(num_seeds):
        cached = load_gt_cache(round_id, seed_idx)
        if cached is not None:
            results.append(cached)
            print(f"  Seed {seed_idx}: loaded from cache")
            continue

        try:
            analysis = client.get_analysis(round_id, seed_idx)
            gt_data = analysis.get("ground_truth")
            if gt_data is None:
                print(f"  Seed {seed_idx}: no GT available")
                continue

            gt = np.array(gt_data, dtype=np.float64)
            grid_data = analysis.get("initial_grid")
            initial_grid = (
                np.array(grid_data, dtype=np.int32)
                if grid_data is not None
                else initial_grids[seed_idx]
            )

            save_gt_cache(round_id, seed_idx, gt, initial_grid)
            results.append((gt, initial_grid))
            print(f"  Seed {seed_idx}: fetched and cached")
        except Exception as e:
            print(f"  Seed {seed_idx}: error - {e}")

    return results


def extract_empirical_priors(
    gt_data: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Extract mean GT distributions grouped by (terrain, distance_bucket).

    Returns: {terrain_name: {distance_bucket: mean_distribution[6]}}
    """
    accum: dict[int, dict[str, tuple[np.ndarray, int]]] = {}
    for terrain_code in DYNAMIC_TERRAINS:
        accum[terrain_code] = {}
        for _, _, bucket_name in DISTANCE_BUCKETS:
            accum[terrain_code][bucket_name] = (np.zeros(NUM_CLASSES), 0)

    for gt, initial_grid in gt_data:
        h, w, _ = gt.shape
        dist_map = compute_settlement_distance_map(initial_grid)

        for y in range(h):
            for x in range(w):
                terrain = initial_grid[y, x]
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                if terrain not in accum:
                    continue

                dist = dist_map[y, x]
                gt_dist = gt[y, x]

                entropy = -np.sum(np.clip(gt_dist, 1e-10, 1.0) * np.log(np.clip(gt_dist, 1e-10, 1.0)))
                if entropy < 1e-8:
                    continue

                for lo, hi, bucket_name in DISTANCE_BUCKETS:
                    if lo <= dist <= hi:
                        old_sum, old_count = accum[terrain][bucket_name]
                        accum[terrain][bucket_name] = (old_sum + gt_dist, old_count + 1)
                        break

    result: dict[str, dict[str, np.ndarray]] = {}
    for terrain_code, terrain_name in DYNAMIC_TERRAINS.items():
        result[terrain_name] = {}
        for _, _, bucket_name in DISTANCE_BUCKETS:
            total_sum, count = accum[terrain_code][bucket_name]
            if count > 0:
                mean_dist = total_sum / count
                result[terrain_name][bucket_name] = mean_dist
            else:
                result[terrain_name][bucket_name] = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)

    return result


def extract_max_prob_stats(
    gt_data: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Analyze max probability distribution in GT to calibrate MAX_DYNAMIC_PROB."""
    all_max_probs: list[float] = []
    forest_max_probs: list[float] = []

    for gt, initial_grid in gt_data:
        h, w, _ = gt.shape
        for y in range(h):
            for x in range(w):
                terrain = initial_grid[y, x]
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue

                max_p = float(np.max(gt[y, x]))
                entropy = -np.sum(np.clip(gt[y, x], 1e-10, 1.0) * np.log(np.clip(gt[y, x], 1e-10, 1.0)))
                if entropy < 1e-8:
                    continue

                all_max_probs.append(max_p)
                if terrain == TERRAIN_FOREST:
                    forest_max_probs.append(max_p)

    arr = np.array(all_max_probs) if all_max_probs else np.array([0.5])
    forest_arr = np.array(forest_max_probs) if forest_max_probs else np.array([0.5])

    return {
        "all_p50": float(np.percentile(arr, 50)),
        "all_p75": float(np.percentile(arr, 75)),
        "all_p90": float(np.percentile(arr, 90)),
        "all_p95": float(np.percentile(arr, 95)),
        "all_p99": float(np.percentile(arr, 99)),
        "all_max": float(np.max(arr)),
        "all_count": len(all_max_probs),
        "forest_p50": float(np.percentile(forest_arr, 50)),
        "forest_p75": float(np.percentile(forest_arr, 75)),
        "forest_p90": float(np.percentile(forest_arr, 90)),
        "forest_p95": float(np.percentile(forest_arr, 95)),
        "forest_max": float(np.max(forest_arr)),
        "forest_count": len(forest_max_probs),
    }


def extract_forest_detail(
    gt_data: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    forest_becomes = np.zeros(NUM_CLASSES)
    forest_count = 0
    near_settlement_becomes = np.zeros(NUM_CLASSES)
    near_count = 0
    far_becomes = np.zeros(NUM_CLASSES)
    far_count = 0

    for gt, initial_grid in gt_data:
        h, w, _ = gt.shape
        dist_map = compute_settlement_distance_map(initial_grid)

        for y in range(h):
            for x in range(w):
                if initial_grid[y, x] != TERRAIN_FOREST:
                    continue

                entropy = -np.sum(np.clip(gt[y, x], 1e-10, 1.0) * np.log(np.clip(gt[y, x], 1e-10, 1.0)))
                if entropy < 1e-8:
                    continue

                gt_dist = gt[y, x]
                forest_becomes += gt_dist
                forest_count += 1

                d = dist_map[y, x]
                if d <= 3:
                    near_settlement_becomes += gt_dist
                    near_count += 1
                elif d > 6:
                    far_becomes += gt_dist
                    far_count += 1

    return {
        "forest_mean_gt": (forest_becomes / max(forest_count, 1)).tolist(),
        "forest_cell_count": forest_count,
        "near_settlement_mean_gt": (near_settlement_becomes / max(near_count, 1)).tolist(),
        "near_settlement_count": near_count,
        "far_mean_gt": (far_becomes / max(far_count, 1)).tolist(),
        "far_count": far_count,
    }


def print_prior_table(priors: dict[str, dict[str, np.ndarray]]) -> None:
    """Pretty-print the empirical prior table."""
    class_headers = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    header = f"{'Terrain':<12} {'Bucket':<12} " + " ".join(f"{h:>10}" for h in class_headers)
    print(header)
    print("-" * len(header))

    for terrain_name, buckets in priors.items():
        for bucket_name, dist in buckets.items():
            row = f"{terrain_name:<12} {bucket_name:<12} "
            row += " ".join(f"{v:10.4f}" for v in dist)
            print(row)
        print()


def generate_code_priors(priors: dict[str, dict[str, np.ndarray]]) -> None:
    """Generate copy-pasteable Python code for prediction_builder.py."""
    print("\n# === GENERATED PRIORS FOR prediction_builder.py ===")
    print("# Copy these into _terrain_aware_prior() and _prior_from_initial_terrain()")
    print()

    name_to_var = {
        "Forest": "TERRAIN_FOREST",
        "Settlement": "TERRAIN_SETTLEMENT",
        "Port": "TERRAIN_PORT",
        "Ruin": "TERRAIN_RUIN",
        "Plains": "TERRAIN_PLAINS (-> CLASS_EMPTY)",
        "Empty": "TERRAIN_EMPTY (-> CLASS_EMPTY)",
    }

    bucket_to_dist = {
        "dist_0_3": "settlement_dist <= 3",
        "dist_4_6": "settlement_dist <= 6",
        "dist_7_plus": "else (dist > 6)",
    }

    for terrain_name, buckets in priors.items():
        print(f"# {terrain_name} ({name_to_var.get(terrain_name, '?')}):")
        for bucket_name, dist in buckets.items():
            condition = bucket_to_dist.get(bucket_name, bucket_name)
            print(f"#   if {condition}:")
            for i in range(NUM_CLASSES):
                if dist[i] > 0.005:
                    print(f"#     {CLASS_NAMES[i]:>12}: {dist[i]:.4f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract empirical priors from GT data")
    parser.add_argument("--round", required=True, help="Round number or UUID")
    args = parser.parse_args()

    round_id = resolve_round_id(args.round)
    print(f"Calibration for round: {round_id}")

    client = AstarIslandClient()
    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [
        np.array(state["grid"], dtype=np.int32)
        for state in detail["initial_states"]
    ]

    print(f"\nLoading GT data for {num_seeds} seeds...")
    gt_data = load_all_gt(client, round_id, num_seeds, initial_grids)
    print(f"Loaded {len(gt_data)} seeds with GT")

    if not gt_data:
        print("No GT data available. Exiting.")
        return

    print("\n=== EMPIRICAL PRIOR DISTRIBUTIONS ===")
    priors = extract_empirical_priors(gt_data)
    print_prior_table(priors)
    generate_code_priors(priors)

    print("\n=== MAX PROBABILITY STATISTICS ===")
    max_stats = extract_max_prob_stats(gt_data)
    for key, val in max_stats.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    print("\n=== FOREST DETAIL ===")
    forest_detail = extract_forest_detail(gt_data)
    print(f"  Forest cells (with entropy): {forest_detail['forest_cell_count']}")
    print(f"  Forest mean GT distribution:")
    for i, v in enumerate(forest_detail["forest_mean_gt"]):  # type: ignore[arg-type]
        if v > 0.005:
            print(f"    {CLASS_NAMES[i]:>12}: {v:.4f}")

    print(f"\n  Near settlement (d<=3, {forest_detail['near_settlement_count']} cells):")
    for i, v in enumerate(forest_detail["near_settlement_mean_gt"]):  # type: ignore[arg-type]
        if v > 0.005:
            print(f"    {CLASS_NAMES[i]:>12}: {v:.4f}")

    print(f"\n  Far from settlement (d>6, {forest_detail['far_count']} cells):")
    for i, v in enumerate(forest_detail["far_mean_gt"]):  # type: ignore[arg-type]
        if v > 0.005:
            print(f"    {CLASS_NAMES[i]:>12}: {v:.4f}")


if __name__ == "__main__":
    main()
