"""Compute GT priors at fine distance granularity (d3, d4, d5) x coastal/inland x terrain."""
import numpy as np
import os
os.environ.setdefault("ASTAR_TOKEN", "dummy")

from api_client import AstarIslandClient
from prediction_builder import _compute_settlement_distance_map, _compute_ocean_distance_map
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from config import NUM_CLASSES, TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_NAMES

client = AstarIslandClient()

# Key: (terrain, fine_dist, coastal_tag)
gt_sums: dict[tuple[int, int, str], np.ndarray] = {}
gt_counts: dict[tuple[int, int, str], int] = {}

for round_name, round_id in KNOWN_ROUNDS.items():
    detail = client.get_round_detail(round_id)
    num_seeds = detail.get("seeds_count", 5)
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]
    analysis_cache: dict = {}

    for seed_idx in range(num_seeds):
        ig = initial_grids[seed_idx]
        gt, _ = load_or_fetch_gt(client, round_id, seed_idx, ig, analysis_cache)
        if gt is None:
            continue

        dist_map = _compute_settlement_distance_map(ig)
        ocean_dist_map = _compute_ocean_distance_map(ig)
        h, w = ig.shape

        for y in range(h):
            for x in range(w):
                terrain = int(ig[y, x])
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                d = int(dist_map[y, x])
                coastal_tag = "coastal" if ocean_dist_map[y, x] == 1 else "inland"
                key = (terrain, d, coastal_tag)
                if key not in gt_sums:
                    gt_sums[key] = np.zeros(NUM_CLASSES)
                    gt_counts[key] = 0
                gt_sums[key] += gt[y, x]
                gt_counts[key] += 1

# Print results for d3, d4, d5 specifically
print(f"\n{'Terrain':>12} {'Dist':>4} {'Coast':>8} {'Cells':>6}  [Empty, Settlement, Port, Ruin, Forest, Mountain]")
print("-" * 100)

for terrain_code in sorted(set(k[0] for k in gt_sums.keys())):
    t_name = TERRAIN_NAMES.get(terrain_code, str(terrain_code))
    for d in range(0, 12):
        for coastal_tag in ["coastal", "inland"]:
            key = (terrain_code, d, coastal_tag)
            if key not in gt_sums:
                continue
            count = gt_counts[key]
            if count < 5:
                continue
            avg = gt_sums[key] / count
            arr_str = np.array2string(avg, precision=4, separator=', ')
            print(f"{t_name:>12} d{d:>2} {coastal_tag:>8} {count:>6}  {arr_str}")
    print()

# Also print proposed bucket aggregates: d3 vs d4-5
print("\n\n=== PROPOSED SPLIT: d3 vs d4-5 ===")
print(f"{'Terrain':>12} {'Bucket':>6} {'Coast':>8} {'Cells':>6}  [Empty, Settlement, Port, Ruin, Forest, Mountain]")
print("-" * 100)

for terrain_code in sorted(set(k[0] for k in gt_sums.keys())):
    t_name = TERRAIN_NAMES.get(terrain_code, str(terrain_code))
    for bucket_name, dists in [("d3", [3]), ("d4-5", [4, 5])]:
        for coastal_tag in ["coastal", "inland"]:
            total_sum = np.zeros(NUM_CLASSES)
            total_count = 0
            for d in dists:
                key = (terrain_code, d, coastal_tag)
                if key in gt_sums:
                    total_sum += gt_sums[key]
                    total_count += gt_counts[key]
            if total_count < 5:
                continue
            avg = total_sum / total_count
            arr_str = np.array2string(avg, precision=4, separator=', ')
            print(f"{t_name:>12} {bucket_name:>6} {coastal_tag:>8} {total_count:>6}  {arr_str}")
    print()

# Compute KL between d3 and d4-5 for each terrain/coastal combo
print("\n=== KL DIVERGENCE: d3 vs d4-5 ===")
for terrain_code in sorted(set(k[0] for k in gt_sums.keys())):
    t_name = TERRAIN_NAMES.get(terrain_code, str(terrain_code))
    for coastal_tag in ["coastal", "inland"]:
        d3_sum = np.zeros(NUM_CLASSES)
        d3_count = 0
        d45_sum = np.zeros(NUM_CLASSES)
        d45_count = 0
        for d in [3]:
            key = (terrain_code, d, coastal_tag)
            if key in gt_sums:
                d3_sum += gt_sums[key]
                d3_count += gt_counts[key]
        for d in [4, 5]:
            key = (terrain_code, d, coastal_tag)
            if key in gt_sums:
                d45_sum += gt_sums[key]
                d45_count += gt_counts[key]
        if d3_count < 5 or d45_count < 5:
            continue
        p = d3_sum / d3_count
        q = d45_sum / d45_count
        # Forward KL
        p_safe = np.maximum(p, 1e-8)
        q_safe = np.maximum(q, 1e-8)
        kl = np.sum(p_safe * np.log(p_safe / q_safe))
        print(f"  {t_name:>12} {coastal_tag:>8}: KL(d3||d4-5) = {kl:.5f}  (d3: {d3_count} cells, d4-5: {d45_count} cells)")
