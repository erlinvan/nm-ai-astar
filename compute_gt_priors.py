import numpy as np
import os
os.environ.setdefault("ASTAR_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3YjY3NWJlYy1mZDhkLTQ5YmEtOTNmNy1kYjRmNTA1NDNjODQiLCJlbWFpbCI6Im1ybGlpbmcxMDFAZ21haWwuY29tIiwiaXNfYWRtaW4iOmZhbHNlLCJleHAiOjE3NzQ2NDExMjJ9.BvhwRzzc_tZDisQhzIUtwHPw-GIgLLZ0L80t9DYX-3I")

from api_client import AstarIslandClient
from prediction_builder import _compute_settlement_distance_map, _dist_bucket
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from config import NUM_CLASSES, TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_NAMES, CLASS_NAMES

def fine_bucket(d):
    if d <= 0:
        return "d0"
    elif d <= 1:
        return "d1"
    elif d <= 2:
        return "d2"
    elif d <= 3:
        return "d3"
    elif d <= 5:
        return "d4-5"
    elif d <= 8:
        return "d6-8"
    return "d9+"

client = AstarIslandClient()

gt_sums: dict[tuple[int, str], np.ndarray] = {}
gt_counts: dict[tuple[int, str], int] = {}

for round_name, round_id in [("R13", KNOWN_ROUNDS["13"]), ("R15", KNOWN_ROUNDS["15"])]:
    detail = client.get_round_detail(round_id)
    num_seeds = detail.get("seeds_count", 5)
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]
    analysis_cache = {}

    for seed_idx in range(num_seeds):
        ig = initial_grids[seed_idx]
        gt, _ = load_or_fetch_gt(client, round_id, seed_idx, ig, analysis_cache)
        if gt is None:
            continue
        
        dist_map = _compute_settlement_distance_map(ig)
        h, w = ig.shape
        
        for y in range(h):
            for x in range(w):
                terrain = int(ig[y, x])
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                bucket = _dist_bucket(dist_map[y, x])
                key = (terrain, bucket)
                if key not in gt_sums:
                    gt_sums[key] = np.zeros(NUM_CLASSES)
                    gt_counts[key] = 0
                gt_sums[key] += gt[y, x]
                gt_counts[key] += 1

                fb = fine_bucket(dist_map[y, x])
                fine_key = (terrain, fb)
                if fine_key not in gt_sums:
                    gt_sums[fine_key] = np.zeros(NUM_CLASSES)
                    gt_counts[fine_key] = 0
                gt_sums[fine_key] += gt[y, x]
                gt_counts[fine_key] += 1

print(f"\nGT-derived optimal priors from R13+R15 (all seeds):")
print(f"{'Terrain':>12} {'Bucket':>6} {'Cells':>6}  [Empty, Settlement, Port, Ruin, Forest, Mountain]")
print("-" * 90)

for key in sorted(gt_sums.keys(), key=lambda k: (k[0], k[1])):
    terrain, bucket = key
    t_name = TERRAIN_NAMES.get(terrain, str(terrain))
    avg = gt_sums[key] / gt_counts[key]
    count = gt_counts[key]
    arr_str = np.array2string(avg, precision=4, separator=', ')
    print(f"{t_name:>12} {bucket:>6} {count:>6}  {arr_str}")

print("\nAs Python code for _gt_calibrated_prior():")
for terrain_code in sorted(set(k[0] for k in gt_sums.keys())):
    t_name = TERRAIN_NAMES.get(terrain_code, str(terrain_code))
    print(f"\n    # {t_name} (terrain={terrain_code})")
    for bucket in ["d0-2", "d3-5", "d6+"]:
        key = (terrain_code, bucket)
        if key in gt_sums:
            avg = gt_sums[key] / gt_counts[key]
            avg[-1] = 0.001  # Mountain = 0.001
            avg = avg / avg.sum()
            vals = ", ".join(f"{v:.3f}" for v in avg)
            print(f"    # {bucket}: np.array([{vals}])  ({gt_counts[key]} cells)")
