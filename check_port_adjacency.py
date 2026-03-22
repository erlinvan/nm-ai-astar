import numpy as np
import os
os.environ.setdefault("ASTAR_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3YjY3NWJlYy1mZDhkLTQ5YmEtOTNmNy1kYjRmNTA1NDNjODQiLCJlbWFpbCI6Im1ybGlpbmcxMDFAZ21haWwuY29tIiwiaXNfYWRtaW4iOmZhbHNlLCJleHAiOjE3NzQ2NDExMjJ9.BvhwRzzc_tZDisQhzIUtwHPw-GIgLLZ0L80t9DYX-3I")

from api_client import AstarIslandClient
from prediction_builder import _compute_settlement_distance_map
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from config import NUM_CLASSES, TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_PORT, TERRAIN_RUIN, TERRAIN_SETTLEMENT, TERRAIN_NAMES

client = AstarIslandClient()

adj_port: dict[tuple[int, bool], list] = {}

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
                if dist_map[y, x] > 2:
                    continue
                
                has_port_neighbor = False
                has_settlement_neighbor = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            nt = int(ig[ny, nx])
                            if nt == TERRAIN_PORT:
                                has_port_neighbor = True
                            if nt == TERRAIN_SETTLEMENT:
                                has_settlement_neighbor = True
                
                key = (terrain, has_port_neighbor)
                if key not in adj_port:
                    adj_port[key] = []
                adj_port[key].append(gt[y, x])

print(f"\nPort class GT average by (terrain, adjacent_to_port_terrain):")
print(f"{'Terrain':>12} {'AdjPort':>8} {'Cells':>6} {'Port_class':>10} {'Full distribution':>50}")
print("-" * 100)

for key in sorted(adj_port.keys()):
    terrain, has_port = key
    t_name = TERRAIN_NAMES.get(terrain, str(terrain))
    gt_arr = np.array(adj_port[key])
    avg = gt_arr.mean(axis=0)
    count = len(adj_port[key])
    arr_str = np.array2string(avg, precision=4, separator=', ')
    print(f"{t_name:>12} {'Yes' if has_port else 'No':>8} {count:>6} {avg[2]:>10.4f} {arr_str}")
