import numpy as np
import os
os.environ.setdefault("ASTAR_TOKEN", "dummy")

from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import (
    build_prediction, compute_round_priors, _compute_settlement_distance_map,
    _compute_ocean_distance_map, _dist_bucket,
)
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from config import NUM_CLASSES, TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_NAMES, CLASS_NAMES

def kl_cell(gt, pred):
    eps = 1e-15
    return np.sum(np.maximum(gt, eps) * np.log(np.maximum(gt, eps) / np.maximum(pred, eps)))

client = AstarIslandClient()

round_id = KNOWN_ROUNDS["17"]
detail = client.get_round_detail(round_id)
num_seeds = detail.get("seeds_count", 5)
initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]
obs_store = ObservationStore.load(round_id)
round_priors = compute_round_priors(obs_store, initial_grids, num_seeds)
analysis_cache: dict = {}

print("R17 — Coastal cell analysis (worst 30 by weighted KL)")
print(f"{'Cell':>8} {'Terrain':>10} {'Bucket':>5} {'Obs':>4} {'KL':>8} {'GT':>55} {'Pred':>55}")
print("-" * 160)

worst = []
for seed_idx in range(1):
    gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
    if gt is None:
        continue
    pred = build_prediction(seed_idx, obs_store, ig, round_priors=round_priors)
    dist_map = _compute_settlement_distance_map(ig)
    ocean_dist_map = _compute_ocean_distance_map(ig)
    h, w = ig.shape

    for y in range(h):
        for x in range(w):
            terrain = int(ig[y, x])
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue
            if ocean_dist_map[y, x] != 1:
                continue
            ent = -np.sum(np.maximum(gt[y, x], 1e-15) * np.log(np.maximum(gt[y, x], 1e-15)))
            if ent < 1e-8:
                continue
            kl = kl_cell(gt[y, x], pred[y, x])
            obs = int(obs_store.observation_count[seed_idx, y, x])
            bucket = _dist_bucket(dist_map[y, x])
            t_name = TERRAIN_NAMES.get(terrain, str(terrain))
            worst.append((ent * kl, y, x, t_name, bucket, obs, kl, gt[y, x], pred[y, x]))

worst.sort(key=lambda r: -r[0])
for w_kl, y, x, t_name, bucket, obs, kl, gt_v, pred_v in worst[:30]:
    gt_str = " ".join(f"{v:.3f}" for v in gt_v)
    pred_str = " ".join(f"{v:.3f}" for v in pred_v)
    print(f"({y:2d},{x:2d}) {t_name:>10} {bucket:>5} {obs:>4} {kl:>8.4f} [{gt_str}] [{pred_str}]")

print("\n\nAvg GT vs Avg Pred for coastal Plains d0-2:")
gt_sum = np.zeros(NUM_CLASSES)
pred_sum = np.zeros(NUM_CLASSES)
count = 0
for seed_idx in range(num_seeds):
    gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
    if gt is None:
        continue
    pred = build_prediction(seed_idx, obs_store, ig, round_priors=round_priors)
    dist_map = _compute_settlement_distance_map(ig)
    ocean_dist_map = _compute_ocean_distance_map(ig)
    h, w = ig.shape
    for y in range(h):
        for x in range(w):
            terrain = int(ig[y, x])
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue
            if ocean_dist_map[y, x] != 1:
                continue
            bucket = _dist_bucket(dist_map[y, x])
            if bucket != "d0-2":
                continue
            if terrain not in (11,):
                continue
            gt_sum += gt[y, x]
            pred_sum += pred[y, x]
            count += 1

if count > 0:
    gt_avg = gt_sum / count
    pred_avg = pred_sum / count
    print(f"  Cells: {count}")
    for i in range(NUM_CLASSES):
        diff = pred_avg[i] - gt_avg[i]
        print(f"  {CLASS_NAMES[i]:>12}: GT={gt_avg[i]:.4f} Pred={pred_avg[i]:.4f} diff={diff:+.4f}")

print("\n\nAvg GT vs Avg Pred for coastal Forest d0-2:")
gt_sum = np.zeros(NUM_CLASSES)
pred_sum = np.zeros(NUM_CLASSES)
count = 0
for seed_idx in range(num_seeds):
    gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
    if gt is None:
        continue
    pred = build_prediction(seed_idx, obs_store, ig, round_priors=round_priors)
    dist_map = _compute_settlement_distance_map(ig)
    ocean_dist_map = _compute_ocean_distance_map(ig)
    h, w = ig.shape
    for y in range(h):
        for x in range(w):
            terrain = int(ig[y, x])
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue
            if ocean_dist_map[y, x] != 1:
                continue
            bucket = _dist_bucket(dist_map[y, x])
            if bucket != "d0-2":
                continue
            if terrain != 4:
                continue
            gt_sum += gt[y, x]
            pred_sum += pred[y, x]
            count += 1

if count > 0:
    gt_avg = gt_sum / count
    pred_avg = pred_sum / count
    print(f"  Cells: {count}")
    for i in range(NUM_CLASSES):
        diff = pred_avg[i] - gt_avg[i]
        print(f"  {CLASS_NAMES[i]:>12}: GT={gt_avg[i]:.4f} Pred={pred_avg[i]:.4f} diff={diff:+.4f}")
