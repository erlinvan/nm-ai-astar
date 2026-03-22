import numpy as np
import os
os.environ.setdefault("ASTAR_TOKEN", "dummy")

from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import (
    build_prediction, compute_round_priors, _compute_settlement_distance_map,
    _compute_ocean_distance_map,
)
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from config import NUM_CLASSES, TERRAIN_OCEAN, TERRAIN_MOUNTAIN, CLASS_NAMES

def per_class_kl(gt_cell, pred_cell):
    eps = 1e-15
    result = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        g = max(gt_cell[i], eps)
        p = max(pred_cell[i], eps)
        result[i] = g * np.log(g / p)
    return result

def entropy_cell(gt_cell):
    eps = 1e-15
    return -np.sum(np.maximum(gt_cell, eps) * np.log(np.maximum(gt_cell, eps)))

client = AstarIslandClient()

for round_name in ["17", "15", "13"]:
    round_id = KNOWN_ROUNDS[round_name]
    detail = client.get_round_detail(round_id)
    num_seeds = detail.get("seeds_count", 5)
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

    obs_store = ObservationStore.load(round_id)
    if obs_store is None:
        continue
    round_priors = compute_round_priors(obs_store, initial_grids, num_seeds)

    analysis_cache: dict = {}

    class_contrib = np.zeros(NUM_CLASSES)
    total_entropy_sum = 0.0

    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        if gt is None:
            continue
        pred = build_prediction(seed_idx, obs_store, ig, round_priors=round_priors)
        h, w = ig.shape

        for y in range(h):
            for x in range(w):
                terrain = int(ig[y, x])
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                ent = entropy_cell(gt[y, x])
                if ent < 1e-8:
                    continue
                ckl = per_class_kl(gt[y, x], pred[y, x])
                class_contrib += ckl * ent
                total_entropy_sum += ent

    print(f"\nRound {round_name} — Per-class contribution to weighted KL:")
    overall = class_contrib.sum() / total_entropy_sum
    for i in range(NUM_CLASSES):
        c = class_contrib[i] / total_entropy_sum
        pct = 100 * class_contrib[i] / class_contrib.sum()
        print(f"  {CLASS_NAMES[i]:>12}: weighted_kl_contrib={c:.5f} ({pct:.1f}%)")
    print(f"  {'TOTAL':>12}: weighted_kl={overall:.5f}, score={100*np.exp(-3*overall):.2f}")
