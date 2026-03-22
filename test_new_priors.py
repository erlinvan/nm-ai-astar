import numpy as np
from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import (
    build_prediction, _gt_calibrated_prior, _compute_settlement_distance_map,
    _apply_floor_and_normalize,
)
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from utils import score_prediction
from config import *
import prediction_builder

client = AstarIslandClient()

gt_by_key = {}
for round_name, round_id in KNOWN_ROUNDS.items():
    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]
    analysis_cache = {}
    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        if gt is None:
            continue
        dist_map = _compute_settlement_distance_map(ig)
        for y in range(gt.shape[0]):
            for x in range(gt.shape[1]):
                terrain = ig[y, x]
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                sd = dist_map[y, x]
                if sd <= 2:
                    db = "d0-2"
                elif sd <= 5:
                    db = "d3-5"
                else:
                    db = "d6+"
                key = (terrain, db)
                if key not in gt_by_key:
                    gt_by_key[key] = []
                gt_by_key[key].append(gt[y, x])

new_priors = {}
for key in gt_by_key:
    samples = np.array(gt_by_key[key])
    p = samples.mean(axis=0)
    p = np.maximum(p, 0.001)
    new_priors[key] = p / p.sum()

original_fn = prediction_builder._gt_calibrated_prior

def new_gt_prior(terrain, settlement_dist):
    if settlement_dist <= 2:
        db = "d0-2"
    elif settlement_dist <= 5:
        db = "d3-5"
    else:
        db = "d6+"
    key = (terrain, db)
    if key in new_priors:
        return new_priors[key].copy()
    return original_fn(terrain, settlement_dist)

PRIOR_STRENGTHS = [3, 5, 10, 20, 30, 50, 100]

for round_name, round_id in KNOWN_ROUNDS.items():
    print(f"\n{'='*60}")
    print(f"ROUND {round_name}")
    print(f"{'='*60}")

    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

    try:
        store = ObservationStore.load(round_id)
    except FileNotFoundError:
        print("  No observations")
        continue

    analysis_cache = {}
    gts = []
    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        gts.append(gt)

    if any(gt is None for gt in gts):
        print("  Missing GT")
        continue

    for prior_label, prior_fn in [("OLD", original_fn), ("NEW", new_gt_prior)]:
        prediction_builder._gt_calibrated_prior = prior_fn
        print(f"\n  {prior_label} priors:")
        for ps in PRIOR_STRENGTHS:
            prediction_builder._adaptive_prior_strength = lambda sd, oc, _ps=ps: _ps
            store.aggregated_counts = None
            store.aggregated_obs_count = None
            scores = []
            for seed_idx in range(num_seeds):
                pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
                scores.append(score_prediction(gts[seed_idx], pred))
            avg = np.mean(scores)
            print(f"    ps={ps:3d}: avg={avg:6.2f} [{' '.join(f'{s:5.1f}' for s in scores)}]")

prediction_builder._gt_calibrated_prior = original_fn
prediction_builder._adaptive_prior_strength = lambda sd, oc: 3.0
print("\nDone.")
