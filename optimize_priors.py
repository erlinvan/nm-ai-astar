import numpy as np
from api_client import AstarIslandClient
from prediction_builder import _compute_settlement_distance_map, _gt_calibrated_prior
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from utils import score_prediction, compute_kl_divergence, compute_entropy
from config import *
import prediction_builder

client = AstarIslandClient()

gt_by_key = {}
terrain_names = {0: "Empty", 11: "Plains", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain", 10: "Ocean"}

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

print("=" * 70)
print("OPTIMAL PRIORS (computed from all GT data)")
print("=" * 70)
print()

for key in sorted(gt_by_key.keys()):
    terrain, db = key
    if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
        continue
    samples = np.array(gt_by_key[key])
    avg = samples.mean(axis=0)
    
    tname = terrain_names.get(terrain, f"T{terrain}")
    arr_str = ", ".join(f"{v:.3f}" for v in avg)
    print(f"  {tname:12s} {db:5s} (n={len(samples):5d}): np.array([{arr_str}])")

print()
print("=" * 70)
print("Now testing: updated priors vs old priors")
print("=" * 70)

new_priors = {}
for key in gt_by_key:
    terrain, db = key
    samples = np.array(gt_by_key[key])
    new_priors[key] = samples.mean(axis=0)

def new_gt_calibrated_prior(terrain, settlement_dist):
    if settlement_dist <= 2:
        db = "d0-2"
    elif settlement_dist <= 5:
        db = "d3-5"
    else:
        db = "d6+"
    key = (terrain, db)
    if key in new_priors:
        p = new_priors[key].copy()
        p = np.maximum(p, 0.001)
        return p / p.sum()
    return _gt_calibrated_prior(terrain, settlement_dist)

old_prior_fn = prediction_builder._gt_calibrated_prior

for round_name, round_id in KNOWN_ROUNDS.items():
    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]
    analysis_cache = {}

    gts = []
    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        gts.append(gt)

    prediction_builder._adaptive_prior_strength = lambda sd, oc: 50.0

    old_scores = []
    new_scores = []
    for seed_idx in range(num_seeds):
        if gts[seed_idx] is None:
            continue

        prediction_builder._gt_calibrated_prior = old_prior_fn
        from observation_store import ObservationStore
        empty_store = ObservationStore(num_seeds, 40, 40)
        pred_old = prediction_builder.build_prediction(seed_idx, empty_store, initial_grids[seed_idx])
        old_scores.append(score_prediction(gts[seed_idx], pred_old))

        prediction_builder._gt_calibrated_prior = new_gt_calibrated_prior
        pred_new = prediction_builder.build_prediction(seed_idx, empty_store, initial_grids[seed_idx])
        new_scores.append(score_prediction(gts[seed_idx], pred_new))

    print(f"\n  Round {round_name}: OLD avg={np.mean(old_scores):.2f}  NEW avg={np.mean(new_scores):.2f}  delta={np.mean(new_scores)-np.mean(old_scores):+.2f}")
    for i, (o, n) in enumerate(zip(old_scores, new_scores)):
        print(f"    Seed {i}: old={o:.2f} new={n:.2f} delta={n-o:+.2f}")

prediction_builder._gt_calibrated_prior = old_prior_fn
prediction_builder._adaptive_prior_strength = lambda sd, oc: 3.0
print("\nDone.")
