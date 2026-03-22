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

original_fn = prediction_builder._gt_calibrated_prior

def make_observation_priors(store, initial_grids, num_seeds):
    """Build priors from the actual observations of the current round."""
    obs_by_key = {}
    for seed_idx in range(num_seeds):
        ig = initial_grids[seed_idx]
        dist_map = _compute_settlement_distance_map(ig)
        for y in range(store.height):
            for x in range(store.width):
                terrain = ig[y, x]
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                obs_count = store.observation_count[seed_idx, y, x]
                if obs_count == 0:
                    continue
                sd = dist_map[y, x]
                if sd <= 2:
                    db = "d0-2"
                elif sd <= 5:
                    db = "d3-5"
                else:
                    db = "d6+"
                key = (terrain, db)
                if key not in obs_by_key:
                    obs_by_key[key] = np.zeros(NUM_CLASSES)
                obs_by_key[key] += store.class_counts[seed_idx, y, x]
    
    priors = {}
    for key, counts in obs_by_key.items():
        total = counts.sum()
        if total > 0:
            p = counts / total
            p = np.maximum(p, 0.001)
            priors[key] = p / p.sum()
    return priors

def make_prior_fn(priors):
    def fn(terrain, settlement_dist):
        if settlement_dist <= 2:
            db = "d0-2"
        elif settlement_dist <= 5:
            db = "d3-5"
        else:
            db = "d6+"
        key = (terrain, db)
        if key in priors:
            return priors[key].copy()
        return original_fn(terrain, settlement_dist)
    return fn

gt_all = {}
for round_name, round_id in KNOWN_ROUNDS.items():
    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]
    analysis_cache = {}
    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        if gt is None:
            continue
        samples = np.array([gt[y, x] for y in range(gt.shape[0]) for x in range(gt.shape[1])])
    gt_all[round_name] = {}

# Now test "observation-based priors" (like using R17 obs as the prior for R17)
# Simulate on R13 and R15: use that round's own observations as the prior
for round_name, round_id in [("13", KNOWN_ROUNDS["13"]), ("15", KNOWN_ROUNDS["15"])]:
    print(f"\n{'='*60}")
    print(f"ROUND {round_name} — observation-derived priors")
    print(f"{'='*60}")

    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

    store = ObservationStore.load(round_id)

    analysis_cache = {}
    gts = []
    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        gts.append(gt)

    obs_priors = make_observation_priors(store, initial_grids, num_seeds)
    obs_prior_fn = make_prior_fn(obs_priors)

    print("\n  Observation-derived priors for this round:")
    for key in sorted(obs_priors.keys()):
        terrain, db = key
        terrain_names = {0: 'Empty', 1: 'Settlement', 2: 'Port', 3: 'Ruin', 4: 'Forest', 11: 'Plains'}
        tname = terrain_names.get(terrain, f'T{terrain}')
        print(f"    {tname:12s} {db}: [{' '.join(f'{v:.3f}' for v in obs_priors[key])}]")

    for prior_label, prior_fn in [("OLD", original_fn), ("OBS-DERIVED", obs_prior_fn)]:
        prediction_builder._gt_calibrated_prior = prior_fn
        print(f"\n  {prior_label} priors (per-seed, no agg):")
        for ps in [5, 10, 20, 30, 50, 100]:
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
