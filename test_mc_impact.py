"""Test MC blending impact on per-seed predictions at optimal ps."""
import numpy as np
from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, build_prediction_with_mc, _apply_floor_and_normalize
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from monte_carlo import run_monte_carlo
from world_builder import build_world_from_state, calibrate_settlements_from_observations
from simulator.params import SimParams
from utils import score_prediction
from config import PROBABILITY_FLOOR
import prediction_builder

client = AstarIslandClient()
round_id = KNOWN_ROUNDS["13"]
detail = client.get_round_detail(round_id)
num_seeds = detail["seeds_count"]
width, height = detail["map_width"], detail["map_height"]
initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

store = ObservationStore.load(round_id)
store.aggregated_counts = None
store.aggregated_obs_count = None

analysis_cache = {}
gts = []
for seed_idx in range(num_seeds):
    gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
    gts.append(gt)

params = SimParams.gt_tuned()
mc_preds = []
for seed_idx in range(num_seeds):
    world = build_world_from_state(detail["initial_states"][seed_idx], width, height, rng_seed=seed_idx)
    calibrate_settlements_from_observations(world, store, seed_idx)
    mc_pred = run_monte_carlo(world, params, num_runs=15, years=50, base_seed=seed_idx*10000, param_noise=0.15)
    _apply_floor_and_normalize(mc_pred, PROBABILITY_FLOOR)
    mc_preds.append(mc_pred)
    print(f"MC seed {seed_idx} done")

for ps in [30.0, 50.0]:
    prediction_builder._adaptive_prior_strength = lambda sd, oc, _ps=ps: _ps
    
    scores_no_mc = []
    for seed_idx in range(num_seeds):
        pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
        scores_no_mc.append(score_prediction(gts[seed_idx], pred))
    
    for mc_w in [0.0, 0.04, 0.08, 0.15]:
        scores_mc = []
        for seed_idx in range(num_seeds):
            if mc_w == 0.0:
                pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
            else:
                pred = build_prediction_with_mc(seed_idx, store, initial_grids[seed_idx], mc_preds[seed_idx], mc_weight=mc_w)
            scores_mc.append(score_prediction(gts[seed_idx], pred))
        print(f"ps={ps:4.0f} mc_w={mc_w:.2f}: avg={np.mean(scores_mc):6.2f} [{' '.join(f'{s:5.1f}' for s in scores_mc)}]")

prediction_builder._adaptive_prior_strength = lambda sd, oc: 3.0
print("\nDone.")
