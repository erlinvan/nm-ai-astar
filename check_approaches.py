"""Check what our current R17 submission looks like vs new approach."""
import numpy as np
from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, build_prediction_with_mc, _apply_floor_and_normalize
from monte_carlo import run_monte_carlo
from world_builder import build_world_from_state, calibrate_settlements_from_observations
from simulator.params import SimParams
from submission import _validate_prediction
from config import PROBABILITY_FLOOR
import prediction_builder

ROUND_ID = "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9"

client = AstarIslandClient()
detail = client.get_round_detail(ROUND_ID)
num_seeds = detail["seeds_count"]
width, height = detail["map_width"], detail["map_height"]
initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

store = ObservationStore.load(ROUND_ID)

# APPROACH 1: Replicate current submission (aggregated, old ps, MC blend)
# The original submission used the old code: ps = max(100, 1000/(obs+0.5))
# With 7 total obs (aggregated across 5 seeds): ps = max(100, 1000/7.5) = max(100, 133) = 133
# Actually with agg: total_obs per cell ≈ 7 (5 seeds * 1.4), so ps ≈ max(100, 133) = 133
print("APPROACH 1: Simulated original submission (agg, ps~133, mc_weight=0.08)")
prediction_builder._adaptive_prior_strength = lambda sd, oc: max(100.0, 1000.0 / (oc + 0.5))
store.aggregate_across_seeds()
for seed_idx in range(num_seeds):
    pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
    max_p = pred.max(axis=-1).mean()
    print(f"  Seed {seed_idx}: mean_max_prob={max_p:.4f}")

# APPROACH 2: New optimal (per-seed, ps=30, no MC)
print("\nAPPROACH 2: New optimal (per-seed, ps=30, no MC)")
prediction_builder._adaptive_prior_strength = lambda sd, oc: 30.0
store.aggregated_counts = None
store.aggregated_obs_count = None
for seed_idx in range(num_seeds):
    pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
    max_p = pred.max(axis=-1).mean()
    print(f"  Seed {seed_idx}: mean_max_prob={max_p:.4f}")

# APPROACH 3: Per-seed, ps=30, WITH MC (to see if MC helps)
print("\nAPPROACH 3: Per-seed, ps=30, with MC (mc_weight=0.08)")
mc_preds = []
params = SimParams.gt_tuned()
for seed_idx in range(num_seeds):
    world = build_world_from_state(detail["initial_states"][seed_idx], width, height, rng_seed=seed_idx)
    calibrate_settlements_from_observations(world, store, seed_idx)
    mc_pred = run_monte_carlo(world, params, num_runs=15, years=50, base_seed=seed_idx*10000, param_noise=0.15)
    _apply_floor_and_normalize(mc_pred, PROBABILITY_FLOOR)
    mc_preds.append(mc_pred)
    print(f"  Seed {seed_idx}: MC done")

for seed_idx in range(num_seeds):
    pred = build_prediction_with_mc(seed_idx, store, initial_grids[seed_idx], mc_preds[seed_idx], mc_weight=0.08)
    max_p = pred.max(axis=-1).mean()
    print(f"  Seed {seed_idx}: mean_max_prob={max_p:.4f}")

prediction_builder._adaptive_prior_strength = lambda sd, oc: 3.0
print("\nDone.")
