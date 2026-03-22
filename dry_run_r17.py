"""Dry run for R17: generate predictions with optimal settings and show what we'd submit."""
import numpy as np
from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, build_prediction_with_mc, _apply_floor_and_normalize
from submission import _validate_prediction
import prediction_builder

ROUND_ID = "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9"

client = AstarIslandClient()
detail = client.get_round_detail(ROUND_ID)
num_seeds = detail["seeds_count"]
initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

store = ObservationStore.load(ROUND_ID)
print(f"Loaded {store.total_observations()} observations")
for s in range(num_seeds):
    print(f"  Seed {s}: {store.coverage_ratio(s):.1%}, avg {store.observation_count[s][store.observation_count[s]>0].mean():.1f} obs/cell")

prediction_builder._adaptive_prior_strength = lambda sd, oc: 30.0

# Per-seed, no aggregation
store.aggregated_counts = None
store.aggregated_obs_count = None

predictions = []
for seed_idx in range(num_seeds):
    pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
    _validate_prediction(pred)
    predictions.append(pred)
    
    max_prob = pred.max(axis=-1)
    print(f"  Seed {seed_idx}: min_prob={pred.min():.4f} max_prob={max_prob.max():.4f} mean_max={max_prob.mean():.4f}")

print(f"\nPredictions ready. Shape: {predictions[0].shape}")
print(f"Settings: per-seed (no aggregation), prior_strength=30, no MC")

# Compare with pure prior baseline
empty_store = ObservationStore(num_seeds, 40, 40)
for seed_idx in range(num_seeds):
    pred_obs = predictions[seed_idx]
    pred_prior = build_prediction(seed_idx, empty_store, initial_grids[seed_idx])
    
    diff = np.abs(pred_obs - pred_prior).mean()
    max_diff = np.abs(pred_obs - pred_prior).max()
    print(f"  Seed {seed_idx}: avg diff from pure prior = {diff:.6f}, max diff = {max_diff:.4f}")

prediction_builder._adaptive_prior_strength = lambda sd, oc: 3.0
print("\nDry run complete. Ready to submit if approved.")
