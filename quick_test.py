"""Quick test: aggregated vs per-seed predictions at various prior strengths."""
import numpy as np
from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, _apply_floor_and_normalize, build_prediction_with_mc
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from utils import score_prediction
import prediction_builder

# Rounds with GT cached
TEST_ROUNDS = {
    "15": KNOWN_ROUNDS["15"],
    "13": KNOWN_ROUNDS["13"],
}

PRIOR_STRENGTHS = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0]

client = AstarIslandClient()

for round_name, round_id in TEST_ROUNDS.items():
    print(f"\n{'='*60}")
    print(f"ROUND {round_name} (ID: {round_id[:12]}...)")
    print(f"{'='*60}")

    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

    # Load observations
    try:
        store = ObservationStore.load(round_id)
    except FileNotFoundError:
        print(f"  No observations cached for round {round_name}")
        continue

    # Load GT for all seeds
    analysis_cache = {}
    gts = []
    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        gts.append(gt)

    if any(gt is None for gt in gts):
        print(f"  Missing GT for some seeds, skipping")
        continue

    for ps in PRIOR_STRENGTHS:
        # Monkey-patch prior strength
        prediction_builder._adaptive_prior_strength = lambda sd, oc, _ps=ps: _ps

        # --- AGGREGATED (current approach) ---
        store.aggregate_across_seeds()
        agg_scores = []
        for seed_idx in range(num_seeds):
            pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
            score = score_prediction(gts[seed_idx], pred)
            agg_scores.append(score)

        # --- PER-SEED (no aggregation) ---
        store.aggregated_counts = None
        store.aggregated_obs_count = None
        per_seed_scores = []
        for seed_idx in range(num_seeds):
            pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
            score = score_prediction(gts[seed_idx], pred)
            per_seed_scores.append(score)

        avg_agg = np.mean(agg_scores)
        avg_ps = np.mean(per_seed_scores)
        print(f"  ps={ps:5.1f}  AGG avg={avg_agg:6.2f} [{' '.join(f'{s:5.1f}' for s in agg_scores)}]  "
              f"PER-SEED avg={avg_ps:6.2f} [{' '.join(f'{s:5.1f}' for s in per_seed_scores)}]  "
              f"{'*** PER-SEED WINS' if avg_ps > avg_agg else ''}")

    # Restore
    prediction_builder._adaptive_prior_strength = lambda sd, oc: 3.0

print("\nDone.")
