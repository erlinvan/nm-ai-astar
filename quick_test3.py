"""Test: pure prior (no observations) vs per-seed with different ps values."""
import numpy as np
from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, _apply_floor_and_normalize
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from utils import score_prediction
import prediction_builder

client = AstarIslandClient()

for round_name, round_id in [("15", KNOWN_ROUNDS["15"]), ("13", KNOWN_ROUNDS["13"])]:
    print(f"\n{'='*60}")
    print(f"ROUND {round_name}")
    print(f"{'='*60}")

    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

    analysis_cache = {}
    gts = []
    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        gts.append(gt)

    # Pure prior — empty store, no observations at all
    empty_store = ObservationStore(num_seeds, 40, 40)
    prediction_builder._adaptive_prior_strength = lambda sd, oc: 50.0
    scores_pure = []
    for seed_idx in range(num_seeds):
        pred = build_prediction(seed_idx, empty_store, initial_grids[seed_idx])
        score = score_prediction(gts[seed_idx], pred)
        scores_pure.append(score)
    print(f"  PURE PRIOR (no obs):  avg={np.mean(scores_pure):6.2f} [{' '.join(f'{s:5.1f}' for s in scores_pure)}]")

    # Per-seed with ps=50, 1 obs
    store = ObservationStore.load(round_id)
    sub_store = ObservationStore(num_seeds, store.width, store.height)
    for s in range(num_seeds):
        for y in range(store.height):
            for x in range(store.width):
                total = store.observation_count[s, y, x]
                if total == 0:
                    continue
                dist = store.class_counts[s, y, x] / total
                chosen_class = np.random.choice(6, p=dist)
                sub_store.class_counts[s, y, x, chosen_class] = 1
                sub_store.observation_count[s, y, x] = 1

    for ps in [5, 10, 20, 30, 50]:
        prediction_builder._adaptive_prior_strength = lambda sd, oc, _ps=ps: _ps
        scores = []
        for seed_idx in range(num_seeds):
            pred = build_prediction(seed_idx, sub_store, initial_grids[seed_idx])
            score = score_prediction(gts[seed_idx], pred)
            scores.append(score)
        print(f"  Per-seed ps={ps:3d} (1obs):  avg={np.mean(scores):6.2f} [{' '.join(f'{s:5.1f}' for s in scores)}]")

prediction_builder._adaptive_prior_strength = lambda sd, oc: 3.0
print("\nDone.")
