"""Test: simulate R17 conditions (1 obs/cell per seed) on R15/R13 to find optimal prior strength."""
import numpy as np
from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, _apply_floor_and_normalize
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from utils import score_prediction
import prediction_builder

PRIOR_STRENGTHS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]

client = AstarIslandClient()

for round_name, round_id in [("15", KNOWN_ROUNDS["15"]), ("13", KNOWN_ROUNDS["13"])]:
    print(f"\n{'='*60}")
    print(f"ROUND {round_name} — Simulated 1 obs/cell per seed (no aggregation)")
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

    if any(gt is None for gt in gts):
        print(f"  Missing GT")
        continue

    # Create a subsampled store with 1 obs/cell per seed
    sub_store = ObservationStore(num_seeds, store.width, store.height)
    for s in range(num_seeds):
        for y in range(store.height):
            for x in range(store.width):
                total = store.observation_count[s, y, x]
                if total == 0:
                    continue
                # Pick a random single observation (simulate having 1 obs)
                dist = store.class_counts[s, y, x] / total
                chosen_class = np.random.choice(6, p=dist)
                sub_store.class_counts[s, y, x, chosen_class] = 1
                sub_store.observation_count[s, y, x] = 1

    # Also test with ~1.4 obs/cell (like R17 with overlap)
    sub14_store = ObservationStore(num_seeds, store.width, store.height)
    for s in range(num_seeds):
        for y in range(store.height):
            for x in range(store.width):
                total = store.observation_count[s, y, x]
                if total == 0:
                    continue
                dist = store.class_counts[s, y, x] / total
                # 40% of cells get 2 obs, 60% get 1
                n_obs = 2 if np.random.random() < 0.4 else 1
                for _ in range(n_obs):
                    chosen_class = np.random.choice(6, p=dist)
                    sub14_store.class_counts[s, y, x, chosen_class] += 1
                    sub14_store.observation_count[s, y, x] += 1

    for label, test_store in [("1.0 obs/cell", sub_store), ("1.4 obs/cell", sub14_store)]:
        print(f"\n  --- {label} ---")
        for ps in PRIOR_STRENGTHS:
            prediction_builder._adaptive_prior_strength = lambda sd, oc, _ps=ps: _ps
            test_store.aggregated_counts = None
            test_store.aggregated_obs_count = None

            scores = []
            for seed_idx in range(num_seeds):
                pred = build_prediction(seed_idx, test_store, initial_grids[seed_idx])
                score = score_prediction(gts[seed_idx], pred)
                scores.append(score)

            avg = np.mean(scores)
            print(f"    ps={ps:5.1f}  avg={avg:6.2f} [{' '.join(f'{s:5.1f}' for s in scores)}]")

    # Also test full 7 obs/cell per-seed for reference
    print(f"\n  --- Full ~7 obs/cell (per-seed, no agg) ---")
    for ps in [10.0, 30.0, 50.0, 100.0]:
        prediction_builder._adaptive_prior_strength = lambda sd, oc, _ps=ps: _ps
        store.aggregated_counts = None
        store.aggregated_obs_count = None

        scores = []
        for seed_idx in range(num_seeds):
            pred = build_prediction(seed_idx, store, initial_grids[seed_idx])
            score = score_prediction(gts[seed_idx], pred)
            scores.append(score)

        avg = np.mean(scores)
        print(f"    ps={ps:5.1f}  avg={avg:6.2f} [{' '.join(f'{s:5.1f}' for s in scores)}]")

prediction_builder._adaptive_prior_strength = lambda sd, oc: 3.0
print("\nDone.")
