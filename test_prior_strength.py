import numpy as np
import os
os.environ.setdefault("ASTAR_TOKEN", "dummy")

from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, compute_round_priors
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from utils import score_prediction
import prediction_builder as pb

client = AstarIslandClient()

orig_fn = pb._adaptive_prior_strength

configs = [
    ("fixed 100", lambda sd, oc: 100.0),
    ("fixed 50", lambda sd, oc: 50.0),
    ("max(30, 100-10*obs)", lambda sd, oc: max(30.0, 100.0 - 10.0 * oc)),
    ("max(50, 150-20*obs)", lambda sd, oc: max(50.0, 150.0 - 20.0 * oc)),
    ("100/sqrt(obs+1)", lambda sd, oc: 100.0 / np.sqrt(oc + 1)),
    ("200/sqrt(obs+1)", lambda sd, oc: 200.0 / np.sqrt(oc + 1)),
    ("50+50/(obs+1)", lambda sd, oc: 50.0 + 50.0 / (oc + 1)),
]

for name, fn in configs:
    pb._adaptive_prior_strength = fn

    scores = []
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

        round_scores = []
        for seed_idx in range(num_seeds):
            gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
            if gt is None:
                continue
            pred = build_prediction(seed_idx, obs_store, ig, round_priors=round_priors)
            s = score_prediction(gt, pred)
            round_scores.append(s)
        avg = np.mean(round_scores)
        scores.append((round_name, avg))

    parts = " | ".join(f"R{n}={s:.2f}" for n, s in scores)
    overall = np.mean([s for _, s in scores])
    print(f"{name:>25}: {parts} | avg={overall:.2f}")

pb._adaptive_prior_strength = orig_fn
