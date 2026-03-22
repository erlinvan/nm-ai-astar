"""Diagnose: how much KL loss comes from observed vs unobserved cells?"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import TERRAIN_OCEAN, TERRAIN_MOUNTAIN, NUM_CLASSES
from observation_store import ObservationStore
from prediction_builder import build_prediction, compute_round_priors
from utils import compute_kl_divergence, compute_entropy

KNOWN_ROUNDS = {
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    17: "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9",
}

for rnd, round_id in sorted(KNOWN_ROUNDS.items()):
    gt_path = f".gt_cache/{round_id}.npz"
    obs_path = f".observations/{round_id}.npz"
    if not os.path.exists(gt_path) or not os.path.exists(obs_path):
        continue

    gt_data = np.load(gt_path)
    ground_truth = gt_data["ground_truth"]
    initial_grids = gt_data["initial_grid"]
    if initial_grids.ndim == 2:
        initial_grids = np.stack([initial_grids] * 5)

    store = ObservationStore(40, 40, 5)
    obs_data = np.load(obs_path)
    store.class_counts = obs_data["class_counts"]
    store.observation_count = obs_data["observation_count"]

    round_priors = compute_round_priors(store, list(initial_grids), 5)

    print(f"\n=== Round {rnd} ===")
    seed = 0
    ig = initial_grids[seed]
    pred = build_prediction(seed, store, ig, round_priors=round_priors)
    oc = store.observation_count[seed]  # (40,40)

    obs_kl_sum = 0.0; obs_entropy_sum = 0.0; obs_cells = 0
    unobs_kl_sum = 0.0; unobs_entropy_sum = 0.0; unobs_cells = 0

    for y in range(40):
        for x in range(40):
            terrain = ig[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue
            gt_cell = ground_truth[y, x]
            ent = compute_entropy(gt_cell)
            if ent < 1e-8:
                continue
            kl = compute_kl_divergence(gt_cell, pred[y, x])

            if oc[y, x] > 0:
                obs_kl_sum += ent * kl
                obs_entropy_sum += ent
                obs_cells += 1
            else:
                unobs_kl_sum += ent * kl
                unobs_entropy_sum += ent
                unobs_cells += 1

    total_entropy = obs_entropy_sum + unobs_entropy_sum
    obs_pct = obs_cells / max(1, obs_cells + unobs_cells) * 100
    print(f"  Coverage: {obs_cells} observed, {unobs_cells} unobserved ({obs_pct:.1f}%)")

    obs_wkl = obs_kl_sum / total_entropy if total_entropy > 0 else 0
    unobs_wkl = unobs_kl_sum / total_entropy if total_entropy > 0 else 0
    total_wkl = (obs_kl_sum + unobs_kl_sum) / total_entropy if total_entropy > 0 else 0

    obs_loss = 100 * (1 - np.exp(-3 * obs_wkl))
    unobs_loss = 100 * (1 - np.exp(-3 * unobs_wkl))

    print(f"  Observed:   wkl={obs_wkl:.6f}  score_loss={obs_loss:.2f} pts  ({obs_kl_sum/max(1e-9,obs_kl_sum+unobs_kl_sum)*100:.1f}% of total KL)")
    print(f"  Unobserved: wkl={unobs_wkl:.6f}  score_loss={unobs_loss:.2f} pts  ({unobs_kl_sum/max(1e-9,obs_kl_sum+unobs_kl_sum)*100:.1f}% of total KL)")
    print(f"  Total:      wkl={total_wkl:.6f}  score={100*np.exp(-3*total_wkl):.2f}")

    # Break down observed by obs count
    print(f"\n  By observation count:")
    for lo, hi, label in [(0,0,"0 (unobs)"), (1,1,"1 obs"), (2,3,"2-3 obs"), (4,10,"4-10 obs"), (11,999,"11+ obs")]:
        bkl = 0.0; bent = 0.0; bn = 0
        for y in range(40):
            for x in range(40):
                terrain = ig[y, x]
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                c = oc[y, x]
                if c < lo or c > hi:
                    continue
                gt_cell = ground_truth[y, x]
                ent = compute_entropy(gt_cell)
                if ent < 1e-8:
                    continue
                kl = compute_kl_divergence(gt_cell, pred[y, x])
                bkl += ent * kl
                bent += ent
                bn += 1
        if bn > 0:
            contrib = bkl / total_entropy
            avg_kl = bkl / bent if bent > 0 else 0
            print(f"    {label:>10}: {bn:4d} cells, wkl_contrib={contrib:.6f}, avg_kl={avg_kl:.6f}")
