import numpy as np
import os
os.environ.setdefault("ASTAR_TOKEN", "dummy")

from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import (
    build_prediction, compute_round_priors, _compute_settlement_distance_map,
    _compute_ocean_distance_map, _dist_bucket,
)
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from config import NUM_CLASSES, TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_NAMES, CLASS_NAMES

def kl_per_cell(gt, pred):
    eps = 1e-15
    gt_safe = np.clip(gt, eps, 1.0)
    pred_safe = np.clip(pred, eps, 1.0)
    return np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=-1)

def entropy_per_cell(gt):
    eps = 1e-15
    gt_safe = np.clip(gt, eps, 1.0)
    return -np.sum(gt_safe * np.log(gt_safe), axis=-1)

client = AstarIslandClient()

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

    print(f"\n{'='*80}")
    print(f"ROUND {round_name} — KL breakdown by (terrain, bucket, coastal)")
    print(f"{'='*80}")

    agg_by_key: dict[tuple[str, str, str], list] = {}

    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        if gt is None:
            continue

        pred = build_prediction(seed_idx, obs_store, ig, round_priors=round_priors)
        kl = kl_per_cell(gt, pred)
        ent = entropy_per_cell(gt)
        dist_map = _compute_settlement_distance_map(ig)
        ocean_dist_map = _compute_ocean_distance_map(ig)
        h, w = ig.shape

        for y in range(h):
            for x in range(w):
                terrain = int(ig[y, x])
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue
                if ent[y, x] < 1e-8:
                    continue
                t_name = TERRAIN_NAMES.get(terrain, str(terrain))
                bucket = _dist_bucket(dist_map[y, x])
                coastal = "coastal" if ocean_dist_map[y, x] == 1 else "inland"
                key = (t_name, bucket, coastal)
                if key not in agg_by_key:
                    agg_by_key[key] = []
                agg_by_key[key].append((kl[y, x], ent[y, x]))

    print(f"\n{'Key':>35} {'Cells':>6} {'Mean KL':>9} {'Wtd KL':>9} {'Contribution%':>14} {'Score':>7}")
    print("-" * 90)

    total_weighted_kl = 0.0
    total_entropy = 0.0
    rows = []
    for key, vals in agg_by_key.items():
        kls = np.array([v[0] for v in vals])
        ents = np.array([v[1] for v in vals])
        weighted = (ents * kls).sum()
        ent_sum = ents.sum()
        total_weighted_kl += weighted
        total_entropy += ent_sum
        rows.append((key, len(vals), kls.mean(), weighted / ent_sum if ent_sum > 0 else 0, weighted))

    for key, count, mean_kl, w_kl, contrib in sorted(rows, key=lambda r: -r[4]):
        pct = 100 * contrib / total_weighted_kl if total_weighted_kl > 0 else 0
        key_str = f"{key[0]:>12} {key[1]:>4} {key[2]:>8}"
        print(f"{key_str:>35} {count:>6} {mean_kl:>9.5f} {w_kl:>9.5f} {pct:>13.1f}% {100*np.exp(-3*w_kl):>7.2f}")

    overall_wkl = total_weighted_kl / total_entropy if total_entropy > 0 else 0
    print(f"\n  TOTAL weighted_kl={overall_wkl:.5f}, score={100*np.exp(-3*overall_wkl):.2f}")

    print(f"\n  Top 10 worst individual cells (seed 0):")
    gt, ig = load_or_fetch_gt(client, round_id, 0, initial_grids[0], analysis_cache)
    pred = build_prediction(0, obs_store, ig, round_priors=round_priors)
    kl = kl_per_cell(gt, pred)
    ent = entropy_per_cell(gt)
    weighted = ent * kl
    flat = np.argsort(weighted.ravel())[::-1][:10]
    for idx in flat:
        y, x = divmod(idx, kl.shape[1])
        terrain = int(ig[y, x])
        t_name = TERRAIN_NAMES.get(terrain, str(terrain))
        obs_count = int(obs_store.observation_count[0, y, x])
        gt_str = " ".join(f"{v:.3f}" for v in gt[y, x])
        pred_str = " ".join(f"{v:.3f}" for v in pred[y, x])
        print(f"    ({y:2d},{x:2d}) {t_name:>10} kl={kl[y,x]:.4f} ent={ent[y,x]:.3f} w={weighted[y,x]:.4f} obs={obs_count}")
        print(f"      GT:   [{gt_str}]")
        print(f"      Pred: [{pred_str}]")
