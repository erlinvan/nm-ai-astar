import numpy as np
import os
os.environ.setdefault("ASTAR_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3YjY3NWJlYy1mZDhkLTQ5YmEtOTNmNy1kYjRmNTA1NDNjODQiLCJlbWFpbCI6Im1ybGlpbmcxMDFAZ21haWwuY29tIiwiaXNfYWRtaW4iOmZhbHNlLCJleHAiOjE3NzQ2NDExMjJ9.BvhwRzzc_tZDisQhzIUtwHPw-GIgLLZ0L80t9DYX-3I")

from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, compute_round_priors, _compute_settlement_distance_map
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from config import NUM_CLASSES, TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_NAMES

def kl_per_cell(gt, pred):
    eps = 1e-15
    gt_safe = np.clip(gt, eps, 1.0)
    pred_safe = np.clip(pred, eps, 1.0)
    return np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=-1)

def entropy_per_cell(gt):
    eps = 1e-15
    gt_safe = np.clip(gt, eps, 1.0)
    return -np.sum(gt_safe * np.log(gt_safe), axis=-1)

def diagnose_round(client, round_id, round_name):
    detail = client.get_round_detail(round_id)
    num_seeds = detail.get("seeds_count", 5)
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

    obs_store = ObservationStore.load(round_id)
    round_priors = compute_round_priors(obs_store, initial_grids, num_seeds)

    analysis_cache = {}
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: Round {round_name}")
    print(f"{'='*60}")

    for seed_idx in range(min(2, num_seeds)):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        if gt is None:
            continue
        
        pred = build_prediction(seed_idx, obs_store, ig, round_priors=round_priors)
        
        kl = kl_per_cell(gt, pred)
        ent = entropy_per_cell(gt)
        dist_map = _compute_settlement_distance_map(ig)
        
        flat_indices = np.argsort(kl.ravel())[::-1][:20]
        
        print(f"\n--- Seed {seed_idx}: Top 20 worst KL cells ---")
        print(f"{'Cell':>8} {'Terrain':>12} {'Dist':>5} {'KL':>8} {'Entropy':>8} {'GT dist':>45} {'Pred dist':>45}")
        for idx in flat_indices:
            y, x = divmod(idx, kl.shape[1])
            terrain = ig[y, x]
            if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue
            t_name = TERRAIN_NAMES.get(int(terrain), str(terrain))
            gt_str = np.array2string(gt[y, x], precision=3, separator=',')
            pred_str = np.array2string(pred[y, x], precision=3, separator=',')
            obs_count = int(obs_store.observation_count[seed_idx, y, x])
            print(f"({y:2d},{x:2d}) {t_name:>12} {dist_map[y,x]:5.0f} {kl[y,x]:8.4f} {ent[y,x]:8.4f} GT={gt_str} P={pred_str} obs={obs_count}")

        print(f"\n--- Seed {seed_idx}: KL by terrain type ---")
        for terrain_code in sorted(set(ig.ravel())):
            if terrain_code in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue
            mask = ig == terrain_code
            t_name = TERRAIN_NAMES.get(terrain_code, str(terrain_code))
            cells = mask.sum()
            mean_kl = kl[mask].mean()
            total_weighted = (ent[mask] * kl[mask]).sum()
            total_entropy = ent[mask].sum()
            w_kl = total_weighted / total_entropy if total_entropy > 0 else 0
            print(f"  {t_name:>12}: {cells:4d} cells, mean_kl={mean_kl:.5f}, weighted_kl={w_kl:.5f}, score={100*np.exp(-3*w_kl):.2f}")

        total_weighted = (ent * kl).sum()
        total_entropy = ent.sum()
        w_kl = total_weighted / total_entropy
        print(f"  {'TOTAL':>12}: weighted_kl={w_kl:.5f}, score={100*np.exp(-3*w_kl):.2f}")


if __name__ == "__main__":
    client = AstarIslandClient()
    for name, rid in [("R13", KNOWN_ROUNDS["13"]), ("R15", KNOWN_ROUNDS["15"])]:
        diagnose_round(client, rid, name)
