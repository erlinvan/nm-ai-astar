import numpy as np
from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import (
    _gt_calibrated_prior, _compute_settlement_distance_map,
    _apply_floor_and_normalize, build_prediction,
)
from backtest import load_or_fetch_gt, KNOWN_ROUNDS
from utils import score_prediction, compute_kl_divergence, compute_entropy
from config import (
    TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_FOREST, TERRAIN_SETTLEMENT,
    TERRAIN_PORT, TERRAIN_RUIN, NUM_CLASSES, CLASS_NAMES, PROBABILITY_FLOOR,
)
import prediction_builder

client = AstarIslandClient()

terrain_names = {
    TERRAIN_OCEAN: "Ocean", TERRAIN_MOUNTAIN: "Mountain", TERRAIN_FOREST: "Forest",
    TERRAIN_SETTLEMENT: "Settlement", TERRAIN_PORT: "Port", TERRAIN_RUIN: "Ruin",
    0: "Empty", 11: "Plains",
}

# Collect GT stats across R13, R14, R15 (all rounds with GT)
all_gt_by_terrain_dist = {}

for round_name, round_id in KNOWN_ROUNDS.items():
    detail = client.get_round_detail(round_id)
    num_seeds = detail["seeds_count"]
    initial_grids = [np.array(s["grid"], dtype=np.int32) for s in detail["initial_states"]]

    analysis_cache = {}
    for seed_idx in range(num_seeds):
        gt, ig = load_or_fetch_gt(client, round_id, seed_idx, initial_grids[seed_idx], analysis_cache)
        if gt is None:
            continue

        dist_map = _compute_settlement_distance_map(ig)

        for y in range(gt.shape[0]):
            for x in range(gt.shape[1]):
                terrain = ig[y, x]
                if terrain in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue

                sd = dist_map[y, x]
                if sd <= 2:
                    dist_bucket = "d0-2"
                elif sd <= 5:
                    dist_bucket = "d3-5"
                else:
                    dist_bucket = "d6+"

                key = (terrain, dist_bucket)
                if key not in all_gt_by_terrain_dist:
                    all_gt_by_terrain_dist[key] = []
                all_gt_by_terrain_dist[key].append(gt[y, x])

print(f"{'Terrain':12s} {'Dist':5s} {'N':>5s}  {'Empty':>7s} {'Settl':>7s} {'Port':>7s} {'Ruin':>7s} {'Forest':>7s} {'Mount':>7s}  {'Entropy':>7s}")
print("-" * 90)

for (terrain, dist_bucket) in sorted(all_gt_by_terrain_dist.keys()):
    samples = np.array(all_gt_by_terrain_dist[(terrain, dist_bucket)])
    avg_gt = samples.mean(axis=0)
    avg_entropy = np.mean([compute_entropy(s) for s in samples])
    n = len(samples)

    prior = _gt_calibrated_prior(terrain, {"d0-2": 1.0, "d3-5": 4.0, "d6+": 8.0}[dist_bucket])
    prior = prior / prior.sum()

    tname = terrain_names.get(terrain, f"T{terrain}")
    gt_str = " ".join(f"{v:7.4f}" for v in avg_gt)
    print(f"{tname:12s} {dist_bucket:5s} {n:5d}  {gt_str}  {avg_entropy:7.4f}")
    prior_str = " ".join(f"{v:7.4f}" for v in prior)
    print(f"{'  PRIOR':12s} {'':5s} {'':5s}  {prior_str}")

    kl = compute_kl_divergence(avg_gt, prior)
    print(f"{'  KL':12s} {'':5s} {'':5s}  {kl:.6f}")
    print()
