import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from api_client import AstarIslandClient
from observation_store import ObservationStore
from prediction_builder import build_prediction, build_prediction_with_mc, _apply_floor_and_normalize
from utils import score_prediction, compute_kl_divergence, compute_entropy
from config import NUM_CLASSES, PROBABILITY_FLOOR, CLASS_NAMES
from main import _print_per_class_analysis, _print_per_distance_analysis


KNOWN_ROUNDS: dict[str, str] = {
    "11": "324fde07-1670-4202-b199-7aa92ecb40ee",
    "13": "7b4bda99-6165-4221-97cc-27880f5e6d95",
    "14": "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    "15": "cc5442dd-bc5d-418b-911b-7eb960cb0390",
}

GT_CACHE_DIR = Path(__file__).parent / ".gt_cache"
OBS_DIR = Path(__file__).parent / ".observations"


def gt_cache_path(round_id: str, seed_index: int) -> Path:
    return GT_CACHE_DIR / f"{round_id}_seed{seed_index}.npz"


def load_gt_cache(round_id: str, seed_index: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    path = gt_cache_path(round_id, seed_index)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return data["ground_truth"], data["initial_grid"]


def save_gt_cache(round_id: str, seed_index: int, ground_truth: np.ndarray, initial_grid: np.ndarray) -> None:
    GT_CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        gt_cache_path(round_id, seed_index),
        ground_truth=ground_truth,
        initial_grid=initial_grid,
    )


def fetch_analysis(
    client: AstarIslandClient,
    round_id: str,
    seed_index: int,
    analysis_cache: dict[int, dict],
) -> Optional[dict]:
    if seed_index in analysis_cache:
        return analysis_cache[seed_index]
    try:
        resp = client.get_analysis(round_id, seed_index)
        analysis_cache[seed_index] = resp
        return resp
    except Exception as e:
        print(f"    [WARNING] Could not fetch analysis for seed {seed_index}: {e}")
        return None


def load_or_fetch_gt(
    client: AstarIslandClient,
    round_id: str,
    seed_index: int,
    fallback_initial_grid: np.ndarray,
    analysis_cache: dict[int, dict],
) -> tuple[Optional[np.ndarray], np.ndarray]:
    cached = load_gt_cache(round_id, seed_index)
    if cached is not None:
        return cached[0], cached[1]

    analysis = fetch_analysis(client, round_id, seed_index, analysis_cache)
    if analysis is None:
        return None, fallback_initial_grid

    gt_data = analysis.get("ground_truth")
    if gt_data is None:
        print(f"    [WARNING] No ground_truth in analysis response for seed {seed_index}")
        return None, fallback_initial_grid

    gt = np.array(gt_data, dtype=np.float64)
    grid_data = analysis.get("initial_grid")
    initial_grid = np.array(grid_data, dtype=np.int32) if grid_data is not None else fallback_initial_grid

    save_gt_cache(round_id, seed_index, gt, initial_grid)
    return gt, initial_grid


def get_submitted_pred(
    analysis_cache: dict[int, dict],
    client: AstarIslandClient,
    round_id: str,
    seed_index: int,
) -> Optional[np.ndarray]:
    analysis = fetch_analysis(client, round_id, seed_index, analysis_cache)
    if analysis is None:
        return None
    pred_data = analysis.get("prediction")
    if pred_data is None:
        return None
    return np.array(pred_data, dtype=np.float64)


def has_cached_observations(round_id: str) -> bool:
    return (OBS_DIR / f"{round_id}.npz").exists()


def round_short_name(round_id: str) -> str:
    for name, rid in KNOWN_ROUNDS.items():
        if rid == round_id:
            return name
    return round_id[:8] + "..."


def resolve_round_id(round_arg: str) -> str:
    return KNOWN_ROUNDS.get(round_arg, round_arg)


def backtest_round(
    client: AstarIslandClient,
    round_id: str,
    detailed: bool = False,
    compare: bool = False,
) -> None:
    print(f"\n=== Backtest: Round {round_short_name(round_id)} ===")
    print(f"Round ID: {round_id}")

    try:
        detail = client.get_round_detail(round_id)
    except Exception as e:
        print(f"  [ERROR] Could not fetch round detail: {e}")
        return

    num_seeds: int = detail.get("seeds_count", 0)
    if num_seeds == 0:
        print("  [ERROR] No seeds found in round detail.")
        return

    initial_grids_from_detail: list[np.ndarray] = [
        np.array(state["grid"], dtype=np.int32)
        for state in detail["initial_states"]
    ]

    obs_store: Optional[ObservationStore] = None
    if has_cached_observations(round_id):
        try:
            obs_store = ObservationStore.load(round_id)
            obs_store.aggregate_across_seeds()
        except Exception as e:
            print(f"  [WARNING] Could not load observations: {e}")

    analysis_cache: dict[int, dict] = {}
    submitted_scores: list[Optional[float]] = []
    regen_scores: list[Optional[float]] = []
    regen_preds: list[Optional[np.ndarray]] = []
    gts: list[Optional[np.ndarray]] = []

    for seed_idx in range(num_seeds):
        fallback_grid = initial_grids_from_detail[seed_idx]
        gt, initial_grid = load_or_fetch_gt(client, round_id, seed_idx, fallback_grid, analysis_cache)
        gts.append(gt)

        sub_score: Optional[float] = None
        if gt is not None:
            sub_pred = get_submitted_pred(analysis_cache, client, round_id, seed_idx)
            if sub_pred is not None and gt.shape == sub_pred.shape:
                sub_score = score_prediction(gt, sub_pred)
        submitted_scores.append(sub_score)

        regen_score: Optional[float] = None
        regen_pred: Optional[np.ndarray] = None
        if obs_store is not None and gt is not None:
            try:
                regen_pred = build_prediction(seed_idx, obs_store, initial_grid)
                regen_score = score_prediction(gt, regen_pred)
            except Exception as e:
                print(f"  [WARNING] Could not regenerate prediction for seed {seed_idx}: {e}")
        regen_scores.append(regen_score)
        regen_preds.append(regen_pred)

    print()
    for seed_idx in range(num_seeds):
        sub = submitted_scores[seed_idx]
        regen = regen_scores[seed_idx]
        sub_str = f"{sub:.2f}" if sub is not None else "N/A"
        regen_str = f"{regen:.2f}" if regen is not None else "N/A"

        if regen is not None and sub is not None:
            delta_str = f"  delta={regen - sub:+.2f}"
        elif obs_store is None:
            delta_str = "  (no cached observations)"
        else:
            delta_str = ""

        print(f"  Seed {seed_idx}: submitted={sub_str}  regenerated={regen_str}{delta_str}")

    print()
    valid_sub = [s for s in submitted_scores if s is not None]
    valid_regen = [s for s in regen_scores if s is not None]
    avg_sub = f"{sum(valid_sub)/len(valid_sub):.2f}" if valid_sub else "N/A"
    avg_regen = f"{sum(valid_regen)/len(valid_regen):.2f}" if valid_regen else "N/A"
    print(f"  Average: submitted={avg_sub}  regenerated={avg_regen}")

    if detailed:
        print()
        for seed_idx in range(num_seeds):
            gt = gts[seed_idx]
            if gt is None:
                continue

            pred_for_analysis = regen_preds[seed_idx]
            label = "regenerated"
            if pred_for_analysis is None:
                pred_for_analysis = get_submitted_pred(analysis_cache, client, round_id, seed_idx)
                label = "submitted"

            if pred_for_analysis is None or gt.shape != pred_for_analysis.shape:
                continue

            score = score_prediction(gt, pred_for_analysis)
            print(f"  --- Seed {seed_idx} ({label}) score={score:.2f} ---")
            _print_per_class_analysis(gt, pred_for_analysis, seed_idx)
            _print_per_distance_analysis(gt, pred_for_analysis, seed_idx)

    if compare:
        print()
        print("  --- Comparison: submitted vs regenerated ---")
        for seed_idx in range(num_seeds):
            gt = gts[seed_idx]
            if gt is None:
                continue

            regen_pred = regen_preds[seed_idx]
            sub_pred = get_submitted_pred(analysis_cache, client, round_id, seed_idx)

            if regen_pred is None and sub_pred is None:
                print(f"  Seed {seed_idx}: no predictions available for comparison")
                continue

            if sub_pred is not None and gt.shape == sub_pred.shape:
                sub_score = score_prediction(gt, sub_pred)
                print(f"\n  Seed {seed_idx} — Submitted (score={sub_score:.2f}):")
                _print_per_class_analysis(gt, sub_pred, seed_idx)

            if regen_pred is not None and gt.shape == regen_pred.shape:
                regen_score = score_prediction(gt, regen_pred)
                print(f"\n  Seed {seed_idx} — Regenerated (score={regen_score:.2f}):")
                _print_per_class_analysis(gt, regen_pred, seed_idx)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest prediction pipeline against ground truth from past rounds."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--round", metavar="ROUND", help="Round number or ID (e.g. 14 or full UUID)")
    group.add_argument("--all", action="store_true", help="Backtest all known rounds")
    parser.add_argument("--detailed", action="store_true", help="Show per-class and per-distance breakdown")
    parser.add_argument("--compare", action="store_true", help="Compare submitted vs regenerated predictions")
    args = parser.parse_args()

    client = AstarIslandClient()
    round_ids = list(KNOWN_ROUNDS.values()) if args.all else [resolve_round_id(args.round)]

    for round_id in round_ids:
        backtest_round(client=client, round_id=round_id, detailed=args.detailed, compare=args.compare)

    print("\nDone.")


if __name__ == "__main__":
    main()
