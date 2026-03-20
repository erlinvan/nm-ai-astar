import numpy as np
from api_client import AstarIslandClient
from prediction_builder import build_prediction
from observation_store import ObservationStore


def submit_all_predictions(
    client: AstarIslandClient,
    round_id: str,
    store: ObservationStore,
    initial_grids: list[np.ndarray],
) -> list[dict]:
    results = []
    for seed_idx in range(store.num_seeds):
        prediction = build_prediction(seed_idx, store, initial_grids[seed_idx])
        _validate_prediction(prediction)
        resp = client.submit(round_id, seed_idx, prediction.tolist())
        print(f"Seed {seed_idx}: {resp.get('status', 'unknown')}")
        results.append(resp)
    return results


def submit_single_prediction(
    client: AstarIslandClient,
    round_id: str,
    seed_index: int,
    prediction: np.ndarray,
) -> dict:
    _validate_prediction(prediction)
    return client.submit(round_id, seed_index, prediction.tolist())


def _validate_prediction(prediction: np.ndarray):
    h, w, c = prediction.shape
    assert c == 6, f"Expected 6 classes, got {c}"
    assert np.all(prediction >= 0), "Negative probabilities found"

    sums = prediction.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=0.01), f"Probabilities don't sum to 1.0: min={sums.min()}, max={sums.max()}"

    assert np.all(prediction > 0), "Zero probabilities found — KL divergence will be infinite"
