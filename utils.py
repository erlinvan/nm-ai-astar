import numpy as np


def compute_kl_divergence(p_true: np.ndarray, p_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    p = np.clip(p_true, epsilon, 1.0)
    q = np.clip(p_pred, epsilon, 1.0)
    return float(np.sum(p * np.log(p / q)))


def compute_entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
    p_clipped = np.clip(p, epsilon, 1.0)
    return float(-np.sum(p_clipped * np.log(p_clipped)))


def score_prediction(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    h, w, c = ground_truth.shape
    assert prediction.shape == ground_truth.shape

    total_weighted_kl = 0.0
    total_entropy = 0.0

    for y in range(h):
        for x in range(w):
            cell_entropy = compute_entropy(ground_truth[y, x])
            if cell_entropy < 1e-8:
                continue
            cell_kl = compute_kl_divergence(ground_truth[y, x], prediction[y, x])
            total_weighted_kl += cell_entropy * cell_kl
            total_entropy += cell_entropy

    if total_entropy < 1e-8:
        return 100.0

    weighted_kl = total_weighted_kl / total_entropy
    return max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
