import numpy as np
from config import (
    TERRAIN_TO_CLASS,
    NUM_CLASSES,
    DEFAULT_MAP_WIDTH,
    DEFAULT_MAP_HEIGHT,
)


class ObservationStore:
    def __init__(self, num_seeds: int, width: int = DEFAULT_MAP_WIDTH, height: int = DEFAULT_MAP_HEIGHT):
        self.num_seeds = num_seeds
        self.width = width
        self.height = height
        self.class_counts = np.zeros((num_seeds, height, width, NUM_CLASSES), dtype=np.float64)
        self.observation_count = np.zeros((num_seeds, height, width), dtype=np.int32)
        self.settlement_data: dict[int, dict[tuple[int, int], list[dict]]] = {
            s: {} for s in range(num_seeds)
        }

    def add_observation(self, seed_index: int, viewport: dict, grid: list[list[int]], settlements: list[dict]):
        vx, vy = viewport["x"], viewport["y"]
        vw, vh = viewport["w"], viewport["h"]

        for row_idx, row in enumerate(grid):
            gy = vy + row_idx
            if gy >= self.height:
                break
            for col_idx, cell_val in enumerate(row):
                gx = vx + col_idx
                if gx >= self.width:
                    break
                cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                self.class_counts[seed_index, gy, gx, cls] += 1
                self.observation_count[seed_index, gy, gx] += 1

        for s in settlements:
            key = (s["x"], s["y"])
            if key not in self.settlement_data[seed_index]:
                self.settlement_data[seed_index][key] = []
            self.settlement_data[seed_index][key].append(s)

    def get_observed_distribution(self, seed_index: int, x: int, y: int):
        count = self.observation_count[seed_index, y, x]
        if count == 0:
            return None
        return self.class_counts[seed_index, y, x] / count

    def get_observation_count(self, seed_index: int, x: int, y: int) -> int:
        return int(self.observation_count[seed_index, y, x])

    def get_all_counts(self, seed_index: int) -> np.ndarray:
        return self.class_counts[seed_index]

    def get_coverage_mask(self, seed_index: int) -> np.ndarray:
        return self.observation_count[seed_index] > 0

    def coverage_ratio(self, seed_index: int) -> float:
        mask = self.get_coverage_mask(seed_index)
        return float(mask.sum()) / (self.width * self.height)

    def total_observations(self) -> int:
        return int(self.observation_count.sum())
