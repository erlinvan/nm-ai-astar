import json
import os
from pathlib import Path

from typing import Optional

import numpy as np
from config import (
    TERRAIN_TO_CLASS,
    NUM_CLASSES,
    DEFAULT_MAP_WIDTH,
    DEFAULT_MAP_HEIGHT,
)


OBSERVATIONS_DIR = Path(__file__).parent / ".observations"


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
        self.aggregated_counts: Optional[np.ndarray] = None
        self.aggregated_obs_count: Optional[np.ndarray] = None

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

    def save(self, round_id: str) -> Path:
        OBSERVATIONS_DIR.mkdir(exist_ok=True)
        path = OBSERVATIONS_DIR / f"{round_id}.npz"

        settlement_serializable = {}
        for seed_idx, cells in self.settlement_data.items():
            seed_key = str(seed_idx)
            settlement_serializable[seed_key] = {}
            for (x, y), entries in cells.items():
                settlement_serializable[seed_key][f"{x},{y}"] = entries

        np.savez_compressed(
            path,
            class_counts=self.class_counts,
            observation_count=self.observation_count,
            settlement_json=json.dumps(settlement_serializable),
            meta=np.array([self.num_seeds, self.width, self.height]),
        )
        return path

    @classmethod
    def load(cls, round_id: str) -> "ObservationStore":
        path = OBSERVATIONS_DIR / f"{round_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"No saved observations for round {round_id}")

        data = np.load(path, allow_pickle=False)
        meta = data["meta"]
        num_seeds, width, height = int(meta[0]), int(meta[1]), int(meta[2])

        store = cls(num_seeds, width, height)
        store.class_counts = data["class_counts"].copy()
        store.observation_count = data["observation_count"].copy()

        settlement_raw = json.loads(str(data["settlement_json"]))
        for seed_key, cells in settlement_raw.items():
            seed_idx = int(seed_key)
            for coord_key, entries in cells.items():
                x, y = coord_key.split(",")
                store.settlement_data[seed_idx][(int(x), int(y))] = entries

        return store

    def merge(self, other: "ObservationStore") -> None:
        self.class_counts += other.class_counts
        self.observation_count += other.observation_count
        for seed_idx in range(self.num_seeds):
            for key, entries in other.settlement_data[seed_idx].items():
                if key not in self.settlement_data[seed_idx]:
                    self.settlement_data[seed_idx][key] = []
                self.settlement_data[seed_idx][key].extend(entries)

    def aggregate_across_seeds(self) -> None:
        self.aggregated_counts = self.class_counts.sum(axis=0)
        self.aggregated_obs_count = self.observation_count.sum(axis=0)
