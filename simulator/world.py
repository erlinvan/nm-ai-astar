from __future__ import annotations
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_EMPTY, TERRAIN_SETTLEMENT,
                    TERRAIN_PORT, TERRAIN_RUIN, TERRAIN_FOREST, TERRAIN_MOUNTAIN,
                    TERRAIN_TO_CLASS, NUM_CLASSES)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulator.settlement import Settlement


def _build_offsets(radius: int) -> list:
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            offsets.append((dx, dy))
    return offsets


_OFFSETS_1 = _build_offsets(1)
_OFFSETS_2 = _build_offsets(2)


class World:
    def __init__(self, width: int = 40, height: int = 40):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), TERRAIN_PLAINS, dtype=np.int32)
        self.settlements: dict[tuple[int, int], "Settlement"] = {}
        self._alive_cache = []
        self._alive_dirty = True

    def get_terrain(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.grid[y, x])
        return TERRAIN_OCEAN

    def set_terrain(self, x: int, y: int, val: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = val

    def get_neighbors(self, x: int, y: int, radius: int = 1) -> list[tuple[int, int]]:
        offsets = _OFFSETS_1 if radius == 1 else _OFFSETS_2 if radius == 2 else _build_offsets(radius)
        w, h = self.width, self.height
        return [(x + dx, y + dy) for dx, dy in offsets
                if 0 <= x + dx < w and 0 <= y + dy < h]

    def get_class_grid(self) -> np.ndarray:
        class_grid = np.zeros((self.height, self.width), dtype=np.int32)
        for terrain_code, class_idx in TERRAIN_TO_CLASS.items():
            class_grid[self.grid == terrain_code] = class_idx
        return class_grid

    def add_settlement(self, settlement) -> None:
        self.settlements[(settlement.x, settlement.y)] = settlement
        self._alive_dirty = True

    def remove_settlement(self, x: int, y: int) -> None:
        self.settlements.pop((x, y), None)
        self._alive_dirty = True

    def get_settlement(self, x: int, y: int):
        return self.settlements.get((x, y))

    def alive_settlements(self) -> list:
        if self._alive_dirty:
            self._alive_cache = [s for s in self.settlements.values() if s.alive]
            self._alive_dirty = False
        return self._alive_cache

    def invalidate_alive_cache(self) -> None:
        self._alive_dirty = True

    def copy_grid(self) -> np.ndarray:
        return self.grid.copy()
