from dataclasses import dataclass, field
import math
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_EMPTY, TERRAIN_SETTLEMENT,
                    TERRAIN_PORT, TERRAIN_RUIN, TERRAIN_FOREST, TERRAIN_MOUNTAIN,
                    TERRAIN_TO_CLASS, NUM_CLASSES)


@dataclass
class Settlement:
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    tech_level: float
    has_port: bool
    has_longship: bool
    alive: bool
    owner_id: int

    def is_coastal(self, world) -> bool:
        for nx, ny in world.get_neighbors(self.x, self.y, radius=1):
            if world.get_terrain(nx, ny) == TERRAIN_OCEAN:
                return True
        return False

    def distance_to(self, other: "Settlement") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_sq_to(self, other: "Settlement") -> int:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    @classmethod
    def create(cls, x: int, y: int, owner_id: int, rng: np.random.Generator,
               has_port: bool = False) -> "Settlement":
        return cls(
            x=x, y=y,
            population=float(rng.uniform(1.0, 3.0)),
            food=float(rng.uniform(0.8, 1.5)),
            wealth=float(rng.uniform(0.3, 0.8)),
            defense=float(rng.uniform(0.3, 0.5)),
            tech_level=float(rng.uniform(0.05, 0.2)),
            has_port=has_port,
            has_longship=False,
            alive=True,
            owner_id=owner_id,
        )
