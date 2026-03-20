import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.world import World
from simulator.params import SimParams
from simulator.phases import (phase_growth, phase_conflict, phase_trade,
                               phase_winter, phase_environment)
from config import TERRAIN_TO_CLASS


class SimulationEngine:
    def __init__(self, world: World, params: SimParams, seed: int = 0,
                 record_history: bool = True):
        self.world = world
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.year = 0
        self._record_history = record_history
        self._history: list[np.ndarray] = []

    def run_step(self) -> None:
        self.world.invalidate_alive_cache()
        phase_growth(self.world, self.params, self.rng)
        self.world.invalidate_alive_cache()
        phase_conflict(self.world, self.params, self.rng)
        phase_trade(self.world, self.params, self.rng)
        self.world.invalidate_alive_cache()
        phase_winter(self.world, self.params, self.rng)
        phase_environment(self.world, self.params, self.rng)
        self.world.invalidate_alive_cache()
        self.year += 1

    def run(self, years: int = 50) -> World:
        if self._record_history:
            self._history = [self.world.copy_grid()]
        for _ in range(years):
            self.run_step()
            if self._record_history:
                self._history.append(self.world.copy_grid())
        return self.world

    def get_grid(self) -> np.ndarray:
        return self.world.grid

    def get_class_grid(self) -> np.ndarray:
        return self.world.get_class_grid()

    def get_history(self) -> list[np.ndarray]:
        return self._history

    def get_stats(self) -> dict:
        alive = self.world.alive_settlements()
        ruins = int(np.sum(self.world.grid == 3))
        factions = len(set(s.owner_id for s in alive))
        return {
            "year": self.year,
            "settlements_alive": len(alive),
            "ruins": ruins,
            "factions": factions,
            "total_population": sum(s.population for s in alive),
        }
