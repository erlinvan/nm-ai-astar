"""Monte Carlo simulation for terrain class probability estimation."""

import os
import multiprocessing as mp
import numpy as np
from simulator.world import World
from simulator.settlement import Settlement
from simulator.params import SimParams
from simulator.engine import SimulationEngine
from config import TERRAIN_TO_CLASS, NUM_CLASSES


def clone_world(world: World) -> World:
    new = World(width=world.width, height=world.height)
    new.grid = world.grid.copy()
    for pos, s in world.settlements.items():
        new.settlements[pos] = Settlement(
            x=s.x, y=s.y,
            population=s.population, food=s.food,
            wealth=s.wealth, defense=s.defense,
            tech_level=s.tech_level,
            has_port=s.has_port, has_longship=s.has_longship,
            alive=s.alive, owner_id=s.owner_id,
        )
    return new


def _mc_worker(args):
    world, params, start_idx, count, years, base_seed, param_noise = args
    h, w = world.height, world.width
    class_counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    param_rng = np.random.default_rng(base_seed + start_idx + 999)

    for i in range(count):
        sim_world = clone_world(world)
        sim_seed = base_seed + (start_idx + i) * 7919

        run_params = params
        if param_noise > 0:
            run_params = params.randomize(param_rng, scale=param_noise)

        engine = SimulationEngine(world=sim_world, params=run_params, seed=sim_seed,
                                  record_history=False)
        engine.run(years=years)

        final_grid = engine.get_grid()
        for terrain_code, class_idx in TERRAIN_TO_CLASS.items():
            mask = final_grid == terrain_code
            class_counts[:, :, class_idx] += mask.astype(np.float64)

    return class_counts


def run_monte_carlo(
    world: World,
    params: SimParams,
    num_runs: int = 200,
    years: int = 50,
    base_seed: int = 0,
    param_noise: float = 0.0,
    workers: int = 0,
) -> np.ndarray:
    """Returns H x W x 6 probability tensor. param_noise > 0 randomizes params each run."""
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 4) - 1)

    if workers == 1 or num_runs < 4:
        return _run_sequential(world, params, num_runs, years, base_seed, param_noise)

    runs_per_worker = num_runs // workers
    remainder = num_runs % workers

    chunks = []
    offset = 0
    for i in range(workers):
        chunk_size = runs_per_worker + (1 if i < remainder else 0)
        if chunk_size == 0:
            continue
        chunks.append((world, params, offset, chunk_size, years, base_seed, param_noise))
        offset += chunk_size

    with mp.Pool(len(chunks)) as pool:
        results = pool.map(_mc_worker, chunks)

    h, w = world.height, world.width
    total_counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    for counts in results:
        total_counts += counts

    return total_counts / num_runs


def _run_sequential(world, params, num_runs, years, base_seed, param_noise):
    return _mc_worker((world, params, 0, num_runs, years, base_seed, param_noise)) / num_runs
