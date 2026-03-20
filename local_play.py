"""Run and visualize the local Norse civilization simulator.

Usage:
    python local_play.py [--seed 42] [--years 50] [--size 40] [--animate]
"""

import argparse
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')

from simulator.map_gen import generate_map
from simulator.params import SimParams
from simulator.engine import SimulationEngine
from simulator.visualizer import (visualize_world, visualize_comparison,
                                   visualize_animation)
from simulator.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description="Astar Island local simulator")
    parser.add_argument("--seed", type=int, default=42, help="Map seed")
    parser.add_argument("--years", type=int, default=50, help="Years to simulate")
    parser.add_argument("--size", type=int, default=40, help="Grid size (N×N)")
    parser.add_argument("--animate", action="store_true", help="Save year-by-year animation")
    parser.add_argument("--out", type=str, default="", help="Output directory for images")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating map  seed={args.seed}  size={args.size}×{args.size}")
    world = generate_map(seed=args.seed, width=args.size, height=args.size)

    world_before = World(width=args.size, height=args.size)
    world_before.grid = world.copy_grid()
    for pos, s in world.settlements.items():
        world_before.settlements[pos] = copy.copy(s)

    before_path = os.path.join(out_dir, f"map_seed{args.seed}_year0.png")
    visualize_world(world_before, title=f"Seed {args.seed} — Year 0", save_path=before_path)
    print(f"Saved initial map → {before_path}")

    params = SimParams()
    engine = SimulationEngine(world=world, params=params, seed=args.seed + 1)

    print(f"Running simulation for {args.years} years…")
    engine.run(years=args.years)

    stats = engine.get_stats()
    print(f"\n=== Final stats (year {stats['year']}) ===")
    print(f"  Settlements alive : {stats['settlements_alive']}")
    print(f"  Ruins             : {stats['ruins']}")
    print(f"  Factions          : {stats['factions']}")
    print(f"  Total population  : {stats['total_population']:.1f}")

    after_path = os.path.join(out_dir, f"map_seed{args.seed}_year{args.years}.png")
    visualize_world(world, title=f"Seed {args.seed} — Year {args.years}", save_path=after_path)
    print(f"Saved final map   → {after_path}")

    compare_path = os.path.join(out_dir, f"map_seed{args.seed}_comparison.png")
    visualize_comparison(world_before, world, save_path=compare_path)
    print(f"Saved comparison  → {compare_path}")

    if args.animate:
        anim_path = os.path.join(out_dir, f"map_seed{args.seed}_animation.gif")
        print(f"Saving animation  → {anim_path}  (may take a moment…)")
        visualize_animation(engine, save_path=anim_path)
        print("Animation saved.")


if __name__ == "__main__":
    main()
