from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_EMPTY, TERRAIN_SETTLEMENT,
                    TERRAIN_PORT, TERRAIN_RUIN, TERRAIN_FOREST, TERRAIN_MOUNTAIN,
                    TERRAIN_NAMES)
from simulator.world import World
from simulator.engine import SimulationEngine

_TERRAIN_CODES = [
    TERRAIN_EMPTY,       # 0
    TERRAIN_SETTLEMENT,  # 1
    TERRAIN_PORT,        # 2
    TERRAIN_RUIN,        # 3
    TERRAIN_FOREST,      # 4
    TERRAIN_MOUNTAIN,    # 5
    TERRAIN_OCEAN,       # 10
    TERRAIN_PLAINS,      # 11
]

_COLORS = {
    TERRAIN_OCEAN: (0.05, 0.18, 0.45),
    TERRAIN_PLAINS: (0.72, 0.90, 0.58),
    TERRAIN_EMPTY: (0.95, 0.91, 0.79),
    TERRAIN_SETTLEMENT: (0.85, 0.10, 0.10),
    TERRAIN_PORT: (1.00, 0.55, 0.10),
    TERRAIN_RUIN: (0.55, 0.55, 0.55),
    TERRAIN_FOREST: (0.13, 0.45, 0.13),
    TERRAIN_MOUNTAIN: (0.50, 0.35, 0.20),
}

_MAX_CODE = 12
_COLOR_ARRAY = np.zeros((_MAX_CODE + 1, 3))
for code, color in _COLORS.items():
    _COLOR_ARRAY[code] = color


def _grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    rgb = np.zeros((h, w, 3))
    for code, color in _COLORS.items():
        mask = grid == code
        rgb[mask] = color
    return rgb


def _make_legend_patches() -> list:
    labels = {
        TERRAIN_OCEAN: "Ocean",
        TERRAIN_PLAINS: "Plains",
        TERRAIN_EMPTY: "Empty",
        TERRAIN_SETTLEMENT: "Settlement",
        TERRAIN_PORT: "Port",
        TERRAIN_RUIN: "Ruin",
        TERRAIN_FOREST: "Forest",
        TERRAIN_MOUNTAIN: "Mountain",
    }
    return [
        mpatches.Patch(color=_COLORS[code], label=name)
        for code, name in labels.items()
    ]


def visualize_world(world: World, title: str = "", save_path: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    rgb = _grid_to_rgb(world.grid)
    ax.imshow(rgb, origin='upper', interpolation='nearest')
    ax.set_title(title or "Astar Island", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    legend = _make_legend_patches()
    ax.legend(handles=legend, loc='upper right', fontsize=7,
              framealpha=0.8, ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def visualize_comparison(world_before: World, world_after: World,
                          save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, world, title in zip(axes,
                                  [world_before, world_after],
                                  ["Year 0 (Initial)", "Year 50 (Final)"]):
        rgb = _grid_to_rgb(world.grid)
        ax.imshow(rgb, origin='upper', interpolation='nearest')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    legend = _make_legend_patches()
    fig.legend(handles=legend, loc='lower center', fontsize=8,
               ncol=8, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def visualize_animation(engine: SimulationEngine, save_path: Optional[str] = None) -> None:
    history = engine.get_history()
    if not history:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    dummy_world = World(engine.world.width, engine.world.height)
    dummy_world.grid = history[0]
    rgb0 = _grid_to_rgb(history[0])
    im = ax.imshow(rgb0, origin='upper', interpolation='nearest')
    title = ax.set_title("Year 0", fontsize=14)
    legend = _make_legend_patches()
    ax.legend(handles=legend, loc='upper right', fontsize=7,
              framealpha=0.8, ncol=2)

    def update(frame: int):
        rgb = _grid_to_rgb(history[frame])
        im.set_data(rgb)
        title.set_text(f"Year {frame}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(history),
                          interval=200, blit=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        plt.close(fig)
    else:
        plt.show()
