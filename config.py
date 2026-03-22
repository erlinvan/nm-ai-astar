"""Configuration constants for Astar Island competition."""

import os

# API Configuration
API_BASE_URL = "https://api.ainm.no"
API_TOKEN = os.environ.get("ASTAR_TOKEN", "")

# Map defaults
DEFAULT_MAP_WIDTH = 40
DEFAULT_MAP_HEIGHT = 40
DEFAULT_SEEDS_COUNT = 5
DEFAULT_YEARS = 50

# Query budget
MAX_QUERIES = 50
MAX_VIEWPORT_SIZE = 15
MIN_VIEWPORT_SIZE = 5

# Terrain codes (internal representation)
TERRAIN_OCEAN = 10
TERRAIN_PLAINS = 11
TERRAIN_EMPTY = 0
TERRAIN_SETTLEMENT = 1
TERRAIN_PORT = 2
TERRAIN_RUIN = 3
TERRAIN_FOREST = 4
TERRAIN_MOUNTAIN = 5

# Prediction class indices
CLASS_EMPTY = 0       # Ocean, Plains, Empty
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5
NUM_CLASSES = 6

# Mapping from terrain code to prediction class
TERRAIN_TO_CLASS = {
    TERRAIN_OCEAN: CLASS_EMPTY,
    TERRAIN_PLAINS: CLASS_EMPTY,
    TERRAIN_EMPTY: CLASS_EMPTY,
    TERRAIN_SETTLEMENT: CLASS_SETTLEMENT,
    TERRAIN_PORT: CLASS_PORT,
    TERRAIN_RUIN: CLASS_RUIN,
    TERRAIN_FOREST: CLASS_FOREST,
    TERRAIN_MOUNTAIN: CLASS_MOUNTAIN,
}

# Static terrain codes (never change during simulation)
STATIC_TERRAINS = {TERRAIN_OCEAN, TERRAIN_MOUNTAIN}

# Prediction smoothing
PROBABILITY_FLOOR = 0.001  # Minimum probability per class to avoid infinite KL
HIGH_CONFIDENCE = 0.94     # Probability for observed class (leaves 0.012 for each of 5 others)
STATIC_CONFIDENCE = 0.99   # Probability for static cells

# Terrain class names for display
CLASS_NAMES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
}

TERRAIN_NAMES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
    10: "Ocean",
    11: "Plains",
}
