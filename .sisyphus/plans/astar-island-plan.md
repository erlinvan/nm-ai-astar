# Astar Island — Solution Architecture Plan

## Problem Summary
- 40x40 grid, 6 prediction classes, 5 seeds per round
- 50 queries total (shared across seeds), each 15x15 viewport of final state
- Score: entropy-weighted KL divergence (only dynamic cells matter)
- Time window: ~2h45m per round

## Architecture: Two-Layer System

### Layer 1: Observation Pipeline (PRIMARY — gets you 80%+ score)
Direct observation via API queries. With 3x3 tiling per seed (9 queries × 5 seeds = 45), 
we can observe the ENTIRE final map for all seeds.

**Components:**
1. `api_client.py` — Auth, rate limiting, all API endpoints
2. `query_strategy.py` — Viewport tiling, adaptive query allocation
3. `observation_store.py` — Store and stitch viewport results
4. `prediction_builder.py` — Convert observations to probability tensors
5. `submission.py` — Format and submit predictions

**Query Strategy:**
- Default: 3×3 grid tiling with starts at x,y ∈ {0, 13, 25} → covers full 40×40
- 45 queries for full coverage, 5 remaining for adaptive re-queries on high-interest areas
- If map is larger or different, adapt tiling dynamically

**Probability Construction:**
- Observed cells: 0.94 for observed class, 0.012 for each other class (sums to 1.0)
- Static cells (ocean/mountain): 0.95 for known class, 0.01 for others
- Forest cells: 0.90 for forest, small prob for ruin/settlement/empty
- Unobserved cells: Dirichlet posterior from neighboring observations

### Layer 2: Local Simulator (SECONDARY — for understanding, testing, fallback)
Approximate simulator matching the described mechanics.

**Components:**
1. `simulator/world.py` — Grid world, terrain types, cell states
2. `simulator/map_gen.py` — Procedural map generation from seed
3. `simulator/settlement.py` — Settlement entity with all properties
4. `simulator/phases.py` — Growth, Conflict, Trade, Winter, Environment phases
5. `simulator/engine.py` — Main simulation loop (50 years)
6. `simulator/params.py` — Hidden parameter definitions + defaults
7. `simulator/visualizer.py` — Matplotlib/terminal visualization

**Simulation Phases (per year):**
1. Growth: food from adjacent terrain, population growth, expansion, port development
2. Conflict: raiding within range (longships extend), looting, faction changes
3. Trade: port-to-port if in range + not at war, wealth/food/tech exchange
4. Winter: variable severity, food loss, collapse → ruin
5. Environment: ruin reclamation by nearby settlements or forest overgrowth

### Layer 3: Parameter Inference (BONUS — if time permits)
Calibrate local simulator to match API observations.

**Approach:**
- ABC (Approximate Bayesian Computation) or grid search over parameter space
- Use observations from Layer 1 as calibration targets
- Run Monte Carlo with fitted parameters for unobserved cells

## File Structure
```
island/
  main.py                    # Entry point — orchestrates full pipeline
  config.py                  # API keys, round config, constants
  api_client.py              # REST client for all endpoints
  query_strategy.py          # Viewport positioning logic
  observation_store.py       # Collect + stitch observations
  prediction_builder.py      # Build H×W×6 probability tensors
  submission.py              # Submit predictions
  simulator/
    __init__.py
    world.py                 # Grid world representation
    map_gen.py               # Procedural map generation
    settlement.py            # Settlement entity
    phases.py                # All 5 simulation phases
    engine.py                # Main sim loop
    params.py                # Parameter definitions
    visualizer.py            # Visualization utilities
  local_play.py              # Run local simulation standalone
  utils.py                   # Shared utilities
```

## Implementation Order
1. API client + round fetching (need this first to understand data)
2. Local simulator (world, map gen, settlement, all phases, engine)
3. Visualizer (see what's happening locally)
4. Query strategy + observation store
5. Prediction builder (Dirichlet smoothing, static cell handling)
6. Submission pipeline
7. Main orchestrator (ties it all together)
8. Parameter inference (if time permits)

## Key Technical Decisions
- Pure Python + numpy (no heavy frameworks)
- Probability floor: 0.01 minimum per class, renormalize
- Dirichlet-multinomial for cells with multiple observations
- Spatial smoothing for unobserved cells (kernel-weighted neighbor borrowing)
- All randomness via numpy.random.Generator for reproducibility
