import numpy as np
from simulator.world import World
from simulator.settlement import Settlement
from observation_store import ObservationStore


_DEFAULT_POP_RANGE = (1.0, 3.0)
_DEFAULT_FOOD_RANGE = (0.8, 1.5)
_DEFAULT_WEALTH_RANGE = (0.3, 0.8)
_DEFAULT_DEFENSE_RANGE = (0.3, 0.5)
_DEFAULT_TECH_RANGE = (0.05, 0.2)


def build_world_from_state(
    state: dict,
    width: int,
    height: int,
    rng_seed: int = 42,
) -> World:
    world = World(width=width, height=height)
    world.grid = np.array(state["grid"], dtype=np.int32)

    rng = np.random.default_rng(rng_seed)

    for i, s_data in enumerate(state["settlements"]):
        x, y = s_data["x"], s_data["y"]
        settlement = Settlement(
            x=x, y=y,
            population=float(s_data.get("population", rng.uniform(*_DEFAULT_POP_RANGE))),
            food=float(s_data.get("food", rng.uniform(*_DEFAULT_FOOD_RANGE))),
            wealth=float(s_data.get("wealth", rng.uniform(*_DEFAULT_WEALTH_RANGE))),
            defense=float(s_data.get("defense", rng.uniform(*_DEFAULT_DEFENSE_RANGE))),
            tech_level=float(s_data.get("tech_level", rng.uniform(*_DEFAULT_TECH_RANGE))),
            has_port=s_data.get("has_port", False),
            has_longship=False,
            alive=s_data.get("alive", True),
            owner_id=i,
        )
        world.add_settlement(settlement)

    return world


def calibrate_settlements_from_observations(
    world: World,
    store: ObservationStore,
    seed_index: int,
) -> None:
    """Update settlement initial stats using observed settlement data from simulation queries.

    The API's simulate endpoint returns detailed settlement stats (population, food,
    wealth, defense, owner_id) within each viewport. While these are post-simulation
    values (year 50), averaging across multiple observations gives us a sense of
    typical settlement trajectories. We use the earliest-seeming stats (smallest
    population values) as proxy for initial conditions.
    """
    for (sx, sy), obs_list in store.settlement_data[seed_index].items():
        settlement = world.get_settlement(sx, sy)
        if settlement is None or not obs_list:
            continue

        avg_pop = float(np.mean([o["population"] for o in obs_list if "population" in o]))
        avg_food = float(np.mean([o["food"] for o in obs_list if "food" in o]))
        avg_wealth = float(np.mean([o["wealth"] for o in obs_list if "wealth" in o]))
        avg_defense = float(np.mean([o["defense"] for o in obs_list if "defense" in o]))

        if not np.isnan(avg_pop):
            settlement.population = float(max(0.5, avg_pop * 0.3))
        if not np.isnan(avg_food):
            settlement.food = float(max(0.3, avg_food * 0.5))
        if not np.isnan(avg_wealth):
            settlement.wealth = float(max(0.1, avg_wealth * 0.4))
        if not np.isnan(avg_defense):
            settlement.defense = float(max(0.1, avg_defense * 0.8))

        if any("owner_id" in o for o in obs_list):
            owner_ids = [o["owner_id"] for o in obs_list if "owner_id" in o]
            from collections import Counter
            most_common_owner = Counter(owner_ids).most_common(1)[0][0]
            settlement.owner_id = most_common_owner

        if any("has_port" in o for o in obs_list):
            settlement.has_port = any(o.get("has_port", False) for o in obs_list)
