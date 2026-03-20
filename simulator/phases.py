import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_EMPTY, TERRAIN_SETTLEMENT,
                    TERRAIN_PORT, TERRAIN_RUIN, TERRAIN_FOREST, TERRAIN_MOUNTAIN)
from simulator.world import World
from simulator.settlement import Settlement
from simulator.params import SimParams


def _is_ocean_adjacent(world: World, x: int, y: int) -> bool:
    for nx, ny in world.get_neighbors(x, y, radius=1):
        if world.get_terrain(nx, ny) == TERRAIN_OCEAN:
            return True
    return False


def _can_reach_by_water(world: World, ax: int, ay: int, tx: int, ty: int,
                         max_dist: int) -> bool:
    """BFS check if two positions are connected via ocean cells."""
    if abs(ax - tx) + abs(ay - ty) > max_dist * 2:
        return False
    visited = {(ax, ay)}
    queue = [(ax, ay, 0)]
    i = 0
    while i < len(queue):
        cx, cy, dist = queue[i]
        i += 1
        if dist >= max_dist:
            continue
        for nx, ny in world.get_neighbors(cx, cy, radius=1):
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            t = world.get_terrain(nx, ny)
            if nx == tx and ny == ty:
                return True
            if t == TERRAIN_OCEAN:
                queue.append((nx, ny, dist + 1))
    return False


def phase_growth(world: World, params: SimParams, rng: np.random.Generator) -> None:
    new_settlements: list[Settlement] = []

    for s in list(world.alive_settlements()):
        food_production = 0.0
        for nx, ny in world.get_neighbors(s.x, s.y, radius=1):
            t = world.get_terrain(nx, ny)
            if t == TERRAIN_FOREST:
                food_production += params.forest_food_yield
            elif t in (TERRAIN_PLAINS, TERRAIN_EMPTY):
                food_production += params.plains_food_yield

        if s.has_port:
            food_production *= params.port_food_multiplier

        food_needed = s.population * params.food_consumption
        food_surplus = food_production - food_needed
        s.food += food_production

        if food_surplus > 0:
            s.population += params.growth_rate * food_surplus
            s.wealth += params.growth_rate * food_surplus * 0.5

        if (s.population > params.expansion_threshold
                and rng.random() < params.expansion_prob):
            candidates = [
                (nx, ny)
                for nx, ny in world.get_neighbors(s.x, s.y, radius=2)
                if world.get_terrain(nx, ny) in (TERRAIN_PLAINS, TERRAIN_EMPTY, TERRAIN_FOREST)
                and (nx, ny) not in world.settlements
            ]
            if candidates:
                idx = int(rng.integers(0, len(candidates)))
                nx, ny = candidates[idx]
                is_coastal = _is_ocean_adjacent(world, nx, ny)
                new_s = Settlement(
                    x=nx, y=ny,
                    population=1.0, food=0.5, wealth=0.3,
                    defense=0.3, tech_level=s.tech_level * 0.8,
                    has_port=is_coastal, has_longship=False,
                    alive=True, owner_id=s.owner_id,
                )
                terrain = TERRAIN_PORT if is_coastal else TERRAIN_SETTLEMENT
                world.set_terrain(nx, ny, terrain)
                new_settlements.append(new_s)

        if (not s.has_port and s.population > params.port_threshold
                and _is_ocean_adjacent(world, s.x, s.y)):
            s.has_port = True
            world.set_terrain(s.x, s.y, TERRAIN_PORT)

        if s.has_port and not s.has_longship and s.wealth > params.longship_threshold:
            s.has_longship = True

    for ns in new_settlements:
        world.add_settlement(ns)


def phase_conflict(world: World, params: SimParams, rng: np.random.Generator) -> None:
    alive = world.alive_settlements()
    if len(alive) < 2:
        return

    for attacker in alive:
        if not attacker.alive:
            continue

        desperation = 0.0
        if attacker.food < attacker.population * 0.5:
            desperation = 1.5

        raid_prob = params.base_raid_prob * (1.0 + desperation)
        if rng.random() > raid_prob:
            continue

        effective_range = params.raid_range
        if attacker.has_longship:
            effective_range += params.longship_range_bonus

        targets = [
            s for s in alive
            if s is not attacker and s.alive and s.owner_id != attacker.owner_id
        ]

        land_range_sq = params.raid_range ** 2
        eff_range_sq = effective_range ** 2

        reachable = []
        for t in targets:
            dist_sq = attacker.distance_sq_to(t)
            if dist_sq <= land_range_sq:
                reachable.append(t)
            elif (attacker.has_longship and dist_sq <= eff_range_sq
                  and _can_reach_by_water(world, attacker.x, attacker.y,
                                          t.x, t.y, effective_range)):
                reachable.append(t)

        if not reachable:
            continue

        target = reachable[int(rng.integers(0, len(reachable)))]

        attacker_str = attacker.population * attacker.defense * (1.0 + attacker.tech_level * 0.1)
        defender_str = target.population * target.defense * (1.0 + target.tech_level * 0.1)

        if attacker_str > defender_str * params.raid_threshold:
            loot_food = target.food * params.loot_fraction
            loot_wealth = target.wealth * params.loot_fraction
            target.food -= loot_food
            target.wealth -= loot_wealth
            attacker.food += loot_food
            attacker.wealth += loot_wealth

            target.defense = max(0.1, target.defense - params.defense_damage)
            pop_loss = target.population * params.pop_damage_fraction
            target.population = max(0.1, target.population - pop_loss)

            if rng.random() < params.conquest_prob:
                target.owner_id = attacker.owner_id


def phase_trade(world: World, params: SimParams, rng: np.random.Generator) -> None:
    ports = [s for s in world.alive_settlements() if s.has_port]
    if len(ports) < 2:
        return

    trade_range_sq = params.trade_range ** 2

    for i, a in enumerate(ports):
        for b in ports[i + 1:]:
            if not (a.alive and b.alive):
                continue
            if a.owner_id != b.owner_id:
                continue
            if a.distance_sq_to(b) > trade_range_sq:
                continue

            eff = params.trade_efficiency
            food_avg = (a.food + b.food) / 2.0
            wealth_avg = (a.wealth + b.wealth) / 2.0

            a.food = a.food + (food_avg - a.food) * eff
            b.food = b.food + (food_avg - b.food) * eff
            a.wealth = a.wealth + (wealth_avg - a.wealth) * eff
            b.wealth = b.wealth + (wealth_avg - b.wealth) * eff

            if a.tech_level < b.tech_level:
                a.tech_level += (b.tech_level - a.tech_level) * params.tech_diffusion_rate
            else:
                b.tech_level += (a.tech_level - b.tech_level) * params.tech_diffusion_rate


def phase_winter(world: World, params: SimParams, rng: np.random.Generator) -> None:
    severity = params.base_winter_severity + float(rng.normal(0, params.winter_variance))
    severity = max(0.05, severity)

    collapse_list: list[Settlement] = []
    for s in world.alive_settlements():
        s.food -= severity * s.population * params.winter_food_loss
        if s.food <= 0 or s.population < params.min_population:
            collapse_list.append(s)

    for s in collapse_list:
        if not s.alive:
            continue
        s.alive = False
        world.set_terrain(s.x, s.y, TERRAIN_RUIN)

        survivors = [
            o for o in world.alive_settlements()
            if o.owner_id == s.owner_id
        ]
        if not survivors:
            survivors = world.alive_settlements()
        if survivors:
            nearest = min(survivors, key=lambda o: s.distance_to(o))
            if s.food > 0:
                nearest.food += s.food * 0.5
            if s.population > params.min_population:
                nearest.population += s.population * 0.3


def phase_environment(world: World, params: SimParams, rng: np.random.Generator) -> None:
    ruin_cells = [
        (x, y)
        for y in range(world.height)
        for x in range(world.width)
        if world.get_terrain(x, y) == TERRAIN_RUIN
    ]

    for rx, ry in ruin_cells:
        nearby_alive = [
            s for s in world.alive_settlements()
            if abs(s.x - rx) + abs(s.y - ry) <= 2
        ]

        if nearby_alive and rng.random() < params.reclaim_prob:
            patron = nearby_alive[int(rng.integers(0, len(nearby_alive)))]
            is_coastal = _is_ocean_adjacent(world, rx, ry)
            new_s = Settlement(
                x=rx, y=ry,
                population=1.0,
                food=patron.food * 0.2,
                wealth=patron.wealth * 0.2,
                defense=0.3,
                tech_level=patron.tech_level * 0.9,
                has_port=is_coastal,
                has_longship=False,
                alive=True,
                owner_id=patron.owner_id,
            )
            terrain = TERRAIN_PORT if is_coastal else TERRAIN_SETTLEMENT
            world.set_terrain(rx, ry, terrain)
            world.remove_settlement(rx, ry)
            world.add_settlement(new_s)
            patron.food *= 0.8
            patron.wealth *= 0.8
        elif rng.random() < params.forest_regrowth_prob:
            world.set_terrain(rx, ry, TERRAIN_FOREST)
        elif rng.random() < params.plains_regrowth_prob:
            world.set_terrain(rx, ry, TERRAIN_PLAINS)
