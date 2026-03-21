from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class SimParams:
    # Growth
    forest_food_yield: float = 0.8
    plains_food_yield: float = 0.3
    port_food_multiplier: float = 1.3
    growth_rate: float = 0.15
    food_consumption: float = 0.5
    expansion_threshold: float = 4.0
    expansion_prob: float = 0.3
    port_threshold: float = 3.0
    longship_threshold: float = 2.0

    # Conflict
    raid_range: int = 3
    longship_range_bonus: int = 4
    base_raid_prob: float = 0.15
    raid_threshold: float = 0.8
    conquest_prob: float = 0.2
    loot_fraction: float = 0.3
    defense_damage: float = 0.2
    pop_damage_fraction: float = 0.25

    # Trade
    trade_range: float = 8.0
    trade_efficiency: float = 0.15
    tech_diffusion_rate: float = 0.05

    # Winter
    base_winter_severity: float = 0.4
    winter_variance: float = 0.15
    winter_food_loss: float = 0.3
    min_population: float = 0.3
    base_collapse_prob: float = 0.0

    # Food spoilage
    food_decay: float = 1.0  # 1.0 = no spoilage (default preserves old behavior)

    # Environment
    reclaim_prob: float = 0.12
    forest_regrowth_prob: float = 0.08
    plains_regrowth_prob: float = 0.15
    forest_spread_prob: float = 0.0

    def randomize(self, rng: np.random.Generator, scale: float = 0.3) -> "SimParams":
        d = asdict(self)
        new_d = {}
        for k, v in d.items():
            if isinstance(v, float):
                noise = rng.normal(0, abs(v) * scale)
                new_val = v + noise
                if k in ("base_raid_prob", "expansion_prob", "conquest_prob",
                         "reclaim_prob", "forest_regrowth_prob", "plains_regrowth_prob",
                         "forest_spread_prob", "food_decay", "base_collapse_prob",
                         "trade_efficiency", "tech_diffusion_rate"):
                    new_val = float(np.clip(new_val, 0.01, 0.99))
                elif k in ("port_food_multiplier",):
                    new_val = max(1.0, new_val)
                else:
                    new_val = max(0.01, new_val)
                new_d[k] = new_val
            else:
                new_d[k] = v
        return SimParams(**new_d)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def gt_tuned(cls) -> "SimParams":
        return cls(
            expansion_prob=0.30,
            expansion_threshold=3.5,
            food_consumption=0.7,
            growth_rate=0.10,
            base_raid_prob=0.20,
            raid_threshold=0.7,
            pop_damage_fraction=0.30,
            defense_damage=0.25,
            loot_fraction=0.35,
            base_winter_severity=0.6,
            winter_variance=0.20,
            winter_food_loss=0.45,
            min_population=0.5,
            base_collapse_prob=0.03,
            food_decay=0.5,
            reclaim_prob=0.10,
            forest_regrowth_prob=0.08,
            plains_regrowth_prob=0.25,
            forest_spread_prob=0.001,
        )
