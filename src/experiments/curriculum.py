from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, Dict, List

EnvKwargs = Dict[str, float | bool]


@dataclass(frozen=True)
class Stage:
    name: str
    timesteps: int
    sampler: Callable[[np.random.Generator], EnvKwargs]


def uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def final_distribution_sampler(rng: np.random.Generator) -> EnvKwargs:
    return {
        "enable_wind": True,
        "gravity": uniform(rng, -11.5, -9.5),
        "wind_power": uniform(rng, 8.0, 18.0),
        "turbulence_power": uniform(rng, 0.4, 1.6),
    }


def baseline_stages(total_timesteps: int) -> List[Stage]:
    return [
        Stage(
            name="baseline_final_dist",
            timesteps=total_timesteps,
            sampler=final_distribution_sampler,
        )
    ]


def parameter_wise_stages(total_timesteps: int) -> List[Stage]:
    # Split budget into 4 stages, can always change
    s0 = int(0.2 * total_timesteps)
    s1 = int(0.25 * total_timesteps)
    s2 = int(0.25 * total_timesteps)
    s3 = total_timesteps - (s0 + s1 + s2)

    def easy(rng):      # No wind, default-ish gravity
        return {"enable_wind": False, "gravity": -10.0}

    def wind_only(rng):
        return {
            "enable_wind": True,
            "gravity": -10.0,
            "wind_power": uniform(rng, 0.0, 10.0),
            "turbulence_power": 0.0,
        }

    def turbulence_only(rng):
        return {
            "enable_wind": True,
            "gravity": -10.0,
            "wind_power": 10.0,
            "turbulence_power": uniform(rng, 0.0, 1.0),
        }

    def final_hard(rng):
        return final_distribution_sampler(rng)

    return [
        Stage("easy_no_wind", s0, easy),
        Stage("ramp_wind", s1, wind_only),
        Stage("ramp_turbulence", s2, turbulence_only),
        Stage("final_distribution", s3, final_hard),
    ]


def joint_stages(total_timesteps: int) -> List[Stage]:
    s0 = int(0.2 * total_timesteps)
    s1 = int(0.3 * total_timesteps)
    s2 = int(0.3 * total_timesteps)
    s3 = total_timesteps - (s0 + s1 + s2)

    def easy(rng):
        return {"enable_wind": False, "gravity": -10.0}

    def medium(rng):
        return {
            "enable_wind": True,
            "gravity": uniform(rng, -10.5, -9.5),
            "wind_power": uniform(rng, 0.0, 10.0),
            "turbulence_power": uniform(rng, 0.0, 0.8),
        }

    def hardish(rng):
        return {
            "enable_wind": True,
            "gravity": uniform(rng, -11.0, -9.5),
            "wind_power": uniform(rng, 6.0, 16.0),
            "turbulence_power": uniform(rng, 0.2, 1.4),
        }

    def final_hard(rng):
        return final_distribution_sampler(rng)

    return [
        Stage("easy_no_wind", s0, easy),
        Stage("joint_medium", s1, medium),
        Stage("joint_hardish", s2, hardish),
        Stage("final_distribution", s3, final_hard),
    ]
