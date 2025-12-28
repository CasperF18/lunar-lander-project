"""
Final training distribution:
    enable_wind = True
    gravity ~ Uniform(-11.5, -9.5)
    wind_power ~ Uniform(8.0, 18.0)
    turbulence_power ~ Uniform(0.4, 1.6)

Unseen ranges will then be:
    gravity down to -12
    wind up to 20
    turbulence up to 2
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

EnvKwargs = Dict [str, float | bool]


@dataclass(frozen=True)
class EvalCase:
    name: str
    env_kwargs: EnvKwargs


def training_like_cases() -> List[EvalCase]:
    # Just a small test set inside the final eval distribution to see if the training went well on seen parameters
    return [
        EvalCase("trainlike_1", {"enable_wind": True, "gravity": -10.0, "wind_power": 10.0, "turbulence_power": 0.6}),
        EvalCase("trainlike_2", {"enable_wind": True, "gravity": -11.0, "wind_power": 16.0, "turbulence_power": 1.2}),
        EvalCase("trainlike_3", {"enable_wind": True, "gravity": -9.7,  "wind_power": 12.0, "turbulence_power": 1.0}),
        EvalCase("trainlike_4", {"enable_wind": True, "gravity": -11.3, "wind_power": 9.0,  "turbulence_power": 0.5}),
    ]


def unseen_combo_cases() -> List[EvalCase]:
    # Anti-correlated combos that curriculum might not have seen often during training
    return [
        EvalCase("combo_high_wind_low_turb", {"enable_wind": True, "gravity": -10.5, "wind_power": 18.0, "turbulence_power": 0.1}),
        EvalCase("combo_low_wind_high_turb", {"enable_wind": True, "gravity": -10.5, "wind_power": 2.0,  "turbulence_power": 1.8}),
        EvalCase("combo_mid_wind_extreme_turb", {"enable_wind": True, "gravity": -10.0, "wind_power": 10.0, "turbulence_power": 2.0}),
    ]


def unseen_range_cases() -> List[EvalCase]:
    # Slight extrapolation beyond FINAL TRAIN distribution but within env intended bounds
    return [
        EvalCase("range_max_wind", {"enable_wind": True, "gravity": -10.5, "wind_power": 19.95, "turbulence_power": 1.0}),
        EvalCase("range_max_turb", {"enable_wind": True, "gravity": -10.5, "wind_power": 12.0, "turbulence_power": 1.95}),
        EvalCase("range_max_gravity", {"enable_wind": True, "gravity": -11.95, "wind_power": 14.0, "turbulence_power": 1.2}),
    ]


def all_eval_cases() -> List[EvalCase]:
    return training_like_cases() + unseen_combo_cases() + unseen_range_cases()