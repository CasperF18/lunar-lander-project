from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


def make_env(env_id: str, seed: int, max_steps: int, render_mode: str | None = None, env_kwargs: dict | None = None):
    env_kwargs = env_kwargs or {}

    def _init():
        env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=max_steps,
            **env_kwargs,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init
