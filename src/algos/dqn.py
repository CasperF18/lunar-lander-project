from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils.logger import RunConfig
from src.scripts.evaluate import evaluate_policy


def make_env(env_id: str, seed: int, max_steps: int):
    def _init():
        env = gym.make(env_id)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def train_dqn(config: RunConfig, run_dir: Path, eval_episodes: int, render_eval: bool):
    vec_env = DummyVecEnv([make_env(config.env_id, config.seed, config.max_steps_per_episode)])

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        seed=config.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb")
    )

    model.learn(
        total_timesteps=int(config.extra["timesteps"]),
        tb_log_name=str("train"),
    )

    eval_result = evaluate_policy(
        model=model,
        env_id=config.env_id,
        seed=config.seed,
        max_steps=config.max_steps_per_episode,
        episodes=eval_episodes,
        render=render_eval,
    )

    return model, eval_result
