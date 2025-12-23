from __future__ import annotations

from typing import Any, Dict, List
import gymnasium as gym
from gymnasium.wrappers import TimeLimit


def evaluate_policy(model, env_id: str, seed: int, max_steps: int, episodes: int, render: bool) -> Dict[str, Any]:
    env = gym.make(env_id, render_mode="human" if render else None)
    env = TimeLimit(env, max_episode_steps=max_steps)

    returns = []
    lengths = []

    obs, info = env.reset(seed=seed)
    for ep in range(episodes):
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_len)
        obs, info = env.reset()

    env.close()

    mean_return = sum(returns) / len(returns)
    mean_length = sum(lengths) / len(lengths)
    return {
        "eval_episodes": episodes,
        "mean_return": mean_return,
        "mean_length": mean_length,
        "returns": returns,
        "lengths": lengths
    }