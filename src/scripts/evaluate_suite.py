from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.scripts.evaluate import evaluate_policy
from src.scripts.make_env import make_env


def make_eval_vec_env(env_id: str, seed: int, max_steps: int, vecnorm_path: Path, env_kwargs: dict, render: bool):
    render_mode = "human" if render else None
    venv = DummyVecEnv([make_env(env_id, seed, max_steps, render_mode=render_mode, env_kwargs=env_kwargs)])
    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def _group_of_case(name: str) -> str:
    n = name.lower()
    if n.startswith("trainlike"):
        return "trainlike"
    if n.startswith("combo") or "combo" in n:
        return "combo"
    if n.startswith("range") or "range" in n:
        return "range"
    return "other"


def evaluate_suite(model, env_id: str, seed: int, max_steps: int, vecnorm_path: Path, cases: List[tuple[str, dict]],
                   episodes_per_case: int, render: bool = False, render_episodes: int = 4) -> Dict[str, Any]:
    results = []

    # render only the first case of each group
    rendered_groups = set()

    for i, (name, env_kwargs) in enumerate(cases):
        group = _group_of_case(name)

        render_this = False
        eps = episodes_per_case

        if render and group in {"trainlike", "combo", "range"} and group not in rendered_groups:
            render_this = True
            eps = render_episodes
            rendered_groups.add(group)

        env = make_eval_vec_env(env_id, seed + 1000 + i, max_steps, vecnorm_path, env_kwargs, render_this)
        r = evaluate_policy(model=model, env=env, seed=seed + 1000 + i, episodes=eps)
        env.close()

        results.append({
            "case": name,
            "env_kwargs": env_kwargs,
            "mean_return": r["mean_return"],
            "success_rate": r["success_rate"],
            "mean_length": r["mean_length"],
            "episodes_used": eps,
            "rendered": render_this,
        })

    mean_return = sum(x["mean_return"] for x in results) / len(results)
    mean_success = sum(x["success_rate"] for x in results) / len(results)
    mean_length = sum(x["mean_length"] for x in results) / len(results)

    return {
        "episodes_per_case": episodes_per_case,
        "num_cases": len(results),
        "mean_return_over_cases": mean_return,
        "mean_success_over_cases": mean_success,
        "mean_length_over_cases": mean_length,
        "cases": results,
    }
