from __future__ import annotations

from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from src.utils.logger import RunConfig, RunLogger
from src.scripts.evaluate_suite import evaluate_suite
from src.scripts.make_env import make_env
from src.experiments.curriculum import baseline_stages, parameter_wise_stages, joint_stages
from src.experiments.eval_suites import all_eval_cases


def _build_train_env(config: RunConfig, stage_seed_offset: int, env_kwargs: dict, vecnorm_path: Path | None):
    n_envs = 8

    raw_env = SubprocVecEnv([
        make_env(
            config.env_id,
            config.seed + i + stage_seed_offset,
            config.max_steps_per_episode,
            render_mode=None,
            env_kwargs=env_kwargs,
        )
        for i in range(n_envs)
    ])

    if vecnorm_path is not None and vecnorm_path.exists():
        train_env = VecNormalize.load(str(vecnorm_path), raw_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    return train_env


def train_ppo(config: RunConfig, run_dir: Path, eval_episodes: int, render_eval: bool, logger: RunLogger):
    total_timesteps = int(config.extra["timesteps"])
    curriculum_type = (config.extra or {}).get("curriculum", "baseline")

    eval_every = int((config.extra or {}).get("eval_every", 100_000))
    checkpoint_episodes = int((config.extra or {}).get("checkpoint_episodes", 5))

    if curriculum_type == "param":
        stages = parameter_wise_stages(total_timesteps)
    elif curriculum_type == "joint":
        stages = joint_stages(total_timesteps)
    else:
        stages = baseline_stages(total_timesteps)

    rng = np.random.default_rng(config.seed)
    vecnorm_path = run_dir / "vec_normalize.pkl"

    model: PPO | None = None
    trained_steps = 0

    cases = [(c.name, c.env_kwargs) for c in all_eval_cases()]
    last_suite = None

    for si, stage in enumerate(stages):
        stage_env_kwargs = stage.sampler(rng)

        # Building and reloading VecNormalize so stats persist across stages
        train_env = _build_train_env(
            config=config,
            stage_seed_offset=si * 10_000,
            env_kwargs=stage_env_kwargs,
            vecnorm_path=vecnorm_path if vecnorm_path.exists() else None,
        )

        if model is None:
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                seed=config.seed,
                verbose=1,
                tensorboard_log=str(run_dir / "tb"),
            )
        else:
            model.set_env(train_env)

        remaining = stage.timesteps
        is_final_stage = (si == len(stages) - 1)

        while remaining > 0:
            chunk = min(eval_every, remaining)

            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                tb_log_name="train",
            )

            trained_steps += chunk
            remaining -= chunk

            train_env.save(str(vecnorm_path))

            is_final_checkpoint = is_final_stage and (remaining == 0)
            episodes_this_eval = eval_episodes if is_final_checkpoint else checkpoint_episodes

            last_suite = evaluate_suite(
                model=model,
                env_id=config.env_id,
                seed=config.seed,
                max_steps=config.max_steps_per_episode,
                vecnorm_path=vecnorm_path,
                cases=cases,
                episodes_per_case=episodes_this_eval,
                render=render_eval and is_final_stage,
                render_episodes=4
            )

            logger.log_episode(
                episode_return=float(last_suite["mean_return_over_cases"]),
                episode_length=int(last_suite["mean_length_over_cases"]),
                terminated=True,
                truncated=False,
                info={
                    "type": "eval_checkpoint",
                    "algo": "ppo",
                    "curriculum": curriculum_type,
                    "run_seed": config.seed,
                    "stage_index": si,
                    "stage_name": stage.name,
                    "stage_timesteps": stage.timesteps,
                    "trained_steps": trained_steps,
                    "actual_timesteps": int(model.num_timesteps),
                    "chunk_timesteps": chunk,
                    "checkpoint_kind": "final" if is_final_checkpoint else "mid",
                    "episodes_per_case": episodes_this_eval,
                    "train_env_kwargs": stage_env_kwargs,
                    "suite": last_suite,
                },
            )

        train_env.close()

    assert model is not None
    assert last_suite is not None

    eval_result = {
        "actual_timesteps": int(model.num_timesteps),
        "eval_episodes": last_suite["episodes_per_case"] * last_suite["num_cases"],
        "mean_return": last_suite["mean_return_over_cases"],
        "mean_length": last_suite["mean_length_over_cases"],
        "strict_success_rate": last_suite["mean_strict_success_over_cases"],
        "soft_success_rate": last_suite["mean_soft_success_over_cases"],
        "suite": last_suite,
        "curriculum": curriculum_type,
        "stages": [{"name": s.name, "timesteps": s.timesteps} for s in stages],
    }

    return model, eval_result
