from __future__ import annotations

from typing import Any, Dict
import numpy as np


def evaluate_policy(model, env, seed: int, episodes: int) -> Dict[str, Any]:
    returns = []
    lengths = []
    episode_details = []
    successes = []

    # Success window around pad center (x = 0)
    PAD_X_TOLERANCE = 0.2

    # “Stable landing” thresholds (diagnostic)
    MAX_ABS_VX = 0.5
    MAX_ABS_VY = 0.5
    MAX_ABS_ANGLE = 0.2

    for ep in range(episodes):
        env.env_method("reset", seed=seed + ep)
        obs = env.reset()

        done = np.array([False])

        ep_return = 0.0
        ep_len = 0

        last_obs_norm = obs
        last_truncated = False
        last_info = None

        best_contact = None
        best_contact_step = None
        best_contact_score = float("inf")

        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            ep_return += float(reward[0])
            ep_len += 1

            last_obs_norm = obs
            last_info = infos[0]

            # I am checking here when an episode ends if it was because of time-limit
            if bool(done[0]):
                last_truncated = bool(last_info.get("TimeLimit.truncated", False))

            if hasattr(env, "unnormalize_obs"):
                raw_obs = env.unnormalize_obs(obs.copy())
            else:
                raw_obs = obs

            # obs: [x, y, vx, vy, angle, ang_vel, leg1, leg2] - taken from their documentation
            raw = raw_obs[0]
            x = float(raw[0])
            y = float(raw[1])
            vx = float(raw[2])
            vy = float(raw[3])
            angle = float(raw[4])
            ang_vel = float(raw[5])
            leg1 = bool(raw[6])
            leg2 = bool(raw[7])

            if leg1 and leg2:
                score = abs(x) + MAX_ABS_VX * abs(vx) + MAX_ABS_VY * abs(vy) + MAX_ABS_ANGLE * abs(angle)
                if score < best_contact_score:
                    best_contact_score = score
                    best_contact_step = ep_len
                    best_contact = {"x": x, "y": y, "vx": vx, "vy": vy, "angle": angle, "angle_vel": ang_vel}

        if hasattr(env, "unnormalize_obs"):
            final_raw_obs = env.unnormalize_obs(last_obs_norm.copy())[0]
        else:
            final_raw_obs = last_obs_norm[0]

        final_leg1 = bool(final_raw_obs[6])
        final_leg2 = bool(final_raw_obs[7])

        centered_strict = (best_contact is not None) and (abs(best_contact["x"]) <= PAD_X_TOLERANCE)
        episode_success = (not last_truncated) and centered_strict and final_leg1 and final_leg2

        episode_details.append({
            "episode": ep,
            "return": ep_return,
            "length": ep_len,
            "best_contact_step": best_contact_step,
            "best_contact": best_contact,
            "centered_final": centered_strict,
            "truncated": last_truncated,
            "success_eval": episode_success,
        })

        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(episode_success)

    mean_return = sum(returns) / len(returns)
    mean_length = sum(lengths) / len(lengths)
    success_rate = sum(1 for s in successes if s) / len(successes)

    return {
        "eval_episodes": episodes,
        "mean_return": mean_return,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
        "episode_details": episode_details
    }
