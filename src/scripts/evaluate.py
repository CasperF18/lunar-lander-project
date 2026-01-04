from __future__ import annotations

from typing import Any, Dict
import numpy as np


def evaluate_policy(model, env, seed: int, episodes: int) -> Dict[str, Any]:
    fail_counts = {"truncated": 0, "not_centered": 0, "not_both_legs": 0}

    returns = []
    lengths = []
    episode_details = []
    strict_successes = []
    soft_successes = []

    # Success window around pad center (x = 0)
    PAD_X_TOLERANCE = 0.2

    for ep in range(episodes):
        reset_results = env.env_method("reset", seed=seed + ep)
        raw_obs_list = [r[0] for r in reset_results]
        raw_obs = np.stack(raw_obs_list, axis=0).astype(np.float32)
        obs = env.normalize_obs(raw_obs) if hasattr(env, "normalize_obs") else raw_obs

        done = np.array([False])

        ep_return = 0.0
        ep_len = 0

        last_obs = obs
        last_truncated = False

        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            ep_return += float(reward[0])
            ep_len += 1

            info0 = infos[0]

            # I am checking here when an episode ends if it was because of time-limit
            if bool(done[0]):
                last_truncated = bool(info0.get("TimeLimit.truncated", False))
                terminal_obs = info0.get("terminal_observation", None)

                if terminal_obs is not None:
                    last_obs = np.array([terminal_obs], dtype=np.float32)
                else:
                    last_obs = obs
            else:
                last_obs = obs

        if hasattr(env, "unnormalize_obs"):
            final_raw_obs = env.unnormalize_obs(last_obs)[0]
        else:
            final_raw_obs = last_obs[0]

        # obs: [x, y, vx, vy, angle, ang_vel, leg1, leg2] - taken from their documentation
        final_leg1 = float(final_raw_obs[6]) > 0.5
        final_leg2 = float(final_raw_obs[7]) > 0.5
        final_x = float(final_raw_obs[0])

        final_both_legs = final_leg1 and final_leg2
        final_centered = abs(final_x) <= PAD_X_TOLERANCE

        if last_truncated:
            fail_counts["truncated"] += 1
        if not final_centered:
            fail_counts["not_centered"] += 1
        if not final_both_legs:
            fail_counts["not_both_legs"] += 1

        episode_strict_success = (
                (not last_truncated)
                and final_both_legs
                and final_centered
        )

        episode_soft_success = (
            (not last_truncated)
            and (final_leg1 or final_leg2)
            and abs(final_x) <= 0.3
        )

        # Mainly for debugging
        episode_details.append({
#            "episode": ep,
#            "return": ep_return,
#            "length": ep_len,
#            "centered_final": final_centered,
#            "truncated": last_truncated,
#           "success_eval": episode_success,
#            "leg1": final_leg1,
#            "leg2": final_leg2,
#            "leg1_value": final_leg1_value,
#            "leg2_value": final_leg2_value,
#            "final raw": final_raw,
        })

        returns.append(ep_return)
        lengths.append(ep_len)
        strict_successes.append(episode_strict_success)
        soft_successes.append(episode_soft_success)

    mean_return = sum(returns) / len(returns)
    mean_length = sum(lengths) / len(lengths)
    strict_success_rate = sum(strict_successes) / len(strict_successes)
    soft_success_rate = sum(soft_successes) / len(soft_successes)

    return {
        "eval_episodes": episodes,
        "mean_return": mean_return,
        "mean_length": mean_length,
        "strict_success_rate": strict_success_rate,
        "soft_success_rate": soft_success_rate,
        "returns": returns,
        "lengths": lengths,
        "strict_successes": strict_successes,
        "soft_successes": soft_successes,
        "episode_details": episode_details,
        "fail_counts": fail_counts,
    }
