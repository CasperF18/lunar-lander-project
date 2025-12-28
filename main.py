import argparse
from dataclasses import asdict

from src.algos.dqn import train_dqn
from src.utils.logger import RunConfig, RunLogger
from src.algos.ppo import train_ppo


def positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected an integer, got: {value}")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got: {value}")
    return ivalue


def parse_args():
    WIND = 0.0
    GRAVITY = -10
    TURBULENCE_POWER = 0.0

    p = argparse.ArgumentParser(
        description="Train and evaluate RL agents on LunarLander using Stable-Baselines3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Algorithm selection ---
    p.add_argument(
        "--algo",
        choices=["dqn", "ppo"],
        default="ppo",
        help="RL algorithm to use",
    )

    # --- Training parameters ---
    p.add_argument(
        "--gravity",
        type=float,
        default=GRAVITY,
    )

    p.add_argument(
        "--enable-wind",
        action="store_true",
    )

    p.add_argument(
        "--wind",
        type=float,
        default=WIND,
    )

    p.add_argument(
        "--turbulence",
        type=float,
        default=TURBULENCE_POWER,
    )

    # --- Curriculum ---
    p.add_argument(
        "--curriculum",
        choices=["baseline", "param", "joint"],
        default="baseline",
        help="Training schedule: baseline(final dist from start), param-wise curriculum, or joint curriculum"
    )

    # --- Environment ---
    p.add_argument(
        "--env-id",
        default="LunarLander-v3",
        help="Gymnasium environment ID",
    )

    # --- Training ---
    p.add_argument(
        "--timesteps",
        type=positive_int,
        default=200_000,
        help="Total training timesteps",
    )

    # --- Reproducibility / repeats ---
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed",
    )
    p.add_argument(
        "--runs",
        type=positive_int,
        default=1,
        help="Number of independent runs",
    )

    # --- Evaluation ---
    p.add_argument(
        "--eval-episodes",
        type=positive_int,
        default=20,
        help="Number of evaluation episodes after training",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes"
    )

    # --- Metadata ---
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional custom name for the run (otherwise auto-generated)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    for i in range(args.runs):
        run_seed = args.seed + i
        run_name = args.run_name or f"{args.algo}_{args.env_id}_{args.curriculum}_seed{run_seed}"

        config = RunConfig(
            run_name=run_name,
            env_id=args.env_id,
            seed=run_seed,
            max_steps_per_episode=1000,     # We can expose this as an argument later if wanted
            notes="SB3 training run",
            extra={
                "algo": args.algo,
                "timesteps": args.timesteps,
                "eval_episodes": args.eval_episodes,
                "render_eval": args.render,
                "env_kwargs": {
                    "gravity": args.gravity,
                    "enable_wind": args.enable_wind,
                    "wind_power": args.wind,
                    "turbulence_power": args.turbulence,
                },
                "curriculum": args.curriculum,
            },
        )

        logger = RunLogger(config)

        # Train + eval
        if args.algo == "ppo":
            model, eval_result = train_ppo(
                config=config,
                run_dir=logger.run_dir,
                eval_episodes=args.eval_episodes,
                render_eval=args.render,
            )
            model_filename = "model_ppo.zip"
        else:
            model, eval_result = train_dqn(
                config=config,
                run_dir=logger.run_dir,
                eval_episodes=args.eval_episodes,
                render_eval=args.render,
            )
            model_filename = "model_dqn.zip"

        model_path = logger.run_dir / model_filename
        model.save(str(model_path))

        logger.log_episode(
            episode_return=float(eval_result["mean_return"]),
            episode_length=int(eval_result["mean_length"]),
            terminated=True,
            truncated=False,
            info={
                "type": "evaluation_summary",
                "algo": args.algo,
                "model_path": str(model_path),
                **eval_result,
            },
        )

        logger.close()

        print(f"Finished run {i + 1}/{args.runs}: {run_name}")
        print(f"Saved to: {logger.run_dir}")

    print("All runs complete.")


if __name__ == "__main__":
    main()
