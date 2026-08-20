#!/usr/bin/env python3

import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from s10_stair_env import S10StairEnv


def main():

    print("=" * 70)
    print("S10 STAIR CLIMBING PPO TRAINING")
    print("=" * 70)

    # ----------------------------------------------------------
    # Environment
    # ----------------------------------------------------------

    env = S10StairEnv(
        episode_length=1000
    )

    print("\nChecking environment...")

    check_env(
        env,
        warn=True
    )

    print("\nEnvironment OK")

    print(
        "Observation:",
        env.observation_space
    )

    print(
        "Action:",
        env.action_space
    )

    # ----------------------------------------------------------
    # Create model directory
    # ----------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ----------------------------------------------------------
    # PPO
    # ----------------------------------------------------------

    print("\nCreating PPO model...")

    model = PPO(

        "MlpPolicy",

        env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.005,

        vf_coef=0.5,

        max_grad_norm=0.5,

        verbose=1,

        tensorboard_log="./tensorboard_stair/",
    )

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    total_steps = 1_000_000

    print("\n")
    print("=" * 70)
    print("STARTING STAIR CLIMBING TRAINING")
    print("=" * 70)

    print(
        f"Training for {total_steps:,} steps"
    )

    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    model.save(
        "models/s10_stair_ppo"
    )

    print("\n")
    print("=" * 70)
    print("STAIR TRAINING FINISHED")
    print("=" * 70)

    print(
        "Saved model:"
    )

    print(
        "models/s10_stair_ppo.zip"
    )

    env.close()


if __name__ == "__main__":

    main()
