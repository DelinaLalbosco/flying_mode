#!/usr/bin/env python3

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from s10_stand_env import S10StandEnv


def main():

    print("=" * 60)
    print("S10 STANDING PPO TRAINING")
    print("=" * 60)

    # ----------------------------------------------------------
    # Create environment
    # ----------------------------------------------------------

    env = S10StandEnv(
        episode_length=1000
    )

    print("\nChecking environment...")

    check_env(
        env,
        warn=True
    )

    print("Environment OK")

    print(
        "Observation:",
        env.observation_space
    )

    print(
        "Action:",
        env.action_space
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

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,

        verbose=1,

        tensorboard_log="./tensorboard_stand/",
    )

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    total_steps = 300_000

    print("\n" + "=" * 60)
    print("STARTING STANDING TRAINING")
    print("=" * 60)

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
        "models/s10_stand_ppo"
    )

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        "Saved model:"
        " models/s10_stand_ppo.zip"
    )

    env.close()


if __name__ == "__main__":
    main()
