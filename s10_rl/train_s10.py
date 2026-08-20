#!/usr/bin/env python3

import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from s10_env import S10Env


def main():

    print("=" * 60)
    print("S10 PPO TRAINING")
    print("=" * 60)

    env = S10Env(
        render_mode=None,
        episode_length=1000,
    )

    print("\nChecking Gymnasium environment...")
    check_env(env, warn=True)

    env = Monitor(env)

    print("\nEnvironment:")
    print("Observation:", env.observation_space)
    print("Action:", env.action_space)

    print("\nCreating PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        n_steps=2048,
        batch_size=256,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,
        vf_coef=0.5,

        max_grad_norm=0.5,

        verbose=1,

        device="cuda",

        tensorboard_log="./tensorboard/",
    )

    print("\nStarting short training test...")
    print("Training for 10,000 steps.")

    model.learn(
        total_timesteps=10_000,
        progress_bar=True,
    )

    os.makedirs("models", exist_ok=True)

    model.save(
        "models/s10_ppo_test"
    )

    print("\nTraining finished.")

    print(
        "Saved model:"
        " models/s10_ppo_test.zip"
    )

    env.close()


if __name__ == "__main__":
    main()
