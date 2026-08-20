#!/usr/bin/env python3

from stable_baselines3 import PPO

from s10_stair_env import S10StairEnv


def main():

    env = S10StairEnv(
        render_mode="human",
        episode_length=1000
    )

    model = PPO.load(
        "models/s10_stair_ppo"
    )

    obs, info = env.reset(
        options={
            "deterministic": True
        }
    )

    for step in range(2000):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        env.render()

        if step % 20 == 0:

            print(
                f"step={step:4d} "
                f"x={info['x']:.3f} "
                f"height={info['height']:.3f} "
                f"stair={info['stair_height']:.3f} "
                f"vx={info['x_velocity']:.3f} "
                f"reward={reward:.3f}"
            )

        if terminated or truncated:

            print(
                "Episode finished:",
                info
            )

            obs, info = env.reset(
                options={
                    "deterministic": True
                }
            )

    env.close()


if __name__ == "__main__":

    main()
