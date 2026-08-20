#!/usr/bin/env python3

import time
import numpy as np
from stable_baselines3 import PPO

from s10_env import S10Env


MODEL_PATH = "models/s10_ppo_test.zip"


def main():

    print("=" * 60)
    print("S10 PPO POLICY TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Create environment
    # ---------------------------------------------------------

    env = S10Env(
        render_mode="human",
        episode_length=2000,
    )

    print("Environment created")

    # ---------------------------------------------------------
    # Load trained PPO model
    # ---------------------------------------------------------

    print("Loading:", MODEL_PATH)

    model = PPO.load(
        MODEL_PATH,
        env=env,
    )

    print("Model loaded successfully")

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    obs, info = env.reset(
        options={"deterministic": True}
    )

    # Use a fixed forward command.
    #
    # Observation layout:
    #
    # [0:3]   angular velocity
    # [3:6]   projected gravity
    # [6:9]   velocity command
    # [9:25]  joint position
    # [25:41] joint velocity
    # [41:57] previous action
    #
    # command = [forward, sideways, yaw]

    env.command[:] = np.array(
        [0.30, 0.0, 0.0],
        dtype=np.float32
    )

    # Recalculate observation with the new command
    obs = env._get_observation()

    print()
    print("Command:")
    print("  Forward : 0.30 m/s")
    print("  Side    : 0.00 m/s")
    print("  Yaw     : 0.00 rad/s")
    print()

    print("Starting simulation...")
    print("Close the MuJoCo window or press Ctrl+C to stop.")
    print()

    total_reward = 0.0

    try:

        for step in range(2000):

            # -------------------------------------------------
            # Get action from PPO
            # -------------------------------------------------

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            # -------------------------------------------------
            # Step simulation
            # -------------------------------------------------

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            # -------------------------------------------------
            # Render
            # -------------------------------------------------

            env.render()

            # -------------------------------------------------
            # Print status every 25 policy steps
            # -------------------------------------------------

            if step % 25 == 0:

                print(
                    f"step={step:4d} "
                    f"x_vel={info['x_velocity']:+.3f} "
                    f"y_vel={info['y_velocity']:+.3f} "
                    f"yaw_vel={info['yaw_velocity']:+.3f} "
                    f"reward={reward:+.3f} "
                    f"height={env.data.qpos[2]:.3f}"
                )

            # -------------------------------------------------
            # Stop if robot falls
            # -------------------------------------------------

            if terminated:

                print()
                print("Robot fell.")
                print(
                    f"Episode ended at step {step}"
                )

                break

            if truncated:

                print()
                print("Episode completed.")

                break

            # Small delay so viewer is easier to observe
            time.sleep(0.005)

    except KeyboardInterrupt:

        print()
        print("Stopped by user.")

    finally:

        env.close()

    print()
    print("=" * 60)
    print("TEST FINISHED")
    print("=" * 60)
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()
