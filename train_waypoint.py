from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from s10_waypoint_env import S10WaypointEnv


XML_PATH = "/absolute/path/to/S10_track.xml"


env = S10WaypointEnv(
    xml_path=XML_PATH,
    waypoint_distance=2.0,
    episode_length=1000,
)


# Check that the Gym environment is valid

check_env(env)

print("Observation:", env.observation_space)
print("Action:", env.action_space)


# --------------------------------------------------
# PPO
# --------------------------------------------------

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
    verbose=1,
    tensorboard_log="./tensorboard/"
)


# --------------------------------------------------
# Train
# --------------------------------------------------

model.learn(
    total_timesteps=500_000
)


# --------------------------------------------------
# Save
# --------------------------------------------------

model.save(
    "s10_waypoint_ppo"
)

env.close()

print("Training finished.")
