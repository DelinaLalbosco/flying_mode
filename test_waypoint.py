from stable_baselines3 import PPO

from s10_waypoint_env import S10WaypointEnv


XML_PATH = "/absolute/path/to/S10_track.xml"


env = S10WaypointEnv(
    xml_path=XML_PATH,
    waypoint_distance=2.0,
    episode_length=1000,
    render_mode="human",
)


model = PPO.load(
    "s10_waypoint_ppo"
)


for episode in range(10):

    obs, info = env.reset()

    done = False

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = \
            env.step(action)

        env.render()

        done = (
            terminated
            or truncated
        )

    print(
        f"Episode {episode + 1}: "
        f"distance={info['distance_to_waypoint']:.3f}, "
        f"goal={info['goal_reached']}, "
        f"x={info['x']:.3f}"
    )


env.close()
