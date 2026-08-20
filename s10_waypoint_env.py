import os
import time
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class S10WaypointEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    JOINT_NAMES = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
    ]

    DEFAULT_POS = np.array([
         0.0, -0.30,  0.60, 0.0,
         0.0, -0.30,  0.60, 0.0,
         0.0,  0.30, -0.60, 0.0,
         0.0,  0.30, -0.60, 0.0,
    ], dtype=np.float32)

    ACTION_SCALE = np.array([
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
    ], dtype=np.float32)

    KP = np.array([80.0, 80.0, 80.0, 0.0] * 4,
                  dtype=np.float32)

    KD = np.array([2.0, 2.0, 2.0, 0.6] * 4,
                  dtype=np.float32)

    POLICY_DT = 0.02

    def __init__(
        self,
        xml_path,
        waypoint_distance=2.0,
        episode_length=1000,
        render_mode=None,
    ):

        super().__init__()

        self.xml_path = os.path.abspath(xml_path)
        self.waypoint_distance = waypoint_distance
        self.episode_length = episode_length
        self.render_mode = render_mode

        # --------------------------------------------------
        # MuJoCo
        # --------------------------------------------------

        self.model = mujoco.MjModel.from_xml_path(
            self.xml_path
        )

        self.data = mujoco.MjData(self.model)

        self.physics_dt = float(
            self.model.opt.timestep
        )

        self.decimation = max(
            1,
            int(round(self.POLICY_DT / self.physics_dt))
        )

        # --------------------------------------------------
        # Joint addresses
        # --------------------------------------------------

        self.qpos_addr = np.zeros(16, dtype=np.int32)
        self.qvel_addr = np.zeros(16, dtype=np.int32)

        for i, name in enumerate(self.JOINT_NAMES):

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name
            )

            if joint_id < 0:
                raise RuntimeError(
                    f"Joint not found: {name}"
                )

            self.qpos_addr[i] = \
                self.model.jnt_qposadr[joint_id]

            self.qvel_addr[i] = \
                self.model.jnt_dofadr[joint_id]

        # --------------------------------------------------
        # Observation
        #
        # 3 angular velocity
        # 3 gravity
        # 16 joint position
        # 16 joint velocity
        # 16 previous action
        # 2 waypoint position
        #
        # Total = 56
        # --------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(56,),
            dtype=np.float32
        )

        # 16 motor commands

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32
        )

        self.previous_action = np.zeros(
            16,
            dtype=np.float32
        )

        self.step_count = 0

        self.goal_reached = False

        self.previous_distance = 0.0

        self.viewer = None

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # Robot starting position

        self.data.qpos[0:3] = [
            0.0,
            0.0,
            0.50
        ]

        # Upright quaternion

        self.data.qpos[3:7] = [
            1.0,
            0.0,
            0.0,
            0.0
        ]

        # Standing pose

        for i in range(16):

            self.data.qpos[
                self.qpos_addr[i]
            ] = self.DEFAULT_POS[i]

        self.data.qvel[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.previous_action[:] = 0.0

        self.step_count = 0

        self.goal_reached = False

        # --------------------------------------------------
        # Waypoint
        # --------------------------------------------------

        self.waypoint = np.array([
            self.waypoint_distance,
            0.0
        ])

        self.previous_distance = \
            self._distance_to_waypoint()

        return (
            self._get_observation(),
            self._get_info()
        )

    # ======================================================
    # WAYPOINT
    # ======================================================

    def _get_waypoint_relative(self):

        robot_x = float(
            self.data.qpos[0]
        )

        robot_y = float(
            self.data.qpos[1]
        )

        dx = self.waypoint[0] - robot_x
        dy = self.waypoint[1] - robot_y

        # Normalize

        dx = np.clip(dx / 2.0, -1.0, 1.0)
        dy = np.clip(dy / 2.0, -1.0, 1.0)

        return np.array(
            [dx, dy],
            dtype=np.float32
        )

    def _distance_to_waypoint(self):

        dx = (
            self.waypoint[0]
            - self.data.qpos[0]
        )

        dy = (
            self.waypoint[1]
            - self.data.qpos[1]
        )

        return float(
            np.sqrt(dx * dx + dy * dy)
        )

    # ======================================================
    # OBSERVATION
    # ======================================================

    def _get_observation(self):

        # Angular velocity

        omega = self.data.qvel[3:6].copy()
        omega *= 0.25

        # Orientation

        rotation = np.zeros(
            (3, 3),
            dtype=np.float64
        )

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            self.data.qpos[3:7]
        )

        gravity = rotation.T @ np.array([
            0.0,
            0.0,
            -1.0
        ])

        # Joint states

        joint_pos = np.zeros(16)
        joint_vel = np.zeros(16)

        for i in range(16):

            joint_pos[i] = \
                self.data.qpos[
                    self.qpos_addr[i]
                ]

            joint_vel[i] = \
                self.data.qvel[
                    self.qvel_addr[i]
                ]

        # Wheel position is ignored

        joint_pos[3] = 0.0
        joint_pos[7] = 0.0
        joint_pos[11] = 0.0
        joint_pos[15] = 0.0

        joint_pos -= self.DEFAULT_POS

        joint_vel *= 0.05

        waypoint = self._get_waypoint_relative()

        obs = np.concatenate([
            omega,
            gravity,
            joint_pos,
            joint_vel,
            self.previous_action,
            waypoint
        ])

        return obs.astype(np.float32)

    # ======================================================
    # ACTION
    # ======================================================

    def _apply_action(self, action):

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # Smooth actions

        filtered_action = (
            0.75 * self.previous_action
            + 0.25 * action
        )

        target = (
            filtered_action
            * self.ACTION_SCALE
            + self.DEFAULT_POS
        )

        torque = np.zeros(16)

        # Legs

        for i in range(12):

            q = self.data.qpos[
                self.qpos_addr[i]
            ]

            dq = self.data.qvel[
                self.qvel_addr[i]
            ]

            torque[i] = (
                self.KP[i]
                * (target[i] - q)
                - self.KD[i] * dq
            )

        # Wheels

        for leg in range(4):

            i = 4 * leg + 3

            dq = self.data.qvel[
                self.qvel_addr[i]
            ]

            desired_velocity = (
                filtered_action[i]
                * self.ACTION_SCALE[i]
            )

            torque[i] = self.KD[i] * (
                desired_velocity - dq
            )

        torque[:12] = np.clip(
            torque[:12],
            -50.0,
            50.0
        )

        torque[3::4] = np.clip(
            torque[3::4],
            -14.0,
            14.0
        )

        self.data.ctrl[:] = torque

        return filtered_action

    # ======================================================
    # FALL
    # ======================================================

    def _is_fallen(self):

        height = float(
            self.data.qpos[2]
        )

        if height < 0.25:
            return True

        rotation = np.zeros(
            (3, 3),
            dtype=np.float64
        )

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            self.data.qpos[3:7]
        )

        upright = rotation[2, 2]

        return upright < 0.45

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        previous_distance = \
            self._distance_to_waypoint()

        filtered_action = \
            self._apply_action(action)

        # Physics

        for _ in range(self.decimation):

            mujoco.mj_step(
                self.model,
                self.data
            )

        self.previous_action = \
            filtered_action.copy()

        self.step_count += 1

        current_distance = \
            self._distance_to_waypoint()

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        progress = (
            previous_distance
            - current_distance
        )

        # --------------------------------------------------
        # Goal
        # --------------------------------------------------

        self.goal_reached = (
            current_distance < 0.20
            and not self._is_fallen()
        )

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        reward = 0.0

        # Getting closer is the main objective

        reward += 20.0 * progress

        # Encourage forward velocity

        vx = float(
            self.data.qvel[0]
        )

        reward += 1.0 * vx

        # Small stability reward

        rotation = np.zeros(
            (3, 3),
            dtype=np.float64
        )

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            self.data.qpos[3:7]
        )

        upright = rotation[2, 2]

        reward += 1.0 * max(
            0.0,
            upright
        )

        # Small action penalty

        reward -= 0.002 * np.sum(
            np.asarray(action) ** 2
        )

        # Falling

        fallen = self._is_fallen()

        if fallen:
            reward -= 20.0

        # Reaching waypoint

        if self.goal_reached:
            reward += 100.0

        terminated = (
            fallen
            or self.goal_reached
        )

        truncated = (
            self.step_count
            >= self.episode_length
        )

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    # ======================================================
    # INFO
    # ======================================================

    def _get_info(self):

        return {
            "x": float(self.data.qpos[0]),
            "y": float(self.data.qpos[1]),
            "distance_to_waypoint":
                self._distance_to_waypoint(),
            "goal_reached":
                self.goal_reached,
            "step":
                self.step_count
        }

    # ======================================================
    # RENDER
    # ======================================================

    def render(self):

        if self.render_mode != "human":
            return

        if self.viewer is None:

            import mujoco.viewer

            self.viewer = \
                mujoco.viewer.launch_passive(
                    self.model,
                    self.data
                )

        if self.viewer.is_running():
            self.viewer.sync()

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        if self.viewer is not None:

            try:
                self.viewer.close()
            except Exception:
                pass

            self.viewer = None
