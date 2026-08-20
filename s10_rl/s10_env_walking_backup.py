#!/usr/bin/env python3

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class S10Env(gym.Env):
    """
    S10 locomotion environment.

    Observation: 57
        3  base angular velocity
        3  projected gravity
        3  velocity command
        16 joint position error
        16 joint velocity
        16 previous action

    Action: 16
        12 leg position actions
        4 wheel velocity actions
    """

    metadata = {"render_modes": ["human"]}

    JOINT_NAMES = [
        "fl_hipx_joint",
        "fl_hipy_joint",
        "fl_knee_joint",
        "fl_wheel_joint",

        "fr_hipx_joint",
        "fr_hipy_joint",
        "fr_knee_joint",
        "fr_wheel_joint",

        "hl_hipx_joint",
        "hl_hipy_joint",
        "hl_knee_joint",
        "hl_wheel_joint",

        "hr_hipx_joint",
        "hr_hipy_joint",
        "hr_knee_joint",
        "hr_wheel_joint",
    ]

    DEFAULT_POS = np.array([
         0.0, -0.3,  0.6, 0.0,
         0.0, -0.3,  0.6, 0.0,
         0.0,  0.3, -0.6, 0.0,
         0.0,  0.3, -0.6, 0.0,
    ], dtype=np.float32)

    ACTION_SCALE = np.array([
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
    ], dtype=np.float32)

    # Same scales used by Deep Robotics policy runner
    OMEGA_SCALE = 0.25
    DOF_VEL_SCALE = 0.05

    KP = np.array(
        [80.0, 80.0, 80.0, 0.0] * 4,
        dtype=np.float32
    )

    KD = np.array(
        [2.0, 2.0, 2.0, 0.6] * 4,
        dtype=np.float32
    )

    def __init__(
        self,
        xml_path=None,
        render_mode=None,
        episode_length=1000,
    ):
        super().__init__()

        if xml_path is None:
            xml_path = os.path.abspath(
                "../src/S10_sdk_deploy/"
                "S10_description/s10_mjcf/mjcf/S10.xml"
            )

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.episode_length = episode_length

        print("Loading:", self.xml_path)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Policy update period = 20 ms = 50 Hz
        self.policy_dt = 0.02

        # MuJoCo timestep comes from XML.
        # We execute enough physics steps to approximately reach 20 ms.
        self.physics_dt = self.model.opt.timestep

        self.decimation = max(
            1,
            int(round(self.policy_dt / self.physics_dt))
        )

        print("MuJoCo timestep:", self.physics_dt)
        print("Policy timestep:", self.policy_dt)
        print("Decimation:", self.decimation)

        # Find joint qpos/qvel addresses.
        self.qpos_addr = np.zeros(16, dtype=np.int32)
        self.qvel_addr = np.zeros(16, dtype=np.int32)

        for i, name in enumerate(self.JOINT_NAMES):
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if joint_id < 0:
                raise RuntimeError(
                    f"Joint not found: {name}"
                )

            self.qpos_addr[i] = self.model.jnt_qposadr[joint_id]
            self.qvel_addr[i] = self.model.jnt_dofadr[joint_id]

        # Base body
        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base_link",
        )

        if self.base_body_id < 0:
            raise RuntimeError("base_link not found")

        # 57-dimensional observation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(57,),
            dtype=np.float32,
        )

        # Policy actions are normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

        self.previous_action = np.zeros(
            16,
            dtype=np.float32,
        )

        self.command = np.zeros(
            3,
            dtype=np.float32,
        )

        self.step_count = 0

        self.viewer = None

    # ------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # Floating base
        self.data.qpos[0:3] = np.array(
            [0.0, 0.0, 0.5],
            dtype=np.float64,
        )

        # Identity orientation quaternion
        self.data.qpos[3:7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        # Default joint configuration
        for i in range(16):
            self.data.qpos[self.qpos_addr[i]] = \
                self.DEFAULT_POS[i]

        # Small randomization
        if options is None or not options.get(
            "deterministic",
            False
        ):
            self.data.qpos[0] += self.np_random.uniform(
                -0.01, 0.01
            )
            self.data.qpos[1] += self.np_random.uniform(
                -0.01, 0.01
            )

            for i in range(12):
                self.data.qpos[self.qpos_addr[i]] += \
                    self.np_random.uniform(-0.02, 0.02)

        self.data.qvel[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # Random velocity command
        self.command = np.array([
            self.np_random.uniform(0.15, 0.40),
            self.np_random.uniform(-0.10, 0.10),
            self.np_random.uniform(-0.30, 0.30),
        ], dtype=np.float32)

        self.previous_action[:] = 0.0
        self.step_count = 0

        obs = self._get_observation()

        info = {}

        return obs, info

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------

    def _get_observation(self):

        # Angular velocity of floating base
        base_omega = self.data.qvel[3:6].copy()

        base_omega *= self.OMEGA_SCALE

        # Base orientation
        quat = self.data.qpos[3:7]

        rotation = np.zeros((3, 3))

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            quat,
        )

        # Gravity expressed in body frame
        world_gravity = np.array(
            [0.0, 0.0, -1.0]
        )

        projected_gravity = rotation.T @ world_gravity

        # Joint positions
        joint_pos = np.zeros(16)

        joint_vel = np.zeros(16)

        for i in range(16):
            joint_pos[i] = self.data.qpos[
                self.qpos_addr[i]
            ]

            joint_vel[i] = self.data.qvel[
                self.qvel_addr[i]
            ]

        # Wheels don't have meaningful position input
        # in the deployed policy.
        joint_pos[12:16] = 0.0

        # Position relative to default pose
        joint_pos -= self.DEFAULT_POS

        # Scale velocity
        joint_vel *= self.DOF_VEL_SCALE

        obs = np.concatenate([
            base_omega,
            projected_gravity,
            self.command,
            joint_pos,
            joint_vel,
            self.previous_action,
        ])

        assert obs.shape == (57,), \
            f"Bad observation shape: {obs.shape}"

        return obs.astype(np.float32)

    # ------------------------------------------------------------
    # ACTION → MOTOR CONTROL
    # ------------------------------------------------------------

    def _apply_action(self, action):

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # Convert normalized action into target values
        target = (
            action * self.ACTION_SCALE
            + self.DEFAULT_POS
        )

        # Current joint state
        q = np.zeros(16)
        dq = np.zeros(16)

        for i in range(16):
            q[i] = self.data.qpos[
                self.qpos_addr[i]
            ]

            dq[i] = self.data.qvel[
                self.qvel_addr[i]
            ]

        # --------------------------------------------------------
        # First 12 joints = position controlled
        # --------------------------------------------------------

        torque = np.zeros(16)

        for i in range(12):

            position_error = target[i] - q[i]

            velocity_error = -dq[i]

            torque[i] = (
                self.KP[i] * position_error
                + self.KD[i] * velocity_error
            )

        # --------------------------------------------------------
        # Last 4 joints = wheel velocity control
        # --------------------------------------------------------

        for leg in range(4):

            i = 4 * leg + 3

            desired_velocity = target[i]

            velocity_error = desired_velocity - dq[i]

            torque[i] = self.KD[i] * velocity_error

        # Torque limits
        torque[:12] = np.clip(
            torque[:12],
            -50.0,
            50.0,
        )

        torque[3::4] = np.clip(
            torque[3::4],
            -14.0,
            14.0,
        )

        self.data.ctrl[:] = torque

    # ------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        self._apply_action(action)

        # Execute MuJoCo physics
        for _ in range(self.decimation):
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.previous_action = action.copy()

        self.step_count += 1

        obs = self._get_observation()

        reward = self._compute_reward(
            action
        )

        terminated = self._is_fallen()

        truncated = (
            self.step_count >= self.episode_length
        )

        info = {
            "x_velocity":
                float(self._base_velocity()[0]),

            "y_velocity":
                float(self._base_velocity()[1]),

            "yaw_velocity":
                float(self.data.qvel[5]),

            "command_x":
                float(self.command[0]),
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------

    def _base_velocity(self):

        # World-frame linear velocity
        return self.data.qvel[0:3]

    def _compute_reward(self, action):

        velocity = self._base_velocity()

        vx = velocity[0]
        vy = velocity[1]

        yaw_rate = self.data.qvel[5]

        # --------------------------------------------------------
        # Velocity tracking
        # --------------------------------------------------------

        vx_error = vx - self.command[0]
        vy_error = vy - self.command[1]
        yaw_error = yaw_rate - self.command[2]

        velocity_reward = np.exp(
            -2.0 * vx_error ** 2
            -2.0 * vy_error ** 2
            -0.5 * yaw_error ** 2
        )

        # --------------------------------------------------------
        # Upright reward
        # --------------------------------------------------------

        quat = self.data.qpos[3:7]

        rotation = np.zeros((3, 3))

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            quat,
        )

        upright = rotation[2, 2]

        upright_reward = max(
            0.0,
            upright
        )

        # --------------------------------------------------------
        # Height
        # --------------------------------------------------------

        height = self.data.qpos[2]

        height_reward = np.exp(
            -20.0 * (height - 0.5) ** 2
        )

        # --------------------------------------------------------
        # Action penalty
        # --------------------------------------------------------

        action_penalty = 0.001 * np.sum(
            action ** 2
        )

        reward = (
            2.0 * velocity_reward
            + 0.5 * upright_reward
            + 0.5 * height_reward
            - action_penalty
        )

        if self._is_fallen():
            reward -= 10.0

        return reward

    # ------------------------------------------------------------
    # FALL DETECTION
    # ------------------------------------------------------------

    def _is_fallen(self):

        height = self.data.qpos[2]

        if height < 0.25:
            return True

        if height > 1.0:
            return True

        quat = self.data.qpos[3:7]

        rotation = np.zeros((3, 3))

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            quat,
        )

        # Robot's local Z axis in world frame
        z_axis = rotation[:, 2]

        # If robot is tilted too far
        if z_axis[2] < 0.5:
            return True

        return False

    # ------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------

    def render(self):

        if self.render_mode != "human":
            return

        if self.viewer is None:

            try:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )

            except Exception as e:
                print("Viewer error:", e)
                return

        if self.viewer.is_running():
            self.viewer.sync()

    # ------------------------------------------------------------
    # CLOSE
    # ------------------------------------------------------------

    def close(self):

        if self.viewer is not None:

            try:
                self.viewer.close()
            except Exception:
                pass

            self.viewer = None
