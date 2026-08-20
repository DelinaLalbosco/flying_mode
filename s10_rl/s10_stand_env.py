#!/usr/bin/env python3

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class S10StandEnv(gym.Env):
    """
    Deep Robotics S10 standing environment.

    Goal:
        Keep the S10 upright and close to its nominal standing pose.

    Observation:
        57 values
          3  base angular velocity
          3  projected gravity
          3  command (zero for standing)
          16 joint position error
          16 joint velocity
          16 previous action

    Action:
        16 normalized actions in [-1, 1]

        12 leg joints:
            position targets

        4 wheel joints:
            velocity targets
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

    # Your current nominal S10 pose
    DEFAULT_POS = np.array([
         0.0, -0.3,  0.6, 0.0,
         0.0, -0.3,  0.6, 0.0,
         0.0,  0.3, -0.6, 0.0,
         0.0,  0.3, -0.6, 0.0,
    ], dtype=np.float32)

    # Same action scaling as your walking environment
    ACTION_SCALE = np.array([
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
    ], dtype=np.float32)

    OMEGA_SCALE = 0.25
    DOF_VEL_SCALE = 0.05

    # Position gains
    KP = np.array(
        [80.0, 80.0, 80.0, 0.0] * 4,
        dtype=np.float32
    )

    # Velocity gains
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

        self.model = mujoco.MjModel.from_xml_path(
            self.xml_path
        )

        self.data = mujoco.MjData(self.model)

        # Policy update rate
        self.policy_dt = 0.02

        # MuJoCo timestep
        self.physics_dt = self.model.opt.timestep

        self.decimation = max(
            1,
            int(round(
                self.policy_dt / self.physics_dt
            ))
        )

        print(
            "MuJoCo timestep:",
            self.physics_dt
        )

        print(
            "Policy timestep:",
            self.policy_dt
        )

        print(
            "Decimation:",
            self.decimation
        )

        # --------------------------------------------------------
        # Joint addresses
        # --------------------------------------------------------

        self.qpos_addr = np.zeros(
            16,
            dtype=np.int32
        )

        self.qvel_addr = np.zeros(
            16,
            dtype=np.int32
        )

        for i, name in enumerate(
            self.JOINT_NAMES
        ):

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if joint_id < 0:
                raise RuntimeError(
                    f"Joint not found: {name}"
                )

            self.qpos_addr[i] = (
                self.model.jnt_qposadr[
                    joint_id
                ]
            )

            self.qvel_addr[i] = (
                self.model.jnt_dofadr[
                    joint_id
                ]
            )

        # --------------------------------------------------------
        # Base
        # --------------------------------------------------------

        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base_link",
        )

        if self.base_body_id < 0:
            raise RuntimeError(
                "base_link not found"
            )

        # --------------------------------------------------------
        # Spaces
        # --------------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(57,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # State
        # --------------------------------------------------------

        self.previous_action = np.zeros(
            16,
            dtype=np.float32
        )

        # Standing command is ALWAYS zero
        self.command = np.zeros(
            3,
            dtype=np.float32
        )

        self.step_count = 0

        self.viewer = None

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # --------------------------------------------------------
        # Floating base
        # --------------------------------------------------------

        self.data.qpos[0:3] = np.array(
            [0.0, 0.0, 0.5],
            dtype=np.float64
        )

        # Identity orientation
        self.data.qpos[3:7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64
        )

        # --------------------------------------------------------
        # Standing joint configuration
        # --------------------------------------------------------

        for i in range(16):

            self.data.qpos[
                self.qpos_addr[i]
            ] = self.DEFAULT_POS[i]

        # --------------------------------------------------------
        # Small randomization
        # --------------------------------------------------------

        deterministic = (
            options is not None
            and options.get(
                "deterministic",
                False
            )
        )

        if not deterministic:

            # Very small base position noise
            self.data.qpos[0] += (
                self.np_random.uniform(
                    -0.005,
                    0.005
                )
            )

            self.data.qpos[1] += (
                self.np_random.uniform(
                    -0.005,
                    0.005
                )
            )

            # Small leg joint noise
            for i in range(12):

                self.data.qpos[
                    self.qpos_addr[i]
                ] += self.np_random.uniform(
                    -0.01,
                    0.01
                )

        # Zero all velocities
        self.data.qvel[:] = 0.0

        # Update kinematics
        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.previous_action[:] = 0.0

        # No movement command
        self.command[:] = 0.0

        self.step_count = 0

        obs = self._get_observation()

        info = {}

        return obs, info

    # ============================================================
    # OBSERVATION
    # ============================================================

    def _get_observation(self):

        # --------------------------------------------------------
        # Base angular velocity
        # --------------------------------------------------------

        base_omega = self.data.qvel[
            3:6
        ].copy()

        base_omega *= self.OMEGA_SCALE

        # --------------------------------------------------------
        # Base orientation
        # --------------------------------------------------------

        quat = self.data.qpos[3:7]

        rotation = np.zeros(
            (3, 3),
            dtype=np.float64
        )

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            quat
        )

        # --------------------------------------------------------
        # Projected gravity
        # --------------------------------------------------------

        world_gravity = np.array(
            [0.0, 0.0, -1.0]
        )

        projected_gravity = (
            rotation.T @ world_gravity
        )

        # --------------------------------------------------------
        # Joint state
        # --------------------------------------------------------

        joint_pos = np.zeros(
            16,
            dtype=np.float64
        )

        joint_vel = np.zeros(
            16,
            dtype=np.float64
        )

        for i in range(16):

            joint_pos[i] = self.data.qpos[
                self.qpos_addr[i]
            ]

            joint_vel[i] = self.data.qvel[
                self.qvel_addr[i]
            ]

        # Wheel positions are not used
        joint_pos[12:16] = 0.0

        # Position error
        joint_pos -= self.DEFAULT_POS

        # Scale velocity
        joint_vel *= self.DOF_VEL_SCALE

        # --------------------------------------------------------
        # Build 57-dimensional observation
        # --------------------------------------------------------

        obs = np.concatenate([
            base_omega,
            projected_gravity,
            self.command,
            joint_pos,
            joint_vel,
            self.previous_action,
        ])

        assert obs.shape == (57,), (
            f"Bad observation shape: {obs.shape}"
        )

        return obs.astype(np.float32)

    # ============================================================
    # ACTION
    # ============================================================

    def _apply_action(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # --------------------------------------------------------
        # Convert normalized action to target
        # --------------------------------------------------------

        target = (
            action * self.ACTION_SCALE
            + self.DEFAULT_POS
        )

        q = np.zeros(16)
        dq = np.zeros(16)

        for i in range(16):

            q[i] = self.data.qpos[
                self.qpos_addr[i]
            ]

            dq[i] = self.data.qvel[
                self.qvel_addr[i]
            ]

        torque = np.zeros(16)

        # --------------------------------------------------------
        # 12 leg joints
        # --------------------------------------------------------

        for i in range(12):

            position_error = (
                target[i] - q[i]
            )

            velocity_error = -dq[i]

            torque[i] = (
                self.KP[i] * position_error
                + self.KD[i] * velocity_error
            )

        # --------------------------------------------------------
        # 4 wheel joints
        # --------------------------------------------------------

        for leg in range(4):

            i = 4 * leg + 3

            desired_velocity = target[i]

            velocity_error = (
                desired_velocity - dq[i]
            )

            torque[i] = (
                self.KD[i] * velocity_error
            )

        # --------------------------------------------------------
        # Torque limits
        # --------------------------------------------------------

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

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        self._apply_action(action)

        # --------------------------------------------------------
        # Physics
        # --------------------------------------------------------

        for _ in range(
            self.decimation
        ):

            mujoco.mj_step(
                self.model,
                self.data
            )

        self.previous_action = (
            action.copy()
        )

        self.step_count += 1

        obs = self._get_observation()

        reward = self._compute_reward(
            action
        )

        terminated = self._is_fallen()

        truncated = (
            self.step_count
            >= self.episode_length
        )

        # --------------------------------------------------------
        # Information
        # --------------------------------------------------------

        velocity = self.data.qvel[0:3]

        info = {
            "height":
                float(self.data.qpos[2]),

            "x_velocity":
                float(velocity[0]),

            "y_velocity":
                float(velocity[1]),

            "roll_rate":
                float(self.data.qvel[3]),

            "pitch_rate":
                float(self.data.qvel[4]),

            "yaw_rate":
                float(self.data.qvel[5]),
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ============================================================
    # REWARD
    # ============================================================

    def _compute_reward(self, action):

        # --------------------------------------------------------
        # Orientation
        # --------------------------------------------------------

        quat = self.data.qpos[3:7]

        rotation = np.zeros(
            (3, 3),
            dtype=np.float64
        )

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            quat
        )

        # Robot Z axis relative to world Z
        upright = rotation[2, 2]

        # --------------------------------------------------------
        # Height
        # --------------------------------------------------------

        height = self.data.qpos[2]

        height_error = (
            height - 0.5
        )

        height_reward = np.exp(
            -30.0 * height_error ** 2
        )

        # --------------------------------------------------------
        # Joint pose
        # --------------------------------------------------------

        joint_error = 0.0

        for i in range(12):

            error = (
                self.data.qpos[
                    self.qpos_addr[i]
                ]
                - self.DEFAULT_POS[i]
            )

            joint_error += error ** 2

        joint_pose_reward = np.exp(
            -5.0 * joint_error
        )

        # --------------------------------------------------------
        # Angular velocity penalty
        # --------------------------------------------------------

        angular_velocity = (
            self.data.qvel[3:6]
        )

        angular_velocity_penalty = (
            np.sum(
                angular_velocity ** 2
            )
        )

        # --------------------------------------------------------
        # Linear velocity penalty
        # --------------------------------------------------------

        linear_velocity = (
            self.data.qvel[0:2]
        )

        linear_velocity_penalty = (
            np.sum(
                linear_velocity ** 2
            )
        )

        # --------------------------------------------------------
        # Action penalty
        # --------------------------------------------------------

        action_penalty = (
            np.sum(action ** 2)
        )

        # --------------------------------------------------------
        # Final reward
        # --------------------------------------------------------

        reward = (
            3.0 * max(0.0, upright)
            + 2.0 * height_reward
            + 1.0 * joint_pose_reward
            - 0.10 * angular_velocity_penalty
            - 0.05 * linear_velocity_penalty
            - 0.001 * action_penalty
        )

        # Strong penalty for falling
        if self._is_fallen():

            reward -= 10.0

        return reward

    # ============================================================
    # FALL DETECTION
    # ============================================================

    def _is_fallen(self):

        height = self.data.qpos[2]

        # Too low
        if height < 0.25:
            return True

        # Unrealistic upward launch
        if height > 1.0:
            return True

        # --------------------------------------------------------
        # Orientation
        # --------------------------------------------------------

        quat = self.data.qpos[3:7]

        rotation = np.zeros(
            (3, 3),
            dtype=np.float64
        )

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            quat
        )

        # Robot local Z axis in world frame
        z_axis = rotation[:, 2]

        # Too much tilt
        if z_axis[2] < 0.5:
            return True

        return False

    # ============================================================
    # RENDER
    # ============================================================

    def render(self):

        if self.render_mode != "human":
            return

        if self.viewer is None:

            try:

                import mujoco.viewer

                self.viewer = (
                    mujoco.viewer.launch_passive(
                        self.model,
                        self.data
                    )
                )

            except Exception as e:

                print(
                    "Viewer error:",
                    e
                )

                return

        if self.viewer.is_running():

            self.viewer.sync()

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self):

        if self.viewer is not None:

            try:
                self.viewer.close()

            except Exception:
                pass

            self.viewer = None
