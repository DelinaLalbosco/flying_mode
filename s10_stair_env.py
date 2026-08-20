#!/usr/bin/env python3

"""
S10 Stair-Climbing Reinforcement Learning Environment
======================================================

Deep Robotics S10 quadruped.

This environment is designed for PPO training on stairs.

Robot:
    16 actuated joints
        12 leg joints
         4 wheel joints

Default terrain:
    2 stairs
    0.08 m step height
    0.35 m step depth

Observation:
    62 values

        3   base angular velocity
        3   projected gravity
        3   command

       16   joint position error
       16   joint velocity
       16   previous action

        5   terrain information

Action:
    16 normalized values [-1, 1]

        12 leg position targets
         4 wheel velocity targets
"""

import os
import time
import numpy as np
import mujoco
import gymnasium as gym

from gymnasium import spaces


class S10StairEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    # ============================================================
    # JOINTS
    # ============================================================

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

    # ============================================================
    # NOMINAL STANDING POSE
    # ============================================================

    DEFAULT_POS = np.array([
         0.0, -0.30,  0.60, 0.0,
         0.0, -0.30,  0.60, 0.0,

         0.0,  0.30, -0.60, 0.0,
         0.0,  0.30, -0.60, 0.0,
    ], dtype=np.float32)

    # ============================================================
    # ACTION SCALE
    # ============================================================

    ACTION_SCALE = np.array([
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
    ], dtype=np.float32)

    # ============================================================
    # OBSERVATION SCALING
    # ============================================================

    OMEGA_SCALE = 0.25

    DOF_VEL_SCALE = 0.05

    # ============================================================
    # PD GAINS
    # ============================================================

    KP = np.array(
        [80.0, 80.0, 80.0, 0.0] * 4,
        dtype=np.float32
    )

    KD = np.array(
        [2.0, 2.0, 2.0, 0.6] * 4,
        dtype=np.float32
    )

    # ============================================================
    # TERRAIN
    # ============================================================

    DEFAULT_STEP_HEIGHT = 0.08

    DEFAULT_STEP_DEPTH = 0.35

    DEFAULT_NUM_STEPS = 2

    DEFAULT_STAIR_START_X = 1.20

    DEFAULT_GOAL_X = 2.50

    # ============================================================
    # ROBOT
    # ============================================================

    START_X = 0.0
    START_Y = 0.0
    START_Z = 0.50

    TARGET_HEIGHT = 0.50

    # ============================================================
    # CONTROL
    # ============================================================

    POLICY_DT = 0.02

    # ============================================================
    # REWARD PARAMETERS
    # ============================================================

    FORWARD_REWARD_SCALE = 8.0

    VELOCITY_REWARD_SCALE = 1.0

    UPRIGHT_REWARD_SCALE = 2.0

    HEIGHT_REWARD_SCALE = 1.0

    STABILITY_REWARD_SCALE = 0.5

    STAIR_PROGRESS_SCALE = 5.0

    GOAL_REWARD = 100.0

    ACTION_PENALTY_SCALE = 0.002

    ACTION_RATE_PENALTY_SCALE = 0.003

    SIDEWAYS_PENALTY_SCALE = 0.15

    ANGULAR_VELOCITY_PENALTY = 0.02

    FALL_PENALTY = 30.0

    # ============================================================
    # FALL LIMITS
    # ============================================================

    MIN_HEIGHT = 0.25

    MAX_HEIGHT = 1.20

    MIN_UPRIGHT = 0.45

    # ============================================================
    # CONSTRUCTOR
    # ============================================================

    def __init__(
        self,
        xml_path=None,
        render_mode=None,

        episode_length=1500,

        step_height=DEFAULT_STEP_HEIGHT,
        step_depth=DEFAULT_STEP_DEPTH,
        num_steps=DEFAULT_NUM_STEPS,
        stair_start_x=DEFAULT_STAIR_START_X,
        goal_x=DEFAULT_GOAL_X,

        curriculum=False,
    ):

        super().__init__()

        print("=" * 70)
        print("S10 STAIR CLIMBING ENVIRONMENT")
        print("=" * 70)

        # --------------------------------------------------------
        # XML
        # --------------------------------------------------------

        if xml_path is None:

            xml_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "src",
                    "S10_sdk_deploy",
                    "S10_description",
                    "s10_mjcf",
                    "mjcf",
                    "S10_stairs_training.xml",
                )
            )

        xml_path = os.path.abspath(xml_path)

        if not os.path.isfile(xml_path):

            raise FileNotFoundError(
                "\nS10 stair XML was not found:\n"
                f"{xml_path}\n\n"
                "Check that S10_stairs_training.xml exists."
            )

        self.xml_path = xml_path

        print("\nStair XML:")
        print(self.xml_path)

        # --------------------------------------------------------
        # Terrain parameters
        # --------------------------------------------------------

        self.step_height = float(step_height)

        self.step_depth = float(step_depth)

        self.num_steps = int(num_steps)

        self.stair_start_x = float(stair_start_x)

        self.stair_end_x = (
            self.stair_start_x
            + self.num_steps * self.step_depth
        )

        self.goal_x = float(goal_x)

        self.curriculum = bool(curriculum)

        # --------------------------------------------------------
        # Episode
        # --------------------------------------------------------

        self.episode_length = int(
            episode_length
        )

        # --------------------------------------------------------
        # Load MuJoCo
        # --------------------------------------------------------

        self.model = mujoco.MjModel.from_xml_path(
            self.xml_path
        )

        self.data = mujoco.MjData(
            self.model
        )

        # --------------------------------------------------------
        # Timing
        # --------------------------------------------------------

        self.policy_dt = self.POLICY_DT

        self.physics_dt = float(
            self.model.opt.timestep
        )

        self.decimation = max(
            1,
            int(
                round(
                    self.policy_dt
                    / self.physics_dt
                )
            )
        )

        print("\nMuJoCo timestep:")
        print(self.physics_dt)

        print("Policy timestep:")
        print(self.policy_dt)

        print("Decimation:")
        print(self.decimation)

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
                    f"Joint not found in XML: {name}"
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
        # Base body
        # --------------------------------------------------------

        self.base_body_id = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "base_link",
            )
        )

        if self.base_body_id < 0:

            raise RuntimeError(
                "base_link was not found in S10 XML"
            )

        # --------------------------------------------------------
        # Observation space
        # --------------------------------------------------------

        # 3 omega
        # 3 gravity
        # 3 command
        # 16 joint position error
        # 16 joint velocity
        # 16 previous action
        # 5 terrain

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(62,),
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # Action
        # --------------------------------------------------------

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

        self.previous_previous_action = np.zeros(
            16,
            dtype=np.float32
        )

        # Desired forward velocity

        self.command = np.array(
            [0.25, 0.0, 0.0],
            dtype=np.float32
        )

        self.step_count = 0

        self.previous_x = 0.0

        self.best_x = 0.0

        self.previous_stair_level = 0

        self.stair_level = 0

        self.goal_reached = False

        self.viewer = None

        self.last_reward_components = {}

        print("\nStair configuration:")
        print(
            f"  Step height: {self.step_height:.2f}"
        )
        print(
            f"  Step depth: {self.step_depth:.2f}"
        )
        print(
            f"  Number of steps: {self.num_steps}"
        )
        print(
            f"  Stair start X: {self.stair_start_x:.2f}"
        )
        print(
            f"  Stair end X: {self.stair_end_x:.2f}"
        )
        print(
            f"  Target X: {self.goal_x:.2f}"
        )

        print("=" * 70)

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # --------------------------------------------------------
        # Base
        # --------------------------------------------------------

        self.data.qpos[0:3] = np.array(
            [
                self.START_X,
                self.START_Y,
                self.START_Z,
            ],
            dtype=np.float64
        )

        # Quaternion:
        # w x y z

        self.data.qpos[3:7] = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64
        )

        # --------------------------------------------------------
        # Standing pose
        # --------------------------------------------------------

        for i in range(16):

            self.data.qpos[
                self.qpos_addr[i]
            ] = self.DEFAULT_POS[i]

        # --------------------------------------------------------
        # Reset velocities
        # --------------------------------------------------------

        self.data.qvel[:] = 0.0

        # --------------------------------------------------------
        # Small randomization
        # --------------------------------------------------------

        deterministic = False

        if options is not None:

            deterministic = bool(
                options.get(
                    "deterministic",
                    False
                )
            )

        if not deterministic:

            self.data.qpos[0] += (
                self.np_random.uniform(
                    -0.01,
                    0.01
                )
            )

            self.data.qpos[1] += (
                self.np_random.uniform(
                    -0.005,
                    0.005
                )
            )

            # Very small joint noise

            for i in range(12):

                self.data.qpos[
                    self.qpos_addr[i]
                ] += self.np_random.uniform(
                    -0.005,
                    0.005
                )

        # --------------------------------------------------------
        # Forward kinematics
        # --------------------------------------------------------

        mujoco.mj_forward(
            self.model,
            self.data
        )

        # --------------------------------------------------------
        # Reset state
        # --------------------------------------------------------

        self.previous_action[:] = 0.0

        self.previous_previous_action[:] = 0.0

        self.step_count = 0

        self.previous_x = float(
            self.data.qpos[0]
        )

        self.best_x = float(
            self.data.qpos[0]
        )

        self.previous_stair_level = 0

        self.stair_level = (
            self._calculate_stair_level()
        )

        self.goal_reached = False

        self.last_reward_components = {}

        obs = self._get_observation()

        info = self._get_info()

        return obs, info

    # ============================================================
    # ORIENTATION
    # ============================================================

    def _get_rotation_matrix(self):

        quat = self.data.qpos[3:7]

        rotation = np.zeros(
            (3, 3),
            dtype=np.float64
        )

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            quat
        )

        return rotation

    # ============================================================
    # STAIR LEVEL
    # ============================================================

    def _calculate_stair_level(self):

        x = float(
            self.data.qpos[0]
        )

        if x < self.stair_start_x:

            return 0

        distance = (
            x - self.stair_start_x
        )

        level = int(
            np.floor(
                distance
                / self.step_depth
            )
        ) + 1

        level = max(
            0,
            min(
                level,
                self.num_steps
            )
        )

        return level

    # ============================================================
    # TERRAIN OBSERVATION
    # ============================================================

    def _get_terrain_observation(self):

        x = float(
            self.data.qpos[0]
        )

        # --------------------------------------------------------
        # Distance to staircase
        # --------------------------------------------------------

        distance_to_stair = (
            self.stair_start_x - x
        )

        # Normalize

        distance_to_stair = np.clip(
            distance_to_stair / 2.0,
            -1.0,
            1.0
        )

        # --------------------------------------------------------
        # Distance to next step
        # --------------------------------------------------------

        if x < self.stair_start_x:

            next_step_distance = (
                self.stair_start_x - x
            )

        else:

            current_level = (
                self._calculate_stair_level()
            )

            next_step_x = (
                self.stair_start_x
                + current_level
                * self.step_depth
            )

            next_step_distance = (
                next_step_x - x
            )

        next_step_distance = np.clip(
            next_step_distance / 2.0,
            -1.0,
            1.0
        )

        # --------------------------------------------------------
        # Upcoming height
        # --------------------------------------------------------

        current_level = (
            self._calculate_stair_level()
        )

        upcoming_height = (
            current_level
            * self.step_height
        )

        upcoming_height = np.clip(
            upcoming_height,
            0.0,
            1.0
        )

        # --------------------------------------------------------
        # Next height difference
        # --------------------------------------------------------

        next_level = min(
            current_level + 1,
            self.num_steps
        )

        height_difference = (
            next_level
            * self.step_height
            - current_level
            * self.step_height
        )

        # Normalize approximately

        height_difference = np.clip(
            height_difference / 0.20,
            0.0,
            1.0
        )

        # --------------------------------------------------------
        # Stair level
        # --------------------------------------------------------

        normalized_level = (
            current_level
            / max(
                1,
                self.num_steps
            )
        )

        return np.array(
            [
                distance_to_stair,
                next_step_distance,
                upcoming_height,
                height_difference,
                normalized_level,
            ],
            dtype=np.float32
        )

    # ============================================================
    # OBSERVATION
    # ============================================================

    def _get_observation(self):

        # --------------------------------------------------------
        # Angular velocity
        # --------------------------------------------------------

        base_omega = self.data.qvel[
            3:6
        ].copy()

        base_omega *= self.OMEGA_SCALE

        # --------------------------------------------------------
        # Gravity
        # --------------------------------------------------------

        rotation = (
            self._get_rotation_matrix()
        )

        world_gravity = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float64
        )

        projected_gravity = (
            rotation.T
            @ world_gravity
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

            joint_pos[i] = (
                self.data.qpos[
                    self.qpos_addr[i]
                ]
            )

            joint_vel[i] = (
                self.data.qvel[
                    self.qvel_addr[i]
                ]
            )

        # Wheel position is not useful

        joint_pos[3] = 0.0
        joint_pos[7] = 0.0
        joint_pos[11] = 0.0
        joint_pos[15] = 0.0

        # Position error

        joint_pos -= self.DEFAULT_POS

        # Velocity scale

        joint_vel *= self.DOF_VEL_SCALE

        # --------------------------------------------------------
        # Terrain
        # --------------------------------------------------------

        terrain = (
            self._get_terrain_observation()
        )

        # --------------------------------------------------------
        # Combine
        # --------------------------------------------------------

        obs = np.concatenate(
            [
                base_omega,
                projected_gravity,
                self.command,
                joint_pos,
                joint_vel,
                self.previous_action,
                terrain,
            ]
        )

        if obs.shape != (62,):

            raise RuntimeError(
                f"Invalid observation shape: "
                f"{obs.shape}, expected (62,)"
            )

        return obs.astype(
            np.float32
        )

    # ============================================================
    # ACTION
    # ============================================================

    def _apply_action(
        self,
        action
    ):

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
        # Action smoothing
        # --------------------------------------------------------

        smoothing = 0.25

        filtered_action = (
            (1.0 - smoothing)
            * self.previous_action
            + smoothing
            * action
        )

        # --------------------------------------------------------
        # Target
        # --------------------------------------------------------

        target = (
            filtered_action
            * self.ACTION_SCALE
            + self.DEFAULT_POS
        )

        # --------------------------------------------------------
        # Current state
        # --------------------------------------------------------

        q = np.zeros(
            16,
            dtype=np.float64
        )

        dq = np.zeros(
            16,
            dtype=np.float64
        )

        for i in range(16):

            q[i] = (
                self.data.qpos[
                    self.qpos_addr[i]
                ]
            )

            dq[i] = (
                self.data.qvel[
                    self.qvel_addr[i]
                ]
            )

        torque = np.zeros(
            16,
            dtype=np.float64
        )

        # --------------------------------------------------------
        # LEG JOINTS
        # --------------------------------------------------------

        for i in range(12):

            position_error = (
                target[i] - q[i]
            )

            velocity_error = (
                -dq[i]
            )

            torque[i] = (
                self.KP[i]
                * position_error
                +
                self.KD[i]
                * velocity_error
            )

        # --------------------------------------------------------
        # WHEELS
        # --------------------------------------------------------

        for leg in range(4):

            i = 4 * leg + 3

            desired_velocity = (
                filtered_action[i]
                * self.ACTION_SCALE[i]
            )

            velocity_error = (
                desired_velocity
                - dq[i]
            )

            torque[i] = (
                self.KD[i]
                * velocity_error
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

        return filtered_action

    # ============================================================
    # STEP
    # ============================================================

    def step(
        self,
        action
    ):

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
        # Previous state
        # --------------------------------------------------------

        previous_x = float(
            self.data.qpos[0]
        )

        previous_level = (
            self._calculate_stair_level()
        )

        # --------------------------------------------------------
        # Apply action
        # --------------------------------------------------------

        filtered_action = (
            self._apply_action(
                action
            )
        )

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

        # --------------------------------------------------------
        # State update
        # --------------------------------------------------------

        self.previous_previous_action = (
            self.previous_action.copy()
        )

        self.previous_action = (
            filtered_action.copy()
        )

        self.step_count += 1

        current_x = float(
            self.data.qpos[0]
        )

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        forward_progress = (
            current_x
            - previous_x
        )

        self.best_x = max(
            self.best_x,
            current_x
        )

        # --------------------------------------------------------
        # Stair level
        # --------------------------------------------------------

        self.stair_level = (
            self._calculate_stair_level()
        )

        # --------------------------------------------------------
        # Goal
        # --------------------------------------------------------

        self.goal_reached = (
            current_x >= self.goal_x
            and not self._is_fallen()
        )

        # --------------------------------------------------------
        # Reward
        # --------------------------------------------------------

        reward = self._compute_reward(
            action=action,
            filtered_action=filtered_action,
            forward_progress=forward_progress,
            previous_level=previous_level,
        )

        # --------------------------------------------------------
        # Termination
        # --------------------------------------------------------

        terminated = (
            self._is_fallen()
            or self.goal_reached
        )

        truncated = (
            self.step_count
            >= self.episode_length
        )

        # --------------------------------------------------------
        # Observation
        # --------------------------------------------------------

        obs = self._get_observation()

        info = self._get_info()

        info["forward_progress"] = float(
            forward_progress
        )

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------

        if terminated or truncated:

            print(
                "Episode ended:",
                info
            )

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

    def _compute_reward(
        self,
        action,
        filtered_action,
        forward_progress,
        previous_level,
    ):

        rotation = (
            self._get_rotation_matrix()
        )

        # --------------------------------------------------------
        # Upright
        # --------------------------------------------------------

        upright = float(
            rotation[2, 2]
        )

        upright_reward = max(
            0.0,
            upright
        )

        # --------------------------------------------------------
        # Height
        # --------------------------------------------------------

        height = float(
            self.data.qpos[2]
        )

        height_error = (
            height
            - self.TARGET_HEIGHT
        )

        height_reward = np.exp(
            -20.0
            * height_error ** 2
        )

        # --------------------------------------------------------
        # Forward progress
        # --------------------------------------------------------

        forward_reward = (
            self.FORWARD_REWARD_SCALE
            * np.clip(
                forward_progress,
                -0.05,
                0.05
            )
        )

        # --------------------------------------------------------
        # Desired forward velocity
        # --------------------------------------------------------

        vx = float(
            self.data.qvel[0]
        )

        velocity_reward = (
            np.exp(
                -8.0
                * (
                    vx
                    - self.command[0]
                ) ** 2
            )
        )

        # --------------------------------------------------------
        # Sideways penalty
        # --------------------------------------------------------

        vy = float(
            self.data.qvel[1]
        )

        sideways_penalty = (
            self.SIDEWAYS_PENALTY_SCALE
            * vy ** 2
        )

        # --------------------------------------------------------
        # Angular velocity
        # --------------------------------------------------------

        angular_velocity = (
            self.data.qvel[3:6]
        )

        angular_penalty = (
            self.ANGULAR_VELOCITY_PENALTY
            * np.sum(
                angular_velocity ** 2
            )
        )

        # --------------------------------------------------------
        # Stair progress
        # --------------------------------------------------------

        current_level = (
            self._calculate_stair_level()
        )

        level_change = (
            current_level
            - previous_level
        )

        stair_reward = (
            self.STAIR_PROGRESS_SCALE
            * max(
                0,
                level_change
            )
        )

        # --------------------------------------------------------
        # Action penalty
        # --------------------------------------------------------

        action_penalty = (
            self.ACTION_PENALTY_SCALE
            * np.sum(
                action ** 2
            )
        )

        # --------------------------------------------------------
        # Action-rate penalty
        # --------------------------------------------------------

        action_difference = (
            filtered_action
            - self.previous_action
        )

        action_rate_penalty = (
            self.ACTION_RATE_PENALTY_SCALE
            * np.sum(
                action_difference ** 2
            )
        )

        # --------------------------------------------------------
        # Stability
        # --------------------------------------------------------

        roll_rate = float(
            self.data.qvel[3]
        )

        pitch_rate = float(
            self.data.qvel[4]
        )

        yaw_rate = float(
            self.data.qvel[5]
        )

        stability_reward = np.exp(
            -0.1
            * (
                roll_rate ** 2
                +
                pitch_rate ** 2
                +
                yaw_rate ** 2
            )
        )

        # --------------------------------------------------------
        # Final reward
        # --------------------------------------------------------

        reward = (

            self.UPRIGHT_REWARD_SCALE
            * upright_reward

            +

            self.HEIGHT_REWARD_SCALE
            * height_reward

            +

            self.STABILITY_REWARD_SCALE
            * stability_reward

            +

            forward_reward

            +

            self.VELOCITY_REWARD_SCALE
            * velocity_reward

            +

            stair_reward

            -

            sideways_penalty

            -

            angular_penalty

            -

            action_penalty

            -

            action_rate_penalty
        )

        # --------------------------------------------------------
        # Fall penalty
        # --------------------------------------------------------

        if self._is_fallen():

            reward -= self.FALL_PENALTY

        # --------------------------------------------------------
        # Goal reward
        # --------------------------------------------------------

        if self.goal_reached:

            reward += self.GOAL_REWARD

        # --------------------------------------------------------
        # Save components
        # --------------------------------------------------------

        self.last_reward_components = {

            "upright":
                float(upright_reward),

            "height":
                float(height_reward),

            "forward":
                float(forward_reward),

            "velocity":
                float(velocity_reward),

            "stair":
                float(stair_reward),

            "stability":
                float(stability_reward),

            "sideways_penalty":
                float(sideways_penalty),

            "angular_penalty":
                float(angular_penalty),

            "action_penalty":
                float(action_penalty),

            "goal":
                float(
                    self.GOAL_REWARD
                    if self.goal_reached
                    else 0.0
                ),
        }

        return float(reward)

    # ============================================================
    # FALL DETECTION
    # ============================================================

    def _is_fallen(self):

        height = float(
            self.data.qpos[2]
        )

        # --------------------------------------------------------
        # Height
        # --------------------------------------------------------

        if height < self.MIN_HEIGHT:

            return True

        if height > self.MAX_HEIGHT:

            return True

        # --------------------------------------------------------
        # Orientation
        # --------------------------------------------------------

        rotation = (
            self._get_rotation_matrix()
        )

        upright = float(
            rotation[2, 2]
        )

        if upright < self.MIN_UPRIGHT:

            return True

        return False

    # ============================================================
    # INFO
    # ============================================================

    def _get_info(self):

        velocity = (
            self.data.qvel[0:3]
        )

        return {

            "height":
                float(
                    self.data.qpos[2]
                ),

            "x":
                float(
                    self.data.qpos[0]
                ),

            "y":
                float(
                    self.data.qpos[1]
                ),

            "x_velocity":
                float(
                    velocity[0]
                ),

            "y_velocity":
                float(
                    velocity[1]
                ),

            "roll_rate":
                float(
                    self.data.qvel[3]
                ),

            "pitch_rate":
                float(
                    self.data.qvel[4]
                ),

            "yaw_rate":
                float(
                    self.data.qvel[5]
                ),

            "goal_x":
                float(
                    self.goal_x
                ),

            "stair_level":
                int(
                    self._calculate_stair_level()
                ),

            "num_steps":
                int(
                    self.num_steps
                ),

            "stair_start_x":
                float(
                    self.stair_start_x
                ),

            "stair_end_x":
                float(
                    self.stair_end_x
                ),

            "goal_reached":
                bool(
                    self.goal_reached
                ),
        }

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

                time.sleep(0.1)

            except Exception as e:

                print(
                    "Viewer error:",
                    e
                )

                self.viewer = None

                return

        try:

            if self.viewer.is_running():

                self.viewer.sync()

        except Exception:

            pass

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self):

        if self.viewer is not None:

            try:

                if self.viewer.is_running():

                    self.viewer.close()

            except Exception:

                pass

            self.viewer = None

        # IMPORTANT:
        # Do NOT write:
        # self.model = NoneO

        self.data = None

        self.model = None


# =================================================================
# DIRECT TEST
# =================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("DIRECT S10 STAIR ENVIRONMENT TEST")
    print("=" * 70)

    env = None

    try:

        env = S10StairEnv(
            episode_length=1000,

            step_height=0.08,

            step_depth=0.35,

            num_steps=2,

            stair_start_x=1.20,

            goal_x=2.50,

            render_mode="human",
        )

        print()
        print("Observation space:")
        print(env.observation_space)

        print()
        print("Action space:")
        print(env.action_space)

        obs, info = env.reset(
            options={
                "deterministic": True
            }
        )

        print()
        print("Initial observation shape:")
        print(obs.shape)

        print()
        print("Initial info:")
        print(info)

        print()
        print("Running 100 test steps...")
        print(
            "NOTE: random actions are only an environment test."
        )
        print(
            "They are NOT expected to climb the stairs."
        )
        print()

        for step in range(100):

            # --------------------------------------------------
            # IMPORTANT:
            #
            # Random actions are deliberately small.
            #
            # This tests whether the environment remains
            # numerically stable without immediately exploding.
            # --------------------------------------------------

            action = (
                env.action_space.sample()
                * 0.10
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            if (
                step == 0
                or
                (step + 1) % 10 == 0
            ):

                print(
                    f"Step {step + 1:03d}: "
                    f"reward={reward:.3f}, "
                    f"x={info['x']:.3f}, "
                    f"height={info['height']:.3f}, "
                    f"level={info['stair_level']}, "
                    f"terminated={terminated}, "
                    f"truncated={truncated}"
                )

            if (
                terminated
                or
                truncated
            ):

                print()
                print(
                    "Episode ended during test."
                )

                break

        print()
        print("=" * 70)
        print("ENVIRONMENT TEST FINISHED")
        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("ENVIRONMENT TEST FAILED")
        print("=" * 70)

        print(
            type(e).__name__,
            ":",
            e
        )

        raise

    finally:

        if env is not None:

            env.close()

            print(
                "Environment closed successfully."
            )
