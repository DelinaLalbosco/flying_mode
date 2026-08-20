#!/usr/bin/env python3

"""
Python-only MuJoCo + ROS2 + ONNX RL simulation for Deep Robotics S10.

The robot walks using the ONNX locomotion policy.

Navigation flow:

    Waypoint
       |
       v
  Navigation controller
       |
       |  vx, vy, wz
       v
  RL observation
       |
       v
    ONNX policy
       |
       | 16 actions
       v
 Joint position/velocity commands
       |
       v
 PD controller
       |
       v
 MuJoCo physics
       |
       v
 Actual robot motion
       |
       v
 Next waypoint

IMPORTANT:
The robot base is NEVER teleported toward a waypoint.
All movement comes from MuJoCo dynamics and the RL policy.
"""

import os
import time
import math
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist

from drdds.msg import (
    ImuData,
    JointsData,
    JointsDataCmd,
    MetaType,
    ImuDataValue,
    JointsDataValue,
    JointData,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "S10"

CURRENT_DIR = Path(__file__).resolve().parent

MJCF_DIR = (
    CURRENT_DIR
    / ".."
    / ".."
    / ".."
    / "S10_description"
    / "s10_mjcf"
    / "mjcf"
).resolve()

SCENE_XML_PATHS = {
    "track": MJCF_DIR / "S10_track.xml"
}

DEFAULT_SCENE_NAME = os.environ.get(
    "S10_MUJOCO_SCENE",
    "track"
)

USE_VIEWER = True

DT = 0.001
RENDER_INTERVAL = 10

TRACK_BODY_NAME = "base_link"

TRACK_START_BASE_POS = np.array(
    [0.0, -2.5, 0.2],
    dtype=np.float64
)

TRACK_WAYPOINT_PREFIX = "track_waypoint_"

WAYPOINT_REACH_RADIUS = 0.40


# ============================================================
# Viewer
# ============================================================

TRACK_VIEWER = False

CAMERA_AZIMUTH = 90
CAMERA_ELEVATION = -25
CAMERA_DISTANCE = 18.0


# ============================================================
# Joint calibration
# ============================================================

JOINT_DIR = np.array(
    [
        1, 1, -1, 1,
        1, -1, 1, -1,
        -1, 1, -1, 1,
        -1, -1, 1, -1,
    ],
    dtype=np.float32,
)


POS_OFFSET_DEG = np.array(
    [
        -35, -145, 156, 0,
         35, -145, 156, 0,
        -35,  145, -156, 0,
         35,  145, -156, 0,
    ],
    dtype=np.float32,
)

POS_OFFSET_RAD = (
    POS_OFFSET_DEG * np.pi / 180.0
).astype(np.float32)


JOINT_INIT = {
    "S10": np.array(
        [
            -0.438, -1.16,  2.76, 0.0,
             0.438, -1.16,  2.76, 0.0,
            -0.438,  1.16, -2.76, 0.0,
             0.438,  1.16, -2.76, 0.0,
        ],
        dtype=np.float32,
    )
}


# ============================================================
# RL policy
# ============================================================

MOTOR_NUM = 16

OBSERVATION_DIM = 57
ACTION_DIM = 16

POLICY_DECIMATION = 4

OMEGA_SCALE = 0.25
DOF_VEL_SCALE = 0.05

GRAVITY_DIRECTION = np.array(
    [0.0, 0.0, -1.0],
    dtype=np.float32,
)


DOF_DEFAULT_POLICY = np.array(
    [
        0.0, -0.3,  0.6,
        0.0, -0.3,  0.6,
        0.0,  0.3, -0.6,
        0.0,  0.3, -0.6,

        0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)


DOF_DEFAULT_ROBOT = np.array(
    [
        0.0, -0.3,  0.6, 0.0,
        0.0, -0.3,  0.6, 0.0,
        0.0,  0.3, -0.6, 0.0,
        0.0,  0.3, -0.6, 0.0,
    ],
    dtype=np.float32,
)


ACTION_SCALE_ROBOT = np.array(
    [
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
        0.125, 0.25, 0.25, 5.0,
    ],
    dtype=np.float32,
)


KP = np.array(
    [
        80, 80, 80, 0,
        80, 80, 80, 0,
        80, 80, 80, 0,
        80, 80, 80, 0,
    ],
    dtype=np.float32,
)


KD = np.array(
    [
        2, 2, 2, 0.6,
        2, 2, 2, 0.6,
        2, 2, 2, 0.6,
        2, 2, 2, 0.6,
    ],
    dtype=np.float32,
)


# ============================================================
# Joint order
# ============================================================

ROBOT_ORDER = [
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


POLICY_ORDER = [
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",

    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",

    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",

    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",

    "fl_wheel_joint",
    "fr_wheel_joint",
    "hl_wheel_joint",
    "hr_wheel_joint",
]


def generate_permutation(source, target):

    mapping = {
        name: i
        for i, name in enumerate(source)
    }

    return [
        mapping[name]
        for name in target
    ]


ROBOT_TO_POLICY = np.array(
    generate_permutation(
        ROBOT_ORDER,
        POLICY_ORDER
    ),
    dtype=np.int32,
)


POLICY_TO_ROBOT = np.array(
    generate_permutation(
        POLICY_ORDER,
        ROBOT_ORDER
    ),
    dtype=np.int32,
)


# ============================================================
# Navigation
# ============================================================

MAX_FORWARD_SPEED = 0.35
MAX_SIDE_SPEED = 0.20
MAX_YAW_SPEED = 0.80

TURN_GAIN = 1.5

HEADING_SLOWDOWN_ANGLE = math.radians(45.0)
HEADING_STOP_ANGLE = math.radians(90.0)

NAV_LOG_PERIOD = 1.0


DEFAULT_WAYPOINTS = [
    (0.0, -1.725),
    (-0.7125, 11.655),
    (-8.73, 11.8425),
    (-10.0, 0.0),
]


# ============================================================
# Helpers
# ============================================================

def normalize_angle(angle):

    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def clamp(value, lo, hi):

    return max(
        lo,
        min(value, hi)
    )


def quaternion_to_euler(q):

    q = np.asarray(
        q,
        dtype=np.float64
    )

    w, x, y, z = q

    roll = math.atan2(
        2.0 * (w*x + y*z),
        1.0 - 2.0 * (x*x + y*y),
    )

    t2 = np.clip(
        2.0 * (w*y - z*x),
        -1.0,
        1.0,
    )

    pitch = math.asin(t2)

    yaw = math.atan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0 * (y*y + z*z),
    )

    return np.array(
        [roll, pitch, yaw],
        dtype=np.float32,
    )


def find_policy_path():

    paths = [
        CURRENT_DIR
        / ".."
        / ".."
        / ".."
        / "policy"
        / "policy.onnx",

        CURRENT_DIR
        / ".."
        / ".."
        / ".."
        / ".."
        / "policy"
        / "policy.onnx",
    ]

    for path in paths:

        path = path.resolve()

        if path.exists():

            return str(path)

    return None


def parse_cli_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_XML_PATHS),
        default=DEFAULT_SCENE_NAME,
    )

    parser.add_argument(
        "--xml-path",
        default=os.environ.get(
            "S10_MUJOCO_XML"
        ),
    )

    parser.add_argument(
        "--model-key",
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--policy",
        default=None,
    )

    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )

    return parser.parse_known_args()


def resolve_xml_path(scene, xml_path):

    if xml_path:

        return str(
            Path(xml_path)
            .expanduser()
            .resolve()
        )

    return str(
        SCENE_XML_PATHS[scene]
        .resolve()
    )


# ============================================================
# MuJoCo node
# ============================================================

class MuJoCoSimulationNode(Node):

    def __init__(
        self,
        model_key=MODEL_NAME,
        xml_path=None,
        policy_path=None,
        use_viewer=True,
    ):

        super().__init__(
            "mujoco_simulation"
        )

        # ----------------------------------------------------
        # Load MJCF
        # ----------------------------------------------------

        if xml_path is None:

            xml_path = str(
                SCENE_XML_PATHS[
                    DEFAULT_SCENE_NAME
                ].resolve()
            )

        if not os.path.isfile(xml_path):

            raise FileNotFoundError(
                f"Cannot find MJCF: {xml_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(
            xml_path
        )

        self.model.opt.timestep = DT

        self.data = mujoco.MjData(
            self.model
        )

        if self.model.nu != MOTOR_NUM:

            raise RuntimeError(
                f"S10 requires {MOTOR_NUM} actuators, "
                f"MJCF has {self.model.nu}"
            )

        self.dof_num = self.model.nu

        # ----------------------------------------------------
        # Commands
        # ----------------------------------------------------

        self.kp_cmd = KP.copy()
        self.kd_cmd = KD.copy()

        self.pos_cmd = np.zeros(
            MOTOR_NUM,
            dtype=np.float32
        )

        self.vel_cmd = np.zeros(
            MOTOR_NUM,
            dtype=np.float32
        )

        self.tau_ff = np.zeros(
            MOTOR_NUM,
            dtype=np.float32
        )

        self.input_tq = np.zeros(
            MOTOR_NUM,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # RL state
        # ----------------------------------------------------

        self.last_action = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        self.current_action = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        self.simulation_time = 0.0

        self.standing_time = 3.0

        self.rl_enabled = False

        # ----------------------------------------------------
        # Navigation state
        # ----------------------------------------------------

        self.navigation_enabled = True

        self.current_waypoint = 0

        self.track_complete = False

        self.track_waypoint_positions = np.empty(
            (0, 3),
            dtype=np.float64
        )

        self.track_body_id = -1

        self.track_waypoint_geom_ids = []

        # ----------------------------------------------------
        # Set initial pose BEFORE waypoint extraction
        # ----------------------------------------------------

        self._set_initial_pose(
            model_key
        )

        self._init_track_progress()

        # ----------------------------------------------------
        # Load ONNX
        # ----------------------------------------------------

        if policy_path is None:

            policy_path = find_policy_path()

        self.policy_path = policy_path

        self.ort_session = None

        if policy_path:

            self._load_policy(
                policy_path
            )

        else:

            self.get_logger().warn(
                "[RL] policy.onnx not found"
            )

            self.get_logger().warn(
                "[RL] Robot will remain in standing mode"
            )

        # ----------------------------------------------------
        # ROS
        # ----------------------------------------------------

        self.imu_pub = self.create_publisher(
            ImuData,
            "/IMU_DATA",
            200,
        )

        self.joints_pub = self.create_publisher(
            JointsData,
            "/JOINTS_DATA",
            200,
        )

        self.nav_pub = self.create_publisher(
            Twist,
            "/NAV_CMD",
            20,
        )

        self.cmd_sub = self.create_subscription(
            JointsDataCmd,
            "/JOINTS_CMD",
            self._cmd_callback,
            50,
        )

        # ----------------------------------------------------
        # Viewer
        # ----------------------------------------------------

        self.viewer = None

        if use_viewer:

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )

            self._configure_viewer()

        # ----------------------------------------------------
        # Startup information
        # ----------------------------------------------------

        self.get_logger().info(
            "=================================================="
        )

        self.get_logger().info(
            "[MuJoCo] S10 simulation started"
        )

        self.get_logger().info(
            f"[MuJoCo] MJCF: {xml_path}"
        )

        self.get_logger().info(
            f"[MuJoCo] DOF: {self.dof_num}"
        )

        self.get_logger().info(
            f"[MuJoCo] RL enabled: {self.rl_enabled}"
        )

        self.get_logger().info(
            f"[MuJoCo] Waypoints: "
            f"{len(self.track_waypoint_positions)}"
        )

        self.get_logger().info(
            "[MuJoCo] Navigation command: BODY FRAME"
        )

        self.get_logger().info(
            "[MuJoCo] Robot movement comes from RL policy"
        )

        self.get_logger().info(
            "=================================================="
        )

    # ========================================================
    # Initial pose
    # ========================================================

    def _set_initial_pose(self, key):

        if key not in JOINT_INIT:

            raise ValueError(
                f"Unknown robot model: {key}"
            )

        q0 = JOINT_INIT[key].astype(
            np.float64
        )

        # Free joint
        self.data.qpos[:3] = (
            TRACK_START_BASE_POS
        )

        self.data.qpos[3:7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        # 16 robot joints
        self.data.qpos[7:23] = q0

        self.data.qvel[:] = 0.0

        self.pos_cmd[:] = q0
        self.vel_cmd[:] = 0.0
        self.tau_ff[:] = 0.0
        self.input_tq[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.get_logger().info(
            "[MuJoCo] Initial standing pose configured"
        )

        self.get_logger().info(
            "[MuJoCo] Base position: "
            + np.array2string(
                self.data.qpos[:3],
                precision=3,
            )
        )

        self.get_logger().info(
            "[MuJoCo] Initial joints: "
            + np.array2string(
                q0,
                precision=3,
                suppress_small=True,
            )
        )

    # ========================================================
    # Waypoints
    # ========================================================

    @staticmethod
    def _track_geom_index(name):

        if (
            not name
            or not name.startswith(
                TRACK_WAYPOINT_PREFIX
            )
        ):

            return None

        suffix = name[
            len(TRACK_WAYPOINT_PREFIX):
        ]

        number = suffix.split(
            "_",
            1
        )[0]

        return (
            int(number)
            if number.isdigit()
            else None
        )

    def _init_track_progress(self):

        self.track_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            TRACK_BODY_NAME,
        )

        if self.track_body_id < 0:

            raise RuntimeError(
                f"Cannot find body '{TRACK_BODY_NAME}'"
            )

        mujoco.mj_forward(
            self.model,
            self.data
        )

        found = {}

        for gid in range(
            self.model.ngeom
        ):

            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                gid,
            )

            idx = self._track_geom_index(
                name
            )

            if idx is not None:

                found[idx] = gid

        # ----------------------------------------------------
        # XML waypoints
        # ----------------------------------------------------

        if not found:

            self.track_waypoint_positions = np.array(
                [
                    [x, y, 0.0]
                    for x, y in DEFAULT_WAYPOINTS
                ],
                dtype=np.float64,
            )

            self.get_logger().warn(
                "[TRACK] No waypoint geoms found"
            )

            self.get_logger().warn(
                "[TRACK] Using DEFAULT_WAYPOINTS"
            )

        else:

            indices = sorted(found)

            self.track_waypoint_geom_ids = [
                found[i]
                for i in indices
            ]

            self.track_waypoint_positions = np.array(
                [
                    self.data.geom_xpos[
                        found[i]
                    ].copy()
                    for i in indices
                ],
                dtype=np.float64,
            )

        # ----------------------------------------------------
        # Print waypoints
        # ----------------------------------------------------

        self.get_logger().info(
            f"[TRACK] {len(self.track_waypoint_positions)} "
            f"waypoints loaded"
        )

        for i, p in enumerate(
            self.track_waypoint_positions
        ):

            self.get_logger().info(
                f"[TRACK] WP {i:02d}: "
                f"x={p[0]:.3f}, "
                f"y={p[1]:.3f}, "
                f"z={p[2]:.3f}"
            )

        self.track_complete = (
            len(self.track_waypoint_positions) == 0
        )

    # ========================================================
    # Robot pose
    # ========================================================

    def _get_robot_pose(self):

        position = self.data.xpos[
            self.track_body_id
        ].copy()

        quaternion = self.data.xquat[
            self.track_body_id
        ].copy()

        rpy = quaternion_to_euler(
            quaternion
        )

        return position, rpy

    # ========================================================
    # Navigation
    # ========================================================

    def _compute_navigation_command(self):

        if (
            not self.navigation_enabled
            or self.track_complete
        ):

            return 0.0, 0.0, 0.0

        if (
            self.current_waypoint
            >= len(
                self.track_waypoint_positions
            )
        ):

            self.track_complete = True

            self.get_logger().info(
                "[NAV] ALL WAYPOINTS REACHED"
            )

            return 0.0, 0.0, 0.0

        # ----------------------------------------------------
        # Actual robot position
        # ----------------------------------------------------

        position, rpy = self._get_robot_pose()

        target = self.track_waypoint_positions[
            self.current_waypoint
        ]

        dx = float(
            target[0] - position[0]
        )

        dy = float(
            target[1] - position[1]
        )

        distance = math.hypot(
            dx,
            dy
        )

        # ----------------------------------------------------
        # Waypoint reached
        #
        # IMPORTANT:
        # only ONE waypoint is advanced.
        # ----------------------------------------------------

        if distance <= WAYPOINT_REACH_RADIUS:

            reached = self.current_waypoint

            self.get_logger().info(
                f"[NAV] WAYPOINT {reached} REACHED"
            )

            self.current_waypoint += 1

            if (
                self.current_waypoint
                >= len(
                    self.track_waypoint_positions
                )
            ):

                self.track_complete = True

                self.get_logger().info(
                    "[NAV] ==========================="
                )

                self.get_logger().info(
                    "[NAV] ALL WAYPOINTS REACHED"
                )

                self.get_logger().info(
                    "[NAV] ==========================="
                )

                return 0.0, 0.0, 0.0

            target = self.track_waypoint_positions[
                self.current_waypoint
            ]

            dx = float(
                target[0] - position[0]
            )

            dy = float(
                target[1] - position[1]
            )

            distance = math.hypot(
                dx,
                dy
            )

        # ----------------------------------------------------
        # Desired heading
        # ----------------------------------------------------

        target_yaw = math.atan2(
            dy,
            dx
        )

        robot_yaw = float(
            rpy[2]
        )

        heading_error = normalize_angle(
            target_yaw - robot_yaw
        )

        abs_error = abs(
            heading_error
        )

        # ----------------------------------------------------
        # Turning
        # ----------------------------------------------------

        wz = clamp(
            TURN_GAIN * heading_error,
            -MAX_YAW_SPEED,
            MAX_YAW_SPEED,
        )

        # ----------------------------------------------------
        # Forward speed
        # ----------------------------------------------------

        if abs_error >= HEADING_STOP_ANGLE:

            speed = 0.0

        elif abs_error > HEADING_SLOWDOWN_ANGLE:

            speed = (
                MAX_FORWARD_SPEED
                * clamp(
                    1.0
                    - (
                        abs_error
                        - HEADING_SLOWDOWN_ANGLE
                    )
                    / (
                        HEADING_STOP_ANGLE
                        - HEADING_SLOWDOWN_ANGLE
                    ),
                    0.0,
                    1.0,
                )
            )

        else:

            speed = MAX_FORWARD_SPEED

        # Slow down close to waypoint.

        if distance < 1.0:

            speed *= clamp(
                distance,
                0.15,
                1.0,
            )

        # ----------------------------------------------------
        # World direction -> BODY FRAME
        # ----------------------------------------------------

        if distance > 1e-6:

            ux = dx / distance
            uy = dy / distance

            c = math.cos(
                robot_yaw
            )

            s = math.sin(
                robot_yaw
            )

            vx_dir = (
                c * ux
                + s * uy
            )

            vy_dir = (
                -s * ux
                + c * uy
            )

        else:

            vx_dir = 0.0
            vy_dir = 0.0

        vx = clamp(
            speed * vx_dir,
            -MAX_FORWARD_SPEED,
            MAX_FORWARD_SPEED,
        )

        vy = clamp(
            speed * vy_dir,
            -MAX_SIDE_SPEED,
            MAX_SIDE_SPEED,
        )

        wz = clamp(
            wz,
            -MAX_YAW_SPEED,
            MAX_YAW_SPEED,
        )

        return vx, vy, wz

    # ========================================================
    # Navigation command publisher
    # ========================================================

    def _publish_navigation_command(
        self,
        vx,
        vy,
        wz,
    ):

        msg = Twist()

        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)

        self.nav_pub.publish(
            msg
        )

    # ========================================================
    # RL observation
    # ========================================================

    def _build_observation(
        self,
        command,
    ):

        # ----------------------------------------------------
        # Angular velocity
        # ----------------------------------------------------

        base_omega = (
            self.data.sensordata[7:10]
            .astype(np.float32)
            * OMEGA_SCALE
        )

        # ----------------------------------------------------
        # Projected gravity
        # ----------------------------------------------------

        quat = np.asarray(
            self.data.xquat[
                self.track_body_id
            ],
            dtype=np.float64,
        )

        rotation_matrix = np.zeros(
            9,
            dtype=np.float64
        )

        mujoco.mju_quat2Mat(
            rotation_matrix,
            quat,
        )

        rotation_matrix = (
            rotation_matrix.reshape(
                3,
                3
            )
        )

        projected_gravity = (
            rotation_matrix.T
            @ GRAVITY_DIRECTION.astype(
                np.float64
            )
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Joint positions
        # ----------------------------------------------------

        q = self.data.qpos[
            7:23
        ].astype(
            np.float32
        )

        dq = self.data.qvel[
            6:22
        ].astype(
            np.float32
        )

        q_policy = q[
            ROBOT_TO_POLICY
        ].copy()

        dq_policy = (
            dq[
                ROBOT_TO_POLICY
            ].copy()
            * DOF_VEL_SCALE
        )

        # Wheels do not use joint position
        # in the policy observation.

        q_policy[12:16] = 0.0

        q_policy -= DOF_DEFAULT_POLICY

        # ----------------------------------------------------
        # Observation
        #
        # 3 angular velocity
        # 3 gravity
        # 3 command
        # 16 joint position
        # 16 joint velocity
        # 16 previous action
        #
        # = 57
        # ----------------------------------------------------

        obs = np.concatenate(
            [
                base_omega,
                projected_gravity,
                np.asarray(
                    command,
                    dtype=np.float32
                ),
                q_policy,
                dq_policy,
                self.last_action,
            ]
        ).astype(
            np.float32
        )

        if obs.size != OBSERVATION_DIM:

            raise RuntimeError(
                f"Expected {OBSERVATION_DIM}-D "
                f"observation, got {obs.size}"
            )

        return obs

    # ========================================================
    # Load policy
    # ========================================================

    def _load_policy(
        self,
        path,
    ):

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"Policy not found: {path}"
            )

        self.get_logger().info(
            f"[RL] Loading policy: {path}"
        )

        self.ort_session = (
            ort.InferenceSession(
                path,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.policy_input_name = (
            self.ort_session
            .get_inputs()[0]
            .name
        )

        self.policy_output_name = (
            self.ort_session
            .get_outputs()[0]
            .name
        )

        self.rl_enabled = True

        self.get_logger().info(
            f"[RL] Input: "
            f"{self.policy_input_name}"
        )

        self.get_logger().info(
            f"[RL] Output: "
            f"{self.policy_output_name}"
        )

        self.get_logger().info(
            "[RL] Policy loaded successfully"
        )

        self._print_joint_permutation()

    # ========================================================
    # Joint permutation debug
    # ========================================================

    def _print_joint_permutation(self):

        self.get_logger().info(
            "===== JOINT PERMUTATION ====="
        )

        for i in range(16):

            self.get_logger().info(
                f"i={i} "
                f"robot2policy={ROBOT_TO_POLICY[i]} "
                f"policy2robot={POLICY_TO_ROBOT[i]}"
            )

        self.get_logger().info(
            "============================="
        )

    # ========================================================
    # Run ONNX
    # ========================================================

    def _run_policy(
        self,
        observation,
    ):

        if self.ort_session is None:

            return np.zeros(
                ACTION_DIM,
                dtype=np.float32
            )

        obs = (
            observation
            .reshape(
                1,
                OBSERVATION_DIM
            )
            .astype(
                np.float32
            )
        )

        output = self.ort_session.run(
            [
                self.policy_output_name
            ],
            {
                self.policy_input_name: obs
            },
        )[0]

        action = np.asarray(
            output,
            dtype=np.float32
        ).reshape(-1)

        if action.size != ACTION_DIM:

            raise RuntimeError(
                f"Policy returned "
                f"{action.size} actions; "
                f"expected {ACTION_DIM}"
            )

        # Most locomotion policies expect
        # actions approximately in [-1, 1].

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        return action

    # ========================================================
    # Apply RL action
    # ========================================================

    def _apply_policy_action(
        self,
        policy_action,
    ):

        policy_action = np.asarray(
            policy_action,
            dtype=np.float32
        )

        self.current_action = (
            policy_action.copy()
        )

        # Policy order -> robot order.

        action_robot = (
            policy_action[
                POLICY_TO_ROBOT
            ].copy()
        )

        # Scale policy output.

        action_robot = (
            action_robot
            * ACTION_SCALE_ROBOT
            + DOF_DEFAULT_ROBOT
        )

        # ----------------------------------------------------
        # Leg joints = position control
        # Wheel joints = velocity control
        # ----------------------------------------------------

        for leg in range(4):

            base = leg * 4

            # Hip X
            self.pos_cmd[
                base
            ] = action_robot[
                base
            ]

            # Hip Y
            self.pos_cmd[
                base + 1
            ] = action_robot[
                base + 1
            ]

            # Knee
            self.pos_cmd[
                base + 2
            ] = action_robot[
                base + 2
            ]

            # Wheel velocity
            self.vel_cmd[
                base + 3
            ] = action_robot[
                base + 3
            ]

        # Previous action is the POLICY-order action.

        self.last_action = (
            policy_action.copy()
        )

    # ========================================================
    # ROS joint command callback
    # ========================================================

    def _cmd_callback(
        self,
        msg,
    ):

        try:

            joints = (
                msg.data.joints_data
            )

        except Exception:

            self.get_logger().warn(
                "[CMD] Invalid JointsDataCmd"
            )

            return

        if len(joints) != 16:

            self.get_logger().warn(
                f"[CMD] Invalid joint count: "
                f"{len(joints)}"
            )

            return

        for i, joint in enumerate(
            joints
        ):

            self.kp_cmd[i] = joint.kp
            self.kd_cmd[i] = joint.kd
            self.pos_cmd[i] = joint.position
            self.vel_cmd[i] = joint.velocity
            self.tau_ff[i] = joint.torque

    # ========================================================
    # Apply PD torque
    # ========================================================

    def _apply_joint_torque(self):

        q = self.data.qpos[
            7:23
        ].astype(
            np.float32
        )

        dq = self.data.qvel[
            6:22
        ].astype(
            np.float32
        )

        torque = (
            self.kp_cmd
            * (
                self.pos_cmd
                - q
            )
            +
            self.kd_cmd
            * (
                self.vel_cmd
                - dq
            )
            +
            self.tau_ff
        )

        torque = np.asarray(
            torque,
            dtype=np.float64
        ).reshape(16)

        self.data.ctrl[:] = torque

        self.input_tq[:] = (
            torque.astype(
                np.float32
            )
        )

    # ========================================================
    # IMU
    # ========================================================

    def _publish_imu(self):

        q = self.data.xquat[
            self.track_body_id
        ]

        rpy = quaternion_to_euler(
            q
        )

        body_acc = (
            self.data.sensordata[4:7]
        )

        angvel = (
            self.data.sensordata[7:10]
        )

        msg = ImuData()

        msg.header = MetaType()
        msg.header.frame_id = 0

        sec = int(
            self.simulation_time
        )

        stamp = Time()

        stamp.sec = sec

        stamp.nanosec = int(
            (
                self.simulation_time
                - sec
            )
            * 1e9
        )

        msg.header.stamp = stamp

        msg.data = ImuDataValue()

        msg.data.roll = float(
            math.degrees(rpy[0])
        )

        msg.data.pitch = float(
            math.degrees(rpy[1])
        )

        msg.data.yaw = float(
            math.degrees(rpy[2])
        )

        msg.data.omega_x = float(
            angvel[0]
        )

        msg.data.omega_y = float(
            angvel[1]
        )

        msg.data.omega_z = float(
            angvel[2]
        )

        msg.data.acc_x = float(
            body_acc[0]
        )

        msg.data.acc_y = float(
            body_acc[1]
        )

        msg.data.acc_z = float(
            body_acc[2]
        )

        self.imu_pub.publish(
            msg
        )

    # ========================================================
    # Joint state
    # ========================================================

    def _publish_joint_state(self):

        q = self.data.qpos[
            7:23
        ].astype(
            np.float32
        )

        dq = self.data.qvel[
            6:22
        ].astype(
            np.float32
        )

        pub_pos = (
            q - POS_OFFSET_RAD
        ) * JOINT_DIR

        pub_vel = (
            dq * JOINT_DIR
        )

        pub_tau = (
            self.input_tq
            * JOINT_DIR
        )

        msg = JointsData()

        msg.header = MetaType()
        msg.header.frame_id = 0

        sec = int(
            self.simulation_time
        )

        stamp = Time()

        stamp.sec = sec

        stamp.nanosec = int(
            (
                self.simulation_time
                - sec
            )
            * 1e9
        )

        msg.header.stamp = stamp

        msg.data = JointsDataValue()

        msg.data.joints_data = [
            JointData()
            for _ in range(16)
        ]

        for i, joint in enumerate(
            msg.data.joints_data
        ):

            joint.name = [
                32,
                32,
                32,
                32,
            ]

            joint.data_id = 0
            joint.status_word = 1

            joint.position = float(
                pub_pos[i]
            )

            joint.velocity = float(
                pub_vel[i]
            )

            joint.torque = float(
                pub_tau[i]
            )

            joint.motion_temp = 40.0
            joint.driver_temp = 45.0

        self.joints_pub.publish(
            msg
        )

    # ========================================================
    # Viewer
    # ========================================================

    def _configure_viewer(self):

        with self.viewer.lock():

            bid = self.track_body_id

            if (
                TRACK_VIEWER
                and bid >= 0
            ):

                self.viewer.cam.type = (
                    mujoco.mjtCamera
                    .mjCAMERA_TRACKING
                )

                self.viewer.cam.trackbodyid = (
                    bid
                )

            else:

                self.viewer.cam.type = (
                    mujoco.mjtCamera
                    .mjCAMERA_FREE
                )

                self.viewer.cam.trackbodyid = -1

                self.viewer.cam.lookat[:] = (
                    self.data.xpos[bid]
                )

            self.viewer.cam.fixedcamid = -1

            self.viewer.cam.azimuth = (
                CAMERA_AZIMUTH
            )

            self.viewer.cam.elevation = (
                CAMERA_ELEVATION
            )

            self.viewer.cam.distance = (
                CAMERA_DISTANCE
            )

    # ========================================================
    # Navigation logging
    # ========================================================

    def _log_navigation(
        self,
        vx,
        vy,
        wz,
    ):

        if self.track_complete:

            self.get_logger().info(
                "[NAV] TRACK COMPLETE"
            )

            return

        position, rpy = (
            self._get_robot_pose()
        )

        target = (
            self.track_waypoint_positions[
                self.current_waypoint
            ]
        )

        dx = float(
            target[0] - position[0]
        )

        dy = float(
            target[1] - position[1]
        )

        distance = math.hypot(
            dx,
            dy
        )

        target_yaw = math.atan2(
            dy,
            dx
        )

        heading_error = normalize_angle(
            target_yaw
            - float(rpy[2])
        )

        # Actual body velocity from MuJoCo.

        actual_body_velocity = (
            self.data.qvel[
                0:3
            ].copy()
        )

        self.get_logger().info(
            f"[NAV] "
            f"POS=("
            f"{position[0]:.3f}, "
            f"{position[1]:.3f}, "
            f"{position[2]:.3f}) "
            f"yaw={math.degrees(rpy[2]):.1f}deg | "
            f"target=("
            f"{target[0]:.3f}, "
            f"{target[1]:.3f}) | "
            f"dist={distance:.3f}m | "
            f"WP={self.current_waypoint}/"
            f"{len(self.track_waypoint_positions)} | "
            f"CMD=("
            f"{vx:.3f}, "
            f"{vy:.3f}, "
            f"{wz:.3f}) | "
            f"qvel_base=("
            f"{actual_body_velocity[0]:.3f}, "
            f"{actual_body_velocity[1]:.3f}, "
            f"{actual_body_velocity[2]:.3f})"
        )

    # ========================================================
    # RL diagnostic
    # ========================================================

    def _log_rl_action(
        self,
        command,
        action,
    ):

        self.get_logger().info(
            "[RL] "
            f"CMD=("
            f"{command[0]:.3f}, "
            f"{command[1]:.3f}, "
            f"{command[2]:.3f}) "
            f"ACTION[min={np.min(action):.3f}, "
            f"max={np.max(action):.3f}, "
            f"mean={np.mean(action):.3f}]"
        )

    # ========================================================
    # Main loop
    # ========================================================

    def start(self):

        step = 0

        last_wall = time.perf_counter()

        last_nav_log = 0.0

        last_rl_log = 0.0

        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.0
            )

            now = time.perf_counter()

            if (
                now - last_wall
                < DT
            ):

                time.sleep(
                    0.0001
                )

                continue

            last_wall = now

            step += 1

            self.simulation_time = (
                step * DT
            )

            # =================================================
            # Navigation command
            # =================================================

            vx, vy, wz = (
                self._compute_navigation_command()
            )

            command = np.array(
                [
                    vx,
                    vy,
                    wz,
                ],
                dtype=np.float32,
            )

            self._publish_navigation_command(
                vx,
                vy,
                wz,
            )

            # =================================================
            # Standing phase
            # =================================================

            if (
                self.simulation_time
                < self.standing_time
            ):

                self.pos_cmd[:] = (
                    JOINT_INIT[
                        MODEL_NAME
                    ]
                )

                self.vel_cmd[:] = 0.0

                self.tau_ff[:] = 0.0

                # Make sure policy history
                # starts from zero.

                self.last_action[:] = 0.0

            # =================================================
            # RL walking
            # =================================================

            elif (
                self.rl_enabled
                and step
                % POLICY_DECIMATION
                == 0
            ):

                try:

                    obs = (
                        self._build_observation(
                            command
                        )
                    )

                    action = (
                        self._run_policy(
                            obs
                        )
                    )

                    self._apply_policy_action(
                        action
                    )

                    # Diagnostic once per second.

                    if (
                        self.simulation_time
                        - last_rl_log
                        >= 1.0
                    ):

                        last_rl_log = (
                            self.simulation_time
                        )

                        self._log_rl_action(
                            command,
                            action,
                        )

                except Exception as exc:

                    self.get_logger().error(
                        f"[RL] Policy error: {exc}"
                    )

                    # Freeze safely.

                    self.pos_cmd[:] = (
                        self.data.qpos[
                            7:23
                        ]
                    )

                    self.vel_cmd[:] = 0.0

                    self.last_action[:] = 0.0

            # =================================================
            # PD control
            # =================================================

            self._apply_joint_torque()

            # =================================================
            # MuJoCo physics
            # =================================================

            mujoco.mj_step(
                self.model,
                self.data
            )

            # =================================================
            # ROS publishing
            # =================================================

            if step % 5 == 0:

                try:

                    self._publish_imu()

                    self._publish_joint_state()

                except Exception as exc:

                    self.get_logger().error(
                        f"[ROS] Publish error: {exc}"
                    )

            # =================================================
            # Navigation logging
            # =================================================

            if (
                self.simulation_time
                - last_nav_log
                >= NAV_LOG_PERIOD
            ):

                last_nav_log = (
                    self.simulation_time
                )

                self._log_navigation(
                    vx,
                    vy,
                    wz,
                )

            # =================================================
            # Viewer
            # =================================================

            if (
                self.viewer
                and step
                % RENDER_INTERVAL
                == 0
            ):

                try:

                    self.viewer.sync()

                except Exception:

                    pass

    # ========================================================
    # Shutdown
    # ========================================================

    def shutdown(self):

        try:

            self.data.ctrl[:] = 0.0

        except Exception:

            pass

        try:

            if self.viewer:

                self.viewer.close()

        except Exception:

            pass


# ============================================================
# Main
# ============================================================

def main():

    cli_args, ros_args = (
        parse_cli_args()
    )

    rclpy.init(
        args=ros_args
    )

    xml_path = resolve_xml_path(
        cli_args.scene,
        cli_args.xml_path,
    )

    policy_path = (
        cli_args.policy
        if cli_args.policy
        else find_policy_path()
    )

    node = None

    try:

        node = MuJoCoSimulationNode(
            model_key=cli_args.model_key,
            xml_path=xml_path,
            policy_path=policy_path,
            use_viewer=not cli_args.no_viewer,
        )

        node.start()

    except KeyboardInterrupt:

        print(
            "\n[MuJoCo] Keyboard interrupt"
        )

    except Exception as exc:

        print(
            "\n[MuJoCo] ERROR:"
        )

        print(exc)

        raise

    finally:

        if node is not None:

            node.shutdown()

            node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()