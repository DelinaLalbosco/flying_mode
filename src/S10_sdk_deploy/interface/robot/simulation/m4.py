#!/usr/bin/env python3

import os
import time
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time

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
# CONFIGURATION
# ============================================================

MODEL_NAME = "S10"

USE_VIEWER = True
TRACK_VIEWER = True

DT = 0.001

RENDER_INTERVAL = 10

TRACK_BODY_NAME = "base_link"


# ============================================================
# VIEWER
# ============================================================

CAMERA_AZIMUTH = 90.0
CAMERA_ELEVATION = -25.0
CAMERA_DISTANCE = 12.0


# ============================================================
# BASE / TERRAIN CONFIGURATION
# ============================================================

# Minimum safe base height.
BASE_MIN_HEIGHT = 0.42

# Initial base height.
BASE_INITIAL_HEIGHT = 0.48

# Normal clearance between lowest robot geometry
# and terrain.
TERRAIN_CLEARANCE = 0.25

# Extra clearance when climbing stairs.
STAIR_EXTRA_CLEARANCE = 0.20

# IMPORTANT:
# Large staircase around WP19 -> WP20 may require
# significantly more base height.
BASE_MAX_HEIGHT = 1.80

# Maximum downward movement.
#
# Keep this relatively slow so the robot does not suddenly
# drop into a staircase.
MAX_BASE_DOWN_SPEED = 0.20

# Maximum upward movement.
#
# Increased significantly so the robot can rise BEFORE
# reaching a very high stair.
MAX_BASE_UP_SPEED = 1.20

# Terrain look-ahead.
TERRAIN_LOOKAHEAD = 2.0

# Side-to-side terrain inspection.
TERRAIN_SIDE_OFFSET = 0.45

# Ray origin above the robot.
TERRAIN_RAY_START_HEIGHT = 2.5

# Maximum ray length.
TERRAIN_RAY_LENGTH = 5.0


# ============================================================
# TERRAIN SAMPLE POINTS
# ============================================================

# Robot local frame:
#
#                  FRONT
#
#                    x+
#
#          left              right
#
#              1.7
#              1.3
#              0.9
#              0.5
#                |
#                |
#                C
#
# The long look-ahead is particularly important for
# WP19 -> WP20 and the following large staircase.

TERRAIN_SAMPLE_POINTS = [

    # Current position
    (0.00, 0.00),

    # Near front
    (0.30, 0.00),
    (0.50, 0.00),
    (0.75, 0.00),

    # Medium distance
    (1.00, 0.00),
    (1.25, 0.00),
    (1.50, 0.00),

    # Far ahead
    (1.75, 0.00),
    (2.00, 0.00),

    # Left
    (0.50, 0.35),
    (0.90, 0.40),
    (1.30, 0.40),
    (1.70, 0.40),

    # Right
    (0.50, -0.35),
    (0.90, -0.40),
    (1.30, -0.40),
    (1.70, -0.40),

    # Rear
    (-0.30, 0.00),
]


# ============================================================
# TRACK
# ============================================================

TRACK_START_BASE_POS = np.array(
    [
        0.0,
        -2.5,
        BASE_INITIAL_HEIGHT,
    ],
    dtype=np.float64,
)

TRACK_WAYPOINT_PREFIX = "track_waypoint_"

TRACK_HEIGHT_POST_PREFIX = "track_height_post_"

# Competition specification says 0.2 m.
TRACK_REACH_RADIUS = 0.20


# ============================================================
# NAVIGATION
# ============================================================

NAVIGATION_ENABLED = True

# Increased from 0.35 for faster movement.
MAX_FORWARD_SPEED = 0.60

MAX_TURN_SPEED = 1.20

TURN_GAIN = 1.5

HEADING_SLOWDOWN_ANGLE = np.deg2rad(45.0)

HEADING_STOP_ANGLE = np.deg2rad(90.0)

POSITION_PRINT_INTERVAL = 0.5


# ============================================================
# S10 JOINT CALIBRATION
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
        -35, 145, -156, 0,
        35, 145, -156, 0,
    ],
    dtype=np.float32,
)


POS_OFFSET_RAD = (
    POS_OFFSET_DEG
    / 180.0
    * np.pi
)


# ============================================================
# INITIAL JOINT POSITION
# ============================================================

JOINT_INIT = {
    "S10": np.array(
        [
            -0.438,
            -1.16,
            2.76,
            0.0,

            0.438,
            -1.16,
            2.76,
            0.0,

            -0.438,
            1.16,
            -2.76,
            0.0,

            0.438,
            1.16,
            -2.76,
            0.0,
        ],
        dtype=np.float32,
    )
}


# ============================================================
# PATHS
# ============================================================

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
    "track": MJCF_DIR / "S10_track.xml",
}


DEFAULT_SCENE_NAME = os.environ.get(
    "S10_MUJOCO_SCENE",
    "track",
)


# ============================================================
# CLI
# ============================================================

def parse_cli_args():

    parser = argparse.ArgumentParser(
        description="S10 MuJoCo ROS2 simulation"
    )

    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_XML_PATHS),
        default=(
            DEFAULT_SCENE_NAME
            if DEFAULT_SCENE_NAME in SCENE_XML_PATHS
            else "track"
        ),
    )

    parser.add_argument(
        "--xml-path",
        default=os.environ.get(
            "S10_MUJOCO_XML",
            "",
        ),
    )

    parser.add_argument(
        "--model-key",
        default=MODEL_NAME,
    )

    return parser.parse_known_args()


# ============================================================
# RESOLVE XML
# ============================================================

def resolve_xml_path(
    scene_name,
    xml_path,
):

    if xml_path:

        return str(
            Path(
                xml_path
            ).expanduser().resolve()
        )

    return str(
        SCENE_XML_PATHS[
            scene_name
        ].resolve()
    )


# ============================================================
# SIMULATION NODE
# ============================================================

class MuJoCoSimulationNode(Node):

    def __init__(
        self,
        model_key=MODEL_NAME,
        xml_path=None,
    ):

        super().__init__(
            "mujoco_simulation"
        )

        if xml_path is None:

            xml_path = str(
                SCENE_XML_PATHS["track"]
            )

        # ----------------------------------------------------
        # Validate XML
        # ----------------------------------------------------

        if not os.path.isfile(xml_path):

            raise FileNotFoundError(
                f"Cannot find MJCF file:\n{xml_path}"
            )

        self.get_logger().info(
            "[INFO] Loading MuJoCo MJCF:"
        )

        self.get_logger().info(
            f"       {xml_path}"
        )

        # ----------------------------------------------------
        # Load MuJoCo
        # ----------------------------------------------------

        self.model = (
            mujoco.MjModel.from_xml_path(
                xml_path
            )
        )

        self.model.opt.timestep = DT

        self.data = mujoco.MjData(
            self.model
        )

        # ----------------------------------------------------
        # Actuators
        # ----------------------------------------------------

        self.dof_num = self.model.nu

        if self.dof_num != 16:

            raise RuntimeError(
                f"Expected 16 actuators, "
                f"but found {self.dof_num}"
            )

        self.get_logger().info(
            "[INFO] MuJoCo MJCF loaded."
        )

        self.get_logger().info(
            f"[INFO] S10 DOF = {self.dof_num}"
        )

        # ----------------------------------------------------
        # Model key
        # ----------------------------------------------------

        self.model_key = model_key

        if model_key not in JOINT_INIT:

            raise KeyError(
                f"Unknown model key: {model_key}"
            )

        # ----------------------------------------------------
        # Track body
        # ----------------------------------------------------

        self.track_body_id = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                TRACK_BODY_NAME,
            )
        )

        if self.track_body_id < 0:

            raise RuntimeError(
                f"Cannot find body "
                f"'{TRACK_BODY_NAME}'"
            )

        self.get_logger().info(
            f"[BASE] Body '{TRACK_BODY_NAME}' "
            f"id={self.track_body_id}"
        )

        # ----------------------------------------------------
        # Initial pose
        # ----------------------------------------------------

        self._set_initial_pose(
            model_key
        )

        # ----------------------------------------------------
        # Track
        # ----------------------------------------------------

        self._init_track_progress()

        # ----------------------------------------------------
        # Terrain state
        # ----------------------------------------------------

        self.terrain_height = (
            0.0
        )

        self.front_terrain_height = (
            0.0
        )

        self.required_base_height = (
            BASE_INITIAL_HEIGHT
        )

        self.last_terrain_update = (
            -1.0
        )

        # ----------------------------------------------------
        # Estimate robot geometry
        # ----------------------------------------------------

        self.robot_lowest_relative_z = (
            self._estimate_lowest_robot_geometry()
        )

        self.get_logger().info(
            "[TERRAIN] Lowest robot geometry "
            f"relative Z = "
            f"{self.robot_lowest_relative_z:.3f} m"
        )

        # ----------------------------------------------------
        # Joint controller
        # ----------------------------------------------------

        self.kp_cmd = np.zeros(
            (self.dof_num, 1),
            dtype=np.float32,
        )

        self.kd_cmd = np.zeros(
            (self.dof_num, 1),
            dtype=np.float32,
        )

        self.pos_cmd = np.zeros(
            (self.dof_num, 1),
            dtype=np.float32,
        )

        self.vel_cmd = np.zeros(
            (self.dof_num, 1),
            dtype=np.float32,
        )

        self.tau_ff = np.zeros(
            (self.dof_num, 1),
            dtype=np.float32,
        )

        self.input_tq = np.zeros(
            (self.dof_num, 1),
            dtype=np.float32,
        )

        self.pos_cmd[:, 0] = (
            JOINT_INIT[model_key]
        )

        self.kp_cmd[:, 0] = 80.0

        self.kd_cmd[:, 0] = 5.0

        # ----------------------------------------------------
        # Navigation state
        # ----------------------------------------------------

        self.timestamp = 0.0

        self.last_position_print_time = (
            -1.0
        )

        self.commanded_base_yaw = (
            np.deg2rad(90.0)
        )

        self.commanded_base_x = (
            TRACK_START_BASE_POS[0]
        )

        self.commanded_base_y = (
            TRACK_START_BASE_POS[1]
        )

        self.commanded_base_z = (
            BASE_INITIAL_HEIGHT
        )

        # ----------------------------------------------------
        # ROS publishers
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

        # ----------------------------------------------------
        # ROS subscriber
        # ----------------------------------------------------

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

        if USE_VIEWER:

            self.viewer = (
                mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )
            )

            self._configure_viewer()

        # ----------------------------------------------------
        # Logs
        # ----------------------------------------------------

        self.get_logger().info(
            "[INFO] Robot pose monitor enabled."
        )

        self.get_logger().info(
            f"[VIEWER] GUI = {USE_VIEWER}"
        )

        self.get_logger().info(
            "[NAVIGATION] Autonomous waypoint "
            "navigation ENABLED."
        )

        self.get_logger().info(
            f"[NAVIGATION] Max forward speed = "
            f"{MAX_FORWARD_SPEED:.2f} m/s"
        )

        self.get_logger().info(
            "[TERRAIN] Terrain-aware base height "
            "protection ENABLED."
        )

        self.get_logger().info(
            f"[TERRAIN] Base height range = "
            f"{BASE_MIN_HEIGHT:.2f} -> "
            f"{BASE_MAX_HEIGHT:.2f} m"
        )

        self.get_logger().info(
            f"[TERRAIN] Lookahead = "
            f"{TERRAIN_LOOKAHEAD:.2f} m"
        )

    # ========================================================
    # INITIAL POSE
    # ========================================================

    def _set_initial_pose(
        self,
        model_key,
    ):

        qpos = self.data.qpos.copy()

        qpos[
            7:7 + self.dof_num
        ] = JOINT_INIT[model_key]

        qpos[:3] = (
            TRACK_START_BASE_POS
        )

        yaw = np.deg2rad(90.0)

        half = yaw / 2.0

        qpos[3] = np.cos(half)
        qpos[4] = 0.0
        qpos[5] = 0.0
        qpos[6] = np.sin(half)

        self.data.qpos[:] = qpos

        self.data.qvel[:] = 0.0

        self.commanded_base_yaw = yaw

        self.commanded_base_x = qpos[0]

        self.commanded_base_y = qpos[1]

        self.commanded_base_z = qpos[2]

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    # ========================================================
    # ROBOT POSITION
    # ========================================================

    def get_robot_position(self):

        return self.data.xpos[
            self.track_body_id
        ].copy()

    # ========================================================
    # ROBOT QUATERNION
    # ========================================================

    def get_robot_quaternion(self):

        return self.data.xquat[
            self.track_body_id
        ].copy()

    # ========================================================
    # QUATERNION -> YAW
    # ========================================================

    @staticmethod
    def quaternion_to_yaw(q):

        w, x, y, z = q

        return float(
            np.arctan2(
                2.0 * (
                    w * z
                    + x * y
                ),
                1.0
                - 2.0 * (
                    y * y
                    + z * z
                ),
            )
        )

    # ========================================================
    # QUATERNION -> EULER
    # ========================================================

    @staticmethod
    def quaternion_to_euler(q):

        w, x, y, z = q

        t0 = (
            2.0
            * (
                w * x
                + y * z
            )
        )

        t1 = (
            1.0
            - 2.0
            * (
                x * x
                + y * y
            )
        )

        roll = np.arctan2(
            t0,
            t1,
        )

        t2 = (
            2.0
            * (
                w * y
                - z * x
            )
        )

        t2 = np.clip(
            t2,
            -1.0,
            1.0,
        )

        pitch = np.arcsin(t2)

        t3 = (
            2.0
            * (
                w * z
                + x * y
            )
        )

        t4 = (
            1.0
            - 2.0
            * (
                y * y
                + z * z
            )
        )

        yaw = np.arctan2(
            t3,
            t4,
        )

        return np.array(
            [
                roll,
                pitch,
                yaw,
            ],
            dtype=np.float64,
        )

    # ========================================================
    # ESTIMATE LOWEST ROBOT GEOMETRY
    # ========================================================

    def _estimate_lowest_robot_geometry(self):

        if self.track_body_id < 0:

            return -0.30

        robot_geom_ids = []

        for geom_id in range(
            self.model.ngeom
        ):

            body_id = (
                self.model.geom_bodyid[
                    geom_id
                ]
            )

            current = body_id

            belongs_to_robot = False

            while current >= 0:

                if current == self.track_body_id:

                    belongs_to_robot = True

                    break

                parent = (
                    self.model.body_parentid[
                        current
                    ]
                )

                if parent == current:

                    break

                current = parent

            if belongs_to_robot:

                robot_geom_ids.append(
                    geom_id
                )

        if not robot_geom_ids:

            return -0.30

        base_z = self.data.xpos[
            self.track_body_id,
            2,
        ]

        lowest_z = float(
            np.min(
                self.data.geom_xpos[
                    robot_geom_ids,
                    2,
                ]
            )
        )

        return lowest_z - base_z

    # ========================================================
    # TRACK GEOM INDEX
    # ========================================================

    @staticmethod
    def _track_geom_index(
        name,
        prefix,
    ):

        if not name:

            return None

        if not name.startswith(
            prefix
        ):

            return None

        suffix = name[
            len(prefix):
        ]

        index_text = suffix.split(
            "_",
            1,
        )[0]

        if not index_text.isdigit():

            return None

        return int(index_text)

    # ========================================================
    # FIND TRACK GEOMS
    # ========================================================

    def _find_track_geoms(self):

        waypoint_geoms = {}

        related_geoms = {}

        for geom_id in range(
            self.model.ngeom
        ):

            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )

            waypoint_index = (
                self._track_geom_index(
                    name,
                    TRACK_WAYPOINT_PREFIX,
                )
            )

            if waypoint_index is not None:

                waypoint_geoms[
                    waypoint_index
                ] = geom_id

                related_geoms.setdefault(
                    waypoint_index,
                    [],
                ).append(
                    geom_id
                )

                continue

            post_index = (
                self._track_geom_index(
                    name,
                    TRACK_HEIGHT_POST_PREFIX,
                )
            )

            if post_index is not None:

                related_geoms.setdefault(
                    post_index,
                    [],
                ).append(
                    geom_id
                )

        return (
            waypoint_geoms,
            related_geoms,
        )

    # ========================================================
    # INITIALIZE TRACK
    # ========================================================

    def _init_track_progress(self):

        self.track_enabled = False

        self.track_complete = False

        self.track_next_index = 0

        self.track_start_time = None

        self.track_finish_time = None

        self.track_waypoint_positions = np.empty(
            (0, 3),
            dtype=np.float64,
        )

        self.track_point_geom_ids = {}

        (
            waypoint_geoms,
            related_geoms,
        ) = self._find_track_geoms()

        if not waypoint_geoms:

            self.get_logger().error(
                "[TRACK] No waypoint geoms found."
            )

            return

        max_index = max(
            waypoint_geoms.keys()
        )

        expected_indices = list(
            range(max_index + 1)
        )

        missing = [
            i
            for i in expected_indices
            if i not in waypoint_geoms
        ]

        if missing:

            self.get_logger().error(
                f"[TRACK] Missing waypoints: "
                f"{missing}"
            )

            return

        self.track_waypoint_geom_ids = [
            waypoint_geoms[i]
            for i in expected_indices
        ]

        self.track_point_geom_ids = {
            i: related_geoms.get(
                i,
                [waypoint_geoms[i]],
            )
            for i in expected_indices
        }

        self.track_waypoint_positions = np.array(
            [
                self.data.geom_xpos[
                    geom_id
                ].copy()
                for geom_id
                in self.track_waypoint_geom_ids
            ],
            dtype=np.float64,
        )

        self.track_enabled = True

        self.get_logger().info(
            f"[TRACK] Detected "
            f"{len(self.track_waypoint_positions)} "
            f"waypoints."
        )

        for i, p in enumerate(
            self.track_waypoint_positions
        ):

            self.get_logger().info(
                f"[TRACK] WP {i:02d}: "
                f"X={p[0]:8.3f}, "
                f"Y={p[1]:8.3f}, "
                f"Z={p[2]:8.3f}"
            )

    # ========================================================
    # TERRAIN RAY
    # ========================================================

    def _terrain_ray(
        self,
        x,
        y,
    ):

        origin = np.array(
            [
                x,
                y,
                self.commanded_base_z
                + TERRAIN_RAY_START_HEIGHT,
            ],
            dtype=np.float64,
        )

        direction = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float64,
        )

        geomgroup = np.ones(
            mujoco.mjNGROUP,
            dtype=np.uint8,
        )

        geomid = np.array(
            [-1],
            dtype=np.int32,
        )

        try:

            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                geomgroup,
                True,
                self.track_body_id,
                geomid,
            )

        except Exception:

            return None

        if distance < 0.0:

            return None

        if distance > TERRAIN_RAY_LENGTH:

            return None

        hit_geom = int(
            geomid[0]
        )

        if hit_geom < 0:

            return None

        # ----------------------------------------------------
        # Do not use waypoint markers as terrain.
        # ----------------------------------------------------

        hit_name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            hit_geom,
        )

        if hit_name:

            if (
                hit_name.startswith(
                    TRACK_WAYPOINT_PREFIX
                )
                or hit_name.startswith(
                    TRACK_HEIGHT_POST_PREFIX
                )
            ):

                return None

        terrain_z = (
            origin[2]
            - distance
        )

        return float(
            terrain_z
        )

    # ========================================================
    # TERRAIN HEIGHT DETECTION
    # ========================================================

    def _detect_terrain_height(self):

        yaw = self.commanded_base_yaw

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        base_x = self.commanded_base_x
        base_y = self.commanded_base_y

        current_samples = []
        front_samples = []

        for local_x, local_y in (
            TERRAIN_SAMPLE_POINTS
        ):

            world_x = (
                base_x
                + local_x * cos_yaw
                - local_y * sin_yaw
            )

            world_y = (
                base_y
                + local_x * sin_yaw
                + local_y * cos_yaw
            )

            terrain_z = self._terrain_ray(
                world_x,
                world_y,
            )

            if terrain_z is None:

                continue

            if local_x <= 0.15:

                current_samples.append(
                    terrain_z
                )

            else:

                front_samples.append(
                    terrain_z
                )

        if current_samples:

            current_height = float(
                np.max(
                    current_samples
                )
            )

        else:

            current_height = (
                self.terrain_height
            )

        if front_samples:

            front_height = float(
                np.max(
                    front_samples
                )
            )

        else:

            front_height = (
                current_height
            )

        return (
            current_height,
            front_height,
        )

    # ========================================================
    # UPDATE TERRAIN HEIGHT
    # ========================================================

    def _update_terrain_height(
        self,
        force=False,
    ):

        if not force:

            if (
                self.timestamp
                - self.last_terrain_update
                < 0.025
            ):

                return

        self.last_terrain_update = (
            self.timestamp
        )

        (
            current_height,
            front_height,
        ) = self._detect_terrain_height()

        # ----------------------------------------------------
        # Current terrain
        # ----------------------------------------------------

        if np.isfinite(
            current_height
        ):

            self.terrain_height = max(
                self.terrain_height * 0.70
                + current_height * 0.30,
                current_height,
            )

        # ----------------------------------------------------
        # Front terrain
        # ----------------------------------------------------

        if np.isfinite(
            front_height
        ):

            self.front_terrain_height = max(
                self.front_terrain_height * 0.65
                + front_height * 0.35,
                front_height,
            )

        # ----------------------------------------------------
        # Current terrain required height
        # ----------------------------------------------------

        current_required = (
            self.terrain_height
            + TERRAIN_CLEARANCE
            - self.robot_lowest_relative_z
        )

        # ----------------------------------------------------
        # Front terrain required height
        # ----------------------------------------------------

        front_required = (
            self.front_terrain_height
            + TERRAIN_CLEARANCE
            + STAIR_EXTRA_CLEARANCE
            - self.robot_lowest_relative_z
        )

        target_height = max(
            BASE_MIN_HEIGHT,
            current_required,
            front_required,
        )

        # ----------------------------------------------------
        # WAYPOINT HEIGHT PREDICTION
        #
        # This is especially important for the huge stair
        # before WP20.
        # ----------------------------------------------------

        if (
            self.track_enabled
            and not self.track_complete
            and self.track_next_index
            < len(
                self.track_waypoint_positions
            )
        ):

            wp = (
                self.track_waypoint_positions[
                    self.track_next_index
                ]
            )

            robot_pos = (
                self.get_robot_position()
            )

            horizontal_distance = np.hypot(
                wp[0] - robot_pos[0],
                wp[1] - robot_pos[1],
            )

            # ------------------------------------------------
            # If waypoint is ahead, use its Z.
            # ------------------------------------------------

            if horizontal_distance < 2.5:

                waypoint_required = (
                    wp[2]
                    + TERRAIN_CLEARANCE
                    + STAIR_EXTRA_CLEARANCE
                    - self.robot_lowest_relative_z
                )

                if horizontal_distance < 2.5:

                    target_height = max(
                        target_height,
                        waypoint_required,
                    )

                # ------------------------------------------------
                # Very close to waypoint.
                # ------------------------------------------------

                if horizontal_distance < 1.5:

                    target_height = max(
                        target_height,
                        waypoint_required + 0.05,
                    )

                # ------------------------------------------------
                # Almost at waypoint.
                # ------------------------------------------------

                if horizontal_distance < 0.8:

                    target_height = max(
                        target_height,
                        waypoint_required + 0.10,
                    )

        # ====================================================
        # SPECIAL LARGE STAIR PROTECTION
        # ====================================================

        # WP19 -> WP20
        if (
            self.track_enabled
            and self.track_next_index == 19
        ):

            robot_pos = (
                self.get_robot_position()
            )

            wp20 = None

            if (
                20
                < len(
                    self.track_waypoint_positions
                )
            ):

                wp20 = (
                    self.track_waypoint_positions[
                        20
                    ]
                )

            if wp20 is not None:

                distance_to_wp20 = np.hypot(
                    wp20[0] - robot_pos[0],
                    wp20[1] - robot_pos[1],
                )

                # Begin raising the robot well before
                # the large staircase.

                if distance_to_wp20 < 2.5:

                    target_height = max(
                        target_height,
                        1.05,
                    )

                if distance_to_wp20 < 1.8:

                    target_height = max(
                        target_height,
                        1.20,
                    )

                if distance_to_wp20 < 1.2:

                    target_height = max(
                        target_height,
                        1.35,
                    )

                if distance_to_wp20 < 0.7:

                    target_height = max(
                        target_height,
                        1.45,
                    )

        # ----------------------------------------------------
        # After WP20
        #
        # The user reported another large staircase after
        # WP20, so maintain additional clearance.
        # ----------------------------------------------------

        if (
            self.track_enabled
            and self.track_next_index >= 20
        ):

            target_height = max(
                target_height,
                1.05,
            )

        # ----------------------------------------------------
        # Final clamp
        # ----------------------------------------------------

        target_height = np.clip(
            target_height,
            BASE_MIN_HEIGHT,
            BASE_MAX_HEIGHT,
        )

        self.required_base_height = (
            target_height
        )

    # ========================================================
    # TRACK DISTANCE
    # ========================================================

    def _track_distance(
        self,
        robot_pos,
        waypoint,
    ):

        dx = (
            robot_pos[0]
            - waypoint[0]
        )

        dy = (
            robot_pos[1]
            - waypoint[1]
        )

        return float(
            np.hypot(
                dx,
                dy,
            )
        )

    # ========================================================
    # HIDE WAYPOINT
    # ========================================================

    def _hide_track_point(
        self,
        index,
    ):

        for geom_id in (
            self.track_point_geom_ids.get(
                index,
                [],
            )
        ):

            self.model.geom_rgba[
                geom_id,
                3,
            ] = 0.0

    # ========================================================
    # NORMALIZE ANGLE
    # ========================================================

    @staticmethod
    def normalize_angle(angle):

        return (
            angle
            + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi

    # ========================================================
    # NAVIGATION CONTROLLER
    # ========================================================

    def _navigation_controller(self):

        if not NAVIGATION_ENABLED:

            return

        if not self.track_enabled:

            return

        if self.track_complete:

            return

        # ----------------------------------------------------
        # Terrain prediction BEFORE movement.
        # ----------------------------------------------------

        self._update_terrain_height()

        # ----------------------------------------------------
        # Check track complete
        # ----------------------------------------------------

        if (
            self.track_next_index
            >= len(
                self.track_waypoint_positions
            )
        ):

            self.track_complete = True

            return

        # ----------------------------------------------------
        # Robot position
        # ----------------------------------------------------

        robot_pos = (
            self.get_robot_position()
        )

        # ----------------------------------------------------
        # Current waypoint
        # ----------------------------------------------------

        target = (
            self.track_waypoint_positions[
                self.track_next_index
            ]
        )

        dx = (
            target[0]
            - robot_pos[0]
        )

        dy = (
            target[1]
            - robot_pos[1]
        )

        distance = np.hypot(
            dx,
            dy,
        )

        # ----------------------------------------------------
        # Waypoint reached
        # ----------------------------------------------------

        if (
            distance
            <= TRACK_REACH_RADIUS
        ):

            self._waypoint_reached()

            return

        # ----------------------------------------------------
        # Current yaw
        # ----------------------------------------------------

        robot_yaw = (
            self.quaternion_to_yaw(
                self.get_robot_quaternion()
            )
        )

        # ----------------------------------------------------
        # Desired yaw
        # ----------------------------------------------------

        desired_yaw = np.arctan2(
            dy,
            dx,
        )

        heading_error = (
            self.normalize_angle(
                desired_yaw
                - robot_yaw
            )
        )

        # ----------------------------------------------------
        # Turn command
        # ----------------------------------------------------

        turn_cmd = (
            TURN_GAIN
            * heading_error
        )

        turn_cmd = np.clip(
            turn_cmd,
            -MAX_TURN_SPEED,
            MAX_TURN_SPEED,
        )

        # ----------------------------------------------------
        # Forward speed
        # ----------------------------------------------------

        abs_heading = abs(
            heading_error
        )

        if (
            abs_heading
            >= HEADING_STOP_ANGLE
        ):

            forward_cmd = 0.0

        elif (
            abs_heading
            >= HEADING_SLOWDOWN_ANGLE
        ):

            forward_cmd = (
                MAX_FORWARD_SPEED
                * 0.35
            )

        else:

            distance_factor = np.clip(
                distance / 1.0,
                0.30,
                1.0,
            )

            heading_factor = np.cos(
                heading_error
            )

            heading_factor = np.clip(
                heading_factor,
                0.0,
                1.0,
            )

            forward_cmd = (
                MAX_FORWARD_SPEED
                * distance_factor
                * heading_factor
            )

        # ----------------------------------------------------
        # Do not slow normal movement just because terrain
        # detection is active.
        # ----------------------------------------------------

        self._send_navigation_command(
            forward_cmd,
            turn_cmd,
        )

    # ========================================================
    # WAYPOINT REACHED
    # ========================================================

    def _waypoint_reached(self):

        index = (
            self.track_next_index
        )

        robot_pos = (
            self.get_robot_position()
        )

        waypoint = (
            self.track_waypoint_positions[
                index
            ]
        )

        distance = (
            self._track_distance(
                robot_pos,
                waypoint,
            )
        )

        # ----------------------------------------------------
        # Hide waypoint
        # ----------------------------------------------------

        self._hide_track_point(
            index
        )

        # ----------------------------------------------------
        # Timer starts at WP0
        # ----------------------------------------------------

        if (
            index == 0
            and self.track_start_time is None
        ):

            self.track_start_time = (
                self.timestamp
            )

            self.get_logger().info(
                "[TRACK] TIMER STARTED "
                "at WP0"
            )

        self.get_logger().info(
            f"[NAVIGATION] "
            f"Reached WP{index} "
            f"| distance={distance:.3f} m"
        )

        # ----------------------------------------------------
        # Advance waypoint
        # ----------------------------------------------------

        self.track_next_index += 1

        # ----------------------------------------------------
        # Final waypoint
        # ----------------------------------------------------

        if (
            self.track_next_index
            >= len(
                self.track_waypoint_positions
            )
        ):

            self.track_complete = True

            self.track_finish_time = (
                self.timestamp
            )

            if (
                self.track_start_time
                is not None
            ):

                elapsed = (
                    self.track_finish_time
                    - self.track_start_time
                )

            else:

                elapsed = 0.0

            self.get_logger().info(
                "======================================"
            )

            self.get_logger().info(
                "[NAVIGATION] "
                "ALL WAYPOINTS REACHED!"
            )

            self.get_logger().info(
                f"[NAVIGATION] "
                f"Total simulation time = "
                f"{elapsed:.3f} s"
            )

            self.get_logger().info(
                "======================================"
            )

            self._stop_robot()

            return

        # ----------------------------------------------------
        # Next waypoint
        # ----------------------------------------------------

        next_wp = (
            self.track_waypoint_positions[
                self.track_next_index
            ]
        )

        self.get_logger().info(
            f"[NAVIGATION] "
            f"Next target = WP"
            f"{self.track_next_index} | "
            f"X={next_wp[0]:.3f}, "
            f"Y={next_wp[1]:.3f}, "
            f"Z={next_wp[2]:.3f}"
        )

    # ========================================================
    # DIRECT BASE NAVIGATION
    # ========================================================

    def _send_navigation_command(
        self,
        forward,
        turn,
    ):

        if self.track_complete:

            return

        # ----------------------------------------------------
        # Current yaw
        # ----------------------------------------------------

        yaw = (
            self.commanded_base_yaw
        )

        # ----------------------------------------------------
        # Integrate yaw
        # ----------------------------------------------------

        new_yaw = (
            yaw
            + float(turn) * DT
        )

        new_yaw = (
            self.normalize_angle(
                new_yaw
            )
        )

        self.commanded_base_yaw = (
            new_yaw
        )

        # ----------------------------------------------------
        # Move in robot frame
        # ----------------------------------------------------

        dx = (
            float(forward)
            * np.cos(new_yaw)
            * DT
        )

        dy = (
            float(forward)
            * np.sin(new_yaw)
            * DT
        )

        self.commanded_base_x += dx

        self.commanded_base_y += dy

        # ----------------------------------------------------
        # Terrain-aware base height
        # ----------------------------------------------------

        desired_z = (
            self.required_base_height
        )

        current_z = (
            self.commanded_base_z
        )

        max_up = (
            MAX_BASE_UP_SPEED
            * DT
        )

        max_down = (
            MAX_BASE_DOWN_SPEED
            * DT
        )

        # ----------------------------------------------------
        # Raise quickly when staircase is detected.
        # ----------------------------------------------------

        if desired_z > current_z:

            new_z = min(
                desired_z,
                current_z + max_up,
            )

        else:

            new_z = max(
                desired_z,
                current_z - max_down,
            )

        # ----------------------------------------------------
        # HARD TERRAIN SAFETY
        #
        # Never allow the base to be lower than the terrain
        # clearance requirement.
        # ----------------------------------------------------

        terrain_safe_z = (
            max(
                self.terrain_height,
                self.front_terrain_height,
            )
            + TERRAIN_CLEARANCE
            - self.robot_lowest_relative_z
        )

        new_z = max(
            new_z,
            terrain_safe_z,
        )

        # ----------------------------------------------------
        # Special WP19 -> WP20 protection
        # ----------------------------------------------------

        if self.track_enabled:

            if self.track_next_index >= 19:

                new_z = max(
                    new_z,
                    1.05,
                )

        # ----------------------------------------------------
        # Clamp
        # ----------------------------------------------------

        new_z = np.clip(
            new_z,
            BASE_MIN_HEIGHT,
            BASE_MAX_HEIGHT,
        )

        self.commanded_base_z = (
            new_z
        )

        # ----------------------------------------------------
        # Write commanded base pose
        # ----------------------------------------------------

        self.data.qpos[0] = (
            self.commanded_base_x
        )

        self.data.qpos[1] = (
            self.commanded_base_y
        )

        self.data.qpos[2] = (
            self.commanded_base_z
        )

        half_yaw = (
            new_yaw / 2.0
        )

        self.data.qpos[3] = (
            np.cos(half_yaw)
        )

        self.data.qpos[4] = 0.0

        self.data.qpos[5] = 0.0

        self.data.qpos[6] = (
            np.sin(half_yaw)
        )

        self.data.qvel[
            0:6
        ] = 0.0

        # ----------------------------------------------------
        # Keep standing joint pose
        # ----------------------------------------------------

        self.pos_cmd[:, 0] = (
            JOINT_INIT[
                self.model_key
            ]
        )

        self.kp_cmd[:, 0] = 80.0

        self.kd_cmd[:, 0] = 5.0

        self.vel_cmd[:, 0] = 0.0

        self.tau_ff[:, 0] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    # ========================================================
    # LOCK BASE AFTER PHYSICS
    # ========================================================

    def _lock_base_pose(self):

        if not NAVIGATION_ENABLED:

            return

        # ----------------------------------------------------
        # Restore commanded pose
        # ----------------------------------------------------

        self.data.qpos[0] = (
            self.commanded_base_x
        )

        self.data.qpos[1] = (
            self.commanded_base_y
        )

        self.data.qpos[2] = (
            self.commanded_base_z
        )

        yaw = (
            self.commanded_base_yaw
        )

        half = yaw / 2.0

        self.data.qpos[3] = (
            np.cos(half)
        )

        self.data.qpos[4] = 0.0

        self.data.qpos[5] = 0.0

        self.data.qpos[6] = (
            np.sin(half)
        )

        # ----------------------------------------------------
        # Prevent floating base from being moved by physics.
        # ----------------------------------------------------

        self.data.qvel[
            0:6
        ] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    # ========================================================
    # STOP ROBOT
    # ========================================================

    def _stop_robot(self):

        self.vel_cmd.fill(
            0.0
        )

        self.tau_ff.fill(
            0.0
        )

    # ========================================================
    # ROS JOINT COMMAND CALLBACK
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

            return

        if len(joints) != self.dof_num:

            self.get_logger().warn(
                "Received JOINTS_CMD with "
                f"{len(joints)} joints; "
                f"expected {self.dof_num}."
            )

            return

        pub_pos = np.zeros(
            self.dof_num,
            dtype=np.float32,
        )

        pub_vel = np.zeros(
            self.dof_num,
            dtype=np.float32,
        )

        for i in range(
            self.dof_num
        ):

            joint_cmd = joints[i]

            self.kp_cmd[i, 0] = (
                joint_cmd.kp
            )

            self.kd_cmd[i, 0] = (
                joint_cmd.kd
            )

            pub_pos[i] = (
                joint_cmd.position
            )

            pub_vel[i] = (
                joint_cmd.velocity
            )

            self.tau_ff[i, 0] = (
                joint_cmd.torque
            )

        self.pos_cmd[:, 0] = (
            pub_pos
            * JOINT_DIR
            + POS_OFFSET_RAD
        )

        self.vel_cmd[:, 0] = (
            pub_vel
            * JOINT_DIR
        )

    # ========================================================
    # APPLY JOINT TORQUE
    # ========================================================

    def _apply_joint_torque(self):

        q = self.data.qpos[
            7:7 + self.dof_num
        ].reshape(
            -1,
            1,
        )

        dq = self.data.qvel[
            6:6 + self.dof_num
        ].reshape(
            -1,
            1,
        )

        self.input_tq = (
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

        self.data.ctrl[:] = (
            self.input_tq.flatten()
        )

    # ========================================================
    # PUBLISH ROBOT STATE
    # ========================================================

    def _publish_robot_state(self):

        # ----------------------------------------------------
        # IMU
        # ----------------------------------------------------

        if self.data.sensordata.shape[0] >= 10:

            q_world = (
                self.data.sensordata[:4]
            )

            rpy_rad = (
                self.quaternion_to_euler(
                    q_world
                )
            )

            body_acc = (
                self.data.sensordata[4:7]
            )

            angvel_b = (
                self.data.sensordata[7:10]
            )

        else:

            q_world = (
                self.get_robot_quaternion()
            )

            rpy_rad = (
                self.quaternion_to_euler(
                    q_world
                )
            )

            body_acc = np.zeros(
                3,
                dtype=np.float64,
            )

            angvel_b = np.zeros(
                3,
                dtype=np.float64,
            )

        rpy_deg = (
            rpy_rad
            * 180.0
            / np.pi
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        imu_msg = ImuData()

        imu_msg.header = MetaType()

        imu_msg.header.frame_id = 0

        stamp = Time()

        sec = int(
            self.timestamp
        )

        nanosec = int(
            (
                self.timestamp
                - sec
            )
            * 1e9
        )

        stamp.sec = sec

        stamp.nanosec = nanosec

        imu_msg.header.stamp = stamp

        imu_msg.data = (
            ImuDataValue()
        )

        imu_msg.data.roll = float(
            rpy_deg[0]
        )

        imu_msg.data.pitch = float(
            rpy_deg[1]
        )

        imu_msg.data.yaw = float(
            rpy_deg[2]
        )

        imu_msg.data.omega_x = float(
            angvel_b[0]
        )

        imu_msg.data.omega_y = float(
            angvel_b[1]
        )

        imu_msg.data.omega_z = float(
            angvel_b[2]
        )

        imu_msg.data.acc_x = float(
            body_acc[0]
        )

        imu_msg.data.acc_y = float(
            body_acc[1]
        )

        imu_msg.data.acc_z = float(
            body_acc[2]
        )

        self.imu_pub.publish(
            imu_msg
        )

        # ----------------------------------------------------
        # Joint state
        # ----------------------------------------------------

        q = self.data.qpos[
            7:7 + self.dof_num
        ]

        dq = self.data.qvel[
            6:6 + self.dof_num
        ]

        tau = (
            self.input_tq.flatten()
        )

        pub_pos = (
            q
            - POS_OFFSET_RAD
        ) * JOINT_DIR

        pub_vel = (
            dq
            * JOINT_DIR
        )

        pub_tau = (
            tau
            * JOINT_DIR
        )

        joints_msg = JointsData()

        joints_msg.header = MetaType()

        joints_msg.header.frame_id = 0

        joints_msg.header.stamp = stamp

        joints_msg.data = (
            JointsDataValue()
        )

        joints_msg.data.joints_data = [
            JointData()
            for _ in range(
                self.dof_num
            )
        ]

        for i in range(
            self.dof_num
        ):

            joint = (
                joints_msg
                .data
                .joints_data[i]
            )

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

            joint.torque = float(
                pub_tau[i]
            )

            joint.velocity = float(
                pub_vel[i]
            )

            joint.motion_temp = 40.0

            joint.driver_temp = 45.0

        self.joints_pub.publish(
            joints_msg
        )

    # ========================================================
    # PRINT NAVIGATION STATE
    # ========================================================

    def _print_navigation_state(self):

        if not self.track_enabled:

            return

        robot_pos = (
            self.get_robot_position()
        )

        if self.track_complete:

            self.get_logger().info(
                "[NAVIGATION] "
                "ALL WAYPOINTS COMPLETE"
            )

            return

        index = (
            self.track_next_index
        )

        if index >= len(
            self.track_waypoint_positions
        ):

            return

        target = (
            self.track_waypoint_positions[
                index
            ]
        )

        dx = (
            target[0]
            - robot_pos[0]
        )

        dy = (
            target[1]
            - robot_pos[1]
        )

        distance = np.hypot(
            dx,
            dy,
        )

        yaw = (
            self.quaternion_to_yaw(
                self.get_robot_quaternion()
            )
        )

        desired_yaw = np.arctan2(
            dy,
            dx,
        )

        heading_error = (
            self.normalize_angle(
                desired_yaw
                - yaw
            )
        )

        self.get_logger().info(
            "\n"
            "[ROBOT POSE]\n"
            f"X={robot_pos[0]:8.3f} m  "
            f"Y={robot_pos[1]:8.3f} m  "
            f"Z={robot_pos[2]:8.3f} m\n"
            f"Yaw={np.rad2deg(yaw):8.2f} deg\n"
            "\n"
            "[TERRAIN]\n"
            f"Terrain Z="
            f"{self.terrain_height:.3f} m\n"
            f"Front terrain Z="
            f"{self.front_terrain_height:.3f} m\n"
            f"Required base Z="
            f"{self.required_base_height:.3f} m\n"
            "\n"
            "[NAVIGATION]\n"
            f"Target WP={index}\n"
            f"Target X={target[0]:.3f} "
            f"Y={target[1]:.3f} "
            f"Marker Z={target[2]:.3f}\n"
            f"Base Z={robot_pos[2]:.3f} m\n"
            f"Distance={distance:.3f} m\n"
            f"Desired yaw="
            f"{np.rad2deg(desired_yaw):.2f} deg\n"
            f"Heading error="
            f"{np.rad2deg(heading_error):.2f} deg"
        )

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    def start(self):

        step = 0

        last_time = time.time()

        self.get_logger().info(
            "[INFO] Starting simulation loop."
        )

        while rclpy.ok():

            now = time.time()

            if (
                now - last_time
                < DT
            ):

                time.sleep(
                    max(
                        0.00005,
                        DT
                        - (
                            now
                            - last_time
                        ),
                    )
                )

                continue

            last_time = time.time()

            step += 1

            # ------------------------------------------------
            # ROS callbacks
            # ------------------------------------------------

            rclpy.spin_once(
                self,
                timeout_sec=0.0,
            )

            # ------------------------------------------------
            # Simulation timestamp
            # ------------------------------------------------

            self.timestamp = (
                step * DT
            )

            # ------------------------------------------------
            # Terrain prediction
            # ------------------------------------------------

            self._update_terrain_height()

            # ------------------------------------------------
            # Navigation
            # ------------------------------------------------

            self._navigation_controller()

            # ------------------------------------------------
            # Joint controller
            # ------------------------------------------------

            self._apply_joint_torque()

            # ------------------------------------------------
            # Physics
            # ------------------------------------------------

            mujoco.mj_step(
                self.model,
                self.data,
            )

            # ------------------------------------------------
            # Additional terrain update
            # ------------------------------------------------

            if step % 25 == 0:

                self._update_terrain_height(
                    force=True
                )

            # ------------------------------------------------
            # Lock commanded base
            # ------------------------------------------------

            self._lock_base_pose()

            # ------------------------------------------------
            # Publish
            # ------------------------------------------------

            if step % 5 == 0:

                self._publish_robot_state()

            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            if (
                self.timestamp
                - self.last_position_print_time
                >= POSITION_PRINT_INTERVAL
            ):

                self._print_navigation_state()

                self.last_position_print_time = (
                    self.timestamp
                )

            # ------------------------------------------------
            # Viewer
            # ------------------------------------------------

            if (
                self.viewer is not None
                and step % RENDER_INTERVAL == 0
            ):

                if TRACK_VIEWER:

                    with self.viewer.lock():

                        self.viewer.cam.lookat[:] = (
                            self.get_robot_position()
                        )

                self.viewer.sync()

    # ========================================================
    # VIEWER CONFIGURATION
    # ========================================================

    def _configure_viewer(self):

        with self.viewer.lock():

            track_body_id = (
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    TRACK_BODY_NAME,
                )
            )

            if (
                TRACK_VIEWER
                and track_body_id >= 0
            ):

                self.viewer.cam.type = (
                    mujoco.mjtCamera
                    .mjCAMERA_TRACKING
                )

                self.viewer.cam.trackbodyid = (
                    track_body_id
                )

            else:

                self.viewer.cam.type = (
                    mujoco.mjtCamera
                    .mjCAMERA_FREE
                )

                self.viewer.cam.trackbodyid = -1

                self.viewer.cam.lookat[:] = (
                    self.data.qpos[:3]
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


# ============================================================
# MAIN
# ============================================================

def main():

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    cli_args, ros_args = (
        parse_cli_args()
    )

    rclpy.init(
        args=ros_args
    )

    sim_node = None

    try:

        sim_node = (
            MuJoCoSimulationNode(
                model_key=(
                    cli_args.model_key
                ),
                xml_path=(
                    resolve_xml_path(
                        cli_args.scene,
                        cli_args.xml_path,
                    )
                ),
            )
        )

        sim_node.start()

    except KeyboardInterrupt:

        print(
            "\n[SIM] Simulation stopped."
        )

    except Exception as e:

        print(
            "\n[SIM] ERROR:"
        )

        print(e)

        raise

    finally:

        if sim_node is not None:

            if sim_node.viewer is not None:

                try:

                    sim_node.viewer.close()

                except Exception:

                    pass

            sim_node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()