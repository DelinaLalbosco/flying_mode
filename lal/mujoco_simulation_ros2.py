"""
S10 MuJoCo ROS2 simulation with fast autonomous
terrain-aware waypoint navigation.

DIRECT BASE NAVIGATION VERSION

Waypoint sequence:

    WP0 -> WP1 -> WP2 -> ... -> WP32

The robot base is controlled directly in MuJoCo.
The S10 joints continue using the normal PD controller.

This is navigation validation.
It is NOT the final real S10 RL walking controller.
"""

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

TRACK_BODY_NAME = "base_link"


# ============================================================
# SIMULATION RATE
# ============================================================

DT = 0.002                         # 500 Hz

RENDER_INTERVAL = 10              # 50 Hz
STATE_PUBLISH_INTERVAL = 10       # 50 Hz

# Faster navigation update
NAVIGATION_INTERVAL = 2           # 250 Hz

TERRAIN_UPDATE_INTERVAL = 10      # 50 Hz

PERFORMANCE_PRINT_INTERVAL = 1.0


# ============================================================
# REAL-TIME MODE
# ============================================================

REALTIME_MODE = True


# ============================================================
# VIEWER
# ============================================================

CAMERA_AZIMUTH = 90.0
CAMERA_ELEVATION = -25.0
CAMERA_DISTANCE = 12.0


# ============================================================
# BASE / TERRAIN
# ============================================================

BASE_MIN_HEIGHT = 0.45
BASE_INITIAL_HEIGHT = 0.50

TERRAIN_CLEARANCE = 0.18
STAIR_EXTRA_CLEARANCE = 0.08

BASE_MAX_HEIGHT = 5.00

MAX_BASE_DOWN_SPEED = 0.40
MAX_BASE_UP_SPEED = 1.20

TERRAIN_LOOKAHEAD = 0.80
TERRAIN_SIDE_OFFSET = 0.30

TERRAIN_RAY_START_HEIGHT = 2.0
TERRAIN_RAY_LENGTH = 4.0


# ============================================================
# TERRAIN SAMPLING
# ============================================================

TERRAIN_SAMPLE_POINTS = [
    (0.00, 0.00),
    (0.40, 0.00),
    (0.80, 0.00),
    (0.50, 0.30),
    (0.50, -0.30),
]


# ============================================================
# TRACK
# ============================================================

TRACK_START_BASE_POS = np.array(
    [0.0, -2.5, BASE_INITIAL_HEIGHT],
    dtype=np.float64,
)

TRACK_WAYPOINT_PREFIX = "track_waypoint_"
TRACK_HEIGHT_POST_PREFIX = "track_height_post_"

# Requested 0.2 m waypoint radius
TRACK_REACH_RADIUS = 0.20


# ============================================================
# FAST NAVIGATION
# ============================================================

NAVIGATION_ENABLED = True

# Increased from 0.70 m/s
MAX_FORWARD_SPEED = 1.50

# Increased turning speed
MAX_TURN_SPEED = 2.50

# Stronger turning response
TURN_GAIN = 3.0

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
    POS_OFFSET_DEG / 180.0 * np.pi
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
        description="Fast S10 MuJoCo ROS2 simulation"
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

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run simulation as fast as possible.",
    )

    return parser.parse_known_args()


# ============================================================
# RESOLVE XML PATH
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
        realtime_mode=REALTIME_MODE,
    ):

        super().__init__(
            "mujoco_simulation"
        )

        self.realtime_mode = realtime_mode

        if xml_path is None:

            xml_path = str(
                SCENE_XML_PATHS["track"]
            )

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
        # LOAD MUJOCO
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

        self.dof_num = self.model.nu

        if self.dof_num != 16:

            raise RuntimeError(
                f"Expected 16 actuators, "
                f"but found {self.dof_num}"
            )

        self.model_key = model_key

        if model_key not in JOINT_INIT:

            raise KeyError(
                f"Unknown model key: {model_key}"
            )

        self.get_logger().info(
            "[INFO] MuJoCo MJCF loaded."
        )

        self.get_logger().info(
            f"[INFO] S10 DOF = {self.dof_num}"
        )

        # ----------------------------------------------------
        # BASE BODY
        # ----------------------------------------------------

        self.track_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            TRACK_BODY_NAME,
        )

        if self.track_body_id < 0:

            raise RuntimeError(
                f"Cannot find MuJoCo body "
                f"'{TRACK_BODY_NAME}'.\n"
                f"Check the body name in S10.xml."
            )

        self.get_logger().info(
            f"[BASE] Found body "
            f"'{TRACK_BODY_NAME}' "
            f"id={self.track_body_id}"
        )

        # ----------------------------------------------------
        # INITIAL POSE
        # ----------------------------------------------------

        self._set_initial_pose(
            model_key
        )

        # ----------------------------------------------------
        # TRACK
        # ----------------------------------------------------

        self._init_track_progress()

        # ----------------------------------------------------
        # TERRAIN
        # ----------------------------------------------------

        self.terrain_height = BASE_MIN_HEIGHT

        self.front_terrain_height = (
            BASE_MIN_HEIGHT
        )

        self.required_base_height = (
            BASE_INITIAL_HEIGHT
        )

        self.last_terrain_update = -999.0

        # ----------------------------------------------------
        # ROBOT GEOMETRY
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
        # JOINT CONTROLLER
        # ----------------------------------------------------

        shape = (
            self.dof_num,
            1,
        )

        self.kp_cmd = np.full(
            shape,
            80.0,
            dtype=np.float32,
        )

        self.kd_cmd = np.full(
            shape,
            5.0,
            dtype=np.float32,
        )

        self.pos_cmd = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.vel_cmd = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.tau_ff = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.input_tq = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.pos_cmd[:, 0] = (
            JOINT_INIT[model_key]
        )

        self.standing_pose = (
            JOINT_INIT[model_key].copy()
        )

        # ----------------------------------------------------
        # NAVIGATION STATE
        # ----------------------------------------------------

        self.timestamp = 0.0

        self.last_position_print_time = -1.0

        self.commanded_base_yaw = np.deg2rad(
            90.0
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
        # PERFORMANCE
        # ----------------------------------------------------

        self.performance_last_wall = (
            time.perf_counter()
        )

        self.performance_last_sim = 0.0

        self.sim_fps = 0.0

        self.real_time_factor = 0.0

        # ----------------------------------------------------
        # ROS PUBLISHERS
        # ----------------------------------------------------

        self.imu_pub = self.create_publisher(
            ImuData,
            "/IMU_DATA",
            50,
        )

        self.joints_pub = self.create_publisher(
            JointsData,
            "/JOINTS_DATA",
            50,
        )

        # ----------------------------------------------------
        # ROS SUBSCRIBER
        # ----------------------------------------------------

        self.cmd_sub = self.create_subscription(
            JointsDataCmd,
            "/JOINTS_CMD",
            self._cmd_callback,
            20,
        )

        # ----------------------------------------------------
        # VIEWER
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
        # LOGGING
        # ----------------------------------------------------

        self.get_logger().info(
            "[INFO] Fast simulation initialized."
        )

        self.get_logger().info(
            f"[PERFORMANCE] Physics = "
            f"{1.0 / DT:.0f} Hz"
        )

        self.get_logger().info(
            f"[PERFORMANCE] Navigation = "
            f"{1.0 / (DT * NAVIGATION_INTERVAL):.0f} Hz"
        )

        self.get_logger().info(
            f"[PERFORMANCE] Terrain = "
            f"{1.0 / (DT * TERRAIN_UPDATE_INTERVAL):.0f} Hz"
        )

        self.get_logger().info(
            f"[PERFORMANCE] State publishing = "
            f"{1.0 / (DT * STATE_PUBLISH_INTERVAL):.0f} Hz"
        )

        self.get_logger().info(
            f"[PERFORMANCE] Real-time mode = "
            f"{self.realtime_mode}"
        )

        self.get_logger().info(
            f"[NAVIGATION] Max forward speed = "
            f"{MAX_FORWARD_SPEED:.2f} m/s"
        )

        self.get_logger().info(
            f"[NAVIGATION] Max turn speed = "
            f"{MAX_TURN_SPEED:.2f} rad/s"
        )

        self.get_logger().info(
            f"[NAVIGATION] Waypoint radius = "
            f"{TRACK_REACH_RADIUS:.2f} m"
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

        qpos[:3] = TRACK_START_BASE_POS

        yaw = np.deg2rad(90.0)

        half = yaw * 0.5

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

        self.get_logger().info(
            f"[BASE] Initial position: "
            f"X={qpos[0]:.3f}, "
            f"Y={qpos[1]:.3f}, "
            f"Z={qpos[2]:.3f}"
        )

        self.get_logger().info(
            f"[BASE] Initial yaw = "
            f"{np.rad2deg(yaw):.1f} deg"
        )


    # ========================================================
    # ROBOT POSITION
    # ========================================================

    def get_robot_position(self):

        return self.data.xpos[
            self.track_body_id
        ]


    # ========================================================
    # ROBOT QUATERNION
    # ========================================================

    def get_robot_quaternion(self):

        return self.data.xquat[
            self.track_body_id
        ]


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

        t0 = 2.0 * (
            w * x
            + y * z
        )

        t1 = (
            1.0
            - 2.0 * (
                x * x
                + y * y
            )
        )

        roll = np.arctan2(
            t0,
            t1,
        )

        t2 = 2.0 * (
            w * y
            - z * x
        )

        t2 = np.clip(
            t2,
            -1.0,
            1.0,
        )

        pitch = np.arcsin(t2)

        t3 = 2.0 * (
            w * z
            + x * y
        )

        t4 = (
            1.0
            - 2.0 * (
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
    # LOWEST ROBOT GEOMETRY
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

            while current >= 0:

                if current == self.track_body_id:

                    robot_geom_ids.append(
                        geom_id
                    )

                    break

                parent = (
                    self.model.body_parentid[
                        current
                    ]
                )

                if parent == current:

                    break

                current = parent

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

        if not name.startswith(prefix):
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

        waypoint_geoms, related_geoms = (
            self._find_track_geoms()
        )

        if not waypoint_geoms:

            self.get_logger().error(
                "[TRACK] No waypoint geoms found."
            )

            return

        max_index = max(
            waypoint_geoms.keys()
        )

        expected_indices = range(
            max_index + 1
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
            for i in range(
                max_index + 1
            )
        ]

        self.track_point_geom_ids = {
            i: related_geoms.get(
                i,
                [waypoint_geoms[i]],
            )
            for i in range(
                max_index + 1
            )
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
    # TRACK HEIGHT
    # ========================================================

    def _estimate_track_height_at_xy(
        self,
        x,
        y,
    ):

        if (
            not self.track_enabled
            or len(
                self.track_waypoint_positions
            ) == 0
        ):

            return 0.0

        points = self.track_waypoint_positions

        query_x = float(x)
        query_y = float(y)

        if len(points) == 1:

            return float(
                points[0, 2]
            )

        best_dist2 = np.inf

        best_z = float(
            points[0, 2]
        )

        for i in range(
            len(points) - 1
        ):

            ax = points[i, 0]
            ay = points[i, 1]

            bx = points[i + 1, 0]
            by = points[i + 1, 1]

            abx = bx - ax
            aby = by - ay

            denom = (
                abx * abx
                + aby * aby
            )

            if denom < 1e-12:

                t = 0.0

            else:

                t = (
                    (query_x - ax) * abx
                    + (query_y - ay) * aby
                ) / denom

                t = np.clip(
                    t,
                    0.0,
                    1.0,
                )

            closest_x = ax + t * abx
            closest_y = ay + t * aby

            dx = query_x - closest_x
            dy = query_y - closest_y

            dist2 = (
                dx * dx
                + dy * dy
            )

            if dist2 < best_dist2:

                best_dist2 = dist2

                best_z = (
                    points[i, 2]
                    + t * (
                        points[i + 1, 2]
                        - points[i, 2]
                    )
                )

        return float(best_z)


    # ========================================================
    # TERRAIN RAY
    # ========================================================

    def _terrain_ray(
        self,
        x,
        y,
    ):

        origin_z = (
            self.commanded_base_z
            + TERRAIN_RAY_START_HEIGHT
        )

        origin = np.array(
            [
                x,
                y,
                origin_z,
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
                1,
                self.track_body_id,
                geomid,
            )

        except Exception:

            return None

        if distance < 0.0:
            return None

        if distance > TERRAIN_RAY_LENGTH:
            return None

        return float(
            origin_z - distance
        )


    # ========================================================
    # TERRAIN DETECTION
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
                max(current_samples)
            )

        else:

            current_height = (
                self.terrain_height
            )

        if front_samples:

            front_height = float(
                max(front_samples)
            )

        else:

            front_height = current_height

        track_current_z = (
            self._estimate_track_height_at_xy(
                base_x,
                base_y,
            )
        )

        front_x = (
            base_x
            + TERRAIN_LOOKAHEAD * cos_yaw
        )

        front_y = (
            base_y
            + TERRAIN_LOOKAHEAD * sin_yaw
        )

        track_front_z = (
            self._estimate_track_height_at_xy(
                front_x,
                front_y,
            )
        )

        current_height = max(
            current_height,
            track_current_z,
        )

        front_height = max(
            front_height,
            track_front_z,
        )

        return (
            current_height,
            front_height,
        )


    # ========================================================
    # UPDATE TERRAIN
    # ========================================================

    def _update_terrain_height(
        self,
        force=False,
    ):

        if not force:

            if (
                self.timestamp
                - self.last_terrain_update
                < DT * TERRAIN_UPDATE_INTERVAL
            ):

                return

        self.last_terrain_update = (
            self.timestamp
        )

        (
            current_height,
            front_height,
        ) = self._detect_terrain_height()

        if np.isfinite(current_height):

            self.terrain_height = (
                0.75 * self.terrain_height
                + 0.25 * current_height
            )

        if np.isfinite(front_height):

            self.front_terrain_height = (
                0.70 * self.front_terrain_height
                + 0.30 * front_height
            )

        current_required = (
            self.terrain_height
            + TERRAIN_CLEARANCE
            - self.robot_lowest_relative_z
        )

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

        self.required_base_height = float(
            np.clip(
                target_height,
                BASE_MIN_HEIGHT,
                BASE_MAX_HEIGHT,
            )
        )


    # ========================================================
    # TRACK DISTANCE
    # ========================================================

    @staticmethod
    def _track_distance(
        robot_pos,
        waypoint,
    ):

        return float(
            np.hypot(
                robot_pos[0] - waypoint[0],
                robot_pos[1] - waypoint[1],
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
            angle + np.pi
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

        if (
            self.timestamp
            - self.last_terrain_update
            >= DT * TERRAIN_UPDATE_INTERVAL
        ):

            self._update_terrain_height(
                force=True
            )

        if (
            self.track_next_index
            >= len(
                self.track_waypoint_positions
            )
        ):

            self.track_complete = True

            return

        robot_pos = (
            self.get_robot_position()
        )

        target = (
            self.track_waypoint_positions[
                self.track_next_index
            ]
        )

        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]

        distance = np.hypot(
            dx,
            dy,
        )

        if distance <= TRACK_REACH_RADIUS:

            self._waypoint_reached()

            return

        robot_yaw = (
            self.commanded_base_yaw
        )

        desired_yaw = np.arctan2(
            dy,
            dx,
        )

        heading_error = (
            self.normalize_angle(
                desired_yaw - robot_yaw
            )
        )

        turn_cmd = (
            TURN_GAIN * heading_error
        )

        turn_cmd = np.clip(
            turn_cmd,
            -MAX_TURN_SPEED,
            MAX_TURN_SPEED,
        )

        abs_heading = abs(
            heading_error
        )

        # ----------------------------------------------------
        # FORWARD SPEED
        # ----------------------------------------------------

        if (
            abs_heading
            >= HEADING_STOP_ANGLE
        ):

            forward_cmd = 0.0

        elif (
            abs_heading
            >= HEADING_SLOWDOWN_ANGLE
        ):

            # Still move forward while turning
            forward_cmd = (
                MAX_FORWARD_SPEED * 0.35
            )

        else:

            # Keep high speed for most of the waypoint
            distance_factor = np.clip(
                distance / 0.8,
                0.50,
                1.0,
            )

            heading_factor = np.cos(
                heading_error
            )

            heading_factor = np.clip(
                heading_factor,
                0.25,
                1.0,
            )

            forward_cmd = (
                MAX_FORWARD_SPEED
                * distance_factor
                * heading_factor
            )

        self._send_navigation_command(
            forward_cmd,
            turn_cmd,
        )


    # ========================================================
    # WAYPOINT REACHED
    # ========================================================

    def _waypoint_reached(self):

        index = self.track_next_index

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

        self._hide_track_point(
            index
        )

        if (
            index == 0
            and self.track_start_time is None
        ):

            self.track_start_time = (
                self.timestamp
            )

        self.get_logger().info(
            f"[NAVIGATION] Reached WP{index} "
            f"| distance={distance:.3f} m"
        )

        self.track_next_index += 1

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
                "[NAVIGATION] "
                "ALL WAYPOINTS REACHED!"
            )

            self.get_logger().info(
                f"[NAVIGATION] "
                f"Total time = "
                f"{elapsed:.3f} s"
            )

            self._stop_robot()

            return

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

        yaw = self.commanded_base_yaw

        # ----------------------------------------------------
        # YAW
        # ----------------------------------------------------

        new_yaw = (
            yaw
            + float(turn) * DT * NAVIGATION_INTERVAL
        )

        new_yaw = self.normalize_angle(
            new_yaw
        )

        self.commanded_base_yaw = new_yaw

        # ----------------------------------------------------
        # XY
        # ----------------------------------------------------

        cos_yaw = np.cos(new_yaw)
        sin_yaw = np.sin(new_yaw)

        movement_dt = (
            DT * NAVIGATION_INTERVAL
        )

        self.commanded_base_x += (
            float(forward)
            * cos_yaw
            * movement_dt
        )

        self.commanded_base_y += (
            float(forward)
            * sin_yaw
            * movement_dt
        )

        # ----------------------------------------------------
        # Z
        # ----------------------------------------------------

        desired_z = (
            self.required_base_height
        )

        current_z = (
            self.commanded_base_z
        )

        max_up = (
            MAX_BASE_UP_SPEED
            * movement_dt
        )

        max_down = (
            MAX_BASE_DOWN_SPEED
            * movement_dt
        )

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

        self.commanded_base_z = float(
            np.clip(
                new_z,
                BASE_MIN_HEIGHT,
                BASE_MAX_HEIGHT,
            )
        )

        # ----------------------------------------------------
        # WRITE BASE QPOS
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

        half_yaw = new_yaw * 0.5

        self.data.qpos[3] = (
            np.cos(half_yaw)
        )

        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0

        self.data.qpos[6] = (
            np.sin(half_yaw)
        )

        self.data.qvel[0:6] = 0.0

        # ----------------------------------------------------
        # STANDING JOINT POSE
        # ----------------------------------------------------

        self.pos_cmd[:, 0] = (
            self.standing_pose
        )

        self.vel_cmd[:, 0] = 0.0
        self.tau_ff[:, 0] = 0.0


    # ========================================================
    # LOCK BASE
    # ========================================================

    def _lock_base_pose(self):

        if not NAVIGATION_ENABLED:
            return

        self.data.qpos[0] = (
            self.commanded_base_x
        )

        self.data.qpos[1] = (
            self.commanded_base_y
        )

        self.data.qpos[2] = (
            self.commanded_base_z
        )

        yaw = self.commanded_base_yaw

        half = yaw * 0.5

        self.data.qpos[3] = np.cos(
            half
        )

        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0

        self.data.qpos[6] = np.sin(
            half
        )

        self.data.qvel[0:6] = 0.0


    # ========================================================
    # STOP
    # ========================================================

    def _stop_robot(self):

        self.vel_cmd.fill(0.0)
        self.tau_ff.fill(0.0)


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
                self.pos_cmd - q
            )
            +
            self.kd_cmd
            * (
                self.vel_cmd - dq
            )
            +
            self.tau_ff
        )

        self.data.ctrl[:] = (
            self.input_tq.ravel()
        )


    # ========================================================
    # PUBLISH ROBOT STATE
    # ========================================================

    def _publish_robot_state(self):

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
        # HEADER
        # ----------------------------------------------------

        stamp = Time()

        sec = int(
            self.timestamp
        )

        nanosec = int(
            (
                self.timestamp - sec
            ) * 1e9
        )

        stamp.sec = sec
        stamp.nanosec = nanosec

        # ----------------------------------------------------
        # IMU
        # ----------------------------------------------------

        imu_msg = ImuData()

        imu_msg.header = MetaType()

        imu_msg.header.frame_id = 0
        imu_msg.header.stamp = stamp

        imu_msg.data = ImuDataValue()

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
        # JOINT STATE
        # ----------------------------------------------------

        q = self.data.qpos[
            7:7 + self.dof_num
        ]

        dq = self.data.qvel[
            6:6 + self.dof_num
        ]

        tau = (
            self.input_tq.ravel()
        )

        pub_pos = (
            q - POS_OFFSET_RAD
        ) * JOINT_DIR

        pub_vel = (
            dq * JOINT_DIR
        )

        pub_tau = (
            tau * JOINT_DIR
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
    # NAVIGATION STATE
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

        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]

        distance = np.hypot(
            dx,
            dy,
        )

        yaw = (
            self.commanded_base_yaw
        )

        desired_yaw = np.arctan2(
            dy,
            dx,
        )

        heading_error = (
            self.normalize_angle(
                desired_yaw - yaw
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
            f"Z={target[2]:.3f}\n"
            f"Base Z={robot_pos[2]:.3f} m\n"
            f"Distance={distance:.3f} m\n"
            f"Heading error="
            f"{np.rad2deg(heading_error):.2f} deg\n"
            f"Max speed="
            f"{MAX_FORWARD_SPEED:.2f} m/s\n"
            "\n"
            "[PERFORMANCE]\n"
            f"Simulation FPS="
            f"{self.sim_fps:.1f}\n"
            f"Real-time factor="
            f"{self.real_time_factor:.2f}x"
        )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    def _update_performance(self):

        wall_now = time.perf_counter()

        wall_dt = (
            wall_now
            - self.performance_last_wall
        )

        sim_dt = (
            self.timestamp
            - self.performance_last_sim
        )

        if wall_dt <= 0.0:
            return

        self.sim_fps = (
            sim_dt / wall_dt
        )

        self.real_time_factor = (
            sim_dt / wall_dt
        )

        self.performance_last_wall = (
            wall_now
        )

        self.performance_last_sim = (
            self.timestamp
        )


    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    def start(self):

        step = 0

        wall_start = (
            time.perf_counter()
        )

        self.get_logger().info(
            "[INFO] Starting fast simulation loop."
        )

        self.get_logger().info(
            "[INFO] Press Ctrl+C to stop."
        )

        while rclpy.ok():

            step += 1

            # ------------------------------------------------
            # SIMULATION TIME
            # ------------------------------------------------

            self.timestamp = (
                step * DT
            )

            # ------------------------------------------------
            # ROS CALLBACKS
            # ------------------------------------------------

            rclpy.spin_once(
                self,
                timeout_sec=0.0,
            )

            # ------------------------------------------------
            # NAVIGATION
            # ------------------------------------------------

            if (
                step
                % NAVIGATION_INTERVAL
                == 0
            ):

                self._navigation_controller()

            # ------------------------------------------------
            # JOINT TORQUE
            # ------------------------------------------------

            self._apply_joint_torque()

            # ------------------------------------------------
            # MUJOCO PHYSICS
            # ------------------------------------------------

            mujoco.mj_step(
                self.model,
                self.data,
            )

            # ------------------------------------------------
            # LOCK BASE
            # ------------------------------------------------

            self._lock_base_pose()

            # ------------------------------------------------
            # TERRAIN
            # ------------------------------------------------

            if (
                step
                % TERRAIN_UPDATE_INTERVAL
                == 0
            ):

                self._update_terrain_height(
                    force=True
                )

            # ------------------------------------------------
            # ROS STATE
            # ------------------------------------------------

            if (
                step
                % STATE_PUBLISH_INTERVAL
                == 0
            ):

                self._publish_robot_state()

            # ------------------------------------------------
            # PERFORMANCE
            # ------------------------------------------------

            if (
                self.timestamp
                - self.performance_last_sim
                >= PERFORMANCE_PRINT_INTERVAL
            ):

                self._update_performance()

            # ------------------------------------------------
            # NAVIGATION LOG
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
            # VIEWER
            # ------------------------------------------------

            if (
                self.viewer is not None
                and step
                % RENDER_INTERVAL
                == 0
            ):

                if TRACK_VIEWER:

                    with self.viewer.lock():

                        self.viewer.cam.lookat[0] = (
                            self.commanded_base_x
                        )

                        self.viewer.cam.lookat[1] = (
                            self.commanded_base_y
                        )

                        self.viewer.cam.lookat[2] = (
                            self.commanded_base_z
                        )

                self.viewer.sync()

            # ------------------------------------------------
            # REAL-TIME LIMITER
            # ------------------------------------------------

            if self.realtime_mode:

                target_wall_time = (
                    wall_start
                    + self.timestamp
                )

                sleep_time = (
                    target_wall_time
                    - time.perf_counter()
                )

                if sleep_time > 0:

                    time.sleep(
                        min(
                            sleep_time,
                            0.001,
                        )
                    )


    # ========================================================
    # VIEWER CONFIGURATION
    # ========================================================

    def _configure_viewer(self):

        if self.viewer is None:
            return

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

        realtime_mode = (
            False
            if cli_args.fast
            else REALTIME_MODE
        )

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
                realtime_mode=realtime_mode,
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