

"""Python-only MuJoCo + ROS2 + ONNX RL simulation for the Deep Robotics S10.

Important fixes in this version:
1. The initial pose is set and mj_forward() is called BEFORE waypoint positions
   are read. Previously data.geom_xpos was read before forward kinematics and
   all waypoints could therefore become [0, 0, 0].
2. Waypoint positions are copied once at startup and are never allowed to move
   with the robot.
3. Only one waypoint can be advanced per navigation update.
4. RL command is expressed in the robot/body frame.
5. MuJoCo quaternion-to-matrix conversion uses float64 as required by MuJoCo.
6. pos_cmd/vel_cmd/tau_ff are initialized before _set_initial_pose().
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
# LiDAR configuration
# ============================================================

LIDAR_SITE_NAME = "lidar_site"

LIDAR_NUM_RAYS = 360
LIDAR_MIN_RANGE = 0.05
LIDAR_MAX_RANGE = 10.0

# Horizontal LiDAR
LIDAR_FOV = 2.0 * math.pi

# Ray height direction
LIDAR_HORIZONTAL = True

# Store latest LiDAR scan
LIDAR_DEBUG_PRINT_PERIOD = 1.0

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "S10"
CURRENT_DIR = Path(__file__).resolve().parent
MJCF_DIR = (CURRENT_DIR / ".." / ".." / ".." / "S10_description" /
            "s10_mjcf" / "mjcf").resolve()
SCENE_XML_PATHS = {"track": MJCF_DIR / "S10_track.xml"}
DEFAULT_SCENE_NAME = os.environ.get("S10_MUJOCO_SCENE", "track")

USE_VIEWER = True
DT = 0.001
RENDER_INTERVAL = 10
TRACK_BODY_NAME = "base_link"
TRACK_START_BASE_POS = np.array([0.0, -2.5, 0.2], dtype=np.float64)
TRACK_WAYPOINT_PREFIX = "track_waypoint_"
WAYPOINT_REACH_RADIUS = 0.55
WAYPOINT_PASS_MARGIN = 0.25
WAYPOINT_LOG_EVERY = 0.5

TRACK_VIEWER = False
CAMERA_AZIMUTH = 90
CAMERA_ELEVATION = -25
CAMERA_DISTANCE = 18.0

# ============================================================
# Joint calibration
# ============================================================

JOINT_DIR = np.array([
    1, 1, -1, 1,
    1, -1, 1, -1,
    -1, 1, -1, 1,
    -1, -1, 1, -1,
], dtype=np.float32)

POS_OFFSET_DEG = np.array([
    -35, -145, 156, 0,
     35, -145, 156, 0,
    -35,  145,-156, 0,
     35,  145,-156, 0,
], dtype=np.float32)
POS_OFFSET_RAD = (POS_OFFSET_DEG * np.pi / 180.0).astype(np.float32)

JOINT_INIT = {
    "S10": np.array([
        -0.438, -1.16,  2.76, 0.0,
         0.438, -1.16,  2.76, 0.0,
        -0.438,  1.16, -2.76, 0.0,
         0.438,  1.16, -2.76, 0.0,
    ], dtype=np.float32)
}

# ============================================================
# RL policy configuration
# ============================================================

MOTOR_NUM = 16
OBSERVATION_DIM = 57
ACTION_DIM = 16
POLICY_DECIMATION = 4
OMEGA_SCALE = 0.25
DOF_VEL_SCALE = 0.05
GRAVITY_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=np.float32)

DOF_DEFAULT_POLICY = np.array([
    0.0, -0.3,  0.6,
    0.0, -0.3,  0.6,
    0.0,  0.3, -0.6,
    0.0,  0.3, -0.6,
    0.0,  0.0,  0.0,  0.0,
], dtype=np.float32)

DOF_DEFAULT_ROBOT = np.array([
    0.0, -0.3,  0.6, 0.0,
    0.0, -0.3,  0.6, 0.0,
    0.0,  0.3, -0.6, 0.0,
    0.0,  0.3, -0.6, 0.0,
], dtype=np.float32)

ACTION_SCALE_ROBOT = np.array([
    0.125, 0.25, 0.25, 5.0,
    0.125, 0.25, 0.25, 5.0,
    0.125, 0.25, 0.25, 5.0,
    0.125, 0.25, 0.25, 5.0,
], dtype=np.float32)

KP = np.array([
    80, 80, 80, 0,
    80, 80, 80, 0,
    80, 80, 80, 0,
    80, 80, 80, 0,
], dtype=np.float32)
KD = np.array([
    2, 2, 2, 0.6,
    2, 2, 2, 0.6,
    2, 2, 2, 0.6,
    2, 2, 2, 0.6,
], dtype=np.float32)

ROBOT_ORDER = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
]
POLICY_ORDER = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]


def generate_permutation(source, target):
    m = {name: i for i, name in enumerate(source)}
    return [m[name] for name in target]

ROBOT_TO_POLICY = np.array(generate_permutation(ROBOT_ORDER, POLICY_ORDER), dtype=np.int32)
POLICY_TO_ROBOT = np.array(generate_permutation(POLICY_ORDER, ROBOT_ORDER), dtype=np.int32)

# ============================================================
# Navigation
# ============================================================

MAX_FORWARD_SPEED = 0.30
MAX_SIDE_SPEED = 0.18
MAX_YAW_SPEED = 0.70
TURN_GAIN = 1.8
HEADING_SLOWDOWN_ANGLE = math.radians(45.0)
HEADING_STOP_ANGLE = math.radians(90.0)
NAV_LOG_PERIOD = 1.0

# Exact fallback copy of track_overlay.xml: START + WP01..WP31 + END.
DEFAULT_WAYPOINTS = [
    (0.0000, -1.7250, 0.000),
    (-0.7125, 11.6550, 0.475),
    (-8.7300, 11.8425, 0.475),
    (-10.5375, 16.3275, 0.475),
    (-15.0225, 17.3700, 0.475),
    (-15.6000, 23.2800, 0.475),
    (-15.1200, 31.8600, 0.600),
    (-14.9325, 41.2050, 1.165),
    (-20.4600, 43.1100, 1.165),
    (-20.6550, 47.7825, 1.165),
    (-4.0500, 47.9700, 1.165),
    (-4.2450, 42.5400, 1.165),
    (-13.5000, 41.5500, 1.165),
    (-12.9225, 34.6275, 0.475),
    (2.7225, 34.5300, 0.475),
    (11.1225, 33.0075, 0.100),
    (16.2750, 31.2900, 0.475),
    (17.6100, 29.1900, 0.475),
    (25.7175, 29.9550, 2.200),
    (34.5975, 29.9550, 2.360),
    (33.9225, 24.4275, 2.360),
    (17.8950, 24.6150, 0.475),
    (19.0425, 16.8000, 0.475),
    (26.4825, 17.1825, 1.670),
    (31.6350, 15.4650, 1.670),
    (33.1650, 15.1800, 1.670),
    (33.6000, 21.3750, 2.700),
    (34.8825, 21.3750, 2.700),
    (35.0700, 15.6600, 3.750),
    (30.9350, 14.7150, 3.750),
    (29.5350, 16.3275, 3.750),
    (29.9175, 20.6175, 3.750),
    (32.9250, 18.4500, 3.750),
]


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value, lo, hi):
    return max(lo, min(value, hi))


def quaternion_to_euler(q):
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q
    roll = math.atan2(2.0 * (w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    t2 = np.clip(2.0 * (w*y - z*x), -1.0, 1.0)
    pitch = math.asin(t2)
    yaw = math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def find_policy_path():
    paths = [
        CURRENT_DIR / ".." / ".." / ".." / "policy" / "policy.onnx",
        CURRENT_DIR / ".." / ".." / ".." / ".." / "policy" / "policy.onnx",
    ]
    for p in paths:
        p = p.resolve()
        if p.exists():
            return str(p)
    return None


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(SCENE_XML_PATHS), default=DEFAULT_SCENE_NAME)
    parser.add_argument("--xml-path", default=os.environ.get("S10_MUJOCO_XML"))
    parser.add_argument("--model-key", default=MODEL_NAME)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_known_args()


def resolve_xml_path(scene, xml_path):
    if xml_path:
        return str(Path(xml_path).expanduser().resolve())
    return str(SCENE_XML_PATHS[scene].resolve())


class MuJoCoSimulationNode(Node):

    def __init__(self, model_key=MODEL_NAME, xml_path=None, policy_path=None, use_viewer=True):
        super().__init__("mujoco_simulation")

        if xml_path is None:
            xml_path = str(SCENE_XML_PATHS[DEFAULT_SCENE_NAME].resolve())
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        if self.model.nu != 16:
            raise RuntimeError(f"S10 requires 16 actuators, MJCF has {self.model.nu}")
        self.dof_num = self.model.nu

        # MUST be initialized before _set_initial_pose().
        self.kp_cmd = KP.copy()
        self.kd_cmd = KD.copy()
        self.pos_cmd = np.zeros(16, dtype=np.float32)
        self.vel_cmd = np.zeros(16, dtype=np.float32)
        self.tau_ff = np.zeros(16, dtype=np.float32)
        self.input_tq = np.zeros(16, dtype=np.float32)

        self.last_action = np.zeros(16, dtype=np.float32)
        self.current_action = np.zeros(16, dtype=np.float32)
        self.simulation_time = 0.0
        self.standing_time = 3.0
        self.rl_enabled = False

        # Navigation state.
        self.navigation_enabled = True
        self.current_waypoint = 0
        self.track_complete = False
        self.track_waypoint_positions = np.empty((0, 3), dtype=np.float64)
        self.track_body_id = -1
        self.track_waypoint_geom_ids = []
        self.waypoint_collected = []
        self.last_nav_command = np.zeros(3, dtype=np.float32)
        self.last_robot_xy = None
        self.last_nav_log_time = -1.0

        # FIX: pose first, then forward kinematics, then read waypoint positions.
        self._set_initial_pose(model_key)
        self._init_track_progress()
        
        self.lidar_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            LIDAR_SITE_NAME
        )

        if self.lidar_site_id < 0:
            raise RuntimeError(
                f"Cannot find LiDAR site '{LIDAR_SITE_NAME}'"
            )

        self.lidar_ranges = np.full(
            LIDAR_NUM_RAYS,
            LIDAR_MAX_RANGE,
            dtype=np.float32
        )

        self.last_lidar_log_time = -1.0

        self.get_logger().info(
            f"[LiDAR] Enabled: {LIDAR_NUM_RAYS} rays, "
            f"range={LIDAR_MIN_RANGE:.2f}-{LIDAR_MAX_RANGE:.2f} m"
        )

        if policy_path is None:
            policy_path = find_policy_path()
        self.policy_path = policy_path
        self.ort_session = None
        if policy_path:
            self._load_policy(policy_path)
        else:
            self.get_logger().warn("[RL] policy.onnx not found; standing mode only")

        self.imu_pub = self.create_publisher(ImuData, "/IMU_DATA", 200)
        self.joints_pub = self.create_publisher(JointsData, "/JOINTS_DATA", 200)
        self.nav_pub = self.create_publisher(Twist, "/NAV_CMD", 20)
        self.cmd_sub = self.create_subscription(JointsDataCmd, "/JOINTS_CMD", self._cmd_callback, 50)

        self.viewer = None
        if use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._configure_viewer()

        self.get_logger().info("==================================================")
        self.get_logger().info("[MuJoCo] S10 simulation started")
        self.get_logger().info(f"[MuJoCo] MJCF: {xml_path}")
        self.get_logger().info(f"[MuJoCo] DOF: {self.dof_num}")
        self.get_logger().info("[MuJoCo] Python-only RL control enabled")
        self.get_logger().info("[MuJoCo] Navigation command: BODY FRAME")
        self.get_logger().info("[MuJoCo] /NAV_CMD enabled")
        self.get_logger().info("==================================================")

    # --------------------------------------------------------
    # Initial pose
    # --------------------------------------------------------

    def _set_initial_pose(self, key):
        if key not in JOINT_INIT:
            raise ValueError(f"Unknown robot model: {key}")
        q0 = JOINT_INIT[key].astype(np.float64)
        self.data.qpos[:3] = TRACK_START_BASE_POS
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[7:23] = q0
        self.data.qvel[:] = 0.0
        self.pos_cmd[:] = q0
        self.vel_cmd[:] = 0.0
        self.tau_ff[:] = 0.0
        self.input_tq[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info("[MuJoCo] Initial standing pose configured")
        self.get_logger().info("[MuJoCo] Initial joints: " + np.array2string(q0, precision=3, suppress_small=True))

    # --------------------------------------------------------
    # Waypoints
    # --------------------------------------------------------

    @staticmethod
    def _track_geom_index(name):
        if not name or not name.startswith(TRACK_WAYPOINT_PREFIX):
            return None
        suffix = name[len(TRACK_WAYPOINT_PREFIX):]
        number = suffix.split("_", 1)[0]
        return int(number) if number.isdigit() else None

    def _init_track_progress(self):
        self.track_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY_NAME)
        if self.track_body_id < 0:
            raise RuntimeError(f"Cannot find body '{TRACK_BODY_NAME}'")

        # Ensure all xpos values are valid before reading them.
        mujoco.mj_forward(self.model, self.data)

        found = {}
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            idx = self._track_geom_index(name)
            if idx is not None:
                found[idx] = gid

        if not found:
            self.track_waypoint_positions = np.array(DEFAULT_WAYPOINTS, dtype=np.float64)
            self.get_logger().warn("[TRACK] No waypoint geoms found; using default waypoints")
        else:
            indices = sorted(found)
            self.track_waypoint_geom_ids = [found[i] for i in indices]
            # CRITICAL: copy the world positions at initialization.
            self.track_waypoint_positions = np.array(
                [self.data.geom_xpos[found[i]].copy() for i in indices], dtype=np.float64)
            self.get_logger().info(f"[TRACK] Enabled: {len(indices)} waypoints")

            for i, p in enumerate(self.track_waypoint_positions):
                self.get_logger().info(
                    f"[TRACK] WP {i:02d}: x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}")

        self.track_complete = len(self.track_waypoint_positions) == 0
        self.waypoint_collected = [False] * len(self.track_waypoint_positions)
        self.last_robot_xy = None

        if self.track_waypoint_positions.size:
            self.get_logger().info(
                f"[TRACK] Route loaded: {len(self.track_waypoint_positions)} waypoints "
                f"(START + 31 intermediate + END)"
            )

    def _hide_waypoint(self, index):
        if index < len(self.track_waypoint_geom_ids):
            gid = self.track_waypoint_geom_ids[index]
            try:
                self.model.geom_rgba[gid, 3] = 0.0
            except Exception:
                pass

    def _collect_current_waypoint(self, pos, distance):
        idx = self.current_waypoint
        if idx >= len(self.track_waypoint_positions):
            self.track_complete = True
            return
        if not self.waypoint_collected[idx]:
            self.waypoint_collected[idx] = True
            self._hide_waypoint(idx)
            self.get_logger().info(
                f"[WAYPOINT] ✓ COLLECTED WP{idx:02d} "
                f"at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                f"distance={distance:.3f} m"
            )
        self.current_waypoint += 1
        if self.current_waypoint >= len(self.track_waypoint_positions):
            self.track_complete = True
            self.get_logger().info("[WAYPOINT] 🏁 ALL 33 WAYPOINTS COLLECTED")
        else:
            target = self.track_waypoint_positions[self.current_waypoint]
            self.get_logger().info(
                f"[WAYPOINT] → Target WP{self.current_waypoint:02d}: "
                f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
            )

    def _waypoint_was_passed(self, pos):
        idx = self.current_waypoint
        if idx <= 0 or idx >= len(self.track_waypoint_positions):
            return False
        prev = self.track_waypoint_positions[idx - 1, :2]
        target = self.track_waypoint_positions[idx, :2]
        segment = target - prev
        seg_len = np.linalg.norm(segment)
        if seg_len < 1e-6:
            return False
        # Once the robot crosses the perpendicular plane beyond the target,
        # allow collection even if the locomotion controller overshoots slightly.
        progress = float(np.dot(pos[:2] - target, segment / seg_len))
        return progress > WAYPOINT_PASS_MARGIN and np.linalg.norm(pos[:2] - target) < 1.5

    # --------------------------------------------------------
    # Robot pose / navigation
    # --------------------------------------------------------

    def _get_robot_pose(self):
        p = self.data.xpos[self.track_body_id].copy()
        q = self.data.xquat[self.track_body_id].copy()
        return p, quaternion_to_euler(q)

    def _compute_navigation_command(self):
        if not self.navigation_enabled or self.track_complete:
            return 0.0, 0.0, 0.0

        if self.current_waypoint >= len(self.track_waypoint_positions):
            self.track_complete = True
            return 0.0, 0.0, 0.0

        pos, rpy = self._get_robot_pose()
        target = self.track_waypoint_positions[self.current_waypoint]
        dx = float(target[0] - pos[0])
        dy = float(target[1] - pos[1])
        distance = math.hypot(dx, dy)

        # Collect only the current waypoint. Never skip multiple waypoints.
        if distance <= WAYPOINT_REACH_RADIUS or self._waypoint_was_passed(pos):
            self._collect_current_waypoint(pos, distance)
            if self.track_complete:
                return 0.0, 0.0, 0.0
            target = self.track_waypoint_positions[self.current_waypoint]
            dx = float(target[0] - pos[0])
            dy = float(target[1] - pos[1])
            distance = math.hypot(dx, dy)

        if distance < 1e-6:
            return 0.0, 0.0, 0.0

        target_yaw = math.atan2(dy, dx)
        heading_error = normalize_angle(target_yaw - float(rpy[2]))
        abs_error = abs(heading_error)

        # Rotate toward the target. Keep some forward motion so the RL gait
        # remains active, but stop forward motion for very large errors.
        wz = clamp(TURN_GAIN * heading_error, -MAX_YAW_SPEED, MAX_YAW_SPEED)

        if abs_error >= HEADING_STOP_ANGLE:
            speed = 0.0
        elif abs_error > HEADING_SLOWDOWN_ANGLE:
            ratio = (HEADING_STOP_ANGLE - abs_error) / (
                HEADING_STOP_ANGLE - HEADING_SLOWDOWN_ANGLE)
            speed = MAX_FORWARD_SPEED * clamp(ratio, 0.0, 1.0)
        else:
            speed = MAX_FORWARD_SPEED

        # Slow down near a waypoint so it is actually collected instead of
        # being crossed at high speed.
        if distance < 2.0:
            speed *= clamp(distance / 1.0, 0.25, 1.0)

        # Desired world direction -> S10 body frame.
        ux = dx / distance
        uy = dy / distance
        yaw = float(rpy[2])
        c = math.cos(yaw)
        s = math.sin(yaw)
        vx_dir = c * ux + s * uy
        vy_dir = -s * ux + c * uy

        vx = clamp(speed * vx_dir, -MAX_FORWARD_SPEED, MAX_FORWARD_SPEED)
        vy = clamp(speed * vy_dir, -MAX_SIDE_SPEED, MAX_SIDE_SPEED)

        # Command smoothing prevents abrupt changes that can destabilize the
        # learned gait at sharp corners.
        desired = np.array([vx, vy, wz], dtype=np.float32)
        alpha = 0.12
        self.last_nav_command += alpha * (desired - self.last_nav_command)
        return tuple(self.last_nav_command.astype(float))

    def _publish_navigation_command(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.nav_pub.publish(msg)
    def _get_clean_lidar(self, lidar_data):
        """
        Clean LiDAR distance measurements.

        Returns:
            numpy array of valid distances in meters.
        """

        lidar = np.asarray(lidar_data, dtype=np.float32).copy()

        # Replace NaN / Inf
        lidar[~np.isfinite(lidar)] = LIDAR_MAX_RANGE

        # Clamp to valid range
        lidar = np.clip(
            lidar,
            LIDAR_MIN_RANGE,
            LIDAR_MAX_RANGE
        )

        return lidar
    
    def _get_lidar_distances(self, lidar_data):
        """
        Calculate obstacle distances around the robot.

        Returns:
            front
            left
            right
            rear
            closest
        """

        lidar = self._get_clean_lidar(lidar_data)

        def sector_min(start_deg, end_deg):

            indices = np.arange(
                start_deg,
                end_deg + 1
            ) % LIDAR_NUM_RAYS

            values = lidar[indices]

            return float(np.min(values))

        front = min(
            sector_min(330, 359),
            sector_min(0, 30)
        )

        left = sector_min(45, 135)

        rear = sector_min(150, 210)

        right = sector_min(225, 315)

        closest = float(np.min(lidar))

        return {
            "front": front,
            "left": left,
            "right": right,
            "rear": rear,
            "closest": closest,
        }
    def _read_lidar(self):
        """
        Simulated 360-degree horizontal LiDAR using MuJoCo mj_ray().

        Returns:
            numpy array of shape (LIDAR_NUM_RAYS,)
            containing distance in meters.
        """

        # LiDAR world position
        origin = self.data.site_xpos[self.lidar_site_id].copy()

        # LiDAR orientation matrix
        rotation = self.data.site_xmat[self.lidar_site_id].reshape(3, 3)

        ranges = np.full(
            LIDAR_NUM_RAYS,
            LIDAR_MAX_RANGE,
            dtype=np.float32
        )

        for i in range(LIDAR_NUM_RAYS):

            # Angle around robot
            angle = -math.pi + (
                2.0 * math.pi * i / LIDAR_NUM_RAYS
            )

            # Ray direction in LiDAR local frame
            direction_local = np.array([
                math.cos(angle),
                math.sin(angle),
                0.0
            ], dtype=np.float64)

            # Convert direction from LiDAR frame to world frame
            direction_world = rotation @ direction_local

            # Normalize
            direction_world /= np.linalg.norm(direction_world)

            # MuJoCo ray casting
            geomid = np.array([-1], dtype=np.int32)

            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction_world,
                None,
                1,
                -1,
                geomid
            )

            if distance >= 0.0:
                distance = max(
                    LIDAR_MIN_RANGE,
                    min(float(distance), LIDAR_MAX_RANGE)
                )
                ranges[i] = distance
            else:
                ranges[i] = LIDAR_MAX_RANGE

        self.lidar_ranges = ranges

        return ranges

    # --------------------------------------------------------
    # RL observation
    # --------------------------------------------------------

    def _build_observation(self, command):
        base_omega = self.data.sensordata[7:10].astype(np.float32) * OMEGA_SCALE
        # ranges = self._read_lidar()
        # distances = self._get_lidar_distances(ranges)

        # print("Front :", distances["front"])
        # print("Left  :", distances["left"])
        # print("Right :", distances["right"])
        # print("Rear  :", distances["rear"])
        # print("Closest:", distances["closest"])

        quat = np.asarray(self.data.xquat[self.track_body_id], dtype=np.float64)
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        projected_gravity = (rot.T @ GRAVITY_DIRECTION.astype(np.float64)).astype(np.float32)

        q = self.data.qpos[7:23].astype(np.float32)
        dq = self.data.qvel[6:22].astype(np.float32)
        q_policy = q[ROBOT_TO_POLICY].copy()
        dq_policy = dq[ROBOT_TO_POLICY].copy() * DOF_VEL_SCALE
        q_policy[12:16] = 0.0
        q_policy -= DOF_DEFAULT_POLICY

        obs = np.concatenate([
            base_omega,
            projected_gravity,
            np.asarray(command, dtype=np.float32),
            q_policy,
            dq_policy,
            self.last_action,
        ]).astype(np.float32)

        if obs.size != OBSERVATION_DIM:
            raise RuntimeError(f"Expected 57-D observation, got {obs.size}")
        return obs

    def _load_policy(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Policy not found: {path}")
        self.get_logger().info(f"[RL] Loading policy: {path}")
        self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.policy_input_name = self.ort_session.get_inputs()[0].name
        self.policy_output_name = self.ort_session.get_outputs()[0].name
        self.rl_enabled = True
        self.get_logger().info(f"[RL] Input: {self.policy_input_name}")
        self.get_logger().info(f"[RL] Output: {self.policy_output_name}")
        self.get_logger().info("[RL] Policy loaded successfully")
        self._print_joint_permutation()

    def _print_joint_permutation(self):
        self.get_logger().info("===== JOINT PERMUTATION =====")
        for i in range(16):
            self.get_logger().info(
                f"i={i} robot2policy={ROBOT_TO_POLICY[i]} policy2robot={POLICY_TO_ROBOT[i]}")
        self.get_logger().info("=============================")

    def _run_policy(self, observation):
        if self.ort_session is None:
            return np.zeros(16, dtype=np.float32)
        obs = observation.reshape(1, 57).astype(np.float32)
        out = self.ort_session.run([self.policy_output_name], {self.policy_input_name: obs})[0]
        action = np.asarray(out, dtype=np.float32).reshape(-1)
        if action.size != 16:
            raise RuntimeError(f"Policy returned {action.size} actions; expected 16")
        return action

    def _apply_policy_action(self, policy_action):
        self.current_action = policy_action.copy()
        action_robot = policy_action[POLICY_TO_ROBOT].copy()
        action_robot = action_robot * ACTION_SCALE_ROBOT + DOF_DEFAULT_ROBOT

        for leg in range(4):
            b = leg * 4
            self.pos_cmd[b:b+3] = action_robot[b:b+3]
            self.vel_cmd[b+3] = action_robot[b+3]
        self.last_action = policy_action.copy()

    # --------------------------------------------------------
    # ROS command callback
    # --------------------------------------------------------

    def _cmd_callback(self, msg):
        try:
            joints = msg.data.joints_data
        except Exception:
            self.get_logger().warn("[CMD] Invalid JointsDataCmd")
            return
        if len(joints) != 16:
            self.get_logger().warn(f"[CMD] Invalid joint count: {len(joints)}")
            return
        for i, joint in enumerate(joints):
            self.kp_cmd[i] = joint.kp
            self.kd_cmd[i] = joint.kd
            self.pos_cmd[i] = joint.position
            self.vel_cmd[i] = joint.velocity
            self.tau_ff[i] = joint.torque

    # --------------------------------------------------------
    # Torque / publishing
    # --------------------------------------------------------

    def _apply_joint_torque(self):
        q = self.data.qpos[7:23].astype(np.float32)
        dq = self.data.qvel[6:22].astype(np.float32)
        torque = (self.kp_cmd * (self.pos_cmd - q) +
                  self.kd_cmd * (self.vel_cmd - dq) + self.tau_ff)
        torque = np.asarray(torque, dtype=np.float64).reshape(16)
        self.data.ctrl[:] = torque
        self.input_tq[:] = torque.astype(np.float32)

    def _publish_imu(self):
        q = self.data.xquat[self.track_body_id]
        rpy = quaternion_to_euler(q)
        body_acc = self.data.sensordata[4:7]
        angvel = self.data.sensordata[7:10]
        msg = ImuData()
        msg.header = MetaType()
        msg.header.frame_id = 0
        sec = int(self.simulation_time)
        stamp = Time()
        stamp.sec = sec
        stamp.nanosec = int((self.simulation_time - sec) * 1e9)
        msg.header.stamp = stamp
        msg.data = ImuDataValue()
        msg.data.roll = float(math.degrees(rpy[0]))
        msg.data.pitch = float(math.degrees(rpy[1]))
        msg.data.yaw = float(math.degrees(rpy[2]))
        msg.data.omega_x = float(angvel[0])
        msg.data.omega_y = float(angvel[1])
        msg.data.omega_z = float(angvel[2])
        msg.data.acc_x = float(body_acc[0])
        msg.data.acc_y = float(body_acc[1])
        msg.data.acc_z = float(body_acc[2])
        self.imu_pub.publish(msg)

    def _publish_joint_state(self):
        q = self.data.qpos[7:23].astype(np.float32)
        dq = self.data.qvel[6:22].astype(np.float32)
        pub_pos = (q - POS_OFFSET_RAD) * JOINT_DIR
        pub_vel = dq * JOINT_DIR
        pub_tau = self.input_tq * JOINT_DIR
        msg = JointsData()
        msg.header = MetaType()
        msg.header.frame_id = 0
        sec = int(self.simulation_time)
        stamp = Time()
        stamp.sec = sec
        stamp.nanosec = int((self.simulation_time - sec) * 1e9)
        msg.header.stamp = stamp
        msg.data = JointsDataValue()
        msg.data.joints_data = [JointData() for _ in range(16)]
        for i, joint in enumerate(msg.data.joints_data):
            joint.name = [32, 32, 32, 32]
            joint.data_id = 0
            joint.status_word = 1
            joint.position = float(pub_pos[i])
            joint.velocity = float(pub_vel[i])
            joint.torque = float(pub_tau[i])
            joint.motion_temp = 40.0
            joint.driver_temp = 45.0
        self.joints_pub.publish(msg)

    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    def _configure_viewer(self):
        with self.viewer.lock():
            bid = self.track_body_id
            if TRACK_VIEWER and bid >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = bid
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.trackbodyid = -1
                self.viewer.cam.lookat[:] = self.data.xpos[bid]
            self.viewer.cam.fixedcamid = -1
            self.viewer.cam.azimuth = CAMERA_AZIMUTH
            self.viewer.cam.elevation = CAMERA_ELEVATION
            self.viewer.cam.distance = CAMERA_DISTANCE

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    def _log_navigation(self, vx, vy, wz):
        if self.track_complete:
            self.get_logger().info("[NAV] Track complete")
            return
        pos, rpy = self._get_robot_pose()
        target = self.track_waypoint_positions[self.current_waypoint]
        dx = float(target[0] - pos[0])
        dy = float(target[1] - pos[1])
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        err = normalize_angle(target_yaw - float(rpy[2]))
        self.get_logger().info(
            f"[NAV] x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}, "
            f"yaw={math.degrees(rpy[2]):.1f} deg, "
            f"target_yaw={math.degrees(target_yaw):.1f} deg, "
            f"error={math.degrees(err):.1f} deg, dist={dist:.3f} m, "
            f"waypoint={self.current_waypoint}/{len(self.track_waypoint_positions)}, "
            f"target_z={target[2]:.3f}, "
            f"BODY_CMD=(vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f})")

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def start(self):
        step = 0
        last_wall = time.perf_counter()
        last_nav_log = 0.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            now = time.perf_counter()
            if now - last_wall < DT:
                time.sleep(0.0001)
                continue
            last_wall = now
            step += 1
            self.simulation_time = step * DT

            if self.simulation_time < self.standing_time:
                vx, vy, wz = 0.0, 0.0, 0.0
                self.last_nav_command[:] = 0.0
            else:
                vx, vy, wz = self._compute_navigation_command()
            command = np.array([vx, vy, wz], dtype=np.float32)
            # self._publish_navigation_command(vx, vy, wz)

            if self.simulation_time < self.standing_time:
                self.pos_cmd[:] = JOINT_INIT[MODEL_NAME]
                self.vel_cmd[:] = 0.0
                self.tau_ff[:] = 0.0
            elif self.rl_enabled and step % POLICY_DECIMATION == 0:
                try:
                    obs = self._build_observation(command)
                    action = self._run_policy(obs)
                    self._apply_policy_action(action)
                except Exception as exc:
                    self.get_logger().error(f"[RL] Policy error: {exc}")
                    self.pos_cmd[:] = self.data.qpos[7:23]
                    self.vel_cmd[:] = 0.0

            self._apply_joint_torque()
            mujoco.mj_step(self.model, self.data)

            # if step % 5 == 0:
            #     try:
            #         self._publish_imu()
            #         self._publish_joint_state()
            #     except Exception as exc:
            #         self.get_logger().error(f"[ROS] Publish error: {exc}")

            if self.simulation_time - last_nav_log >= NAV_LOG_PERIOD:
                last_nav_log = self.simulation_time
                self._log_navigation(vx, vy, wz)
                self._print_robot_target_direction()

            if self.viewer and step % RENDER_INTERVAL == 0:
                try:
                    self.viewer.sync()
                except Exception:
                    pass


    def _print_robot_target_direction(self):
        """
        Print where the current waypoint is relative to the robot.

        Possible directions:
            FORWARD
            BACKWARD
            LEFT
            RIGHT
            FORWARD-LEFT
            FORWARD-RIGHT
            BACKWARD-LEFT
            BACKWARD-RIGHT
        """

        if self.track_complete:
            return

        if self.current_waypoint >= len(self.track_waypoint_positions):
            return

        # Robot position and orientation
        pos, rpy = self._get_robot_pose()

        # Current waypoint
        target = self.track_waypoint_positions[self.current_waypoint]

        # World-frame difference
        dx = float(target[0] - pos[0])
        dy = float(target[1] - pos[1])

        distance = math.hypot(dx, dy)

        if distance < 1e-6:
            self.get_logger().info("[DIRECTION] TARGET REACHED")
            return

        # Robot yaw
        yaw = float(rpy[2])

        # World direction -> robot/body frame
        c = math.cos(yaw)
        s = math.sin(yaw)

        forward = c * dx + s * dy
        left = -s * dx + c * dy

        # Normalize
        forward_norm = forward / distance
        left_norm = left / distance

        # Direction thresholds
        threshold = 0.35

        if forward_norm > threshold:
            if left_norm > threshold:
                direction = "FORWARD-LEFT"
            elif left_norm < -threshold:
                direction = "FORWARD-RIGHT"
            else:
                direction = "FORWARD"

        elif forward_norm < -threshold:
            if left_norm > threshold:
                direction = "BACKWARD-LEFT"
            elif left_norm < -threshold:
                direction = "BACKWARD-RIGHT"
            else:
                direction = "BACKWARD"

        else:
            if left_norm > 0:
                direction = "LEFT"
            else:
                direction = "RIGHT"

        self.get_logger().info(
            f"[DIRECTION] {direction} | "
            f"distance={distance:.2f} m | "
            f"forward={forward:.2f} m | "
            f"left={left:.2f} m | "
            f"yaw={math.degrees(yaw):.1f} deg"
        )

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


def main():
    cli_args, ros_args = parse_cli_args()
    rclpy.init(args=ros_args)
    xml_path = resolve_xml_path(cli_args.scene, cli_args.xml_path)
    policy_path = cli_args.policy if cli_args.policy else find_policy_path()
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
        print("\n[MuJoCo] Keyboard interrupt")
    except Exception as exc:
        print("\n[MuJoCo] ERROR:")
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