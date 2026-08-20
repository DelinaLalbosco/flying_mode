"""
@file mujoco_simulation_preview.py
@brief Combined preview build: real navigation/locomotion + camera + LiDAR perception
@author Team preview build (derived from Bo (Percy) Peng's mujoco_simulation.py v1.1)
@date 2026-08-20

------------------------------------------------------------------------------
WHY THIS FILE EXISTS
------------------------------------------------------------------------------
This is a NEW file. It does not modify or replace either teammate's original
script:
    - mujoco_simulation.py            (friend's real nav + locomotion, v1.1)
    - mujoco_simulation_direct_nav.py (my kinematic waypoint-validation script)

It combines:
    1. Friend's navigation + locomotion stack, UNCHANGED in logic:
       pure-pursuit path following, waypoint-snap, depth-camera wall
       avoidance, stuck-recovery. Locomotion is still produced by the real,
       physics-driven RL policy via /JOINTS_CMD -> torque control -> mj_step.
    2. NEW: multi-camera image capture (auto-discovers every camera defined
       in the MJCF) with a live OpenCV preview window per camera.
    3. NEW: full 360-degree LiDAR raycasting with a live sector-distance
       readout (front/left/right/rear/closest), printed to console and
       published as a ROS 2 LaserScan.

WHAT IS DELIBERATELY *NOT* INCLUDED
------------------------------------------------------------------------------
My direct-base-navigation script's core mechanism -- writing the robot's
qpos directly every step to move it -- is NOT included here. That mechanism
bypasses MuJoCo physics; it cannot coexist with the real RL-driven torque
control below without silently overriding or fighting it. Including it would
make the "closed loop" in this file misleading. If you want to run that
kinematic validation tool, run the original script separately and standalone
-- do not merge it into this file.

REQUIREMENTS TO RUN
------------------------------------------------------------------------------
Same as friend's original file: this node only PUBLISHES nav commands on
/AUTO_NAV_CMD. Actual joint torques come from the SDK's RL policy runner
(rl_deploy, C++), which must be running separately and subscribed via
/JOINTS_CMD, exactly as in the existing PROJECT_LOG.md workflow. This file
does not stand up a fake walker -- if rl_deploy isn't running, the robot
will not walk, which is the correct (non-misleading) behavior.

RULES COMPLIANCE NOTES
------------------------------------------------------------------------------
- TRACK_REACH_RADIUS is set to 0.2 m (the original spec), NOT the loosened
  0.4 m that appeared in one intermediate debug build. Confirm this value
  against the official track spec before competition use.
- Known-track waypoint/path coordinates are used for path-following, which
  the handbook explicitly permits; camera/LiDAR perception remains active
  for genuinely unknown obstacles (wall avoidance), not disabled or faked.
- Camera and LiDAR data below are read directly from the live MuJoCo
  simulation each step -- nothing here is pre-recorded or synthetic.
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
from drdds.msg import ImuData, JointsData, JointsDataCmd, MetaType, ImuDataValue, JointsDataValue, JointData

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan

try:
    from cv_bridge import CvBridge
    import cv2
    _HAVE_CV = True
except ImportError:
    _HAVE_CV = False

MODEL_NAME = "S10"
CURRENT_DIR = Path(__file__).resolve().parent
MJCF_DIR = (CURRENT_DIR / ".." / ".." / ".." / "S10_description" / "s10_mjcf" / "mjcf").resolve()

SCENE_XML_PATHS = {
    "track": MJCF_DIR / "S10_track.xml",
}
DEFAULT_SCENE_NAME = os.environ.get("S10_MUJOCO_SCENE", "track")
XML_PATH = str(SCENE_XML_PATHS.get(DEFAULT_SCENE_NAME, SCENE_XML_PATHS["track"]).resolve())
USE_VIEWER = True
TRACK_VIEWER = True
DT = 0.001
RENDER_INTERVAL = 10
TRACK_BODY_NAME = "base_link"
CAMERA_AZIMUTH = 90
CAMERA_ELEVATION = -25
CAMERA_DISTANCE = 18.0
COLLISION_GEOM_GROUP = 1
TRACK_START_BASE_POS = np.array([0.0, -2.5, 0.2])

# Correct official spec -- see "RULES COMPLIANCE NOTES" above.
TRACK_REACH_RADIUS = float(os.environ.get("S10_TRACK_REACH_RADIUS", "0.2"))
TRACK_DISTANCE_MODE = os.environ.get("S10_TRACK_DISTANCE_MODE", "xy").lower()
TRACK_WAYPOINT_PREFIX = "track_waypoint_"
TRACK_HEIGHT_POST_PREFIX = "track_height_post_"

DEPTH_CAMERA_NAME = "front_camera"
DEPTH_IMG_WIDTH = 64
DEPTH_IMG_HEIGHT = 64
DEPTH_RENDER_STEP_INTERVAL = 500  # ~2 Hz at DT=0.001
WALL_DETECT_MAX_RANGE = 1.2
WALL_ROW_TOP_FRAC = 0.15
WALL_ROW_BOTTOM_FRAC = 0.55

# --- NEW: perception preview settings ---
RGB_IMG_WIDTH = 320
RGB_IMG_HEIGHT = 240
RGB_RENDER_STEP_INTERVAL = 500     # ~2 Hz -- matches depth interval, avoids the
                                    # ~50ms/render overhead noted in PROJECT_LOG.md
SHOW_CV_WINDOWS = True              # set False to disable live cv2.imshow windows
LIDAR_NUM_RAYS = 360
LIDAR_UPDATE_INTERVAL = 200         # ~5 Hz
LIDAR_SITE_NAME = "lidar"
LIDAR_PRINT_INTERVAL_SEC = 1.0

JOINT_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1], dtype=np.float32)
POS_OFFSET_DEG = np.array([-35, -145, 156, 0.,
                             35, -145, 156, 0,
                             -35, 145, -156, 0,
                             35, 145, -156, 0])
POS_OFFSET_RAD = POS_OFFSET_DEG / 180.0 * np.pi

JOINT_INIT = {
    "S10": np.array([-0.438, -1.16, 2.76, 0,
                     0.438, -1.16, 2.76, 0,
                     -0.438, 1.16, -2.76, 0,
                     0.438, 1.16, -2.76, 0], dtype=np.float32),
}


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run S10 MuJoCo ROS2 preview simulation.")
    parser.add_argument("--scene", choices=sorted(SCENE_XML_PATHS),
                         default=DEFAULT_SCENE_NAME if DEFAULT_SCENE_NAME in SCENE_XML_PATHS else "track")
    parser.add_argument("--xml-path", default=os.environ.get("S10_MUJOCO_XML"))
    parser.add_argument("--model-key", default=MODEL_NAME)
    parser.add_argument("--no-cv-windows", action="store_true", help="Disable live camera preview windows.")
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def resolve_xml_path(scene_name: str, xml_path):
    if xml_path:
        return str(Path(xml_path).expanduser().resolve())
    return str(SCENE_XML_PATHS[scene_name].resolve())


class MuJoCoPreviewNode(Node):
    def __init__(self, model_key: str = MODEL_NAME, xml_path: str = XML_PATH, show_cv_windows: bool = True):
        super().__init__('mujoco_simulation_preview')

        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        self.actuator_ids = [a for a in range(self.model.nu)]
        self.dof_num = len(self.actuator_ids)
        assert self.dof_num == 16, "Expected 16 DOF for S10"

        self._set_initial_pose(model_key)
        self._init_track_progress()
        self._init_path_polyline()

        self.kp_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.kd_cmd = np.zeros_like(self.kp_cmd)
        self.pos_cmd = np.zeros_like(self.kp_cmd)
        self.vel_cmd = np.zeros_like(self.kp_cmd)
        self.tau_ff = np.zeros_like(self.kp_cmd)
        self.input_tq = np.zeros_like(self.kp_cmd)

        self.timestamp = 0.0

        # Stuck-recovery state (from friend's version, unchanged)
        self.last_check_pos = None
        self.last_check_time = 0.0
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_timer = 0.0

        self.get_logger().info(f"[INFO] MuJoCo MJCF loaded: {xml_path}")
        self.get_logger().info(f"[INFO] MuJoCo model loaded, dof = {self.dof_num}")

        # --- Publishers ---
        self.imu_pub = self.create_publisher(ImuData, '/IMU_DATA', 200)
        self.joints_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 200)
        self.nav_cmd_pub = self.create_publisher(Twist, '/AUTO_NAV_CMD', 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/LIDAR_DATA', 10)

        # --- Subscriber: real joint commands from the RL policy runner (rl_deploy) ---
        self.cmd_sub = self.create_subscription(JointsDataCmd, '/JOINTS_CMD', self._cmd_callback, 50)

        self.viewer = None
        if USE_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._configure_viewer()

        # Depth camera for wall-avoidance (unchanged from friend's version)
        self.depth_renderer = mujoco.Renderer(self.model, height=DEPTH_IMG_HEIGHT, width=DEPTH_IMG_WIDTH)
        self.depth_camera_available = DEPTH_CAMERA_NAME in [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)
        ]
        self.wall_ahead = False
        self.wall_steer_bias = 0.0

        # --- NEW: multi-camera RGB perception preview ---
        self.show_cv_windows = show_cv_windows and _HAVE_CV
        if show_cv_windows and not _HAVE_CV:
            self.get_logger().warn("[WARN] cv_bridge/cv2 not available; camera preview windows disabled, publishing only.")
        self.bridge = CvBridge() if _HAVE_CV else None
        self.rgb_renderer = mujoco.Renderer(self.model, height=RGB_IMG_HEIGHT, width=RGB_IMG_WIDTH)
        self._camera_ids = {}
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                self._camera_ids[name] = i
        if self._camera_ids:
            self.get_logger().info(f"[INFO] Discovered {len(self._camera_ids)} camera(s): {list(self._camera_ids.keys())}")
        else:
            self.get_logger().warn("[WARN] No named cameras found in MJCF; RGB preview disabled.")
        self.camera_pubs = {
            name: self.create_publisher(Image, f"/CAMERA/{name}", 10)
            for name in self._camera_ids
        }

        # --- NEW: full 360-degree LiDAR ---
        self.lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, LIDAR_SITE_NAME)
        self.lidar_available = self.lidar_site_id >= 0
        if self.lidar_available:
            self.lidar_body_id = self.model.site_bodyid[self.lidar_site_id]
            self.lidar_angles = np.linspace(-np.pi, np.pi, LIDAR_NUM_RAYS, endpoint=False)
            self.lidar_local_dirs = np.column_stack([
                np.cos(self.lidar_angles), np.sin(self.lidar_angles), np.zeros(LIDAR_NUM_RAYS),
            ])
            self.lidar_ranges = np.full(LIDAR_NUM_RAYS, 30.0, dtype=np.float32)
            self.get_logger().info(f"[INFO] LiDAR site '{LIDAR_SITE_NAME}' found; {LIDAR_NUM_RAYS}-ray scan enabled.")
        else:
            self.get_logger().warn(f"[WARN] LiDAR site '{LIDAR_SITE_NAME}' not found; LiDAR preview disabled.")
        self._last_lidar_print_time = -999.0

    # ---------------- setup helpers (unchanged from friend's version) ----------------

    def _set_initial_pose(self, key: str):
        qpos0 = self.data.qpos.copy()
        qpos0[7:7 + self.dof_num] = JOINT_INIT[key]
        qpos0[:3] = TRACK_START_BASE_POS
        qpos0[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)

    def _track_geom_index(self, name, prefix):
        if not name or not name.startswith(prefix):
            return None
        suffix = name[len(prefix):]
        index_text = suffix.split("_", 1)[0]
        if not index_text.isdigit():
            return None
        return int(index_text)

    def _find_track_geoms(self):
        waypoint_geoms, point_related_geoms = {}, {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            wi = self._track_geom_index(name, TRACK_WAYPOINT_PREFIX)
            if wi is not None:
                waypoint_geoms[wi] = geom_id
                point_related_geoms.setdefault(wi, []).append(geom_id)
                continue
            pi = self._track_geom_index(name, TRACK_HEIGHT_POST_PREFIX)
            if pi is not None:
                point_related_geoms.setdefault(pi, []).append(geom_id)
        return waypoint_geoms, point_related_geoms

    def _init_track_progress(self):
        self.track_enabled = False
        self.track_complete = False
        self.track_next_index = 0
        self.track_start_time = None
        self.track_finish_time = None
        self.track_waypoint_positions = np.empty((0, 3), dtype=np.float64)
        self.track_point_geom_ids = {}

        waypoint_geoms, point_related_geoms = self._find_track_geoms()
        if not waypoint_geoms:
            return

        expected_indices = list(range(max(waypoint_geoms) + 1))
        missing = [i for i in expected_indices if i not in waypoint_geoms]
        if missing:
            self.get_logger().warn(f"Track progress disabled; missing waypoint geoms: {missing}")
            return

        self.track_waypoint_geom_ids = [waypoint_geoms[i] for i in expected_indices]
        self.track_point_geom_ids = {
            i: point_related_geoms.get(i, [waypoint_geoms[i]]) for i in expected_indices
        }
        self.track_waypoint_positions = np.array(
            [self.data.geom_xpos[gid].copy() for gid in self.track_waypoint_geom_ids], dtype=np.float64,
        )
        self.track_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY_NAME)
        if self.track_body_id < 0:
            self.get_logger().warn(f"Track progress disabled; cannot find body '{TRACK_BODY_NAME}'")
            return

        self.track_enabled = True
        self.get_logger().info(
            f"[INFO] Track progress enabled: {len(self.track_waypoint_positions)} waypoints, "
            f"radius={TRACK_REACH_RADIUS:.3f}m, distance_mode={TRACK_DISTANCE_MODE}"
        )

    def _init_path_polyline(self):
        import re
        self.path_points = np.empty((0, 3), dtype=np.float64)
        try:
            overlay_path = (MJCF_DIR / "track_overlay.xml").resolve()
            with open(overlay_path) as f:
                content = f.read()
            pattern = r'name="track_segment_(\d+)"[^>]*fromto="([\d\.\-\s]+)"'
            matches = re.findall(pattern, content)
            matches.sort(key=lambda m: int(m[0]))
            points = []
            for idx_str, coords in matches:
                nums = [float(x) for x in coords.split()]
                if not points:
                    points.append(nums[0:3])
                points.append(nums[3:6])
            self.path_points = np.array(points, dtype=np.float64)
            self.get_logger().info(f"[INFO] Path polyline loaded: {len(self.path_points)} points")
        except Exception as e:
            self.get_logger().warn(f"[WARN] Failed to load path polyline: {e}")

    def _get_pursuit_target(self, robot_pos, lookahead=1.5):
        if self.path_points is None or len(self.path_points) < 2:
            return None
        best_dist, best_seg_idx, best_t = float("inf"), 0, 0.0
        for i in range(len(self.path_points) - 1):
            a, b = self.path_points[i][:2], self.path_points[i + 1][:2]
            ab = b - a
            ab_len_sq = np.dot(ab, ab)
            if ab_len_sq < 1e-9:
                continue
            t = np.clip(np.dot(robot_pos[:2] - a, ab) / ab_len_sq, 0.0, 1.0)
            proj = a + t * ab
            dist = np.linalg.norm(robot_pos[:2] - proj)
            if dist < best_dist:
                best_dist, best_seg_idx, best_t = dist, i, t

        remaining, seg_idx = lookahead, best_seg_idx
        a, b = self.path_points[seg_idx][:2], self.path_points[seg_idx + 1][:2]
        seg_len = np.linalg.norm(b - a)
        pos_on_seg = best_t * seg_len

        while remaining > 0:
            room_left = seg_len - pos_on_seg
            if remaining <= room_left:
                pos_on_seg += remaining
                remaining = 0
            else:
                remaining -= room_left
                seg_idx += 1
                if seg_idx >= len(self.path_points) - 1:
                    seg_idx = len(self.path_points) - 2
                    pos_on_seg = np.linalg.norm(self.path_points[seg_idx + 1][:2] - self.path_points[seg_idx][:2])
                    break
                a, b = self.path_points[seg_idx][:2], self.path_points[seg_idx + 1][:2]
                seg_len = np.linalg.norm(b - a)
                pos_on_seg = 0.0

        a, b = self.path_points[seg_idx][:2], self.path_points[seg_idx + 1][:2]
        seg_len = np.linalg.norm(b - a)
        frac = 0.0 if seg_len < 1e-9 else pos_on_seg / seg_len
        return a + frac * (b - a)

    def _on_climbing_segment(self, robot_pos, dz_threshold=0.15):
        if self.path_points is None or len(self.path_points) < 2:
            return False
        best_dist, best_seg_idx = float("inf"), 0
        for i in range(len(self.path_points) - 1):
            a, b = self.path_points[i][:2], self.path_points[i + 1][:2]
            ab = b - a
            ab_len_sq = np.dot(ab, ab)
            if ab_len_sq < 1e-9:
                continue
            t = np.clip(np.dot(robot_pos[:2] - a, ab) / ab_len_sq, 0.0, 1.0)
            proj = a + t * ab
            dist = np.linalg.norm(robot_pos[:2] - proj)
            if dist < best_dist:
                best_dist, best_seg_idx = dist, i
        dz = self.path_points[best_seg_idx + 1][2] - self.path_points[best_seg_idx][2]
        return abs(dz) > dz_threshold

    def _hide_track_point(self, waypoint_index):
        for geom_id in self.track_point_geom_ids.get(waypoint_index, []):
            self.model.geom_rgba[geom_id, 3] = 0.0

    def _track_distance(self, robot_pos, waypoint_pos):
        if TRACK_DISTANCE_MODE == "xyz":
            return float(np.linalg.norm(robot_pos - waypoint_pos))
        return float(np.linalg.norm(robot_pos[:2] - waypoint_pos[:2]))

    def _update_track_progress(self):
        if not self.track_enabled or self.track_complete:
            return
        if self.track_next_index >= len(self.track_waypoint_positions):
            return
        robot_pos = self.data.xpos[self.track_body_id]
        waypoint_pos = self.track_waypoint_positions[self.track_next_index]
        distance = self._track_distance(robot_pos, waypoint_pos)
        if distance > TRACK_REACH_RADIUS:
            return

        reached_index = self.track_next_index
        self._hide_track_point(reached_index)

        if reached_index == 0 and self.track_start_time is None:
            self.track_start_time = self.timestamp
            self.get_logger().info(f"[TRACK] Timer started at waypoint 0, sim_time={self.track_start_time:.3f}s")
        else:
            self.get_logger().info(
                f"[TRACK] Reached waypoint {reached_index}, sim_time={self.timestamp:.3f}s, distance={distance:.3f}m"
            )

        self.track_next_index += 1
        if self.track_next_index >= len(self.track_waypoint_positions):
            self.track_complete = True
            self.track_finish_time = self.timestamp
            elapsed = 0.0 if self.track_start_time is None else self.track_finish_time - self.track_start_time
            self.get_logger().info(f"[TRACK] Final waypoint reached. elapsed={elapsed:.3f}s")

    # ---------------- navigation (unchanged from friend's version) ----------------

    def _auto_nav_step(self):
        if not self.track_enabled or self.track_complete:
            return
        if self.track_next_index >= len(self.track_waypoint_positions):
            return

        robot_pos = self.data.xpos[self.track_body_id]
        q_world = self.data.sensordata[:4]
        _, _, yaw = self.quaternion_to_euler(q_world)

        if self.last_check_pos is None:
            self.last_check_pos = robot_pos.copy()
            self.last_check_time = self.timestamp
        elif self.timestamp - self.last_check_time > 3.0:
            disp = np.linalg.norm(robot_pos[:2] - self.last_check_pos[:2])
            if disp < 0.1 and not self.recovery_mode:
                self.stuck_counter += 1
                self.get_logger().warn(f"[NAV-RECOVERY] Stuck detected! disp={disp:.3f}m. Recovery.")
                self.recovery_mode = True
                self.recovery_timer = self.timestamp
            self.last_check_pos = robot_pos.copy()
            self.last_check_time = self.timestamp

        if self.recovery_mode:
            if self.timestamp - self.recovery_timer < 1.5:
                twist = Twist()
                twist.linear.x = -0.2
                twist.angular.z = 0.5
                self.nav_cmd_pub.publish(twist)
                return
            else:
                self.recovery_mode = False

        next_wp = self.track_waypoint_positions[self.track_next_index]
        dist_to_wp = float(np.linalg.norm(robot_pos[:2] - next_wp[:2]))
        WAYPOINT_SNAP_RADIUS = 1.2

        if dist_to_wp < WAYPOINT_SNAP_RADIUS:
            dx, dy = next_wp[0] - robot_pos[0], next_wp[1] - robot_pos[1]
        else:
            pursuit_target = self._get_pursuit_target(robot_pos, lookahead=1.5)
            if pursuit_target is not None:
                dx, dy = pursuit_target[0] - robot_pos[0], pursuit_target[1] - robot_pos[1]
            else:
                dx, dy = next_wp[0] - robot_pos[0], next_wp[1] - robot_pos[1]

        heading_to_target = np.arctan2(dy, dx)
        yaw_error = heading_to_target - yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        turn = float(np.clip(yaw_error * 1.2, -0.6, 0.6))
        forward = 0.0 if abs(yaw_error) > 0.8 else float(0.5 * np.cos(yaw_error))

        climbing = self._on_climbing_segment(robot_pos)
        if abs(yaw_error) < 0.4 and climbing:
            turn = float(np.clip(turn, -0.25, 0.25))
            forward = min(forward, 0.35)

        if self.wall_ahead:
            forward = 0.15
            turn = float(np.clip(turn + self.wall_steer_bias, -0.7, 0.7))

        twist = Twist()
        twist.linear.x = forward
        twist.linear.y = 0.0
        twist.angular.z = turn
        self.nav_cmd_pub.publish(twist)

    def _update_depth_perception(self, step):
        if not self.depth_camera_available:
            return
        if step % DEPTH_RENDER_STEP_INTERVAL != 0:
            return
        self.depth_renderer.update_scene(self.data, camera=DEPTH_CAMERA_NAME)
        self.depth_renderer.enable_depth_rendering()
        depth = self.depth_renderer.render()

        h, w = depth.shape
        top, bottom = int(h * WALL_ROW_TOP_FRAC), int(h * WALL_ROW_BOTTOM_FRAC)
        c0, c1 = w // 3, 2 * w // 3
        wall_band = depth[top:bottom, c0:c1]
        self.wall_ahead = bool(np.min(wall_band) < WALL_DETECT_MAX_RANGE)

        if self.wall_ahead:
            left_band, right_band = depth[top:bottom, :w // 2], depth[top:bottom, w // 2:]
            self.wall_steer_bias = 0.5 if np.min(left_band) > np.min(right_band) else -0.5
        else:
            self.wall_steer_bias = 0.0

    # ---------------- NEW: multi-camera RGB perception preview ----------------

    def _update_rgb_cameras(self, step):
        if not self._camera_ids or step % RGB_RENDER_STEP_INTERVAL != 0:
            return
        for name, cam_id in self._camera_ids.items():
            try:
                self.rgb_renderer.update_scene(self.data, camera=cam_id)
                frame = self.rgb_renderer.render().copy()
                frame = np.asarray(frame, dtype=np.uint8)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    continue

                if name in self.camera_pubs:
                    ros_image = Image()
                    ros_image.header.frame_id = name
                    ros_image.height, ros_image.width = frame.shape[0], frame.shape[1]
                    ros_image.encoding = "rgb8"
                    ros_image.is_bigendian = 0
                    ros_image.step = frame.shape[1] * 3
                    ros_image.data = frame.tobytes()
                    self.camera_pubs[name].publish(ros_image)

                if self.show_cv_windows:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"S10 Camera: {name}", bgr)
                    cv2.waitKey(1)
            except Exception as e:
                self.get_logger().error(f"[CAMERA] '{name}' capture failed: {type(e).__name__}: {e}")

    # ---------------- NEW: full 360-degree LiDAR ----------------

    def _update_lidar(self, step):
        if not self.lidar_available or step % LIDAR_UPDATE_INTERVAL != 0:
            return

        site_pos = self.data.site_xpos[self.lidar_site_id].astype(np.float64)
        site_mat = self.data.site_xmat[self.lidar_site_id].reshape(3, 3)
        world_dirs = (site_mat @ self.lidar_local_dirs.T).T.astype(np.float64)

        geomid = np.zeros(LIDAR_NUM_RAYS, dtype=np.int32)
        dist = np.zeros(LIDAR_NUM_RAYS, dtype=np.float64)

        mujoco.mj_multiRay(
            self.model, self.data, pnt=site_pos, vec=world_dirs.flatten(),
            geomgroup=None, flg_static=1, bodyexclude=self.lidar_body_id,
            geomid=geomid, dist=dist, normal=None, nray=LIDAR_NUM_RAYS, cutoff=30.0,
        )
        dist[dist < 0] = 30.0
        self.lidar_ranges = dist.astype(np.float32)
        self._publish_lidar()

        if self.timestamp - self._last_lidar_print_time >= LIDAR_PRINT_INTERVAL_SEC:
            self._print_lidar_sectors()
            self._last_lidar_print_time = self.timestamp

    def _publish_lidar(self):
        scan_msg = LaserScan()
        scan_msg.header.frame_id = "lidar"
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        scan_msg.header.stamp.sec = sec
        scan_msg.header.stamp.nanosec = nanosec
        scan_msg.angle_min = float(self.lidar_angles[0])
        scan_msg.angle_max = float(self.lidar_angles[-1])
        scan_msg.angle_increment = float(2 * np.pi / LIDAR_NUM_RAYS)
        scan_msg.range_min = 0.05
        scan_msg.range_max = 30.0
        scan_msg.ranges = self.lidar_ranges.tolist()
        self.lidar_pub.publish(scan_msg)

    def _print_lidar_sectors(self):
        ranges = self.lidar_ranges.astype(np.float64)
        angles_deg = np.rad2deg(self.lidar_angles)

        def sector_min(lo, hi):
            mask = (angles_deg >= lo) & (angles_deg <= hi)
            sel = ranges[mask]
            return float(np.min(sel)) if len(sel) else 30.0

        front = sector_min(-30.0, 30.0)
        left = sector_min(30.0, 150.0)
        right = sector_min(-150.0, -30.0)
        rear_mask = (angles_deg >= 150.0) | (angles_deg <= -150.0)
        rear_sel = ranges[rear_mask]
        rear = float(np.min(rear_sel)) if len(rear_sel) else 30.0
        closest = float(np.min(ranges))

        print(
            f"[LIDAR] Front: {front:.3f} m | Left: {left:.3f} m | "
            f"Right: {right:.3f} m | Rear: {rear:.3f} m | Closest: {closest:.3f} m"
        )

    # ---------------- rest: unchanged plumbing ----------------

    def _configure_viewer(self):
        with self.viewer.lock():
            track_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY_NAME)
            if TRACK_VIEWER and track_body_id >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = track_body_id
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.trackbodyid = -1
                self.viewer.cam.lookat[:] = self.data.qpos[:3]
            self.viewer.cam.fixedcamid = -1
            self.viewer.cam.azimuth = CAMERA_AZIMUTH
            self.viewer.cam.elevation = CAMERA_ELEVATION
            self.viewer.cam.distance = CAMERA_DISTANCE
            if COLLISION_GEOM_GROUP < len(self.viewer.opt.geomgroup):
                self.viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = 0

    def _cmd_callback(self, msg: JointsDataCmd):
        if len(msg.data.joints_data) != 16:
            self.get_logger().warn("Received JointsDataCmd with incorrect number of joints")
            return
        pub_pos = np.zeros(self.dof_num, dtype=np.float32)
        pub_vel = np.zeros(self.dof_num, dtype=np.float32)
        for i in range(self.dof_num):
            joint_cmd = msg.data.joints_data[i]
            self.kp_cmd[i] = joint_cmd.kp
            self.kd_cmd[i] = joint_cmd.kd
            pub_pos[i] = joint_cmd.position
            pub_vel[i] = joint_cmd.velocity
            self.tau_ff[i] = joint_cmd.torque
        self.pos_cmd.flat = pub_pos * JOINT_DIR + POS_OFFSET_RAD
        self.vel_cmd.flat = pub_vel * JOINT_DIR

    def start(self):
        step = 0
        last_time = time.time()
        while rclpy.ok():
            if time.time() - last_time >= DT:
                last_time = time.time()
                step += 1
                self._apply_joint_torque()
                mujoco.mj_step(self.model, self.data)

                self.timestamp = step * DT
                self._update_depth_perception(step)
                self._update_rgb_cameras(step)
                self._update_lidar(step)
                self._update_track_progress()
                self._auto_nav_step()

                if step % 5 == 0:
                    self._publish_robot_state(step)

                if self.viewer and step % RENDER_INTERVAL == 0:
                    self.viewer.sync()

            rclpy.spin_once(self, timeout_sec=0.0)

    def _apply_joint_torque(self):
        q = self.data.qpos[7:7 + self.dof_num].reshape(-1, 1)
        dq = self.data.qvel[6:6 + self.dof_num].reshape(-1, 1)
        self.input_tq = self.kp_cmd * (self.pos_cmd - q) + self.kd_cmd * (self.vel_cmd - dq) + self.tau_ff
        self.data.ctrl[:] = self.input_tq.flatten()

    def quaternion_to_euler(self, q):
        w, x, y, z = q
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _publish_robot_state(self, step):
        q_world = self.data.sensordata[:4]
        rpy_rad = self.quaternion_to_euler(q_world)
        rpy_deg = [a * (180.0 / np.pi) for a in rpy_rad]
        body_acc = self.data.sensordata[4:7]
        angvel_b = self.data.sensordata[7:10]

        imu_msg = ImuData()
        imu_msg.header = MetaType()
        imu_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec, stamp.nanosec = sec, nanosec
        imu_msg.header.stamp = stamp
        imu_msg.data = ImuDataValue()
        imu_msg.data.roll, imu_msg.data.pitch, imu_msg.data.yaw = float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])
        imu_msg.data.omega_x, imu_msg.data.omega_y, imu_msg.data.omega_z = float(angvel_b[0]), float(angvel_b[1]), float(angvel_b[2])
        imu_msg.data.acc_x, imu_msg.data.acc_y, imu_msg.data.acc_z = float(body_acc[0]), float(body_acc[1]), float(body_acc[2])
        self.imu_pub.publish(imu_msg)

        q = self.data.qpos[7:7 + self.dof_num]
        dq = self.data.qvel[6:6 + self.dof_num]
        tau = self.input_tq.flatten()
        pub_pos = (q - POS_OFFSET_RAD) * JOINT_DIR
        pub_vel = dq * JOINT_DIR
        pub_tau = tau * JOINT_DIR

        joints_msg = JointsData()
        joints_msg.header = MetaType()
        joints_msg.header.frame_id = 0
        joints_msg.header.stamp = stamp
        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(self.dof_num)]
        for i in range(self.dof_num):
            joint = joints_msg.data.joints_data[i]
            joint.name = [32, 32, 32, 32]
            joint.data_id = 0
            joint.status_word = 1
            joint.position = float(pub_pos[i])
            joint.torque = float(pub_tau[i])
            joint.velocity = float(pub_vel[i])
            joint.motion_temp = 40.0
            joint.driver_temp = 45.0
        self.joints_pub.publish(joints_msg)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    cli_args, ros_args = parse_cli_args()
    rclpy.init(args=ros_args)
    sim_node = MuJoCoPreviewNode(
        model_key=cli_args.model_key,
        xml_path=resolve_xml_path(cli_args.scene, cli_args.xml_path),
        show_cv_windows=SHOW_CV_WINDOWS and not cli_args.no_cv_windows,
    )
    try:
        sim_node.start()
    except KeyboardInterrupt:
        print("\n[SIM] Preview stopped.")
    finally:
        if _HAVE_CV:
            cv2.destroyAllWindows()
        sim_node.destroy_node()
        rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()