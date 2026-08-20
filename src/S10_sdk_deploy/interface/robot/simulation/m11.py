"""
 * @file mujoco_simulation.py
 * @brief simulation in mujoco
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-05
 *
 * @copyright Copyright (c) 2025  DeepRobotics
"""

import os
import time
import socket
import struct
import threading
import argparse
#from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from drdds.msg import ImuData, JointsData, JointsDataCmd, MetaType, ImuDataValue, JointsDataValue, JointData, JointDataCmd
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

MODEL_NAME = "S10"
# Get the directory of the current Python file
CURRENT_DIR = Path(__file__).resolve().parent
MJCF_DIR = (CURRENT_DIR / ".." / ".." / ".." / "S10_description" / "s10_mjcf" / "mjcf").resolve()

SCENE_XML_PATHS = {
    "track": MJCF_DIR / "S10_track.xml",
}
DEFAULT_SCENE_NAME = os.environ.get("S10_MUJOCO_SCENE", "track")
XML_PATH = str(SCENE_XML_PATHS.get(DEFAULT_SCENE_NAME, SCENE_XML_PATHS["track"]).resolve())
USE_VIEWER = True
TRACK_VIEWER = False
DT = 0.001
RENDER_INTERVAL = 10
TRACK_BODY_NAME = "base_link"

CAMERA_AZIMUTH = 90
CAMERA_ELEVATION = -45
CAMERA_DISTANCE = 45.0

COLLISION_GEOM_GROUP = 1
TRACK_START_BASE_POS = np.array([0.0, -2.5, 0.2])
TRACK_REACH_RADIUS = float(os.environ.get("S10_TRACK_REACH_RADIUS", "0.2"))
TRACK_DISTANCE_MODE = os.environ.get("S10_TRACK_DISTANCE_MODE", "xy").lower()
TRACK_WAYPOINT_PREFIX = "track_waypoint_"
TRACK_HEIGHT_POST_PREFIX = "track_height_post_"

# LiDAR parameters
LIDAR_SITE_NAME = "lidar"

LIDAR_NUM_RAYS = 360
LIDAR_MIN_ANGLE = -np.pi
LIDAR_MAX_ANGLE = np.pi

LIDAR_MIN_RANGE = 0.05
LIDAR_MAX_RANGE = 30.0

LIDAR_UPDATE_INTERVAL = 10

# Calibaration parameters (for sim-to-real consistency)
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
    parser = argparse.ArgumentParser(description="Run S10 MuJoCo ROS2 simulation.")
    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_XML_PATHS),
        default=DEFAULT_SCENE_NAME if DEFAULT_SCENE_NAME in SCENE_XML_PATHS else "track",
        help="Built-in MJCF scene to load. Defaults to S10_MUJOCO_SCENE or 'track'.",
    )
    parser.add_argument(
        "--xml-path",
        default=os.environ.get("S10_MUJOCO_XML"),
        help="Custom MJCF path. Overrides --scene and S10_MUJOCO_SCENE.",
    )
    parser.add_argument("--model-key", default=MODEL_NAME, help="Robot key used for initial joint pose.")
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def resolve_xml_path(scene_name: str, xml_path: str | None) -> str:
    if xml_path:
        return str(Path(xml_path).expanduser().resolve())
    return str(SCENE_XML_PATHS[scene_name].resolve())


class MuJoCoSimulationNode(Node):
    def __init__(self,
                 model_key: str = MODEL_NAME,
                 xml_path: str = XML_PATH):

        super().__init__('mujoco_simulation')

        # 加载 MJCF
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        print(self.model)
        print("Numberof cam",self.model.ncam)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        # 机器人自由度列表
        self.actuator_ids = [a for a in range(self.model.nu)]  # 0..15
        self.dof_num = len(self.actuator_ids)
        assert self.dof_num == 16, "Expected 16 DOF for S10"

        # 初始化站立姿态
        self._set_initial_pose(model_key)
        self._init_track_progress()

        # 缓存
        self.kp_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.kd_cmd = np.zeros_like(self.kp_cmd)
        self.pos_cmd = np.zeros_like(self.kp_cmd)
        self.vel_cmd = np.zeros_like(self.kp_cmd)
        self.tau_ff = np.zeros_like(self.kp_cmd)
        self.input_tq = np.zeros_like(self.kp_cmd)

        # IMU
        self.last_base_linvel = np.zeros((3, 1), np.float64)
        self.timestamp = 0.0

        self.get_logger().info(f"[INFO] MuJoCo MJCF loaded: {xml_path}")
        self.get_logger().info(f"[INFO] MuJoCo model loaded, dof = {self.dof_num}")

        # ROS Publishers
        self.imu_pub = self.create_publisher(ImuData, '/IMU_DATA', 200)
        self.joints_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 200)
        self.lidar_pub = self.create_publisher(
            LaserScan,
            '/LIDAR_DATA',
            10
        )

        # ROS Subscriber
        self.cmd_sub = self.create_subscription(
            JointsDataCmd,
            '/JOINTS_CMD',
            self._cmd_callback,
            50
        )

        self.renderer = mujoco.Renderer(
            self.model,
            height=480,
            width=640,
        )

        # 可视化
        self.viewer = None
        if USE_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._configure_viewer()

        self._camera_ids = {
            "front": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_front_camera"),
            "top":   mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_top_camera"),
            "side":  mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_side_camera"),
   
        }

        for name, cam_id in self._camera_ids.items():
            if cam_id < 0:
                self.get_logger().warn(f"Camera '{name}' not found in model")

        self._id = 0

        self._init_lidar()
        
        self.bridge = CvBridge()

        self.camera_pubs = {
            "front": self.create_publisher(
                Image,
                "/CAMERA/front",
                10
            ),
            "top": self.create_publisher(
                Image,
                "/CAMERA/top",
                10
            ),
            "side": self.create_publisher(
                Image,
                "/CAMERA/side",
                10
            ),
            
            "simulator": self.create_publisher(
                Image,
                "/CAMERA/simulator",
                10
            ),
        }
        
    def _take_simulator_camera_pic(self):
        if self.viewer is None:
            return

        try:
            with self.viewer.lock():

                # Copy the viewer camera settings
                cam = mujoco.MjvCamera()

                cam.type = self.viewer.cam.type
                cam.fixedcamid = self.viewer.cam.fixedcamid
                cam.trackbodyid = self.viewer.cam.trackbodyid

                cam.lookat[:] = self.viewer.cam.lookat
                cam.distance = self.viewer.cam.distance
                cam.azimuth = self.viewer.cam.azimuth
                cam.elevation = self.viewer.cam.elevation

            # Render using the simulator-view camera parameters
            self.renderer.update_scene(
                self.data,
                camera=cam
            )

            frame = self.renderer.render().copy()
            frame = np.asarray(frame, dtype=np.uint8)

            if frame.ndim != 3 or frame.shape[2] != 3:
                self.get_logger().error(
                    f"Invalid simulator camera frame shape: {frame.shape}"
                )
                return

            ros_image = Image()

            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "simulator_camera"

            ros_image.height = int(frame.shape[0])
            ros_image.width = int(frame.shape[1])
            ros_image.encoding = "rgb8"
            ros_image.is_bigendian = 0
            ros_image.step = int(frame.shape[1] * 3)
            ros_image.data = frame.tobytes()

            self.camera_pubs["simulator"].publish(ros_image)

        except Exception as e:
            self.get_logger().error(
                f"Failed to publish simulator camera: "
                f"type={type(e).__name__}, "
                f"repr={repr(e)}"
            )
        
    def _take_robot_camera_pic(self):
        self._take_simulator_camera_pic()
        for name, cam_id in self._camera_ids.items():
            if cam_id < 0:
                continue

            self.renderer.update_scene(
                self.data,
                camera=cam_id
            )

            frame = self.renderer.render().copy()
            frame = np.asarray(frame, dtype=np.uint8)

            if frame.ndim != 3 or frame.shape[2] != 3:
                self.get_logger().error(
                    f"Invalid {name} camera frame shape: {frame.shape}"
                )
                continue

            try:
                ros_image = Image()

                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = name

                ros_image.height = int(frame.shape[0])
                ros_image.width = int(frame.shape[1])
                ros_image.encoding = "rgb8"
                ros_image.is_bigendian = 0
                ros_image.step = int(frame.shape[1] * 3)
                ros_image.data = frame.tobytes()

                self.camera_pubs[name].publish(ros_image)

            except Exception as e:
                self.get_logger().error(
                    f"Failed to publish {name} camera: "
                    f"type={type(e).__name__}, "
                    f"repr={repr(e)}"
                )


    def _init_lidar(self):
        self.lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        if self.lidar_site_id < 0:
            raise RuntimeError("LiDAR site 'lidar' not found in MJCF")

        self.lidar_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        self.lidar_local_dirs = np.column_stack([
            np.cos(self.lidar_angles),
            np.sin(self.lidar_angles),
            np.zeros(360),
        ])
        self.lidar_body_id = self.model.site_bodyid[self.lidar_site_id]

    def _update_lidar(self):
        site_pos = self.data.site_xpos[self.lidar_site_id].astype(np.float64)
        site_mat = self.data.site_xmat[self.lidar_site_id].reshape(3, 3)
        world_dirs = (site_mat @ self.lidar_local_dirs.T).T.astype(np.float64)

        geomid = np.zeros(360, dtype=np.int32)
        dist = np.zeros(360, dtype=np.float64)

        mujoco.mj_multiRay(
            self.model, self.data,
            pnt=site_pos,
            vec=world_dirs.flatten(),
            geomgroup=None,
            flg_static=1,
            bodyexclude=self.lidar_body_id,
            geomid=geomid,
            dist=dist,
            normal=None,
            nray=360,
            cutoff=30.0,
        )

        dist[dist < 0] = 30.0
        self.lidar_ranges = dist.astype(np.float32)
        
    def _publish_lidar(self):
        scan_msg = LaserScan()
        scan_msg.header.frame_id = "lidar"

        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        scan_msg.header.stamp.sec = sec
        scan_msg.header.stamp.nanosec = nanosec

        scan_msg.angle_min = float(self.lidar_angles[0])
        scan_msg.angle_max = float(self.lidar_angles[-1])
        scan_msg.angle_increment = float(2 * np.pi / len(self.lidar_angles))
        scan_msg.range_min = 0.05
        scan_msg.range_max = 30.0
        scan_msg.ranges = self.lidar_ranges.tolist()

        self.lidar_pub.publish(scan_msg)

    def _set_initial_pose(self, key: str):
        """关节位置设置为与 PyBullet 脚本一致的初始角度"""
        qpos0 = self.data.qpos.copy()
        qpos0[7:7 + self.dof_num] = JOINT_INIT[key]  # ,3-6 basequat，0-2 basepos
        qpos0[:3] = TRACK_START_BASE_POS
        qpos0[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)

    def _track_geom_index(self, name: str, prefix: str):
        if not name or not name.startswith(prefix):
            return None
        suffix = name[len(prefix):]
        index_text = suffix.split("_", 1)[0]
        if not index_text.isdigit():
            return None
        return int(index_text)

    def _find_track_geoms(self):
        waypoint_geoms = {}
        point_related_geoms = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            waypoint_index = self._track_geom_index(name, TRACK_WAYPOINT_PREFIX)
            if waypoint_index is not None:
                waypoint_geoms[waypoint_index] = geom_id
                point_related_geoms.setdefault(waypoint_index, []).append(geom_id)
                continue

            post_index = self._track_geom_index(name, TRACK_HEIGHT_POST_PREFIX)
            if post_index is not None:
                point_related_geoms.setdefault(post_index, []).append(geom_id)

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
        missing = [index for index in expected_indices if index not in waypoint_geoms]
        if missing:
            self.get_logger().warn(f"Track progress disabled; missing waypoint geoms: {missing}")
            return

        self.track_waypoint_geom_ids = [waypoint_geoms[index] for index in expected_indices]
        self.track_point_geom_ids = {
            index: point_related_geoms.get(index, [waypoint_geoms[index]])
            for index in expected_indices
        }
        self.track_waypoint_positions = np.array(
            [self.data.geom_xpos[geom_id].copy() for geom_id in self.track_waypoint_geom_ids],
            dtype=np.float64,
        )
        self.track_original_rgba = {
            geom_id: self.model.geom_rgba[geom_id].copy()
            for geom_ids in self.track_point_geom_ids.values()
            for geom_id in geom_ids
        }
        self.track_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY_NAME)
        if self.track_body_id < 0:
            self.get_logger().warn(f"Track progress disabled; cannot find body '{TRACK_BODY_NAME}'")
            return

        self.track_enabled = True
        self.get_logger().info(
            f"[INFO] Track progress enabled: {len(self.track_waypoint_positions)} waypoints, "
            f"radius={TRACK_REACH_RADIUS:.3f}m, distance_mode={TRACK_DISTANCE_MODE}"
        )

    def _hide_track_point(self, waypoint_index: int):
        for geom_id in self.track_point_geom_ids.get(waypoint_index, []):
            self.model.geom_rgba[geom_id, 3] = 0.0

    def _track_distance(self, robot_pos: np.ndarray, waypoint_pos: np.ndarray) -> float:
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
            self.get_logger().info(
                f"[TRACK] Timer started at waypoint 0, sim_time={self.track_start_time:.3f}s"
            )
        else:
            self.get_logger().info(
                f"[TRACK] Reached waypoint {reached_index}, sim_time={self.timestamp:.3f}s, "
                f"distance={distance:.3f}m"
            )

        self.track_next_index += 1
        if self.track_next_index >= len(self.track_waypoint_positions):
            self.track_complete = True
            self.track_finish_time = self.timestamp
            elapsed = 0.0 if self.track_start_time is None else self.track_finish_time - self.track_start_time
            self.get_logger().info(
                f"[TRACK] Final waypoint reached. Timer stopped at sim_time={self.track_finish_time:.3f}s, "
                f"elapsed={elapsed:.3f}s"
            )

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
    

    def _get_camera_image(self):
        with self.viewer.lock():
            # Use exactly the same camera configuration as the viewer
            self.renderer.update_scene(
                self.data,
                camera=self.viewer.cam,
            )

            frame = self.renderer.render().copy()

        return frame

    def _cmd_callback(self, msg: JointsDataCmd):
        """Convert received (published) positions/velocities to internal (raw)"""
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
            self.tau_ff[i] = joint_cmd.torque  # tau_ff no processing

        # Convert: raw = published * dir + offset_rad
        self.pos_cmd.flat = pub_pos * JOINT_DIR + POS_OFFSET_RAD
        self.vel_cmd.flat = pub_vel * JOINT_DIR

    def start(self):
        # 主模拟循环
        step = 0
        last_time = time.time()
        while rclpy.ok():
            if time.time() - last_time >= DT:
                last_time = time.time()
                step += 1
                # 控制律
                self._apply_joint_torque()
                # 模拟一步
                mujoco.mj_step(self.model, self.data)

                self.timestamp = step * DT
                self._update_track_progress()

                # 采样 & 发送观测 (every 5 steps for 200 Hz)
                if step % 5 == 0:
                    self._publish_robot_state(step)
                    #self._take_robot_camera_pic()
                    
                # -------------------------
                # Camera: 10 Hz
                # -------------------------
                if step % 400 == 0:
                    self._take_robot_camera_pic()
                    self._update_lidar()
                    self._publish_lidar()


                # if step % 5000 == 0:
                #     # self._take_robot_camera_pic()
                #     self._update_lidar()
                #     self._publish_lidar()

                # 可视化
                if self.viewer and step % RENDER_INTERVAL == 0:
                    self.viewer.sync()

            # Handle ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.0)

    def _apply_joint_torque(self):
        # 当前关节状态
        q = self.data.qpos[7:7 + self.dof_num].reshape(-1, 1)
        dq = self.data.qvel[6:6 + self.dof_num].reshape(-1, 1)
        self.input_tq = (
                self.kp_cmd * (self.pos_cmd - q) +
                self.kd_cmd * (self.vel_cmd - dq) +
                self.tau_ff
        )

        # 写入 control 缓冲区
        self.data.ctrl[:] = self.input_tq.flatten()

    # --------------------------------------------------------
    def quaternion_to_euler(self, q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).
        """
        w, x, y, z = q

        # roll (X-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        # pitch (Y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # 防止数值漂移导致 |t2|>1
        pitch = np.arcsin(t2)

        # yaw (Z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    # --------------------------------------------------------

    def _publish_robot_state(self, step: int):
        # ----- IMU -----
        q_world = self.data.sensordata[:4]  # quaternion (w, x, y, z) in MuJoCo convention
        rpy_rad = self.quaternion_to_euler(q_world)  # returns [roll, pitch, yaw] in radians

        # Convert to degrees
        rpy_deg = [angle * (180.0 / 3.141592653589793) for angle in rpy_rad]

        body_acc = self.data.sensordata[4:7]
        angvel_b = self.data.sensordata[7:10]  # body frame

        imu_msg = ImuData()
        imu_msg.header = MetaType()
        imu_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        imu_msg.header.stamp = stamp
        imu_msg.data = ImuDataValue()
        imu_msg.data.roll = float(rpy_deg[0])
        imu_msg.data.pitch = float(rpy_deg[1])
        imu_msg.data.yaw = float(rpy_deg[2])
        imu_msg.data.omega_x = float(angvel_b[0])
        imu_msg.data.omega_y = float(angvel_b[1])
        imu_msg.data.omega_z = float(angvel_b[2])
        imu_msg.data.acc_x = float(body_acc[0])
        imu_msg.data.acc_y = float(body_acc[1])
        imu_msg.data.acc_z = float(body_acc[2])
        self.imu_pub.publish(imu_msg)

        # ----- 关节 -----
        q = self.data.qpos[7:7 + self.dof_num]
        dq = self.data.qvel[6:6 + self.dof_num]
        tau = self.input_tq.flatten()

        # Convert raw to published: published = (raw - offset_rad) * dir
        pub_pos = (q - POS_OFFSET_RAD) * JOINT_DIR
        pub_vel = dq * JOINT_DIR
        pub_tau = tau * JOINT_DIR  # Torque also needs direction flip
        
        joints_msg = JointsData()
        joints_msg.header = MetaType()
        joints_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        joints_msg.header.stamp = stamp
        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(self.dof_num)]
        for i in range(self.dof_num):
            joint = joints_msg.data.joints_data[i]
            joint.name = [32, 32, 32, 32]  # Dummy name (four spaces)
            joint.data_id = 0  # Dummy
            joint.status_word = 1  # Normal
            joint.position = float(pub_pos[i])
            joint.torque = float(pub_tau[i])
            joint.velocity = float(pub_vel[i])
            joint.motion_temp = 40.0  # Dummy normal temp
            joint.driver_temp = 45.0  # Dummy normal temp
        self.joints_pub.publish(joints_msg)

      #  self._take_robot_camera_pic()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    cli_args, ros_args = parse_cli_args()
    rclpy.init(args=ros_args)
    sim_node = MuJoCoSimulationNode(
        model_key=cli_args.model_key,
        xml_path=resolve_xml_path(cli_args.scene, cli_args.xml_path),
    )
    sim_node.start()
    sim_node.destroy_node()
    rclpy.shutdown()
