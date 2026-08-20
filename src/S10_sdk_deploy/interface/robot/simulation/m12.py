
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import threading
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

class CollectRobotInformation(Node):

    def __init__(self):
        super().__init__('camera_viewer')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/CAMERA/front',
            self.camera_callback,
            10
        )

        self.top_subscription = self.create_subscription(
            Image,
            '/CAMERA/top',
            self.camera_top_callback,
            10
        )
        
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/LIDAR_DATA',
            self.lidar_data_callback,
            10
        )
        
        self.side_subscription = self.create_subscription(
            Image,
            '/CAMERA/side',
            self.camera_side_callback,
            10
        )
        
        
        self.side_subscription = self.create_subscription(
            Image,
            '/CAMERA/simulator',
            self.camera_simulator_callback,
            10
        )
        
        self.publisher_ = self.create_publisher(String, '/ROBOT_CMD', 10)

        self.get_logger().info(
            'Waiting for camera frames...'
        )
        
        # self.input_thread = threading.Thread( target=self.publish_char, daemon=True )
        # self.input_thread.start()
        
    def lidar_data_callback(self, data):

        # =========================================================
        # Get LiDAR ranges
        # =========================================================

        ranges = np.asarray(
            data.ranges,
            dtype=np.float64
        )

        # Replace invalid values
        ranges[~np.isfinite(ranges)] = data.range_max

        # Clamp to valid LiDAR range
        ranges = np.clip(
            ranges,
            data.range_min,
            data.range_max
        )

        # =========================================================
        # Calculate angle for every LiDAR ray
        # =========================================================

        angles = (
            data.angle_min
            + np.arange(len(ranges))
            * data.angle_increment
        )

        angles_deg = np.rad2deg(angles)

        # =========================================================
        # Helper: minimum distance in an angular sector
        # =========================================================

        def get_sector_distance(min_angle, max_angle):

            mask = (
                (angles_deg >= min_angle)
                &
                (angles_deg <= max_angle)
            )

            sector_ranges = ranges[mask]

            if len(sector_ranges) == 0:
                return data.range_max

            return float(
                np.min(sector_ranges)
            )

        # =========================================================
        # FRONT
        # -30° to +30°
        # =========================================================

        front_distance = get_sector_distance(
            -30.0,
            30.0
        )

        # =========================================================
        # LEFT
        # +30° to +150°
        # =========================================================

        left_distance = get_sector_distance(
            30.0,
            150.0
        )

        # =========================================================
        # RIGHT
        # -150° to -30°
        # =========================================================

        right_distance = get_sector_distance(
            -150.0,
            -30.0
        )

        # =========================================================
        # REAR
        # +150° to +180°
        # -180° to -150°
        # =========================================================

        rear_mask = (
            (angles_deg >= 150.0)
            |
            (angles_deg <= -150.0)
        )

        rear_ranges = ranges[rear_mask]

        if len(rear_ranges) > 0:
            rear_distance = float(
                np.min(rear_ranges)
            )
        else:
            rear_distance = data.range_max

        # =========================================================
        # CLOSEST OBJECT
        # =========================================================

        closest_distance = float(
            np.min(ranges)
        )

        # =========================================================
        # Store distances
        # =========================================================

        distances = {
            "front": front_distance,
            "left": left_distance,
            "right": right_distance,
            "rear": rear_distance,
            "closest": closest_distance,
        }

        # =========================================================
        # Print
        # =========================================================

        print(
            f"[LIDAR] "
            f"Front: {front_distance:.3f} m | "
            f"Left: {left_distance:.3f} m | "
            f"Right: {right_distance:.3f} m | "
            f"Rear: {rear_distance:.3f} m | "
            f"Closest: {closest_distance:.3f} m"
        )


    def publish_char(self):
        # TODO: Control the robot  
        while True:
            command = input("Enter command: ")
            msg = String()
            msg.data = command
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published: '{command}'")

    def _camera_callback(self, msg, name):

            # Convert ROS Image -> OpenCV image
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='rgb8'
            )

            # OpenCV uses BGR
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            # Display image
            cv2.imshow(
                name,
                frame
            )

            # Required for OpenCV window to update
            cv2.waitKey(1)

    def camera_simulator_callback(self, msg):
        self._camera_callback(msg, 'S10 similliator Camera')

    def camera_side_callback(self, msg): 
        self._camera_callback(msg, 'S10 Side Camera')

    def camera_top_callback(self, msg):
        self._camera_callback(msg, 'S10 Top Camera')

    def camera_callback(self, msg):
        self._camera_callback(msg, 'S10 Front Camera')


def main():

    rclpy.init()

    node = CollectRobotInformation()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

