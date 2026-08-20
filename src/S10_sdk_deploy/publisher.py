#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from drdds.msg import JointsDataCmd, JointDataCmd, ImuData, JointsData


class JointCommandPublisher(Node):

    def __init__(self):
        super().__init__("joint_command_publisher")

        # Publish to /JOINTS_CMD
        self.publisher = self.create_publisher(
            JointsDataCmd,
            "/JOINTS_CMD",
            50
        )

        self.latest_joint_data = None
        self.latest_imu_data = None

        self.join_data_sub = self.create_subscription(
            JointsData,
            "/JOINTS_DATA",
            self.join_data_callback,
            200
        )

        self.imu_sub = self.create_subscription(
            ImuData,
            "/IMU_DATA",
            self.imu_sub_callback,
            200
        )

        # Publish every 0.01 seconds (100 Hz)
        self.timer = self.create_timer(5.0, self.publish_command)

    def imu_sub_callback(self, imu_data):
        self.latest_imu_data = imu_data

        # 'Imu data: drdds.msg.ImuData(
        # header=drdds.msg.MetaType(
        # frame_id=0, 
        # stamp=builtin_interfaces.msg.Time(sec=220, nanosec=950000000)),
        #  data=drdds.msg.ImuDataValue(
        # roll=-13.644125938415527, 
        # pitch=-0.00021894088422413915, 
        # yaw=-0.6242533922195435,
        #  omega_x=-1.659954023125465e-07,
        #  omega_y=-4.05989997176448e-09, 
        # omega_z=4.286613375370507e-08, 
        # acc_x=3.8243961171247065e-05, 
        # acc_y=-2.3140954971313477, 
        # acc_z=9.53315544128418))

    def join_data_callback(self, join_data):
        self.latest_joint_data = join_data

    def publish_command(self):

        msg = JointsDataCmd()

        # Create commands for 16 joints
        msg.data.joints_data = []

        for i in range(16):

            joint = JointDataCmd()

            joint.kp = 30.0
            joint.kd = 1.0
            joint.position = 0.0
            joint.velocity = 0.0
            joint.torque = 0.0

            msg.data.joints_data.append(joint)

        # self.publisher.publish(msg)

        # self.get_logger().info("Published Joint Command")

        print(f"Imu data: {self.latest_imu_data}")
        print(f"Joint data: {self.latest_joint_data}")



def main():

    rclpy.init()

    node = JointCommandPublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()