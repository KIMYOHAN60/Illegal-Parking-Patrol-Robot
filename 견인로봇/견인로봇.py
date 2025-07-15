#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import datetime
import os

marker_topic_name = '/robot2/detected_object_position'


class TurtleBotMarkerFollower(Node):
    def __init__(self):
        super().__init__('turtlebot_marker_follower')

        self.navigator = TurtleBot4Navigator(namespace='/robot3')

        self.subscription = self.create_subscription(
            PointStamped,
            marker_topic_name,
            self.marker_callback,
            10
        )

        # 이미지 관련 초기화
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # 실제 카메라 토픽에 맞게 수정하세요
            self.image_callback,
            10
        )

        self.received_marker = False
        self.marker_position = None

        # 현재 저장 위치 출력
        self.get_logger().info(f"📂 Current working directory: {os.getcwd()}")

    def marker_callback(self, msg: PointStamped):
        if not self.received_marker:
            x = msg.point.x
            y = msg.point.y
            z = msg.point.z
            self.get_logger().info(f"Received Marker at x={x:.2f}, y={y:.2f}, z={z:.2f}")

            self.marker_position = [x, y]
            self.received_marker = True

    def image_callback(self, msg: Image):
        self.latest_image = msg

    def take_picture_with_context(self, position):
        if self.latest_image is not None:
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                x_str = f"{position[0]:.2f}".replace('.', '_').replace('-', 'm')
                y_str = f"{position[1]:.2f}".replace('.', '_').replace('-', 'm')

                filename = f"photo_{timestamp}_x{x_str}_y{y_str}.jpg"

                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
                cv2.imwrite(filename, cv_image)

                self.get_logger().info(f"✅ Image saved to {filename}")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to save image: {e}")
        else:
            self.get_logger().warn("⚠ No image received yet.")

    def wait_for_marker_and_navigate(self):
        self.get_logger().info("Waiting for Marker message...")

        while not self.received_marker:
            rclpy.spin_once(self)

        if not self.navigator.getDockedStatus():
            self.navigator.info('Docking before initializing pose')
            self.navigator.dock()

        initial_pose = self.navigator.getPoseStamped([-3.0, -3.65], 270.0)
        self.navigator.setInitialPose(initial_pose)

        self.navigator.waitUntilNav2Active()
        self.navigator.undock()

        # 첫 번째 목적지: 마커 좌표
        goal_pose_1 = self.navigator.getPoseStamped(self.marker_position, 0.0)
        self.navigator.startToPose(goal_pose_1)

        self.get_logger().info("Reached marker. Taking picture...")
        time.sleep(1.0)  # 이미지 안정화 대기
        self.take_picture_with_context(self.marker_position)

        self.get_logger().info("Proceeding to (-4.7, -3.3).")

        time.sleep(5)

        # 두 번째 목적지: (-4.7, -3.3)
        goal_pose_2 = self.navigator.getPoseStamped([-4.7, -3.3], 270.0)
        self.navigator.startToPose(goal_pose_2)

        self.get_logger().info("Navigation complete.")


def main():
    rclpy.init()
    node = TurtleBotMarkerFollower()

    try:
        node.wait_for_marker_and_navigate()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
