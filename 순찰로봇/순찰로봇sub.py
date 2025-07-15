#!/usr/bin/env python3

import rclpy
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator

# ======================
# 초기 설정 (파일 안에서 직접 정의)
# ======================
INITIAL_POSE_POSITION = [-0.2612, -0.0462]
INITIAL_POSE_DIRECTION = 0.0

GOAL_POSES = [
    ([-0.47, 1.92], 0.0),
    ([-0.5109, 0.5069], 0.0),
    ([-2.7932, 0.5308], 0.0),
    ([-1.79, 1.92], 0.0),
    ([-3.03, 2.01], 0.0),
    ([-2.98, -0.12], 270.0),
    ([-2.7094, 0.6813], 270.0),
    ([-2.1615, 0.1634], 180.0),
    ([-0.60, 0.06], 0.0),
]
# ======================

def main():
    rclpy.init()
    navigator = TurtleBot4Navigator()

    if not navigator.getDockedStatus():
        navigator.info('Docking before initializing pose')
        navigator.dock()

    initial_pose = navigator.getPoseStamped(INITIAL_POSE_POSITION, INITIAL_POSE_DIRECTION)
    navigator.setInitialPose(initial_pose)

    navigator.waitUntilNav2Active()

    navigator.undock()

    goal_pose_msgs = [navigator.getPoseStamped(position, direction) for position, direction in GOAL_POSES]
    navigator.startFollowWaypoints(goal_pose_msgs)

    navigator.dock()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
