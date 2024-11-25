#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point
import actionlib
import gridbased_dmpc.transformations as u_tf
from gridbased_dmpc.msg import AngleTargetPointAction, AngleTargetPointGoal, OSCSTargetAction, OSCSTargetGoal
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import LaserScan
import time
from threading import Semaphore

# We create some constants with the corresponding values from the SimpleGoalState class
PENDING = 0
ACTIVE = 1
PREEMPTED = 2
SUCCEEDED = 3
ERROR = 4


class LocalPlanner(object):

    def __init__(self, action_server_name, tb3_id, number_tb3s, increment, inflate_radius, min_distance):
        self.tb3_id = tb3_id
        self.neighbors = set(range(1, number_tb3s + 1))
        self.neighbors.difference_update({tb3_id})
        self.action_server_name = action_server_name
        self.client = actionlib.SimpleActionClient(action_server_name, AngleTargetPointAction)
        self.current_p = []
        self.current_p_neighbors = {i: [] for i in self.neighbors}
        self.current_path = []
        self.current_waypoint = []
        self.R = increment * 0.05
        self.min_distance = min_distance
        self.inflate_radius = inflate_radius
        self.got_scan = False
        self.angles = []
        self.point_clouds = set()
        self.yaw = 0
        self.plot_oscs = actionlib.SimpleActionClient("/oscs_plot_service", OSCSTargetAction)
        self.semaphore_scan = Semaphore(1)
        self.admissible_angles = np.round(np.array([0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0])*np.pi/180, 3)
        rospy.Subscriber("global_planner/path", Path, self.path_callback)
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        rospy.Subscriber("scan", LaserScan, self.scan_callback)
        self.path_pub = rospy.Publisher("local_planner/path", Path, queue_size=2)
        for i in self.neighbors:
            rospy.Subscriber("/tb3_" + str(i) + "/odom", Odometry, self.odom_neighbor_callback, i)

    def path_callback(self, msg):
        self.current_path = []
        for i in range(0, len(msg.poses)):
            p = [msg.poses[i].pose.position.x, msg.poses[i].pose.position.y, msg.poses[i].pose.position.z]
            self.current_path.append(p)

    def scan_callback(self, msg):
        if not self.got_scan:
            self.got_scan = True
            angles_range = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
            for i in range(0, len(angles_range)):
                self.angles.append(np.array([np.cos(angles_range[i]), np.sin(angles_range[i])]))

        R_z = u_tf.rot_z(self.yaw)
        with self.semaphore_scan:
            self.point_clouds = set()
            for i in range(0, len(self.angles)):
                if  msg.ranges[i] == float('inf'):
                    continue
                pc = msg.ranges[i] * R_z @ self.angles[i] + self.current_p
                self.point_clouds.add((pc[0], pc[1]))

    @staticmethod
    def feedback_callback(feedback):
        rospy.loginfo(feedback.state)

    def publish_oscs(self, angle, p):
        goal = OSCSTargetGoal()
        goal.point = Point(p[0], p[1], 0.07)
        goal.angle = angle
        goal.tb3_id = self.tb3_id
        self.plot_oscs.send_goal(goal)

    def publish_path(self, x):
        if len(x) < 1:
            return

        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()

        path_msg = Path()
        path_msg.header = header
        for i in range(0, len(x)):
            pose_stamped = PoseStamped()

            pose_stamped.pose.position.x = x[i][0]
            pose_stamped.pose.position.y = x[i][1]
            pose_stamped.pose.position.z = x[i][2]

            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

    def direction_distance(self, angle):
        # Normalize angle within [0, 2*pi]
        angle = angle % (2 * np.pi)

        # Check if angle is a multiple of pi/2
        if np.isclose(angle % (np.pi / 2), 0):
            return self.R * 1.1
        return np.sqrt(2) * self.R * 1.15

    def closest_angle(self, angle):
        return min(self.admissible_angles, key=lambda x: abs(x - angle))

    def send_goal(self, angle):
        goal = AngleTargetPointGoal()
        goal.angle = angle
        goal.waypoint = Vector3(self.current_waypoint[0], self.current_waypoint[1], self.current_waypoint[2])
        self.client.send_goal(goal)
        self.publish_oscs(angle, self.current_waypoint)

    def is_waypoint_viable(self, waypoint):
        for neighbor in self.neighbors:
            if np.linalg.norm(np.array(self.current_p_neighbors[neighbor]) - np.array(waypoint[0:2])) <= self.inflate_radius:
                return False
        with self.semaphore_scan:
            for p in self.point_clouds:
                if np.linalg.norm(np.array(p) - np.array(waypoint[0:2])) <= self.inflate_radius:
                    return False
        return True

    def select_waypoint(self):
        start_time = time.time()
        path = np.array(self.current_path)
        distances = np.linalg.norm(path - np.array(self.current_waypoint), axis=1)
        idx = np.argmin(distances)

        idx_next = min(idx + 1, path.shape[0] - 1)
        next_point = self.current_path[idx_next]

        if np.linalg.norm(next_point[0:2] - np.array(self.current_p)) < self.min_distance:
            self.current_path = []
            return

        points = [self.current_waypoint, next_point]
        angle = round(np.arctan2(next_point[1] - self.current_waypoint[1], next_point[0] - self.current_waypoint[0]), 3)

        new_selected_waypoint = False
        if self.is_waypoint_viable(next_point) and np.linalg.norm(next_point[0:2] - np.array(self.current_p)) <= self.direction_distance(angle):
            self.current_waypoint = next_point
            new_selected_waypoint = True

        end_time = time.time() - start_time
        rospy.loginfo('Waypoint selected in t = %.4f sec' % end_time)

        if new_selected_waypoint:
            if angle not in self.admissible_angles:
                rospy.logwarn("Found a non admissible direction, it is replaced with the closest admissible one")
                angle = self.closest_angle(angle)
            self.send_goal(angle)
        self.publish_path(points)

    def odom_callback(self, msg):
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.current_p = p
        self.yaw = u_tf.quaternion_to_yaw(msg.pose.pose.orientation)

    def odom_neighbor_callback(self, msg, neighbor):
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.current_p_neighbors[neighbor] = p

def classify_angle(angle):
    # Normalize the angle between 0 and 2*pi
    angle = angle % (2 * np.pi)

    # Define arrays of right and inclined angles
    right_angles = np.array([k * np.pi / 2 for k in range(4)])
    inclined_angles = np.array([np.pi / 4 + k * np.pi / 2 for k in range(4)])

    # Calculate the minimum distance to each category
    min_dist_right = np.min(np.abs(angle - right_angles))
    min_dist_inclined = np.min(np.abs(angle - inclined_angles))
    return 0 if min_dist_right < min_dist_inclined else np.pi / 4

def main():
    rospy.init_node("local_planner_node", anonymous=True)
    tb3_id = rospy.get_param('~tb3_id')
    Ts = rospy.get_param('~Ts_robot')
    rate = rospy.Rate(1 / Ts)

    number_tb3s = rospy.get_param('~number_tb3s')
    increment = rospy.get_param('~increment')
    min_distance = rospy.get_param('~min_distance')
    inflate_radius = rospy.get_param('~inflate_radius')
    rospy.wait_for_message("odom", Odometry)

    action_server_name = "local_gb_mpc"
    node = LocalPlanner(action_server_name, tb3_id, number_tb3s, increment, inflate_radius, min_distance)

    rospy.loginfo('Waiting for Local Grid Based MPC ' + action_server_name)
    node.client.wait_for_server()
    rospy.loginfo('Local Grid Based MPC found...' + action_server_name)

    # Send the first waypoint
    rospy.wait_for_message("global_planner/path", Path)
    node.current_waypoint = node.current_path[0]
    angle = classify_angle(np.arctan2(node.current_waypoint[1] - node.current_p[1], node.current_waypoint[0] - node.current_p[0]))
    node.send_goal(angle)


    while not rospy.is_shutdown():
        if len(node.current_path):
            node.select_waypoint()

        rate.sleep()


if __name__ == '__main__':
    main()
