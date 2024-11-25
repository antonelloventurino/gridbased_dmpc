#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import numpy as np


class FakeScheduler(object):

    def __init__(self, number_tb3s):
        self.number_tb3s = number_tb3s
        self.data_saved = False
        self.p0 = {i: [] for i in range(0, number_tb3s)}
        self.goal_pub = []
        self.goals = {i: {} for i in range(0, number_tb3s)}
        self.last_goals = {i: [] for i in range(0, number_tb3s)}
        self.goals[0] = {18: [-1.5, -9.5], 100: [8, -8], 180: [-4.5, 11]}
        self.goals[1] = {18: [8, -8], 110: [-4.5, 11], 234: [-1.5, -9.5]}
        self.goals[2] = {18: [-4.5, 11], 90: [-1.5, -9.5], 240: [8, -8]}

        for i in range(1, number_tb3s + 1):
            rospy.Subscriber("/tb3_" + str(i) + "/odom", Odometry, self.odom_callback, i - 1)
            rospy.Subscriber("/tb3_" + str(i) + "/global_planner/path", Path, self.path_callback, i - 1)
            self.goal_pub.append(rospy.Publisher("/tb3_" + str(i) + "/goal", PoseStamped, queue_size=2))

    def path_callback(self, msg, tb3_id):
        self.last_goals[tb3_id] = [msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y]

    def send_goal(self, p, tb3_id):
        goal = PoseStamped()
        goal.pose.position.x = p[0]
        goal.pose.position.y = p[1]

        self.goal_pub[tb3_id].publish(goal)

    def odom_callback(self, msg, tb3_id):
        self.p0[tb3_id] = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    def run(self):
        now = rospy.get_rostime()
        for tb3_id in range(0, self.number_tb3s):
            times = list(self.goals[tb3_id].keys())
            if len(times) < 1:
                continue
            if now.secs >= times[0]:
                self.send_goal(self.goals[tb3_id][times[0]], tb3_id)
                del self.goals[tb3_id][times[0]]


def main():
    rospy.init_node("scheduler_node", anonymous=True)
    fs_scheduler = rospy.get_param('~fs_scheduler')
    rate = rospy.Rate(fs_scheduler)
    number_tb3s = rospy.get_param('~number_tb3s')
    node = FakeScheduler(number_tb3s)

    while not rospy.is_shutdown():
        node.run()
        rate.sleep()


if __name__ == '__main__':
    main()
