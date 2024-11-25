#!/usr/bin/env python3
import rospy
import actionlib
import numpy as np
import gridbased_dmpc.transformations as u_tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gridbased_dmpc.msg import AngleTargetPointFeedback, AngleTargetPointResult, AngleTargetPointAction
from gridbased_dmpc.srv import OSCSList
import cvxpy as cp


class LocalGridBasedMPC(object):
    _feedback = AngleTargetPointFeedback()
    _result = AngleTargetPointResult()

    def __init__(self, name, tb3_id, A, B, min_distance, Ts, b, Hu, u_bounds_max=None, x_bounds_max=None):
        self.id = tb3_id
        self.angle = None
        self._action_name = name
        if x_bounds_max is None:
            x_bounds_max = []
        if u_bounds_max is None:
            u_bounds_max = []
        self.sampling_frequency = None
        self.A = A
        self.B = B
        self.b = b
        self.n_x = A.shape[0]
        self.n_u = B.shape[1]
        self.u_prev = np.zeros((self.n_u, 1))
        self.min_distance = min_distance
        self.Hu = Hu
        self.x0 = np.array([[0.0, 0.0, 0.0]]).T
        self.u_bounds_max = u_bounds_max
        self.x_bounds_max = x_bounds_max
        self.Ts = Ts  # [s]
        self.waypoint = np.zeros((self.n_x, 1))
        self.K = None
        self.to_compute = False
        self._as = actionlib.SimpleActionServer(self._action_name, AngleTargetPointAction, execute_cb=self.goal_callback,
                                                auto_start=False)
        self.oscs_already_computed = {}  # key = angle, values = [Q_0, Q_1, ..., Q_n]
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.get_odom)
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=5)
        rospy.wait_for_service("/oscs_service", timeout=60)
        oscs_service = rospy.ServiceProxy('/oscs_service', OSCSList)
        res = oscs_service()
        for i in range(0, res.n):
            angle = round(res.oscs[i].angle, 3)
            self.oscs_already_computed[angle] = []
            for j in range(0, len(res.oscs[i].oscs)):
                self.oscs_already_computed[angle].append(np.array(res.oscs[i].oscs[j].Q).reshape((self.n_x, self.n_x)))

        self._as.start()
        rospy.loginfo("TB3_" + str(self.id) + " Server On")

    def get_odom(self, msg):
        self.x0[0, 0] = msg.pose.pose.position.x
        self.x0[1, 0] = msg.pose.pose.position.y
        self.x0[2, 0] = u_tf.quaternion_to_yaw(msg.pose.pose.orientation)

    def get_current_one_step_controllable_set(self):
        idx = 0
        xT = self.x0[0:2, [0]].T - self.waypoint[0:2, [0]].T
        x = self.x0[0:2, [0]] - self.waypoint[0:2, [0]]

        for i in range(0, len(self.oscs_already_computed[self.angle])):
            if xT @ np.linalg.inv(self.oscs_already_computed[self.angle][i][0:self.n_x, 0:self.n_x]) @ x <= 1:
                idx = i
                break

        return idx

    def compute_online_control_law(self):
        theta = self.x0[2, 0]

        U = cp.Variable(shape=(self.n_u, 1))
        gamma = cp.Variable(shape=(1, 1))
        idx_Q = self.get_current_one_step_controllable_set()
        Q11 = self.oscs_already_computed[self.angle][idx_Q][0:self.n_x, 0:self.n_x]

        T_FL = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta) / self.b, np.cos(theta) / self.b]])
        H_theta = self.Hu @ T_FL

        matrix1 = cp.bmat(H_theta @ U[:, [0]])
        x = self.A @ self.x0[0:2, [0]] + self.B @ U[:, [0]] - self.waypoint[0:2, [0]]

        matrix2 = cp.bmat([[gamma, x.T], [x, Q11]])

        constraints = [gamma >= np.finfo(np.float32).eps, gamma <= 1, matrix1 <= np.ones((self.n_u*2, 1)), matrix2 >> 0]

        opt_prob = cp.Problem(cp.Minimize(gamma), constraints)
        opt_prob.solve(verbose=False, solver=cp.SCS)

        u = U[0:self.n_u, 0].value
        if u is None:
            rospy.logwarn("u is None")
            u = 0 * self.u_prev
        else:
            self.u_prev = u
        return T_FL @ u

    def compute_control_input(self):
        if not self.to_compute:
            return [0, 0]

        theta = self.x0[2, 0]
        T_FL = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta) / self.b, np.cos(theta) / self.b]])
        diff = np.array([[self.x0[0, 0] - self.waypoint[0, 0]], [self.x0[1, 0] - self.waypoint[1, 0]]])
        u = T_FL @ self.K @ diff

        return u

    def goal_callback(self, goal):
        self.to_compute = True

        self.sampling_frequency = rospy.Rate(1 / self.Ts)

        self.waypoint[0, 0] = goal.waypoint.x
        self.waypoint[1, 0] = goal.waypoint.y

        self.angle = round(goal.angle, 3)
        twist = Twist()
        while np.linalg.norm(self.x0[0:2] - self.waypoint[0:2, [0]]) > self.min_distance:
            if self._as.is_preempt_requested():
                rospy.loginfo('The goal has been cancelled/preempted')
                self._as.set_preempted()
                return

            distance = np.linalg.norm(self.x0[0:2] - self.waypoint[0:2, [0]])
            self._feedback.state = "Distance from goal = %s" % distance
            self._as.publish_feedback(self._feedback)

            u = self.compute_online_control_law()
            twist.linear.x = u[0]
            twist.angular.z = u[1]
            self.cmd_pub.publish(twist)

            self.sampling_frequency.sleep()

        distance = np.linalg.norm(self.x0[0:2] - self.waypoint[0:2, [0]])
        self._feedback.state = "Last distance from goal = %s" % distance
        self._as.publish_feedback(self._feedback)

        #if success:
        twist.linear.x = 0
        twist.angular.z = 0
        self.cmd_pub.publish(twist)
        self._result = 0
        self.to_compute = False
        self._as.set_succeeded(self._result)


if __name__ == '__main__':
    rospy.init_node("local_gb_mpc")
    _tb3_id = rospy.get_param('~tb3_id')
    _min_distance = rospy.get_param('~min_distance')
    _Ts = rospy.get_param('~Ts_robot')
    v_max = rospy.get_param('~v_max')
    w_max = rospy.get_param('~w_max')
    _b = rospy.get_param('~increment') * rospy.get_param('~map_resolution')
    _Hu = np.array([[-1/v_max, 0], [1/v_max, 0], [0, 1/w_max], [0, -1/w_max]])
    _A = np.identity(2)
    _B = _Ts * np.identity(2)

    LocalGridBasedMPC(rospy.get_name(), _tb3_id, _A, _B, _min_distance, _Ts, _b, _Hu)
    rospy.spin()
