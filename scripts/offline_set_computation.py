#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
import actionlib
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from gridbased_dmpc.srv import EdgesList, EdgesListResponse, OSCSList, OSCSListResponse
from gridbased_dmpc.msg import OSCS, QEllipse, OSCSTargetAction, OSCSTargetResult
from nav_msgs.msg import OccupancyGrid
import time


class OfflineSetComputation(object):
    _oscs_result = OSCSTargetResult()

    def __init__(self, increment, min_distance, Ts, b, A, B, R, Q, u_bounds_max=None, x_bounds_max=None):
        self.increment = increment
        self.min_distance = min_distance
        self.b = b
        self.A = A
        self.B = B
        self.R = R
        self.Q = Q
        self.u_bounds_max = u_bounds_max
        self.x_bounds_max = x_bounds_max
        self.Ts = Ts  # [s]
        self.diff = np.array([[0.0, 0.0, 0.0]]).T
        self.mpc_computed = False
        self.K = None
        self.pi_set_already_computed = {}  # key = angle, values = [K, Q]
        self.oscs_already_computed = {}  # key = angle, values = [Q_0, Q_1, ..., Q_n]
        self.last_published_ids = {}  # key = namespace, values = [id_0, id_1, ..., id_n]
        self.marker_pub = rospy.Publisher('ellipsoids', MarkerArray, queue_size=10)
        self.resolution = None
        self.origin = None
        rospy.Subscriber("global_costmap", OccupancyGrid, self.costmap_callback)
        rospy.Service('set_computation_service', EdgesList, self.compute_offline_sets)
        self._as_oscs = actionlib.SimpleActionServer('oscs_plot_service', OSCSTargetAction, execute_cb=self.plot_oscs,
                                                     auto_start=False)
        self._as_oscs.start()

    def _plot_markers(self, Qs, points, alpha=0.5, red=0.86, green=0.86, blue=0.86, ns="pi", scale=0.03, tb3_id=None):
        namespace = ns
        if tb3_id is not None:
            namespace += "_" + str(tb3_id)
        if namespace in self.last_published_ids.keys():
            marker_array = MarkerArray()
            for i in range(0, len(self.last_published_ids[namespace])):
                marker = Marker()
                marker.id = self.last_published_ids[namespace][i]
                marker.ns = namespace
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)
            self.marker_pub.publish(marker_array)

        self.last_published_ids[namespace] = []

        marker_array = MarkerArray()
        transition = np.linspace(0, 1, points.shape[0])
        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = transition
        colors[:, 1] = 0.0
        colors[:, 2] = 1.0

        for i in range(0, points.shape[0]):
            self.last_published_ids[namespace].append(i)

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace
            marker.id = i

            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = scale
            marker.color.a = alpha
            if ns == "oscs":
                marker.color.r = colors[i, 0]
                marker.color.g = colors[i, 1]
                marker.color.b = colors[i, 2]
            else:
                marker.color.r = red
                if tb3_id == 1:
                    marker.color.r = 1.0
                else:
                    marker.color.r = 0.0

                marker.color.b = blue
                if tb3_id == 2:
                    marker.color.b = 1.0
                else:
                    marker.color.b = 0.0

                marker.color.g = green

            num_points = 20
            theta = np.linspace(0, 2 * np.pi, num_points)
            ellipse_points = np.column_stack((np.cos(theta), np.sin(theta)))
            ellipse_points = np.dot(ellipse_points, sqrtm(Qs[i]))
            ellipse_points = np.column_stack((ellipse_points[:, 0] + points[i, 0], ellipse_points[:, 1] + points[i, 1]))

            for point in ellipse_points:
                point_msg = Point()
                point_msg.x = point[0]
                point_msg.y = point[1]
                point_msg.z = 0.1
                marker.points.append(point_msg)
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def costmap_callback(self, msg):
        magnitude = np.floor(np.log10(np.abs(msg.info.resolution)))
        factor = 10 ** (magnitude - 2)
        self.resolution = np.round(msg.info.resolution / factor) * factor
        self.origin = msg.info.origin

    def plot_oscs(self, req):
        angle = round(req.angle, 3)
        n_ell = len(self.oscs_already_computed[angle])
        p = np.repeat(np.array([[req.point.x, req.point.y, req.point.z]]), n_ell, axis=0)

        Qs = []
        for i in range(0, n_ell):
            Qs.append(self.oscs_already_computed[angle][i])

        self._plot_markers(Qs, p, tb3_id=req.tb3_id, ns="oscs", scale=0.02)
        self._as_oscs.set_succeeded(self._oscs_result)

    @staticmethod
    def get_ellipsoidal_constraints(QQ, P):
        n_p = P.shape[0]
        rank_P = np.linalg.matrix_rank(P)
        _, Sigma, Vt = np.linalg.svd(P)

        Sigma_matrix = np.diag(Sigma[0:rank_P])
        selector_matrix = np.block([np.identity(rank_P), np.zeros((rank_P, n_p - rank_P))])

        return cp.bmat(selector_matrix @ Vt.T @ QQ @ Vt @ selector_matrix.T) << np.linalg.inv(Sigma_matrix)

    def compute_one_step_controllable_sets(self, angle, current_Q):
        nx = self.A.shape[0]

        singular_values = np.linalg.svd(current_Q, compute_uv=False)
        radius = np.max(singular_values)

        self.oscs_already_computed[angle] = []
        idx = 0
        while True:
            radius = radius + self.Ts * self.u_bounds_max
            Q_shaping = np.identity(nx) * radius ** 2

            idx += 1
            self.oscs_already_computed[angle].append(Q_shaping)
            if self.diff.T @ np.linalg.inv(Q_shaping) @ self.diff <= 1:
                break

    def send_oscs(self, _):
        oscs_list = []
        for angle, list_Q in self.oscs_already_computed.items():
            oscs = OSCS()
            oscs.angle = round(angle, 3)
            oscs.oscs = []
            for QQ in list_Q:
                oscs.oscs.append(QEllipse(QQ.flatten().tolist()))
            oscs_list.append(oscs)

        return OSCSListResponse(len(self.oscs_already_computed), oscs_list)

    def compute_offline_sets(self, req):
        start_time = time.time()
        Qs = []
        for i in range(0, len(req.edges)):
            vi = (req.edges[i].vix, req.edges[i].viy)
            vf = (req.edges[i].vfx, req.edges[i].vfy)

            angle = np.arctan2(vf[1] - vi[1], vf[0] - vi[0])
            if angle not in self.pi_set_already_computed:
                self.diff = np.array([[vi[0] - vf[0]], [vi[1] - vf[1]]]) * self.resolution * self.increment * 1.5
                # R_epsilon equal to min_distance
                p_epsilon_R = np.array([[vi[0] - vf[0]], [vi[1] - vf[1]]]) * self.min_distance

                nx = self.A.shape[0]
                nu = self.B.shape[1]

                Q = cp.Variable((nx, nx))
                Y = cp.Variable((nu, nx))
                gamma = cp.Variable()

                constraints = [Q >> 0, cp.PSD(Q)]

                zeros_nx = np.zeros((nx, nx))
                zeros_nx_nu = np.zeros((nx, nu))
                zeros_nu_nx = zeros_nx_nu.T 

                constraints.append(cp.bmat([[Q, (self.A @ Q + self.B @ Y).T],
                                            [self.A @ Q + self.B @ Y, Q]]) >> 0)

                constraints.append(
                    cp.bmat([[Q, (self.A @ Q + self.B @ Y).T, (sqrtm(self.Q) @ Q).T, (sqrtm(self.R) @ Y).T],
                             [self.A @ Q + self.B @ Y, Q, zeros_nx, zeros_nx_nu],
                             [sqrtm(self.Q) @ Q, zeros_nx, gamma * np.identity(nx), zeros_nx_nu],
                             [sqrtm(self.R) @ Y, zeros_nu_nx, zeros_nu_nx, gamma * np.identity(nu)]]) >> 0)

                Blk1 = np.hstack(([[1]], p_epsilon_R.T))
                Blk2 = cp.bmat([[p_epsilon_R, Q]])

                constraints.append(cp.bmat([[Blk1], [Blk2]]) >> 0)

                if self.u_bounds_max.size == 1:
                    constraints.append(cp.bmat([[self.u_bounds_max ** 2 * np.identity(nu), Y],
                                                [Y.T, Q]]) >> 0)
                else:
                    U = cp.diag(cp.Variable((nu, 1)))
                    constraints.append(cp.bmat([[U, Y],
                                                [Y.T, Q]]) >> 0)
                    for j in range(0, nu):
                        constraints.append(U[j, j] <= self.u_bounds_max[j] * self.u_bounds_max[j])

                # create optimization problem
                opt_prob = cp.Problem(cp.Minimize(gamma), constraints)

                # solve optimization problem
                opt_prob.solve(verbose=False, solver=cp.SCS)

                Q = Q.value
                K = Y.value @ np.linalg.inv(Q)
                self.pi_set_already_computed[angle] = [K, Q]
                rospy.logdebug("A PI set computed")
                self.compute_one_step_controllable_sets(round(angle, 3), Q)

            else:
                [_, Q] = self.pi_set_already_computed[angle]
            Qs.append(Q)
        end_time = time.time() - start_time
        rospy.loginfo('Sets constructed in t = %.4f sec' % end_time)

        rospy.Service("oscs_service", OSCSList, self.send_oscs)

        return EdgesListResponse(True)


if __name__ == '__main__':
    rospy.init_node('offline_set_computation_node')
    _increment = rospy.get_param('~increment')
    _Ts = rospy.get_param('~Ts_robot')
    _min_distance = rospy.get_param('~min_distance')
    v_max = rospy.get_param('~v_max')
    w_max = rospy.get_param('~w_max')
    _b = rospy.get_param('~increment') * rospy.get_param('~map_resolution')
    r_q = _b * v_max * w_max / (np.sqrt(_b**2 * w_max**2 + v_max**2))

    _A = np.identity(2)
    _B = _Ts * np.identity(2)
    _R = np.identity(2)
    _Q = np.identity(2)
    OfflineSetComputation(_increment, _min_distance, _Ts, _b, _A, _B, _R, _Q, u_bounds_max=r_q)

    rospy.spin()
