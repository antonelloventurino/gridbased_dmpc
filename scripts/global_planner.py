#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from gridbased_dmpc.graph import Graph
import gridbased_dmpc.transformations as u_tf
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point
from gridbased_dmpc.srv import EdgesList, EdgesListRequest
from gridbased_dmpc.msg import Edge
from threading import Semaphore
from visualization_msgs.msg import Marker
from scipy.ndimage import binary_dilation
import time
import traceback


class GlobalPlanner(object):

    def __init__(self, inflate_radius, increment, min_distance, number_tb3s):
        self.inflate_radius = inflate_radius
        self.min_distance = min_distance
        self.number_tb3s = number_tb3s
        self.tb3s = set(range(1, number_tb3s + 1))
        self.path_pub = []
        self.lidar_segments_pub = []
        self.remove_vertex_pub = []
        self.add_vertex_pub = []
        self.graph_pub = rospy.Publisher("global_planner/nodes", PointCloud2, queue_size=2)
        self.width = 0
        self.height = 0
        self.resolution = 0
        self.directions = [(inflate_radius * x, inflate_radius * y) for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]]
        self.origin = None
        self.costmap = None
        self.costmap_pub = rospy.Publisher("global_costmap", OccupancyGrid, queue_size=1)
        self.g = Graph()
        self.p0 = {i: [] for i in range(0, number_tb3s)}
        self.yaw = {i: [] for i in range(0, number_tb3s)}
        self.goal = {i: [] for i in range(0, number_tb3s)}
        self.shortest_paths = {i: {} for i in range(0, number_tb3s)}
        self.graph_built = False
        self.increment = increment
        self.semaphore_graph = Semaphore(1)
        self.map_received = False
        self.got_scan = False
        self.angles = []
        for i in self.tb3s:
            rospy.Subscriber("/tb3_" + str(i) + "/scan", LaserScan, self.scan_callback, i-1)
            rospy.Subscriber("/tb3_" + str(i) + "/odom", Odometry, self.odom_callback, i-1)
            rospy.Subscriber("/tb3_" + str(i) + "/goal", PoseStamped, self.goal_callback, i-1)
            self.path_pub.append(rospy.Publisher("/tb3_" + str(i) + "/global_planner/path", Path, queue_size=2))
            self.lidar_segments_pub.append(rospy.Publisher("/tb3_" + str(i) + "/lidar_segments", Marker, queue_size=10))
            self.remove_vertex_pub.append(rospy.Publisher("/tb3_" + str(i) + "/global_planner/remove_vertex", PointCloud2, queue_size=2))
            self.add_vertex_pub.append(rospy.Publisher("/tb3_" + str(i) + "/global_planner/add_vertex", PointCloud2, queue_size=2))

    def goal_callback(self, msg, tb3_id):
        rospy.loginfo('New target goal received!')
        self.goal[tb3_id] = [msg.pose.position.x, msg.pose.position.y]

    def odom_callback(self, msg, tb3_id):
        self.p0[tb3_id] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.yaw[tb3_id] = u_tf.quaternion_to_yaw(msg.pose.pose.orientation)

    def find_vertex(self, p):
        cx = -(p[0] + self.origin.position.x) / (self.resolution * self.increment)
        cy = -(p[1] + self.origin.position.y) / (self.resolution * self.increment)

        return round(cx), round(cy)

    def publish_lidar_segments(self, p0, points, rgbs, tb3_id):
        marker = Marker()
        marker.ns = "tb3_" + str(tb3_id)
        marker.header.frame_id = "map"
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.colors = []
        start_point = Point()
        start_point.x = p0[0]
        start_point.y = p0[1]
        start_point.z = 0

        for i in range(0, len(points)):
            marker.colors.append(ColorRGBA(rgbs[i][0], rgbs[i][1], rgbs[i][2], 0.4))
            marker.colors.append(ColorRGBA(rgbs[i][0], rgbs[i][1], rgbs[i][2], 0.4))

            end_point = Point()
            end_point.x = points[i][0]
            end_point.y = points[i][1]
            end_point.z = 0

            marker.points.append(start_point)
            marker.points.append(end_point)

        self.lidar_segments_pub[tb3_id-1].publish(marker)

    def generate_points_along_segment(self, p, p_l, R):
        cell_p_x = round(-(p[0] + self.origin.position.x) / R)
        cell_p_y = round(-(p[1] + self.origin.position.y) / R)
        cell_p_l_x = round(-(p_l[0] + self.origin.position.x) / R)
        cell_p_l_y = round(-(p_l[1] + self.origin.position.y) / R)

        x0, y0 = cell_p_x, cell_p_y
        x1, y1 = cell_p_l_x, cell_p_l_y

        points = set()
        cells = set()

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0

        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cell = (x, y)
                cells.add(cell)
                point = (-R * x - self.origin.position.x, -R * y - self.origin.position.y)
                points.add(point)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cell = (x, y)
                cells.add(cell)
                point = (-R * x - self.origin.position.x, -R * y - self.origin.position.y)
                points.add(point)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        # add last point
        cell = (x1, y1)
        cells.add(cell)
        point = (-R * x1 - self.origin.position.x, -R * y1 - self.origin.position.y)
        points.add(point)

        return points, cells

    def scan_callback(self, msg, tb3_id):
        if not self.got_scan:
            self.got_scan = True
            angles_range = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
            for i in range(0, len(angles_range)):
                self.angles.append(np.array([np.cos(angles_range[i]), np.sin(angles_range[i])]))
        if not self.graph_built:
            return

        max_range = msg.range_max
        R_z = u_tf.rot_z(self.yaw[tb3_id])

        rgbs = []
        points_lines = []
        points_to_be_removed = set()
        nodes_to_be_removed = set()
        nodes_to_be_added = set()
        points_to_be_added = set()
        candidate_point_segments = []
        candidate_node_segments = []
        R = self.resolution * self.increment
        for i in range(0, len(self.angles)):
            obj_detected = True
            if  msg.ranges[i] == float('inf'):
                r = max_range
                rgbs.append([0, 1, 0])
                obj_detected = False
            else:
                r = msg.ranges[i]
                rgbs.append([1, 0, 0])
            pc = r * R_z @ self.angles[i] + self.p0[tb3_id]
            points_lines.append(pc)
            points_segment, cells_segment = self.generate_points_along_segment(self.p0[tb3_id], pc, R)
            if obj_detected:
                points_to_be_added = points_to_be_added.union(points_segment)
                nodes_to_be_added = nodes_to_be_added.union(cells_segment)
                c = self.find_vertex(pc)

                for dx, dy in self.directions:
                    p_neighbor = [pc[0] + dx, pc[1] + dy]
                    c_neighbor = self.find_vertex(p_neighbor)
                    points_to_be_removed.add(self.find_point_vertex(c_neighbor))
                    nodes_to_be_removed.add(c_neighbor)

                points_to_be_removed.add(self.find_point_vertex(c))
                nodes_to_be_removed.add((c[0], c[1]))
            else:
                candidate_point_segments.append(points_segment)
                candidate_node_segments.append(cells_segment)

        for tb3_neighbor in self.tb3s.difference({tb3_id+1}):
            if np.linalg.norm(np.array(self.p0[tb3_id]) - np.array(self.p0[tb3_neighbor-1])) > max_range:
                pass
            vertices_neighbor = {self.find_vertex(self.p0[tb3_neighbor - 1])}
            points_neighbor = {(self.p0[tb3_neighbor-1][0], self.p0[tb3_neighbor-1][1])}
            for dx, dy in self.directions:
                c_neighbor = self.find_vertex([self.p0[tb3_neighbor-1][0] + dx, self.p0[tb3_neighbor-1][1] + dy])
                points_neighbor.add(self.find_point_vertex(c_neighbor))
                vertices_neighbor.add(c_neighbor)
            nodes_to_be_removed.difference_update(vertices_neighbor)
            points_to_be_removed.difference_update(points_neighbor)

        for i in range(0, len(candidate_node_segments)):
            nodes = set(candidate_node_segments[i])
            if nodes.isdisjoint(nodes_to_be_removed):
                points_to_be_added = points_to_be_added.union(set(candidate_point_segments[i]))
                nodes_to_be_added = nodes_to_be_added.union(nodes)

        nodes_to_be_added = nodes_to_be_added.difference(nodes_to_be_removed)
        points_to_be_added = points_to_be_added.difference(points_to_be_removed)
        for v in nodes_to_be_added:
            self.g.add_node(v[0], v[1])
        for v in nodes_to_be_removed:
            self.g.remove_existing_node(v)


        self.publish_vertices()
        self.publish_lidar_segments(self.p0[tb3_id], points_lines, rgbs, tb3_id+1)
        self.publish_vertices_laser(points_to_be_removed, tb3_id)
        self.publish_vertices_laser(points_to_be_added, tb3_id,True)


    def receive_map(self):
        msg: OccupancyGrid = rospy.wait_for_message("map", OccupancyGrid) # type: ignore

        self.width = msg.info.width
        self.height = msg.info.height
        magnitude = np.floor(np.log10(np.abs(msg.info.resolution)))
        factor = 10 ** (magnitude - 2)
        self.resolution = np.round(msg.info.resolution / factor) * factor
        self.origin = msg.info.origin
        arr = np.array(msg.data)

        inflate_radius_cell = int(self.inflate_radius / self.resolution)

        arr = np.flip(arr)
        mappa = arr.reshape(msg.info.width, msg.info.height, order='F')
        mappa = np.array(mappa, dtype='float64')
        binary_dilation(mappa, output=mappa, iterations=inflate_radius_cell)

        self.costmap = 100 * np.array(mappa, dtype=np.int8)
        self.map_received = True
        self.publish_global_costmap()

    def publish_global_costmap(self):
        occupancy_grid = OccupancyGrid()

        # Fill in the header
        occupancy_grid.header.stamp = rospy.Time.now()
        occupancy_grid.header.frame_id = "map"

        # Define the map parameters
        occupancy_grid.info.width = self.width
        occupancy_grid.info.height = self.height
        occupancy_grid.info.resolution = self.resolution
        occupancy_grid.info.origin = self.origin

        data = self.costmap.reshape(1, self.width * self.height, order='F')[0]
        occupancy_grid.data = np.flip(data).tolist()

        self.costmap_pub.publish(occupancy_grid)

    def publish_vertices(self):
        if self.width == 0:
            return

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]

        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()

        with self.semaphore_graph:
            points = [self.find_point_vertex(k) for k in self.g.get_all_nodes()]

        points = np.array(points, dtype=np.float32)
        n_points = points.shape[0]
        zeros_col = np.zeros((n_points,))

        points = np.concatenate((points, zeros_col[:, None]), axis=1)

        pc2 = PointCloud2()
        pc2.header = header
        pc2.fields = fields
        pc2.height = 1
        pc2.width = n_points
        pc2.is_bigendian = False
        pc2.point_step = 12
        pc2.row_step = 12 * n_points
        pc2.is_dense = False
        pc2.data = points.astype(np.float32).tobytes()
        self.graph_pub.publish(pc2)

    def publish_vertices_laser(self, points, tb3_id, add=False):
        if self.width == 0 or len(points) == 0:
            return

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]

        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()

        points = np.array(list(points), dtype=np.float32)
        n_points = points.shape[0]
        zeros_col = np.zeros((n_points,))

        points = np.concatenate((points, zeros_col[:, None]), axis=1)

        pc2 = PointCloud2()
        pc2.header = header
        pc2.fields = fields
        pc2.height = 1
        pc2.width = n_points
        pc2.is_bigendian = False
        pc2.point_step = 12
        pc2.row_step = 12 * n_points
        pc2.is_dense = False

        pc2.data = points.astype(np.float32).tobytes()
        if add:
            self.add_vertex_pub[tb3_id].publish(pc2)
            return
        self.remove_vertex_pub[tb3_id].publish(pc2)

    def find_point_vertex(self, v):
        # if self.width == 0:
        #     return

        p0 = -self.resolution * self.increment * v[0] - self.origin.position.x
        p1 = -self.resolution * self.increment * v[1] - self.origin.position.y

        return p0, p1

    def find_closest_node(self, v):
        if self.g.size_nodes() == 0:
            return v

        with self.semaphore_graph:
            nodes = self.g.get_all_nodes()

        size = len(nodes)
        distances = np.zeros((size,))

        v = np.array([v[0], v[1]])
        for i in range(0, size):
            vc = np.array([nodes[i][0], nodes[i][1]])
            distances[i] = np.linalg.norm(vc - v)

        idx = np.argmin(distances)
        return nodes[idx]

    def publish_path(self, path_cells, tb3_id):
        if len(path_cells) < 1:
            return

        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()

        path_msg = Path()
        path_msg.header = header

        for cell in path_cells:
            pose_stamped = PoseStamped()
            p = self.find_point_vertex(cell)
            pose_stamped.pose.position.x = p[0]
            pose_stamped.pose.position.y = p[1]
            pose_stamped.pose.position.z = 0.0
            path_msg.poses.append(pose_stamped)

        self.path_pub[tb3_id].publish(path_msg)

    def compute_offline_sets(self):
        if self.g.size_nodes() == 0:
            rospy.logdebug('Empty graph: impossible to compute the sets!')
            return

        edges = EdgesListRequest()
        edges.edges = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        self.semaphore_graph.acquire()
        rows, cols = self.g.grid.shape
        for i in range(rows):
            for j in range(cols):
                if self.g.grid[i][j] == 1:
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < rows and 0 <= nj < cols and self.g.grid[ni][nj] == 1:
                            edge = Edge(i, j, ni, nj)
                            edges.edges.append(edge)
        self.semaphore_graph.release()

        rospy.wait_for_service('set_computation_service')
        compute_offline_sets = rospy.ServiceProxy('set_computation_service', EdgesList)

        compute_offline_sets(edges)
        rospy.loginfo('Sets computed!')

    def check_safe_space(self, idx):
        i_min = int(max(0, (idx[0] - 1) * self.increment))
        i_max = int(min(self.width - 1, (idx[0] + 1) * self.increment-1))

        j_min = int(max(0, (idx[1] - 1) * self.increment))
        j_max = int(min(self.width - 1, (idx[1] + 1) * self.increment-1))

        space_matrix = self.costmap[i_min:i_max, j_min:j_max]
        return np.sum(space_matrix) < 1

    def build_graph(self):
        if not self.map_received:
            rospy.logdebug('Empty map: impossible to build the graph!')
            return

        rospy.loginfo('Constructing initial grid graph...')
        R_bar = self.increment
        self.g.grid = np.zeros((int(self.width/R_bar) + 1, int(self.height/R_bar) + 1), dtype=bool)

        for i in range(0, self.g.grid.shape[0]):
            for j in range(0, self.g.grid.shape[1]):
                self.g.grid[i, j] = self.check_safe_space((i, j))
        self.graph_built = True
        rospy.loginfo('Graph constructed!')

    def run(self):
        for tb3_i in range(0, self.number_tb3s):
            if len(self.p0[tb3_i]) < 1:
                continue

            if not self.graph_built:
                v0 = self.find_vertex(self.p0[tb3_i])
                self.build_graph()
                self.publish_vertices()
                rospy.loginfo('Computing offline sets...')
                self.compute_offline_sets()
            else:
                v0 = self.find_vertex(self.p0[tb3_i])

            if len(self.goal[tb3_i]) > 0:
                vf = self.find_vertex(self.goal[tb3_i])
                pf = self.find_point_vertex(vf)

                self.goal[tb3_i][0] = pf[0]
                self.goal[tb3_i][1] = pf[1]

                if not self.g.node_exists(vf):
                    rospy.logwarn('You send an unreachable goal, I\'m going to replace it with the nearest one!')
                    vf = self.find_closest_node(vf)
                    pf = self.find_point_vertex(vf)
                    self.goal[tb3_i][0] = pf[0]
                    self.goal[tb3_i][1] = pf[1]

                self.semaphore_graph.acquire()
                computed = False
                self.shortest_paths[tb3_i] = {}
                try:
                    self.shortest_paths[tb3_i] = self.g.a_star_with_constraints(v0, vf, self.shortest_paths)
                    computed = True
                except KeyError:
                    rospy.logwarn(traceback.format_exc())
                    rospy.logwarn('The new global path can\'t be computed!')
                self.semaphore_graph.release()

                if computed:
                    shortest_path = list(self.shortest_paths[tb3_i].keys())
                    self.publish_path(shortest_path, tb3_i)
                    rospy.loginfo('New global path computed!')

                if np.linalg.norm(np.array(self.p0[tb3_i]) - np.array(self.goal[tb3_i])) <= self.min_distance:
                    self.goal[tb3_i] = []


def main():
    rospy.init_node('global_planner_node', anonymous=True, log_level=rospy.DEBUG)
    fq_update_global_path = rospy.get_param('~fq_update_global_path')
    rate = rospy.Rate(fq_update_global_path)  # Hz

    inflate_radius = rospy.get_param('~inflate_radius')
    increment = rospy.get_param('~increment')
    number_tb3s = rospy.get_param('~number_tb3s')
    min_distance = rospy.get_param('~min_distance')
    node = GlobalPlanner(inflate_radius, increment, min_distance, number_tb3s)
    node.receive_map()

    while not rospy.is_shutdown():
        node.run()
        rate.sleep()

if __name__ == '__main__':
    main()
