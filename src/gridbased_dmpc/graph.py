#!/usr/bin/env python3
import numpy as np
import heapq


class Graph(object):
    # Initialize the matrix
    def __init__(self):
        self.grid = np.zeros((1, 1), dtype=bool)

    # Add edges
    def add_node(self, i, j):
        self.grid[i][j] = True

    def remove_existing_node(self, v):
        self.grid[v[0], v[1]] = False

    # Print the list
    def __str__(self):
        return str(self.grid)

    def size_nodes(self):
        return np.sum(np.sum(self.grid))/2

    def node_exists(self, v):
        return self.grid[v[0], v[1]] == True

    def get_all_nodes(self):
        return [(i, j) for i in range(self.grid.shape[0]) for j in range(self.grid.shape[1]) if self.grid[i, j] == 1]

    @staticmethod
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def is_goal_occupied(goal, other_paths):
        # Checks whether the goal node is occupied indefinitely in any other path
        for path in other_paths.values():
            if goal in path and path[goal][1] == float('infinity'):
                return True
        return False

    def a_star_with_constraints(self, start, goal, other_paths=None, u_max=1):
        rows, cols = len(self.grid), len(self.grid[0])

        goal_occupied = self.is_goal_occupied(goal, other_paths)

        # Initialize distances with infinity
        dist = [[float('inf')] * cols for _ in range(rows)]
        dist[start[0]][start[1]] = 0

        # Initialize the 'time' dictionary for each valid cell in the grid
        time = {(i, j): [float('infinity'), float('infinity')] for i in range(rows) for j in range(cols) if
                self.grid[i, j] == 1}
        time[start][0] = 0

        # Predecessors to reconstruct the path
        predecessors = [[None] * cols for _ in range(rows)]

        # Closed list and number of paths of other robots
        closed_list = []
        other_paths = other_paths or []

        # Priority queue for A* (total estimated cost, current distance, current time, location)
        queue = [(0 + self.heuristic(start, goal), 0, start)]
        heapq.heapify(queue)

        while queue:
            _, current_dist, current_node = heapq.heappop(queue)

            # Avoid reviewing nodes in the closed list
            if current_node in closed_list:
                continue
            closed_list.append(current_node)

            # Stop if you have reached the goal
            if current_node == goal:
                path = []

                # Path reconstruction with time intervals
                t0 = time[goal][1]
                tf = float('infinity')
                while current_node != start:
                    path.insert(0, (current_node, [t0, tf]))
                    t0 = time[current_node][0]
                    tf = time[current_node][1]
                    current_node = predecessors[current_node[0]][current_node[1]]

                # Add the starting node with the starting time
                path.insert(0, (start, [0, tf]))
                return dict(path)

            # Examine all neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = current_node[0] + dx, current_node[1] + dy
                neighbor = (nx, ny)

                # Skip if neighbor is off the grid or inaccessible
                if not (0 <= nx < rows and 0 <= ny < cols) or self.grid[nx][ny] == 0:
                    continue

                # Calculate the cost for the neighbor and the arrival and departure time
                step_cost = 1.4 if dx != 0 and dy != 0 else 1
                new_dist = current_dist + step_cost
                if new_dist >= dist[nx][ny]:
                    continue
                t0 = new_dist / u_max
                tf = float('infinity') if neighbor == goal else (new_dist + 1) / u_max

                # Checking time conflicts with other paths
                add = True
                for other_path in other_paths.values():
                    if neighbor in other_path and tf >= other_path[neighbor][0] and other_path[neighbor][1] >= t0:
                        add = False
                        break

                # Update if there are no conflicts and the distance is smaller
                if add and new_dist < dist[nx][ny]:
                    dist[nx][ny] = new_dist
                    time[neighbor] = [current_dist / u_max, new_dist / u_max]
                    predecessors[nx][ny] = current_node
                    total_cost = new_dist + self.heuristic(neighbor, goal)
                    heapq.heappush(queue, (total_cost, new_dist, neighbor))

        # If the goal is occupied indefinitely, construct the path to the nearest node
        if goal_occupied:
            x_goal, y_goal = goal
            neighbors = [
                (x_goal + dx, y_goal + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                if 0 <= x_goal + dx < rows and 0 <= y_goal + dy < cols and dist[x_goal + dx][y_goal + dy] < float('inf')
            ]

            closest_node = min(neighbors, key=lambda nodo: dist[nodo[0]][nodo[1]], default=None)

            path = []
            current_node = closest_node

            # Path reconstruction with time intervals
            t0 = time[closest_node][1]
            tf = float('infinity')
            while current_node != start:
                path.insert(0, (current_node, [t0, tf]))
                t0 = time[current_node][0]
                tf = time[current_node][1]
                current_node = predecessors[current_node[0]][current_node[1]]

            # Add the starting node with the starting time
            path.insert(0, (start, [0, tf]))
            return dict(path)

        # If there is no path, it returns an empty dictionary
        return {}