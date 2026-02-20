#!/usr/bin/env python3

import rospy
import math
import cv2
import numpy as np
from std_msgs.msg import Header
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from priority_queue import PriorityQueue
from tf.transformations import quaternion_from_euler


DIRECTIONS_OF_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS_OF_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class PathPlanner:

    @staticmethod
    def grid_to_index(mapdata, p):
        """
        Returns the index corresponding to the given (x,y) coordinates in the occupancy grid.
        """
        return p[1] * mapdata.info.width + p[0]

    @staticmethod
    def get_cell_value(mapdata, p):
        """
        Returns the cell corresponding to the given (x,y) coordinates in the occupancy grid.
        """
        return mapdata.data[PathPlanner.grid_to_index(mapdata, p)]

    @staticmethod
    def euclidean_distance(p1, p2):
        """
        Calculates the Euclidean distance between two points.
        """
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def grid_to_world(mapdata, p):
        """
        Transforms a cell coordinate in the occupancy grid into a world coordinate.
        """
        x = (p[0] + 0.5) * mapdata.info.resolution + mapdata.info.origin.position.x
        y = (p[1] + 0.5) * mapdata.info.resolution + mapdata.info.origin.position.y
        return Point(x, y, 0)

    @staticmethod
    def world_to_grid(mapdata, wp):
        """
        Transforms a world coordinate into a cell coordinate in the occupancy grid.
        """
        x = int((wp.x - mapdata.info.origin.position.x) / mapdata.info.resolution)
        y = int((wp.y - mapdata.info.origin.position.y) / mapdata.info.resolution)
        return (x, y)

    @staticmethod
    def path_to_poses(mapdata, path):
        """
        Converts the given path into a list of PoseStamped.
        """
        poses = []
        for i in range(len(path) - 1):
            cell = path[i]
            next_cell = path[i + 1]
            if i != len(path) - 1:
                angle_to_next = math.atan2(
                    next_cell[1] - cell[1], next_cell[0] - cell[0]
                )
            q = quaternion_from_euler(0, 0, angle_to_next)
            poses.append(
                PoseStamped(
                    header=Header(frame_id="map"),
                    pose=Pose(
                        position=PathPlanner.grid_to_world(mapdata, cell),
                        orientation=Quaternion(q[0], q[1], q[2], q[3]),
                    ),
                )
            )
        return poses

    @staticmethod
    def is_cell_in_bounds(mapdata, p):
        width = mapdata.info.width
        height = mapdata.info.height
        x = p[0]
        y = p[1]

        if x < 0 or x >= width:
            return False
        if y < 0 or y >= height:
            return False
        return True

    @staticmethod
    def is_cell_walkable(mapdata, p):
        """
        A cell is walkable if it is within bounds and free.
        """
        if not PathPlanner.is_cell_in_bounds(mapdata, p):
            return False

        WALKABLE_THRESHOLD = 50
        return PathPlanner.get_cell_value(mapdata, p) < WALKABLE_THRESHOLD

    @staticmethod
    def neighbors(mapdata, p, directions, must_be_walkable=True):
        """
        Returns the neighbors cells of (x,y) in the occupancy grid.
        """
        neighbors = []
        for direction in directions:
            candidate = (p[0] + direction[0], p[1] + direction[1])
            if (
                must_be_walkable and PathPlanner.is_cell_walkable(mapdata, candidate)
            ) or (
                not must_be_walkable
                and PathPlanner.is_cell_in_bounds(mapdata, candidate)
            ):
                neighbors.append(candidate)
        return neighbors

    @staticmethod
    def neighbors_of_4(mapdata, p, must_be_walkable=True):
        return PathPlanner.neighbors(mapdata, p, DIRECTIONS_OF_4, must_be_walkable)

    @staticmethod
    def neighbors_of_8(mapdata, p, must_be_walkable=True):
        return PathPlanner.neighbors(mapdata, p, DIRECTIONS_OF_8, must_be_walkable)

    @staticmethod
    def neighbors_and_distances(mapdata, p, directions, must_be_walkable=True):
        """
        Returns the neighbors cells and their distances.
        """
        neighbors = []
        for direction in directions:
            candidate = (p[0] + direction[0], p[1] + direction[1])
            if not must_be_walkable or PathPlanner.is_cell_walkable(mapdata, candidate):
                distance = PathPlanner.euclidean_distance(direction, (0, 0))
                neighbors.append((candidate, distance))
        return neighbors

    @staticmethod
    def neighbors_and_distances_of_4(mapdata, p, must_be_walkable=True):
        return PathPlanner.neighbors_and_distances(
            mapdata, p, DIRECTIONS_OF_4, must_be_walkable
        )

    @staticmethod
    def neighbors_and_distances_of_8(mapdata, p, must_be_walkable=True):
        return PathPlanner.neighbors_and_distances(
            mapdata, p, DIRECTIONS_OF_8, must_be_walkable
        )

    @staticmethod
    def get_grid_cells(mapdata, cells):
        world_cells = []
        for cell in cells:
            world_cells.append(PathPlanner.grid_to_world(mapdata, cell))
        resolution = mapdata.info.resolution
        return GridCells(
            header=Header(frame_id="map"),
            cell_width=resolution,
            cell_height=resolution,
            cells=world_cells,
        )

    @staticmethod
    def calc_cspace(mapdata, include_cells):
        """
        Calculates the C-Space, i.e., makes the obstacles in the map thicker.
        """
        PADDING = 5

        width = mapdata.info.width
        height = mapdata.info.height
        map_arr = np.array(mapdata.data).reshape(width, height).astype(np.uint8)

        unknown_area_mask = cv2.inRange(map_arr, 255, 255)
        kernel = np.ones((PADDING, PADDING), dtype=np.uint8)
        unknown_area_mask = cv2.erode(unknown_area_mask, kernel, iterations=1)

        map_arr[map_arr == 255] = 0

        kernel = np.ones((PADDING, PADDING), np.uint8)
        obstacle_mask = cv2.dilate(map_arr, kernel, iterations=1)
        cspace_data = cv2.bitwise_or(obstacle_mask, unknown_area_mask)
        cspace_data = np.array(cspace_data).reshape(width * height).tolist()

        cspace = OccupancyGrid(
            header=mapdata.header, info=mapdata.info, data=cspace_data
        )

        cspace_cells = None
        if include_cells:
            cells = []
            obstacle_indices = np.where(obstacle_mask > 0)
            for y, x in zip(*obstacle_indices):
                cells.append((x, y))
            cspace_cells = PathPlanner.get_grid_cells(mapdata, cells)

        return (cspace, cspace_cells)

    @staticmethod
    def get_cost_map_value(cost_map, p):
        return cost_map[p[1]][p[0]]

    @staticmethod
    def show_map(name, map_arr):
        normalized = cv2.normalize(
            map_arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        cv2.imshow(name, normalized)
        cv2.waitKey(0)

    @staticmethod
    def calc_cost_map(mapdata):
        rospy.loginfo("Calculating cost map")

        width = mapdata.info.width
        height = mapdata.info.height
        map_arr = np.array(mapdata.data).reshape(height, width).astype(np.uint8)
        map_arr[map_arr == 255] = 100

        cost_map = np.zeros_like(map_arr)
        dilated_map = map_arr.copy()
        iterations = 0
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        while np.any(dilated_map == 0):
            iterations += 1
            next_dilated_map = cv2.dilate(dilated_map, kernel, iterations=1)
            difference = next_dilated_map - dilated_map
            difference[difference > 0] = iterations
            cost_map = cv2.bitwise_or(cost_map, difference)
            dilated_map = next_dilated_map

        cost_map = PathPlanner.create_hallway_mask(mapdata, cost_map, iterations // 4)

        dilated_map = cost_map.copy()
        cost = 1
        for i in range(iterations):
            cost += 1
            next_dilated_map = cv2.dilate(dilated_map, kernel, iterations=1)
            difference = next_dilated_map - dilated_map
            difference[difference > 0] = cost
            cost_map = cv2.bitwise_or(cost_map, difference)
            dilated_map = next_dilated_map

        cost_map[cost_map > 0] -= 1
        return cost_map

    @staticmethod
    def create_hallway_mask(mapdata, cost_map, threshold):
        mask = np.zeros_like(cost_map, dtype=bool)
        non_zero_indices = np.nonzero(cost_map)
        for y, x in zip(*non_zero_indices):
            if PathPlanner.is_hallway_cell(mapdata, cost_map, (x, y), threshold):
                mask[y][x] = 1
        return mask.astype(np.uint8)

    @staticmethod
    def is_hallway_cell(mapdata, cost_map, p, threshold):
        cost_map_value = PathPlanner.get_cost_map_value(cost_map, p)
        for neighbor in PathPlanner.neighbors_of_8(mapdata, p, False):
            neighbor_cost_map_value = PathPlanner.get_cost_map_value(cost_map, neighbor)
            if (
                neighbor_cost_map_value < threshold
                or neighbor_cost_map_value > cost_map_value
            ):
                return False
        return True

    @staticmethod
    def get_first_walkable_neighbor(mapdata, start):
        queue = []
        queue.append(start)
        visited = {}

        while queue:
            current = queue.pop(0)
            if PathPlanner.is_cell_walkable(mapdata, current):
                return current

            for neighbor in PathPlanner.neighbors_of_4(mapdata, current, False):
                visited[neighbor] = True
                queue.append(neighbor)

        return start

    @staticmethod
    def a_star(mapdata, cost_map, start, goal):
        COST_MAP_WEIGHT = 1000

        if not PathPlanner.is_cell_walkable(mapdata, start):
            start = PathPlanner.get_first_walkable_neighbor(mapdata, start)

        if not PathPlanner.is_cell_walkable(mapdata, goal):
            goal = PathPlanner.get_first_walkable_neighbor(mapdata, goal)

        pq = PriorityQueue()
        pq.put(start, 0)

        cost_so_far = {}
        distance_cost_so_far = {}
        cost_so_far[start] = 0
        distance_cost_so_far[start] = 0
        came_from = {}
        came_from[start] = None

        while not pq.empty():
            current = pq.get()

            if current == goal:
                break

            for neighbor, distance in PathPlanner.neighbors_and_distances_of_8(
                mapdata, current
            ):
                added_cost = (
                    distance
                    + COST_MAP_WEIGHT
                    * PathPlanner.get_cost_map_value(cost_map, neighbor)
                )
                new_cost = cost_so_far[current] + added_cost
                if not neighbor in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    distance_cost_so_far[neighbor] = (
                        distance_cost_so_far[current] + distance
                    )
                    priority = new_cost + PathPlanner.euclidean_distance(neighbor, goal)
                    pq.put(neighbor, priority)
                    came_from[neighbor] = current

        path = []
        cell = goal

        while cell:
            path.insert(0, cell)
            if cell in came_from:
                cell = came_from[cell]
            else:
                return (None, None, start, goal)

        MIN_PATH_LENGTH = 12
        if len(path) < MIN_PATH_LENGTH:
            return (None, None, start, goal)

        POSES_TO_TRUNCATE = 8
        path = path[:-POSES_TO_TRUNCATE]

        return (path, distance_cost_so_far[goal], start, goal)

    @staticmethod
    def path_to_message(mapdata, path):
        poses = PathPlanner.path_to_poses(mapdata, path)
        return Path(header=Header(frame_id="map"), poses=poses)