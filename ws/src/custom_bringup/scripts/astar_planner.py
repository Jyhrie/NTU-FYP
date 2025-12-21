import heapq
import math

class AStarPlanner:
    def __init__(self, grid, width, height):
        self.grid = grid # 2D numpy array (0-100, 255/100 is lethal)
        self.width = width
        self.height = height

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_path(self, start, goal):
        """
        start/goal: (x, y) pixel coordinates
        Returns: List of (x, y) pixel coordinates
        """
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        while open_list:
            current = heapq.heappop(open_list)[1]

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue
                
                # Check for lethal obstacles in costmap (usually > 90 or 254)
                costmap_value = self.grid[neighbor[1], neighbor[0]]
                if costmap_value >= 95: 
                    continue

                # Cost = distance (1 or 1.4) + costmap inflation value
                move_cost = math.sqrt(dx**2 + dy**2) + (costmap_value / 10.0)
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))
        return None # No path found

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]