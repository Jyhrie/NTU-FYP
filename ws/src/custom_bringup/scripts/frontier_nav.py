class FrontierNavigator:
    def __init__(self, detector, selector, planner):
        self.detector = detector
        self.selector = selector
        self.planner = planner
        self.blacklist = set()

    def coordinate_to_pixel(self, x, y):
        px = int((x - self.detector.origin_x) / self.detector.res)
        py = int((y - self.detector.origin_y) / self.detector.res)
        return (px, py)

    def update(self, robot_pose, map_data):
        # 1. Detect all frontier clusters
        # Returns: list of {'id': id, 'centroid': (world_x, world_y), 'size': n}
        frontiers = self.detector.detect_frontiers(map_data)
        
        # 2. Remove blacklisted frontiers
        active_frontiers = [f for f in frontiers if f['centroid'] not in self.blacklist]
        
        if not active_frontiers:
            print("Exploration Complete or No Reachable Frontiers.")
            return None

        # 3. Select the best frontier based on .env weights
        target_world_coords = self.selector.select_best_frontier(robot_pose, active_frontiers)
        
        if target_world_coords:
            # 4. Convert world target to pixels for A*
            start_px = self.coordinate_to_pixel(robot_pose[0], robot_pose[1])
            goal_px = self.coordinate_to_pixel(target_world_coords[0], target_world_coords[1])
            
            # 5. Generate Path
            path = self.planner.get_path(start_px, goal_px)
            
            if path is None:
                print(f"Goal {target_world_coords} unreachable. Blacklisting.")
                self.blacklist.add(target_world_coords)
                return self.update(robot_pose, map_data) # Recursively try next best
                
            return path # Path for your local controller to follow
        
        return None