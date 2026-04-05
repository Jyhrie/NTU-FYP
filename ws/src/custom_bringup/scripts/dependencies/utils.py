import math

def compare_bbox_centroid(bbox1, bbox2, radius=12):
    # Safety check if either detection is missing
    if bbox1 is None or bbox2 is None:
        return False
    
    # Coordinates are now in tuples: (x_start, y_start, x_len, y_len)
    # Index 0: x_start | Index 1: y_start | Index 2: x_len | Index 3: y_len
    
    # 1. Calculate Centroid for bbox1
    c1_x = bbox1[0] + (bbox1[2] / 2.0)
    c1_y = bbox1[1] + (bbox1[3] / 2.0)
    
    # 2. Calculate Centroid for bbox2
    c2_x = bbox2[0] + (bbox2[2] / 2.0)
    c2_y = bbox2[1] + (bbox2[3] / 2.0)
    
    # 3. Calculate Euclidean Distance
    distance = math.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)
    
    return distance <= radius

def wrap_angle(angle_rad):
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

def project_local_to_world(robot_pose, rel_angle_deg, distance):
    rx, ry, ryaw = robot_pose
    ry = ry
    
    # 1. Total bearing: Robot's orientation + sensor's relative offset
    total_bearing_rad = wrap_angle(ryaw + math.radians(rel_angle_deg))
    
    # 2. Project out from the robot's current (x, y)
    world_x = rx + (distance * math.cos(total_bearing_rad))
    world_y = ry + (distance * math.sin(total_bearing_rad))
    
    return (world_x, world_y)