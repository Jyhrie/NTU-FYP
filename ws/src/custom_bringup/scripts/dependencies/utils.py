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