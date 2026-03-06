import math

def compare_bbox_centroid(d1, d2, radius=12):
    if d1 is None or d2 is None:
        return False
    
    # Extract BBox: [left, top, w, h]
    b1 = d1['bbox']
    b2 = d2['bbox']
    
    # 2. Calculate Centroids
    c1_x, c1_y = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    c2_x, c2_y = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    
    # 3. Euclidean Distance: sqrt((x2-x1)^2 + (y2-y1)^2)
    distance = math.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)
    
    return distance <= radius
