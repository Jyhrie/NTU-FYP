
import math


def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def angle_between(v1, v2):
    dot = v1.x*v2.x + v1.y*v2.y
    cross = v1.x*v2.y - v1.y*v2.x
    return normalize_angle(math.atan2(cross, dot))  # [-pi, pi], signed