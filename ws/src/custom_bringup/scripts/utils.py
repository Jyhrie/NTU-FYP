
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
    dot = v1.x * v2.x + v1.y * v2.y
    mag1 = math.hypot(v1.x, v1.y)
    mag2 = math.hypot(v2.x, v2.y)

    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp for safety

    return normalize_angle(math.acos(cos_theta))  # radians