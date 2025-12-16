import math

class Vector2:
    __slots__ = ('x', 'y')   # reduces memory + access cost

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other):
        if not isinstance(other, Vector2):
            return NotImplemented
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if not isinstance(other, Vector2):
            return NotImplemented
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __repr__(self):
        return "Vector2(x={:.3f}, y={:.3f})".format(self.x, self.y)

    def add(self, other):
        self.x = self.x + other.x
        self.y = self.y + other.y
        return self
    
    def subtract(self, other):
        self.x = self.x - other.x
        self.y = self.y - other.y
        return self

    def copy(self):
        return Vector2(self.x, self.y)
    
    def normalize(self):
        mag = math.hypot(self.x, self.y)   # fast sqrt(x*x + y*y)
        if not mag == 0:
            self.x = self.x / mag
            self.y = self.y / mag
        return self

    def normal(self):
        return Vector2(-self.y, -self.x)
    
    def zero(self):
        if self.x == 0 and self.y == 0:
            return True
        return False