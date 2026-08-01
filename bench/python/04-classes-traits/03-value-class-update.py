class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def translated(self, dx, dy):
        return Point(self.x + dx, self.y + dy)

point = Point(1.0, 2.0).translated(3.0, 4.0)
print(f"{point.x:g}")
print(f"{point.y:g}")
