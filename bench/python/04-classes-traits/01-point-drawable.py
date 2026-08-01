class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def magnitude(self):
        return (self.x * self.x + self.y * self.y) ** 0.5

    def draw(self):
        print(f"point {self.x:g} {self.y:g}")

point = Point(3.0, 4.0)
point.draw()
print(f"{point.magnitude():g}")
