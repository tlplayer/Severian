class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def describe(point):
    match (point.x, point.y):
        case (0, 0):
            print("origin")
        case (x, 0):
            print("x axis")
            print(x)
        case (x, y):
            print(x + y)
