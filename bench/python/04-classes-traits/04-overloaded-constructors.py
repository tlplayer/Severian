class X:
    def __init__(self, x, y=None):
        self.value = x if y is None else x + y

print(X(20, 22).value)
print(X(42).value)
