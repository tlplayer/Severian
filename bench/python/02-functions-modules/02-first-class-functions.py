def add(a, b):
    return a + b

def apply(op, left, right):
    return op(left, right)

print(apply(add, 20, 22))
