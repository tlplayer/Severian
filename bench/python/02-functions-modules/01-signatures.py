def scale(value, factor=1.0):
    return value * factor

def describe(label, value):
    return f"{label}: {value:g}"

print(describe("width", scale(12.0, factor=2.0)))
