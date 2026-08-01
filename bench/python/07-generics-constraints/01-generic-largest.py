def largest(values):
    return max(values, default=None)

result = largest([3, 9, 2])
print("absent" if result is None else f"present({result})")
