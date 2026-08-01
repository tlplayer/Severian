def increment(values):
    for index in range(len(values)):
        values[index] += 1

values = [10, 20, 30]
increment(values)
print(values)
