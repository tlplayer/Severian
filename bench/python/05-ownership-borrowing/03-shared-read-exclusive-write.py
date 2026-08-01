def total(values):
    return sum(values)

def push_value(values, value):
    values.append(value)

values = [1, 2, 3]
print(total(values))
push_value(values, 4)
print(total(values))
