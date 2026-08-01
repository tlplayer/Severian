def total(values):
    return sum(values)

numbers = [1, 2, 3, 4]
copied = numbers.copy()
owned = copied
print(total(numbers))
print(total(owned))
