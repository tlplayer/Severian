values = [4, -2, 7, 1]
summary = [min(values), max(values), sum(values)]
weighted_total = sum(index * value for index, value in enumerate([5, 10, 20]))
dot_product = sum(left * right for left, right in zip([1, 2, 3], [4, 5]))
until_negative = 0
for value in [2, 0, 5, -1, 100]:
    if value < 0:
        break
    until_negative += value

print(" | ".join("alpha,beta,gamma".split(",")))
print(summary)
print(weighted_total)
print(dot_product)
print(until_negative)
