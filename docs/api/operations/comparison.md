## Comparisons

There are eight comparison operations in Python. All comparison operators have the same precedence, which is higher than Boolean operations.

Comparisons can be chained:

```python
x < y <= z
```

This is equivalent to:

```python
x < y and y <= z
```

with two differences:

* `y` is evaluated only once.
* `z` is not evaluated if `x < y` is `False`.

### Comparison operators

| Operation | Meaning                 |
| --------- | ----------------------- |
| `<`       | Strictly less than      |
| `<=`      | Less than or equal      |
| `>`       | Strictly greater than   |
| `>=`      | Greater than or equal   |
| `==`      | Equal                   |
| `!=`      | Not equal               |
| `is`      | Object identity         |
| `is not`  | Negated object identity |

Unless otherwise specified, objects of different types do not compare equal.

The `==` operator is always defined. For some object types, such as class objects, it may be equivalent to `is`.

The ordering operators:

```text
<  <=  >  >=
```

are only defined for types where ordering is meaningful.

For example, comparing complex numbers with ordering operators raises a `TypeError`.
