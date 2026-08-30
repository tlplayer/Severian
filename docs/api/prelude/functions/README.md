# Prelude functions

API ID: `prelude.function.assert`


### Async

| Function  | Purpose                                         |
| --------- | ----------------------------------------------- |
| `async` | Call function asynchronously                    |
| `await` | Await a task created by asyn |

### Attributes and Properties

| Function     | Purpose                           |
| ------------ | --------------------------------- |
| `borrow`  | borrow an object                  |
| `clone`  | copy an object             |
| `view`  | only view but not modify object                  |
| `drop`  | drop memory of an object                  |

### Collections

| Function       | Purpose                           |
| -------------- | --------------------------------- |
| `array()`  | Create an immutable byte array       |
| `vector()`  | Create a mutable byte array       |
| `dict()`       | Create a dictionary               |
| `map()`       | Create a map               |
| `list()`       | Create a list                     |
| `set()`        | Create a set                      |
| `tuple()`      | Create a tuple                    |

### Compilation and Execution

| Function       | Purpose                |
| -------------- | ---------------------- |
| `import()` | Import a module        |
| `compile()`    | Compile source         |
| `eval()`       | Evaluate an expression |
| `exe()`       | Execute code           |

### Conversion and Representation

| Function    | Purpose                                          |
| ----------- | ------------------------------------------------ |
| `ascii()`   | Produce an ASCII representation                  |
| `bin()`     | Convert an integer to binary representation      |
| `bool()`    | Convert to boolean                               |
| `char()`     | Convert an integer to a character                |
| `complex()` | Convert to a complex number                      |
| `float()`   | Convert to floating point                        |
| `format()`  | Format a value                                   |
| `hex()`     | Convert an integer to hexadecimal representation |
| `int()`     | Convert to integer                               |
| `string()`     | Convert to string                                |

### Functional

| Function   | Purpose                        |
| ---------- | ------------------------------ |
| `filter()` | Filter values using a function |
| `apply()`    | Apply a function over values   |

### Input, Output, and Debugging

| Function       | Purpose                |
| -------------- | ---------------------- |
| `breakpoint()` | Enter the debugger     |
| `help()`       | Display help           |
| `input()`      | Read input             |
| `open()`       | Open a file or stream  |
| `print()`      | Write formatted output |

### Introspection and Reflection

| Function       | Purpose                             |
| -------------- | ----------------------------------- |
| `hash()`       | Get a hash value                    |
| `type()`       | Get or construct a type             |

### Iteration and Sequences

| Function      | Purpose                                  |
| ------------- | ---------------------------------------- |
| `all()`       | Test whether all values are true         |
| `any()`       | Test whether any value is true           |
| `enumerate()` | Iterate with indexes                     |
| `iterator()`  | Get an iterator                          |
| `len()`       | Get the number of items                  |
| `size()`      | Get the number of items                  |
| `bytes()`     | Get the number of bytes owned by item   |
| `next()`      | Get the next iterator item               |
| `range()`     | Create an integer range                  |
| `slice()`     | Create a slice                           |
| `sorted()`    | Return values in sorted order            |
| `zip()`       | Iterate over multiple sequences together |

### Numeric

| Function   | Purpose                    |
| ---------- | -------------------------- |
| `abs()`    | Get the absolute value     |
| `max()`    | Get the maximum value      |
| `min()`    | Get the minimum value      |
| `pow()`    | Raise a value to a power   |
| `round()`  | Round a numeric value      |
| `sum()`    | Sum values                 |

