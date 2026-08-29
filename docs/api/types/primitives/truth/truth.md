# Truth Value Testing

Any object can be tested for truth value, for use in an if or while condition or as operand of the Boolean operations below.

By default, an object is considered true unless its class defines either a _bool() method that returns False or a _len() or _size() method that returns zero, when called with the object. [1] If one of the methods raises an error when called, the error is propagated and the object does not have a truth value (for example, NotImplemented). Here are most of the built-in objects considered false:

    constants defined to be false: None and False

    zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)

    empty sequences and collections: '', (), [], {}, set(), range(0)

Operations and built-in functions that have a Boolean result always return 0 or False for false and 1 or True for true, unless otherwise stated. (Important exception: the Boolean operations or and and always return one of their operands.)


