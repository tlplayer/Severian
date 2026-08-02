# Mathematical model symbols

This example keeps the model vocabulary close to its formula. `X` is the input
vector, `Relu(X)` is the activation, `FastSigmoid([0.0])` demonstrates another
named activation, and `J(X)` is the flattened ReLU activation Jacobian.

`@models(Relu, FastSigmoid, J)` is a Severian symbol pack: it makes those
spellings resolve through the `models` package only within `main`. It is not a
Python decorator, does not wrap `main`, and does not select CPU or GPU
execution. Placement stays at the operation that distributes work, for example
`with self and local:`.

The package implements the piecewise scalar formulas as native-lowered
conditional expressions, then builds the vector and Jacobian forms with normal
Severian loops.
