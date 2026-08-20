# Core primitive bootstrap contract

This package is the canonical source-language declaration database for
Severian primitive types. The compiler must load it before user semantic
analysis and must fail bootstrap when it is missing or malformed.

The compiler recognizes the structural `Primitive` protocol. A declaration's
identity comes from the declaration itself; its `category`, `representation`,
`bits`, `signed`, and `default_literal` properties describe the limited facts
needed by semantic analysis and lowering. The compiler must not maintain a
second list of user-visible primitive names.

Adding a primitive consists of adding a declaration here, teaching semantic
capability rules about a genuinely new category if necessary, and adding
backend support for a genuinely new representation. Ordinary expression,
generic, HIR, MIR, parser, and call-analysis code must remain unchanged.

Tensor and other parameterized library abstractions are not primitives.
