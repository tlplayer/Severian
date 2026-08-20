# Literals

Literal syntax is recognized by the lexer and parser. Semantic analysis asks
the primitive catalog for the unique declaration marked `default_literal` in
the corresponding structural category. Literal spelling never fabricates a
compiler-owned language type.
