# Syntax

API ID: `syntax.module`

Severian modules are newline- and indentation-structured. A module contains declarations, bindings, and expressions; indentation owns block extent, while parentheses and brackets own multiline expression extent. The lexer preserves spans so diagnostics and coverage map back to source.

```sev
def syntax_probe(value: int) -> int:
    # The indented suite is the function body.
    return value + 1
```

Lowering begins with tokens and an ordered `ast::Module`; whitespace is never reconstructed later. Current weakness: the complete lexical grammar is not yet generated from this catalogue.
