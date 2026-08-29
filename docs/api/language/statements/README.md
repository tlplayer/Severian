# Statements

API ID: `statement.kind`

Every public AST statement is inventoried: bindings and mutation, defer/return/assert, unsafe and placement blocks, structured errors, branches, loops, match, and select. Mutable declaration (`:=`), update, and error preservation (`?=`) remain distinct flags.

```sev
def statement_probe(values: list[int]) -> int:
    total := 0
    for value in values:
        total += value
    return total
```

MIR makes control transfer explicit. Current weakness: not every placement policy has equivalent executable backend coverage.
