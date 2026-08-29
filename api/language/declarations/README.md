# Declarations

API ID: `declaration.item`

Imports, traits, classes, extensions, enums, bindings, expressions, functions, types, and tests exactly mirror `ast::Item`. Function bodies and generic substitutions remain available for downstream specialization where interfaces support them.

```sev
def declaration_probe[T](value: T) -> T:
    return value
```

Declarations enter the module symbol table before typed HIR lowering. Current weakness: per-declaration public/private visibility is not yet a complete language contract.
