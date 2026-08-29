# Prelude types

API ID: `prelude.type.any`

Always-visible constructors include `Any`, `Result`, `Option`, `Function`, `Channel`, `Buffer`, `Path`, pointer, and collection types. Type arguments resolve through ordinary generic application.

```sev
def optional_value(value: int | None) -> int | None:
    return value
```

Current weakness: several container method contracts still live only in package source rather than generated reference pages.
