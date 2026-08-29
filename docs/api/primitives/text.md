# Characters, strings, and errors

API ID: `primitive.text_and_storage`

`char` is one character value. `string` is text storage. `Error` currently has
a string representation but is semantically the root error type; sharing a
representation does not make the types interchangeable.

```sev
def greet(name: string) -> string:
    return "hello " + name

def require_name(name: string) -> string | Error:
    if name == "":
        throw Error("name is empty")
    return greet(name)
```

Strings are not assumed Copy. Calls and containers follow their declared
borrow/move contract. Escapes and interpolation are source syntax layered over
the same string value.

Current weakness: TextMate highlighting recognizes literals and interpolation,
but semantic escape validation and ownership classifications require the future
language-server path.
