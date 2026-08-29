# Expressions

API ID: `expression.kind`

Every public `ast::ExpressionKind` is an independently queryable member: literals, collections and comprehensions, names, members, indexing, slicing, generic application, calls, async/await, conditionals, fallbacks, throws, and operators.

```sev
def expression_probe(values: list[int]) -> int:
    return values[0] + 1 if size(values) > 0 else 0
```

Semantic analysis produces typed HIR and ownership decisions before structural lowering. Current weakness: experimental mock/concurrency expressions have stable identities but incomplete behavioral-oracle coverage.
