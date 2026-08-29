# Effects

API ID: `effect.placement`

Effects describe observable mutation, I/O, allocation, async suspension, unsafe access, and device execution. They are contract data attached to calls and regions rather than symbol suffixes or backend guesses.

```sev
def pure_effect_probe(left: int, right: int) -> int:
    return left + right
```

Fusion and scheduling must preserve effect order. Current weakness: the public syntax for user-declared effect sets is not yet complete.
