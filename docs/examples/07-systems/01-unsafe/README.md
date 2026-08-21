# Unsafe boundaries

`unsafe` grants access to operations whose invariants the compiler cannot prove;
it does not disable type checking, ownership, or lifetime tracking.

The examples establish that:

- unsafe authority is lexical and does not leak through an ordinary call;
- raw pointer creation, pointer arithmetic, unchecked casts, and direct foreign
  calls require unsafe authority;
- safe functions may contain a small unsafe implementation and expose a checked
  result to callers;
- an unsafe operation still has a precise type and ownership effect.

Keep unsafe regions narrow. Validate lengths, alignment, nullability, ownership,
and lifetime assumptions before entering the region whenever possible.
