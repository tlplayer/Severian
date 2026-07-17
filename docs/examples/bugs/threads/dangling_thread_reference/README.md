# Dangling task reference

A returned task may run after `launch` has released its locals, so borrowed
captures cannot cross that boundary. Moving task-owned data makes its lifetime
independent of the creator.

Mitigation: **compile-time** for escaping captures and **scheduler-level** for
the structured scopes that join non-escaping children.
