# Use after move

`move()` transfers the allocation and invalidates its source binding. The
diagnostic highlights the later `packet` use, not the move site, so the repair is
local and clear.

Mitigation: **compile-time**. Use the moved binding, borrow before the move, or
clone when two independent owners are actually required.
