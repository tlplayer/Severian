# Task outlives scope

Returning a task makes its lifetime explicit. It cannot borrow stack data from
the function that created it; move captured inputs into the task, or create the
task in a structured scope and await it before leaving.

Mitigation: **compile-time** and **scheduler-level**.
