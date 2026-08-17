Ownership in Severian has four operations:

borrow      read without taking ownership
borrow mut  temporarily obtain exclusive write access
clone       create independent ownership
move        transfer ownership

The compiler prevents:
- use after move
- use after drop
- mutable aliasing
- mutation while shared references require stability
- references escaping their owners
- unsafe ownership transfer across tasks and foreign boundaries

Most parameter ownership effects are inferred.
Explicit ownership syntax is used when the operation itself matters.