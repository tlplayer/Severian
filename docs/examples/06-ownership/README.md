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

## Foreign boundaries

FFI ownership is part of the boundary contract rather than ordinary Severian
expression syntax. A foreign declaration must distinguish these four cases:

- Severian-owned values passed to foreign code as borrowed values
- Severian-owned values transferred to foreign code
- foreign-owned values exposed to Severian as borrowed values
- foreign-owned values adopted by Severian

The executable FFI matrix will be added with the boundary declaration syntax;
until then these rules live here rather than in a `.sev` file containing prose.
