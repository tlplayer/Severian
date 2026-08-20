# Package interface boundary

The interface boundary serializes and validates public package information for `.pkg` and `.pkgi` consumers.

## Model

Interface structs are versioned data-transfer objects. They are converted to and from universal definitions at a controlled boundary.

The interface crate may contain:

- Serialized stable declaration IDs.
- Qualified paths.
- Public signatures and constraints.
- Versioned representation metadata.
- Compatibility and integrity validation.

It may not contain:

- The live compiler `TypeStore`.
- Operator or literal resolution algorithms.
- Primitive lookup by mutable process-global catalog.
- Source loading.
- Backend spelling.

A method such as `supports("+")` is semantic resolution and belongs in `compiler/universal`, not on an interface DTO.
