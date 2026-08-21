# Environment

The environment library exposes process environment state through a platform
provider. Reads return an optional string or an explicit fallback. Mutation is
process-global and therefore belongs in isolated integration tests.

An immutable snapshot is preferred when several values must be read
consistently or passed into another component. Environment mutation does not
silently change compiler configuration, package resolution, or published build
semantics.
