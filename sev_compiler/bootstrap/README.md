# Compatibility entry

The compiler package is now [`sev_compiler`](../README.md). Run `sev build`
from its root and use `sev_compiler FILE.sev` from PATH. These source files
retain imports for older callers; the implementation lives in the normal
compiler boundary and frontend modules.
