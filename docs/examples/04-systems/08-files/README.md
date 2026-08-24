# Files

The file library builds typed, owned handles on top of filesystem paths and IO
streams. Opening a file returns a resource whose close operation runs exactly
once, including during error propagation.

The examples cover text and byte access, format dispatch, memory mapping, and
advisory locking. Convenience functions such as `file.read(path)` are composed
from the same reader/writer contracts as explicit handles.

Memory maps and locks are resources. Data borrowed from a mapping cannot escape
the mapping, and a lock token cannot be copied or unlocked twice.
