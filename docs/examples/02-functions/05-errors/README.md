T
    guaranteed value

T | None
    optional value

T | Error
    recoverable failure

T | None | Error
    optional and fallible

=
    take the successful value and propagate errors

?=
    preserve the complete union

.default(value)
    replace an error with a fallback

throw
    create or rethrow a recoverable error

try / catch
    explicitly handle recoverable errors

panic
    unrecoverable invariant failure


Error-handling strictness

Severian is designed to allow error handling to be tightened as a project
matures.

During development, errors may propagate through intermediate wrappers without
requiring every wrapper to add handling logic.

For example:

def leaf() -> Error:
    throw Error("failure")




def wrapper() -> int:
    leaf()
    return 1

The error may continue through wrapper to the caller.

A stricter compiler configuration may require the effective failure type to be
declared:

def wrapper() -> int | Error:
    leaf()
    return 1

Increasing error strictness should require more accurate declarations and
handling at important boundaries. It should not require changing every normal
call into explicit result-handling syntax.