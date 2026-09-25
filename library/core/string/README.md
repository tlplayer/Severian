# Core string

`core.string` owns string-specific library behavior outside the compiler.

- `core.string.format` implements conversions, formatting, quoting, and
  structured text rendering. The default string facade exports this package.
- `core.string.regex` owns regular expressions and its POSIX provider. Select
  this package explicitly so ordinary formatting does not require regex.

Future text-processing facilities, such as sed-style transformations, belong
under this namespace; no `sed` API is implemented here yet. General utilities
such as assertions, zip, map, and sum belong to `core.util`.

Primitive string representation and compiler boundaries remain where they are.
The old compiler formatting path, `core/text/src/format.sev`, and the `regex`
package forward to these implementations for compatibility.
