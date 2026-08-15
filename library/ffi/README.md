# Foreign-function interface

`ffi` owns the source-level vocabulary for Severian's stable C ABI. Domain
packages use these types in package-private `extern` declarations; safe public
APIs should expose domain values instead of raw handles or output parameters.

The compiler recognizes these wrappers at a C v1 boundary and generates the
conversion shim. Providers receive only stable scalars, views, handles, and
output structures—not compiler-owned Severian values.
