# Primitive declarations

`declarations.sev` owns primitive identities, representations and literal
metadata through concrete implementations of `syntax.Primitive`.
`catalog.sev` derives descriptors through that trait; its scalar catalog is a
compatibility view for existing lowering consumers.

`number.sev` recognizes numeric spelling. `literal.sev` normalizes it without
floating-point rounding and constructs a literal carrying its primitive family,
default type and span. Expected-type selection remains semantic analysis work.

There is one default per literal syntax. A primitive accepting a literal is
separate from being that syntax's default. The lexer does not execute ordinary
runtime constructors.

The retained buffer identity encoding and remaining Rust catalog migration are
recorded in [the migration status](../PIPELINE_MIGRATION.md).
