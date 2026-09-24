# CFG construction

This package owns the executable CFG builder used between resolved semantic
bodies and MIR ownership analysis. It creates blocks and terminators and attaches
source/callable provenance to every emitted operation, independently of debugger
instrumentation.

The current carrier types remain re-exported by universal during the HIR/MIR
model migration. Ownership and storage analysis live in `mir/ownership`.
