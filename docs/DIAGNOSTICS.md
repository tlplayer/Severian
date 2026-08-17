# Severian diagnostics contract

Severian diagnostics are source-level explanations, not stringified compiler
failures. Terminal output, JSON output, editor integrations, and `sev explain`
must all be projections of the same structured diagnostic.
| Code | Responsibility |
| --- | --- |
| `E001xxx` | Errors at compile-time or runtime |
| `W001xxx` | Warning |
| `N001xxx` | Note/additional diagnostic data |
| `C001xxx` | Compiler defect / internal compiler failure |
| `[ErrorType:code]` | User defined errors |


## Stable error namespace

Errors use `E` followed by exactly six decimal digits. Severian does not use a
separate runtime prefix: compilation and execution share one searchable
namespace.

| Range | Responsibility |
| --- | --- |
| `E0001xx` | lexing and syntax |
| `E0002xx` | names, calls, types, traits, and contracts |
| `E0003xx` | ownership and borrowing |
| `E0004xx` | memory and bounds safety |
| `E0005xx` | arithmetic safety |
| `E0006xx` | concurrency and task safety |
| `E0007xx` | unsafe capabilities and foreign boundaries |
| `E0008xx` | recoverable results and effects |
| `E0009xx` | runtime failures |
| `E0024xx` | tensors, shapes, and accelerator operations |

Codes are never reused for a different meaning. A broad code may be replaced
by a new, narrower code, but existing documentation remains a stable reference.
Warnings and naming guidance retain their existing `W` and `N` namespaces.
