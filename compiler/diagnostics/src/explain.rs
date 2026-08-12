use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Explanation {
    pub code: &'static str,
    pub title: &'static str,
    pub text: &'static str,
}

pub fn explain(code: &str) -> Option<Explanation> {
    explanations().remove(code)
}

fn explanations() -> BTreeMap<&'static str, Explanation> {
    [
        Explanation {
            code: "N001",
            title: "Variable naming",
            text: "Variables, parameters, fields, and bindings use snake_case. Conventional one-letter coordinates and indices remain valid.",
        },
        Explanation {
            code: "N002",
            title: "Function naming",
            text: "Functions and methods use snake_case. The only mixed-case exception is the narrow coordinate accessor set getX/getY/getZ and setX/setY/setZ.",
        },
        Explanation {
            code: "N003",
            title: "Type naming",
            text: "Concrete types use PascalCase, such as TensorShape and HttpServer, unless a registered scientific spelling such as ReLU applies.",
        },
        Explanation {
            code: "N004",
            title: "Constant naming",
            text: "Top-level typed constants use SCREAMING_SNAKE_CASE.",
        },
        Explanation {
            code: "N005",
            title: "Module naming",
            text: "Packages, modules, and import aliases use snake_case and ordinary words are written in full.",
        },
        Explanation {
            code: "N006",
            title: "Decorator naming",
            text: "Decorator names and their package path segments use snake_case.",
        },
        Explanation {
            code: "N007",
            title: "Deprecated compatibility spelling",
            text: "The spelling remains accepted for migration but is not canonical Severian style. Apply the diagnostic's replacement when practical.",
        },
        Explanation {
            code: "N010",
            title: "Canonical technical spelling",
            text: "A small registry preserves spellings whose capitalization carries established technical meaning, including XLA, MLIR, CUDA, PJRT, BERT, and GPT.",
        },
        Explanation {
            code: "N011",
            title: "Canonical scientific operator spelling",
            text: "Named scientific constructs preserve conventional spellings such as ReLU, GELU, SiLU, LSTM, RMSNorm, and Conv2D; functional operations remain snake_case.",
        },
        Explanation {
            code: "E0201",
            title: "Inferred Any in a type-safe package",
            text: "The package enables `[package] type-safe = true`, but a parameter or field has no annotation and would default to `Any`. Add a concrete type to improve checking and optimization, or write `Any` explicitly when the dynamic boundary is intentional.",
        },
        Explanation {
            code: "E0101",
            title: "Unterminated block string",
            text: "A triple-double-quoted block string reached the end of the source before its closing triple quotes.",
        },
        Explanation {
            code: "E0102",
            title: "Inconsistent indentation",
            text: "Indentation must match an enclosing block and must use spaces rather than tabs.",
        },
        Explanation {
            code: "E0103",
            title: "Invalid package source syntax",
            text: "A dependency source file cannot be parsed by this Severian compiler. The diagnostic identifies the package file, line, column, and rejected source. Fix an editable dependency or select a package/compiler version with compatible syntax.",
        },
        Explanation {
            code: "E0202",
            title: "Incompatible types",
            text: "A value does not satisfy the concrete type required at this assignment, return, or call boundary. Convert the value or correct the declaration.",
        },
        Explanation {
            code: "E0301",
            title: "Use after move",
            text: "Ownership moved to another binding, so the original binding can no longer be read. Use the new owner or clone before moving.",
        },
        Explanation {
            code: "E0302",
            title: "Mutation while viewed",
            text: "A structural mutation conflicts with a live immutable view. End the view's use first or clone an independent value.",
        },
        Explanation {
            code: "E0401",
            title: "Statically known out-of-bounds index",
            text: "The index is outside a collection whose length is known at compile time. Dynamic indices remain runtime bounds checked.",
        },
        Explanation {
            code: "E0501",
            title: "Checked integer overflow",
            text: "Safe arithmetic cannot fit the result in its destination integer type. Choose an explicit wrapping, saturating, or overflow-reporting operation when required.",
        },
        Explanation {
            code: "E0601",
            title: "Mutable task call without a lock",
            text: "Mutable state cannot cross an asynchronous boundary without transferring its lock capability. Use `with self and lock` or pass frozen data.",
        },
        Explanation {
            code: "E0701",
            title: "Unapproved unsafe boundary",
            text: "Unsafe capabilities require a source-scoped package permission, and native ABI declarations are restricted to library targets. Applications and tests should use the safe library API.",
        },
        Explanation {
            code: "E0801",
            title: "Unhandled recoverable result",
            text: "A recoverable Result must be propagated, handled, or explicitly discarded with a reviewable reason.",
        },
        Explanation {
            code: "dead-code::function",
            title: "Unreachable function",
            text: "The function cannot be reached from main, exported functions, tests, native ABI roots, or another reachable function. Remove it or make the intended entry path explicit.",
        },
        Explanation {
            code: "W001",
            title: "Unused binding",
            text: "A binding is created but never read. Remove the binding, use the value, or prefix the name with `_` when the unused value is intentional.",
        },
        Explanation {
            code: "W002",
            title: "Discarded task",
            text: "A task handle is created and immediately discarded. This can hide cancellation, failure, ordering, and lifetime bugs. Bind and await it unless detached execution is intentional.",
        },
        Explanation {
            code: "lint::discarded-send",
            title: "Discarded asynchronous send",
            text: "Channel send returns asynchronous work. Discarding it can make delivery ordering or send failure invisible to the caller.",
        },
        Explanation {
            code: "verify::duplicate-function",
            title: "Duplicate HIR function",
            text: "The compiler produced multiple top-level HIR functions with the same identity. This is an internal compiler invariant failure rather than a normal source error.",
        },
        Explanation {
            code: "coverage::threshold",
            title: "Coverage below threshold",
            text: "Observed source coverage is below the minimum configured for the build. Add tests or intentionally change the package/workspace threshold.",
        },
        Explanation {
            code: "doctor::missing-tool",
            title: "Required compiler tool missing",
            text: "A tool needed by the selected compiler pipeline was not found. `sev doctor` reports the exact missing component before a build reaches code generation.",
        },
    ]
    .into_iter()
    .map(|explanation| (explanation.code, explanation))
    .collect()
}
