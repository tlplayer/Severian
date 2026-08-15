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

pub fn all() -> Vec<Explanation> {
    explanations().into_values().collect()
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
            text: "Functions and methods use a clear single word when possible and snake_case when a boundary is useful. CamelCase spellings may remain callable at compatibility boundaries, but native Severian code receives this warning.",
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
            text: "The spelling remains accepted for migration or external compatibility but is not canonical Severian style. Native Severian declarations and calls prefer one clear word, otherwise snake_case. Apply the diagnostic's replacement when practical.",
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
            code: "E000101",
            title: "Unterminated block string",
            text: "A triple-double-quoted block string reached the end of the source before its closing triple quotes.",
        },
        Explanation {
            code: "E000100",
            title: "Invalid source syntax",
            text: "Severian could not form a valid declaration, statement, expression, or token from this source. The primary label identifies where parsing stopped; repair that construct before acting on later errors, which may be recovery cascades.",
        },
        Explanation {
            code: "E000102",
            title: "Inconsistent indentation",
            text: "Indentation must match an enclosing block and must use spaces rather than tabs.",
        },
        Explanation {
            code: "E000103",
            title: "Invalid package source syntax",
            text: "A dependency source file cannot be parsed by this Severian compiler. The diagnostic identifies the package file, line, column, and rejected source. Fix an editable dependency or select a package/compiler version with compatible syntax.",
        },
        Explanation {
            code: "E000104",
            title: "Required syntax token is missing",
            text: "The parser reached a position where the surrounding construct requires a specific token.\n\nCommon causes:\n  - a missing `:` after a function, conditional, loop, switch header, or switch arm\n  - a missing comma or closing delimiter in a collection or call\n  - an incomplete expression before the newline\n\nWhen the insertion is unambiguous, the diagnostic includes a machine-applicable edit that editors may expose as a Quick Fix.",
        },
        Explanation {
            code: "E000200",
            title: "Invalid program semantics",
            text: "The source is syntactically valid, but its names, types, effects, or language rules are inconsistent. The primary label identifies the expression that could not be resolved. More specific semantic failures use a narrower E0002xx code.",
        },
        Explanation {
            code: "E000202",
            title: "Incompatible types",
            text: "A value does not satisfy the concrete type required at an assignment, return, or call boundary.\n\nHow to read it:\n  - `found` is the type inferred for the highlighted expression\n  - `expected` comes from the receiving declaration or function signature\n  - secondary labels show where that requirement originated when available\n\nTypical fixes:\n  - correct the value when its type is accidental\n  - change the receiving annotation when the declaration is wrong\n  - use an explicit conversion such as `int(value)`, `float(value)`, or `string(value)`\n  - use `int.parse` or `float.parse` when invalid external input must remain recoverable\n\nA suggested conversion is only machine-applicable when it preserves the surrounding syntax and the compiler can construct the complete replacement.",
        },
        Explanation {
            code: "E000203",
            title: "Missing required argument",
            text: "A function or method call omitted a parameter that has no default value. The primary label identifies the incomplete call and a secondary label identifies the parameter declaration when it is in the current source.\n\nThe suggested named argument preserves call syntax, but its placeholder value is marked maybe-incorrect because only the developer can choose the value with the right meaning. Editors may preview the edit but should not apply it silently.",
        },
        Explanation {
            code: "E000204",
            title: "Unknown named argument",
            text: "A call supplies a named argument that is not present in the selected function or method signature. Check the spelling or update the call for the current API. When one parameter name is a close, unambiguous match, Severian provides a machine-applicable rename.",
        },
        Explanation {
            code: "E000205",
            title: "Binding requires initialization",
            text: "Every Severian binding has a value from the moment it enters scope. Add an initializer at the declaration instead of relying on control flow to assign the binding later. This stronger rule prevents possibly-uninitialized reads and keeps all paths deterministic.",
        },
        Explanation {
            code: "E000206",
            title: "Non-exhaustive enum switch",
            text: "A switch over an enum must handle every declared variant, or contain an unguarded wildcard arm. The diagnostic lists variants that are not covered. Exhaustive switches make adding an enum variant a reviewable source change instead of an implicit runtime fallthrough.",
        },
        Explanation {
            code: "E000207",
            title: "Unresolved type escaped semantic analysis",
            text: "The package's `[compiler.type_resolution]` policy rejected a dynamic type that was created by inference fallback, an unresolved name or generic, or lost compiler information. Explicit source-level `Any` remains legal. Fix the originating annotation or inference rule instead of weakening the value after type checking.",
        },
        Explanation {
            code: "E000208",
            title: "Compiler function name is reserved",
            text: "A top-level function declaration reused a name owned by a compiler-provided operation, such as `size`, `len`, `sqrt`, `min`, or `max`. Compiler functions have one stable meaning and cannot be shadowed. Remove the declaration and call the compiler function directly, or choose a domain-specific name such as `element_count`, `square_root`, `minimum`, or `maximum`. Method names remain in their owning type's namespace and are unaffected.",
        },
        Explanation {
            code: "E000209",
            title: "Explicit self parameter",
            text: "Class and trait method receivers are implicit in Severian. Remove the leading `self` parameter; fields, methods, and the receiver value remain available inside the method body.",
        },
        Explanation {
            code: "E000300",
            title: "Invalid ownership operation",
            text: "A read, mutation, move, view, or borrow conflicts with the value's current ownership state. More specific ownership failures use a narrower E0003xx code and identify the operation that established the conflicting state when available.",
        },
        Explanation {
            code: "E000301",
            title: "Use after move",
            text: "Ownership moved to another binding, so the original binding can no longer be read. Use the new owner or clone before moving.",
        },
        Explanation {
            code: "E000302",
            title: "Mutation while viewed",
            text: "A structural mutation conflicts with a live immutable view. End the view's use first or clone an independent value.",
        },
        Explanation {
            code: "E000401",
            title: "Statically known out-of-bounds index",
            text: "The index is outside a collection whose length is known at compile time. Dynamic indices remain runtime bounds checked.",
        },
        Explanation {
            code: "E000501",
            title: "Checked integer overflow",
            text: "Safe arithmetic cannot fit the result in its destination integer type. Choose an explicit wrapping, saturating, or overflow-reporting operation when required.",
        },
        Explanation {
            code: "E000502",
            title: "Compile-time division by zero",
            text: "The divisor is provably zero during semantic analysis, so the expression can never produce a valid result. Remove the operation or define an explicit zero case. E000920 is reserved for divisors that become zero only during execution.",
        },
        Explanation {
            code: "E000601",
            title: "Mutable task call without a lock",
            text: "Mutable state cannot cross an asynchronous boundary without transferring its lock capability. Use `with self and lock` or pass frozen data.",
        },
        Explanation {
            code: "E000701",
            title: "Unapproved unsafe boundary",
            text: "Unsafe capabilities require a source-scoped package permission, and native ABI declarations are restricted to library targets. Applications and tests should use the safe library API.",
        },
        Explanation {
            code: "E000801",
            title: "Unhandled recoverable result",
            text: "A recoverable Result must be propagated, handled, or explicitly discarded with a reviewable reason.",
        },
        Explanation {
            code: "E000902",
            title: "Runtime assertion failed",
            text: "An `assert` condition evaluated to false while the native program was running. Inspect the labeled condition and the values that produced it; assertions should describe invariants rather than replace normal Result handling.",
        },
        Explanation {
            code: "E000910",
            title: "Runtime index out of bounds",
            text: "A dynamic index was outside the valid range of a collection or string. The diagnostic reports both the requested index and the runtime length; validate external indices or check the length before indexing.",
        },
        Explanation {
            code: "E000911",
            title: "Runtime map key not found",
            text: "A map lookup requested a key that was not present. Check membership first, use a default-returning lookup, or handle the missing-key case explicitly.",
        },
        Explanation {
            code: "E000912",
            title: "Runtime slice step is zero",
            text: "A dynamic slice step evaluated to zero. A slice cannot advance with a zero step; use a positive step for forward traversal or a negative step for reverse traversal.",
        },
        Explanation {
            code: "E000920",
            title: "Runtime division by zero",
            text: "A divisor evaluated to zero while native Severian code was executing. Integer division and modulo have no result for a zero divisor; floating-point division is also checked so behavior stays explicit and portable.\n\nCommon causes:\n  - unchecked user, file, network, or model input\n  - a derived count or dimension becoming zero\n  - an empty collection feeding an average or normalization\n  - a conversion producing zero before the division\n\nPossible fixes:\n  - guard the divisor and define the zero case\n  - reject zero at the input boundary\n  - return a recoverable `Result` when zero is expected bad input\n  - use a domain-specific fallback only when it has a clear meaning\n\nExample:\n    def divide(value: int, divisor: int) -> Result[int, DivideError]:\n        if divisor == 0:\n            return failure(DivideError(\"divisor cannot be zero\"))\n        return value / divisor",
        },
        Explanation {
            code: "E000921",
            title: "Invalid runtime conversion",
            text: "A dynamically typed value could not be converted to the concrete type required at this operation or call boundary. Validate the value or use a parsing API that returns Result.",
        },
        Explanation {
            code: "E000980",
            title: "Runtime invariant failure",
            text: "The generated runtime encountered an internal state that valid compiled Severian code should not produce. The first diagnostic includes the available source location and failure detail; report that output with the compiler version.",
        },
        Explanation {
            code: "E000990",
            title: "Unclassified native process failure",
            text: "The native program received a fatal signal before a narrower Severian runtime diagnostic could be produced. Normal execution automatically captures and prints a symbolic call stack; no diagnostics flag or second run is required. Start with the first Severian or `__sev_` frame. Internal mode only adds protocol and artifact metadata for compiler development.",
        },
        Explanation {
            code: "E002401",
            title: "Incompatible tensor dimensions",
            text: "A tensor operation received shapes that violate its source-level dimension requirements. Matrix multiplication requires the left contracting dimension to equal the right contracting dimension.\n\nThe diagnostic prints both operand types and the exact equality that failed. Reshape an operand when its logical shape is wrong, transpose it when the contracting axis is in the wrong position, or correct the declared shape when the annotation is stale. Dynamic dimensions are checked when their concrete runtime shapes become available.\n\nMLIR, StableHLO, LLVM, XLA, and PJRT verifier records are implementation details and appear only in internal diagnostics.",
        },
        Explanation {
            code: "E009900",
            title: "Unclassified compiler diagnostic",
            text: "The compiler identified a source-related failure that has not yet been assigned a narrower stable code. Internal diagnostics include the originating compiler stage. This code should become less common as diagnostic translation coverage expands.",
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
