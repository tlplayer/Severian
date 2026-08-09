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
            code: "dead-code::function",
            title: "Unreachable function",
            text: "The function cannot be reached from main, exported functions, tests, native ABI roots, or another reachable function. Remove it or make the intended entry path explicit.",
        },
        Explanation {
            code: "lint::unused-binding",
            title: "Unused binding",
            text: "A binding is created but never read. Remove the binding, use the value, or prefix the name with `_` when the unused value is intentional.",
        },
        Explanation {
            code: "lint::discarded-task",
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
