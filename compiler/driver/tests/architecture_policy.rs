use severian_package::BuildPolicy;
use std::path::Path;

#[test]
fn repository_respects_declared_hard_file_limits() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let policy = BuildPolicy::for_input(&root).unwrap();
    let findings = severian_driver::architecture::check_file_budgets(&policy).unwrap();
    let errors = findings
        .iter()
        .filter(|finding| finding.severity == "error")
        .map(|finding| format!("{}: {}", finding.path.display(), finding.message))
        .collect::<Vec<_>>();
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
