//! Bootstrap adapter for package.diagnostic.lint's explicit-imports rule.
//! Source import corrections run as the final lint step before compilation.
use severian_ast::Item;
use severian_diagnostics::Diagnostic;
use severian_modules::ModuleGraph;

/// Final lint step shared by build, run, check and test. Plan before writing;
/// callers resolve fresh sources for compilation after corrections are applied.
pub fn correct_imports(graph: &ModuleGraph, root: &std::path::Path) -> Result<crate::explicit_imports::Plan, String> {
    let plan = import_corrections(graph, root, true)?;
    crate::explicit_imports::apply(&plan)?;
    Ok(plan)
}

/// Explicit lint requests and default build lint share the correction engine.
pub fn import_corrections(graph: &ModuleGraph, root: &std::path::Path, automatic: bool) -> Result<crate::explicit_imports::Plan, String> {
    if automatic { crate::explicit_imports::plan_compiled(graph, root) }
    else { crate::explicit_imports::plan(graph, root) }
}

pub fn explicit_imports(graph: &ModuleGraph) -> Result<(), Diagnostic> {
    for module in &graph.modules {
        if module.path.components().any(|p|p.as_os_str()=="package.pkg") {continue;}
        let policy=graph.policies.get(&module.package);
        // Builds narrow these imports automatically. Retain the source-style
        // lint for packages explicitly opting out of automatic narrowing.
        if policy.is_none_or(|p|p.narrow_imports) {continue;}
        if policy.is_some_and(|p|!p.lint_enabled || !p.explicit_imports || p.explicit_imports_level=="off") {continue;}
        let level=policy.map_or("warning",|p|p.explicit_imports_level.as_str());
        let tokens=severian_lexer::scan(&module.source)?;
        let lines:Vec<_>=module.source.text.lines().collect();
        for import in module.ast.items.iter().filter_map(|i|if let Item::Import(i)=i {Some(i)} else {None}).filter(|i|i.is_wildcard() && i.alias.is_none()) {
            let span=tokens.iter().find(|t| t.span.start>=import.span.start && t.span.end<=import.span.end && matches!(t.kind,severian_lexer::TokenKind::Star)).map_or(import.span,|t|t.span);
            if tokens.iter().rev().find(|token| token.span.end <= span.start)
                .is_some_and(|token| matches!(token.kind, severian_lexer::TokenKind::Comma)) { continue; }
            let line=module.source.text[..span.start as usize].bytes().filter(|b|*b==b'\n').count();
            let suppressed=lines.iter().enumerate().any(|(index,text)| {
                let text=text.trim();
                let prefix=if text.starts_with("# sev-lint: allow ") {"# sev-lint: allow "}
                    else if index+1==line && text.starts_with("# sev-lint-next-line: allow ") {"# sev-lint-next-line: allow "}
                    else {return false;};
                let offset=lines[..index].iter().map(|l|l.len()+1).sum::<usize>();
                if tokens.iter().any(|t|t.span.start as usize<=offset && t.span.end as usize>offset) {return false;}
                text[prefix.len()..].split(',').any(|id|id.trim()=="L0015")
            });
            if suppressed {continue;}
            let diagnostic=Diagnostic::new("L0015", "wildcard import hides the names used by this module; run sev --lint to replace * with the names used",Some(span)).with_source(module.source.clone());
            if level=="error" {return Err(diagnostic);}
            eprintln!("{level}: {diagnostic}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod correction_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn lint_corrects_imports_before_strict_policy_including_source_dependencies() {
        let root = std::env::temp_dir().join(format!("sev-lint-correction-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let project = root.join("project");
        let dependency = root.join("dependency");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::create_dir_all(&dependency).unwrap();
        let entry = project.join("main.sev");
        let facade = dependency.join("lib.sev");
        std::fs::write(dependency.join("leaf.sev"), "def answer() -> int:\n    return 42\n").unwrap();
        std::fs::write(&facade, "import * from \"leaf.sev\" as leaf\ndef answer() -> int:\n    return leaf.answer()\n").unwrap();
        std::fs::write(&entry, "import * from \"../dependency/lib.sev\"\ndef main() -> int:\n    return answer()\n").unwrap();
        let mut graph = severian_modules::resolve(&entry).unwrap();
        let package = graph.modules.last().unwrap().package;
        graph.policies.insert(package, severian_modules::PackagePolicy {
            narrow_imports: true, library: facade.clone(), explicit_imports: true,
            lint_enabled: true, explicit_imports_level: "error".into(),
        });
        assert_eq!(severian_modules::validate_import_policy(&graph).unwrap_err().code, "E000125");
        let plan = correct_imports(&graph, &project).unwrap();
        assert_eq!(plan.files.len(), 2);
        assert!(std::fs::read_to_string(&entry).unwrap().contains("import answer from"));
        assert!(std::fs::read_to_string(&facade).unwrap().contains("import \"leaf.sev\" as leaf"));
        let mut corrected = severian_modules::resolve(&entry).unwrap();
        corrected.policies = graph.policies;
        severian_modules::validate_import_policy(&corrected).unwrap();
        assert!(correct_imports(&corrected, &project).unwrap().files.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disabled_package_lint_preserves_source() {
        let root = std::env::temp_dir().join(format!("sev-lint-disabled-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&root).unwrap();
        let entry = root.join("main.sev");
        let text = "import * from \"lib.sev\"\ndef main() -> int:\n    return answer()\n";
        std::fs::write(&entry, text).unwrap();
        std::fs::write(root.join("lib.sev"), "def answer() -> int:\n    return 42\n").unwrap();
        let mut graph = severian_modules::resolve(&entry).unwrap();
        let package = graph.modules.last().unwrap().package;
        graph.policies.insert(package, severian_modules::PackagePolicy {
            narrow_imports: false, library: root.join("lib.sev"), explicit_imports: true,
            lint_enabled: false, explicit_imports_level: "warning".into(),
        });
        assert!(correct_imports(&graph, &root).unwrap().files.is_empty());
        assert_eq!(std::fs::read_to_string(&entry).unwrap(), text);
        std::fs::remove_dir_all(root).unwrap();
    }
}
