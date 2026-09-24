//! Mechanical import edits based on the compiler's resolved module graph.
use serde::Serialize;
use severian_ast::{ImportSubject, Item};
use severian_modules::{ModuleGraph, ModuleId};
use severian_semantic::{ProgramIndex, Resolution};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
pub struct FileEdit {
    pub path: PathBuf,
    pub before: String,
    pub after: String,
    pub imports: usize,
}
#[derive(Debug, Default, Serialize)]
pub struct Plan {
    pub files: Vec<FileEdit>,
    pub notes: Vec<String>,
}

#[derive(Clone)]
struct Word { text: String, start: usize, end: usize }

fn words(module: &severian_modules::ResolvedModule) -> Result<Vec<Word>, String> {
    let tokens = severian_lexer::scan(&module.source).map_err(|e| e.to_string())?;
    Ok(tokens.into_iter().filter(|t| t.span.start < t.span.end).map(|t| Word {
        text: module.source.text[t.span.start as usize..t.span.end as usize].to_owned(),
        start: t.span.start as usize, end: t.span.end as usize,
    }).collect())
}
fn wildcard(import: &severian_ast::ImportDeclaration) -> bool {
    import.is_wildcard()
}
fn selected(import: &severian_ast::ImportDeclaration) -> Option<&str> {
    import.selected_name()
}
fn same_binding(index: &ProgramIndex, module: ModuleId, target: ModuleId, name: &str) -> bool {
    let value = index.modules[&module].scope.bindings.get(name);
    value.is_some() && value == index.exports[&target].get(name)
}

/// Plans all edits before writing. Dependencies outside `root` are read only;
/// generated package.pkg sources are owned by their generators.
pub fn plan(graph: &ModuleGraph, root: &Path) -> Result<Plan, String> {
    plan_with_mode(graph, root, Some(true))
}

/// Successful builds show used names while retaining extensibility by default.
pub fn plan_compiled(graph: &ModuleGraph, root: &Path) -> Result<Plan, String> {
    plan_with_mode(graph, root, None)
}

fn plan_with_mode(graph: &ModuleGraph, root: &Path, explicit: Option<bool>) -> Result<Plan, String> {
    let build_plan = severian_semantic::import_plan(graph).map_err(|e|e.to_string())?;
    let index = severian_semantic::import_index(graph).map_err(|e| e.to_string())?;
    let root = std::fs::canonicalize(root).map_err(|e| e.to_string())?;
    let mut result = Plan::default();
    // Choose one provider for each identity, preferring its defining module.
    // Diamond imports must not turn into repeated named bindings.
    let mut providers = BTreeMap::new();
    for module in &graph.modules {
        for (name,resolution) in &index.modules[&module.id].scope.bindings {
            let owner = match resolution { Resolution::Def(id)=>Some(ModuleId(id.module)), _=>None };
            if owner == Some(module.id) {continue;}
            let imports: Vec<_> = module.ast.items.iter().filter_map(|i|if let Item::Import(i)=i {Some(i)}else{None}).collect();
            if imports.iter().any(|i|selected(i).is_some_and(|n|i.alias.as_deref().unwrap_or(n)==name)) {continue;}
            let provider = imports.iter().filter(|i|wildcard(i)&&i.alias.is_none()).filter_map(|i| {
                let edge=module.imports.iter().find(|e|e.span==i.span)?;
                same_binding(&index,module.id,edge.module,name).then_some((owner!=Some(edge.module),i.span.start))
            }).min();
            if let Some((_,start))=provider {providers.insert((module.id,name.clone()),start);}
        }
    }
    let mut required: BTreeMap<ModuleId, BTreeSet<String>> = graph.modules.iter().map(|m| (m.id, BTreeSet::new())).collect();
    let mut lexed = BTreeMap::new();
    for module in &graph.modules {
        let tokens = words(module)?;
        // Build and source conversion share AST-derived requirements. Include
        // downstream requests that arrive through facade import edges too.
        for name in build_plan.requirements.get(&module.id).into_iter().flatten() {
            required.get_mut(&module.id).unwrap().insert(name.split('.').next().unwrap_or(name).to_owned());
        }
        for ((consumer,_),names) in &build_plan.selections {
            if *consumer == module.id { required.get_mut(&module.id).unwrap().extend(names.iter().cloned()); }
        }
        // Preserve explicitly promised APIs even when this build has no consumers.
        for directory in module.path.ancestors().skip(1) {
            let manifest = directory.join("package.json");
            if !manifest.is_file() { continue; }
            let text = std::fs::read_to_string(&manifest).map_err(|e| e.to_string())?;
            let value: serde_json::Value = json5::from_str(&text).map_err(|e| e.to_string())?;
            let library = value["lib"]["path"].as_str().unwrap_or("src/lib.sev");
            if directory.join(library) == module.path {
                if let Some(exports) = value["package"]["export"].as_array() {
                    required.get_mut(&module.id).unwrap().extend(exports.iter().filter_map(|v|v.as_str().map(str::to_owned)));
                } else if module.ast.items.iter().all(|i|matches!(i, Item::Import(_))) {
                    // A facade without an explicit API cannot be narrowed safely.
                    required.get_mut(&module.id).unwrap().extend(index.exports[&module.id].keys().cloned());
                    result.notes.push(format!("{}: declare package.export to narrow the facade API", manifest.display()));
                }
            }
            break;
        }
        lexed.insert(module.id, tokens);
    }
    // Consumer requirements flow back through re-export chains, including cycles.
    loop {
        let mut changed = false;
        for module in &graph.modules {
            let names = required[&module.id].clone();
            for import in module.ast.items.iter().filter_map(|i|if let Item::Import(i)=i {Some(i)}else{None}) {
                let Some(edge) = module.imports.iter().find(|e|e.span==import.span) else {continue};
                if let Some(name) = selected(import) {
                    let exposed = import.alias.as_deref().unwrap_or(name);
                    if names.contains(exposed) { changed |= required.get_mut(&edge.module).unwrap().insert(name.to_owned()); }
                } else if wildcard(import) && import.alias.is_none() {
                    for name in &names {
                        if providers.get(&(module.id,name.clone()))==Some(&import.span.start) { changed |= required.get_mut(&edge.module).unwrap().insert(name.clone()); }
                    }
                }
            }
        }
        if !changed { break; }
    }
    for module in &graph.modules {
        if !module.path.starts_with(&root) || module.path.components().any(|c|c.as_os_str()=="package.pkg") { continue; }
        // Nested dependency packages have their own sources and build policy.
        if module.path.ancestors().skip(1).take_while(|directory| *directory != root)
            .any(|directory| directory.join("package.json").is_file()) { continue; }
        let tokens = &lexed[&module.id];
        let mut edits: Vec<(usize,usize,String)> = Vec::new();
        let mut supplied = BTreeSet::new();
        let mut count = 0;
        for import in module.ast.items.iter().filter_map(|i|if let Item::Import(i)=i {Some(i)}else{None}) {
            if !wildcard(import) || import.alias.is_some() { continue; }
            if !module.imports.iter().any(|edge| edge.span == import.span) { continue; }
            let locator = match &import.subject {
                ImportSubject::Locator(locator) => locator.as_str(),
                ImportSubject::Name(_) => import.source.as_deref().expect("package wildcard source"),
            };
            let mut members = Vec::new();
            for name in &required[&module.id] {
                if providers.get(&(module.id,name.clone()))==Some(&import.span.start) && supplied.insert(name.clone()) {
                    members.push(name.clone());
                }
            }
            let closed = explicit.unwrap_or_else(|| graph.policies.get(&module.package).is_some_and(|policy| policy.narrow_imports));
            let member_only = module.source.text[import.span.start as usize..import.span.end as usize].trim() == "*";
            // An existing explicit list already displays its known names. In
            // closed mode remove only its wildcard and adjacent comma.
            if closed && member_only && members.is_empty() {
                let start = import.span.start as usize;
                let end = import.span.end as usize;
                let previous = tokens.iter().rev().find(|token| token.end <= start);
                let next = tokens.iter().find(|token| token.start >= end);
                if let Some(comma) = previous.filter(|token| token.text == ",") {
                    if !module.source.text[comma.start..end].contains('#') {
                        edits.push((comma.start, end, String::new()));
                        count += 1;
                    }
                    continue;
                }
                if let Some(comma) = next.filter(|token| token.text == ",") {
                    if !module.source.text[start..comma.end].contains('#') {
                        edits.push((start, comma.end, String::new()));
                        count += 1;
                    }
                    continue;
                }
            }
            if members.is_empty() {
                // Importing can register extensions or execute initializers. Do
                // not erase a dependency merely because it contributes no names.
                result.notes.push(format!("{}: no named uses of `{locator}`; retained for review of initialization/extension effects",module.path.display()));
                continue;
            }
            let source = if matches!(import.subject, ImportSubject::Locator(_)) {serde_json::to_string(locator).map_err(|e|e.to_string())?} else {locator.to_owned()};
            if module.source.text[import.span.start as usize..import.span.end as usize].contains('#') {
                result.notes.push(format!("{}: retained import containing an internal comment", module.path.display()));
                continue;
            }
            if !closed { members.push("*".to_owned()); }
            let replacement = if member_only { members.join(", ") }
                else { format!("import {} from {source}", members.join(", ")) };
            edits.push((import.span.start as usize,import.span.end as usize,replacement));
            count+=1;
        }
        if edits.is_empty() {continue;}
        edits.sort_by_key(|e|e.0);
        if edits.windows(2).any(|pair|pair[0].1>pair[1].0) {return Err(format!("{}: overlapping import edits",module.path.display()));}
        let mut after = module.source.text.to_string();
        for (start,end,text) in edits.into_iter().rev() {after.replace_range(start..end,&text);}
        let source = severian_source::SourceFile::virtual_source(module.path.clone(), after.clone());
        severian_parser::parse(&severian_lexer::scan(&source).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        result.files.push(FileEdit {path:module.path.clone(),before:module.source.text.to_string(),after,imports:count});
    }
    result.notes.sort(); result.notes.dedup();
    Ok(result)
}

/// Refuse stale plans before writing any source. The editor applies the same
/// plan through WorkspaceEdit so its changes can be undone normally.
pub fn apply(plan: &Plan) -> Result<(), String> {
    for edit in &plan.files {
        if std::fs::read_to_string(&edit.path).map_err(|e|e.to_string())? != edit.before {
            return Err(format!("{} changed after import analysis; no edits applied",edit.path.display()));
        }
    }
    for edit in &plan.files {std::fs::write(&edit.path,&edit.after).map_err(|e|format!("{}: {e}",edit.path.display()))?;}
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn fixture(files: &[(&str,&str)]) -> PathBuf {
        let root=std::env::temp_dir().join(format!("sev-explicit-imports-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir_all(&root).unwrap();
        for (name,text) in files {std::fs::write(root.join(name),text).unwrap();}
        root
    }
    #[test]
    fn names_types_aliases_comments_and_local_shadowing() {
        let root=fixture(&[
            ("lib.sev","class Point:\n    x: int\ndef used() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n"),
            ("main.sev","# 😀 untouched\nimport * from \"lib.sev\" # keep this\ndef main(unused: int) -> Point:\n    # unused()\n    print(\"unused\")\n    return Point(used())\n"),
        ]);
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        let plan=plan(&graph,&root).unwrap();
        let edit=plan.files.iter().find(|f|f.path.ends_with("main.sev")).unwrap();
        assert!(edit.after.contains("import Point, used from \"lib.sev\" # keep this"),"{}",edit.after);
        apply(&plan).unwrap();
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        assert!(super::plan(&graph,&root).unwrap().files.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn qualified_imports_preserve_namespaces_and_uses() {
        let root=fixture(&[("lib.sev","def used() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n"),
            ("main.sev","import * from \"lib.sev\" as helpers\ndef main() -> int:\n    # helpers.unused()\n    print(\"helpers.unused()\")\n    return helpers.used()\n")]);
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        let p=plan(&graph,&root).unwrap();
        assert!(p.files.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn reexport_requirements_reach_the_leaf() {
        let root=fixture(&[("leaf.sev","def needed() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n"),
            ("facade.sev","import * from \"leaf.sev\"\n"),
            ("main.sev","import * from \"facade.sev\"\ndef main() -> int:\n    return needed()\n")]);
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        let p=plan(&graph,&root).unwrap();
        assert_eq!(p.files.len(),2);
        assert!(p.files.iter().all(|f|f.after.contains("import needed")));
        apply(&p).unwrap();
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        let index=severian_semantic::import_index(&graph).unwrap();
        assert!(index.modules[&graph.modules.last().unwrap().id].scope.bindings.contains_key("needed"));
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn interpolation_uses_the_build_plan_and_keeps_namespace_source_edits_safe() {
        let root=fixture(&[("lib.sev","def needed() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n"),
            ("main.sev","import * from \"lib.sev\"\ndef run() -> string:\n    return f\"{needed()}\"\n")]);
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        let edits=plan(&graph,&root).unwrap();
        assert!(edits.files[0].after.contains("import needed"));
        std::fs::write(root.join("main.sev"),"import * from \"lib.sev\" as helpers\ndef run() -> string:\n    value = helpers.needed()\n    return f\"{helpers.needed()}\"\n").unwrap();
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        assert!(plan(&graph,&root).unwrap().files.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ordinary_build_automatically_handles_wildcards_even_with_source_lint_errors() {
        let root=fixture(&[("package.json",r#"{"lint":{"rules":{"L0015":"error"}}}"#),
            ("lib.sev","def needed() -> int:\n    return 7\n"),
            ("main.sev","import * from \"lib.sev\"\ndef main() -> int:\n    return needed()\n")]);
        let before=std::fs::read_to_string(root.join("main.sev")).unwrap();
        crate::Compiler::new(severian_target::TargetSpec::host()).unwrap().check_file(&root.join("main.sev")).unwrap();
        assert_eq!(std::fs::read_to_string(root.join("main.sev")).unwrap(),before);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stale_plan_leaves_other_sources_untouched() {
        let root=fixture(&[("lib.sev","def used() -> int:\n    return 1\n"),
            ("main.sev","import * from \"lib.sev\"\ndef main() -> int:\n    return used()\n")]);
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        let p=plan(&graph,&root).unwrap();
        std::fs::write(root.join("main.sev"),"# concurrent edit\n").unwrap();
        assert!(apply(&p).unwrap_err().contains("changed after"));
        assert_eq!(std::fs::read_to_string(root.join("main.sev")).unwrap(),"# concurrent edit\n");
        std::fs::remove_dir_all(root).unwrap();
    }
}
