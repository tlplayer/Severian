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
struct Word { text: String, start: usize, end: usize, identifier: bool, formatted: bool }

fn words(module: &severian_modules::ResolvedModule) -> Result<Vec<Word>, String> {
    let tokens = severian_lexer::scan(&module.source).map_err(|e| e.to_string())?;
    Ok(tokens.into_iter().filter(|t| t.span.start < t.span.end).map(|t| Word {
        text: module.source.text[t.span.start as usize..t.span.end as usize].to_owned(),
        start: t.span.start as usize, end: t.span.end as usize,
        identifier: matches!(t.kind, severian_lexer::TokenKind::Identifier(_)),
        formatted: matches!(t.kind, severian_lexer::TokenKind::FormattedString(_)),
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
        let tokens = &lexed[&module.id];
        let mut edits: Vec<(usize,usize,String)> = Vec::new();
        let mut supplied = BTreeSet::new();
        let mut count = 0;
        for import in module.ast.items.iter().filter_map(|i|if let Item::Import(i)=i {Some(i)}else{None}) {
            if !wildcard(import) { continue; }
            let Some(edge) = module.imports.iter().find(|e|e.span==import.span) else {continue};
            let locator = match &import.subject {
                ImportSubject::Locator(locator) => locator.as_str(),
                ImportSubject::Name(_) => import.source.as_deref().expect("package wildcard source"),
            };
            let mut members = Vec::new();
            let mut uses = Vec::new();
            if let Some(alias) = &import.alias {
                let shadows = shadows(&module.ast);
                let occurrences: Vec<_> = tokens.iter().enumerate().filter(|(p,t)|t.identifier && t.text==*alias
                    && !(*p>0 && tokens[*p-1].text==".")
                    && !(t.start>=import.span.start as usize && t.start<import.span.end as usize)
                    && !shadows.iter().any(|(s,e,n)|t.start>=*s && t.start<*e && n.contains(alias))).collect();
                let unseen = build_plan.requirements.get(&module.id).into_iter().flatten()
                    .filter_map(|name|name.strip_prefix(&format!("{alias}.")))
                    .any(|member| !occurrences.iter().any(|(p,_)|tokens.get(p+2).is_some_and(|t|t.text==member)));
                let embedded = tokens.iter().any(|token|token.formatted && token.text.contains(&format!("{alias}.")));
                if unseen || embedded || occurrences.iter().any(|(p,_)|tokens.get(p+1).is_none_or(|t|t.text!=".")) {
                    result.notes.push(format!("{}: retained module namespace `{alias}`; it is used as a value or re-export", module.path.display()));
                    continue;
                }
                // An exported namespace is part of its consumers' contract.
                if required[&module.id].contains(alias) && occurrences.is_empty() { continue; }
                let mut names = BTreeMap::new();
                for (position, token) in occurrences {
                    if shadows.iter().any(|(s,e,n)|token.start>=*s && token.start<*e && n.contains(alias)) {continue;}
                    let Some(member) = tokens.get(position+2).filter(|t|t.identifier) else {continue};
                    if !index.exports[&edge.module].contains_key(&member.text) {
                        return Err(format!("{}: `{alias}.{}` does not resolve; imports were not changed", module.path.display(), member.text));
                    }
                    let local = format!("{alias}__{}", member.text);
                    if tokens.iter().any(|t|t.identifier && t.text==local) {
                        return Err(format!("{}: generated import alias `{local}` conflicts with an existing name",module.path.display()));
                    }
                    names.insert(member.text.clone(),local.clone());
                    uses.push((token.start,member.end,local));
                }
                members.extend(names.into_iter().map(|(name,alias)|format!("{name} as {alias}")));
            } else {
                for name in &required[&module.id] {
                    if providers.get(&(module.id,name.clone()))==Some(&import.span.start) && supplied.insert(name.clone()) {members.push(name.clone());}
                }
            }
            if members.is_empty() {
                // Importing can register extensions or execute initializers. Do
                // not erase a dependency merely because it contributes no names.
                result.notes.push(format!("{}: no named uses of `{locator}`; retained for review of initialization/extension effects",module.path.display()));
                continue;
            }
            let source = if matches!(import.subject, ImportSubject::Locator(_)) {serde_json::to_string(locator).map_err(|e|e.to_string())?} else {locator.to_owned()};
            let replacement = if module.source.text[import.span.start as usize..import.span.end as usize].trim()=="*" {members.join(", ")}
                else if members.len()==1 {format!("from {source} import {}",members[0])}
                else {format!("from {source} import {{ {} }}",members.join(", "))};
            edits.push((import.span.start as usize,import.span.end as usize,replacement));
            edits.extend(uses);
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

// Scope ranges exclude parameter names and local bindings from imported uses.
fn shadows(module: &severian_ast::Module) -> Vec<(usize,usize,BTreeSet<String>)> {
    fn bindings(body: &[severian_ast::Statement], names: &mut BTreeSet<String>) {
        use severian_ast::Statement::*;
        for statement in body {match statement {
            Binding(b) => {names.insert(b.name.clone());},
            Destructure{names:ns,..} => names.extend(ns.iter().cloned()),
            Unsafe{body,..}|Placement{body,..}|While{body,..} => bindings(body,names),
            For{binding,second_binding,body,..} => {names.insert(binding.clone());names.extend(second_binding.iter().cloned());bindings(body,names);},
            If{then_block,else_block,..}=>{bindings(then_block,names);bindings(else_block,names);},
            Try{body,catch_body,..}=>{bindings(body,names);bindings(catch_body,names);},
            FallibleElse{body,..}=>bindings(body,names),
            Match{cases,..}=>for c in cases {bindings(&c.body,names);},
            Select{cases,error_body,..}=>{for c in cases {bindings(&c.body,names);}bindings(error_body,names);},
            _=>{}
        }}
    }
    let mut scopes=Vec::new();
    let mut add=|f:&severian_ast::FunctionDeclaration| {
        let mut names:BTreeSet<_>=f.parameters.iter().map(|p|p.name.clone()).collect();
        names.extend(f.type_parameters.iter().cloned());
        if let Some(body)=&f.body {bindings(body,&mut names);}
        for parameter in &f.parameters {
            scopes.push((parameter.span.start as usize, parameter.span.start as usize + parameter.name.len(), BTreeSet::from([parameter.name.clone()])));
        }
        if let Some(first) = f.body.as_ref().and_then(|body|body.first()) {
            use severian_ast::Statement::*;
            let start = match first {
                Binding(b)=>b.span.start,
                Expression(e)=>e.span.start,
                Destructure{span,..}|FieldAssignment{span,..}|IndexAssignment{span,..}|Defer{span,..}|Return{span,..}|Yield{span,..}|Assert{span,..}|Unsafe{span,..}|Placement{span,..}|Try{span,..}|FallibleElse{span,..}|If{span,..}|While{span,..}|For{span,..}|Break{span,..}|Continue{span,..}|Match{span,..}|Select{span,..}=>span.start,
            };
            scopes.push((start as usize,f.span.end as usize,names));
        }
    };
    for item in &module.items {match item {
        Item::Function(f)=>add(f),
        Item::Class(c)=>{for f in c.methods.iter().chain(&c.constructors){add(f);}},
        Item::Trait(t)=>{for f in &t.methods {add(f);}},
        _=>{}
    }}
    scopes
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
        assert!(edit.after.contains("from \"lib.sev\" import { Point, used } # keep this"),"{}",edit.after);
        apply(&plan).unwrap();
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        assert!(super::plan(&graph,&root).unwrap().files.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn qualified_imports_rewrite_only_resolved_uses() {
        let root=fixture(&[("lib.sev","def used() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n"),
            ("main.sev","import * from \"lib.sev\" as helpers\ndef main() -> int:\n    # helpers.unused()\n    print(\"helpers.unused()\")\n    return helpers.used()\n")]);
        let graph=severian_modules::resolve(&root.join("main.sev")).unwrap();
        let p=plan(&graph,&root).unwrap();
        let text=&p.files[0].after;
        assert!(text.contains("from \"lib.sev\" import used as helpers__used"));
        assert!(text.contains("return helpers__used()"));
        assert!(text.contains("# helpers.unused()"));
        assert!(text.contains("\"helpers.unused()\""));
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
