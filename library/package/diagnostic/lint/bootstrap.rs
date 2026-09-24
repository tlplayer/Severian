//! Bootstrap adapter for package.diagnostic.lint's explicit-imports rule.
//! Optional source-style linting. Successful builds resolve import spelling
//! afterwards; this lint does not gate semantic compilation.
use severian_ast::Item;
use severian_diagnostics::Diagnostic;
use severian_modules::ModuleGraph;

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
            let diagnostic=Diagnostic::new("L0015", "wildcard import hides the names used by this module; run sev build --explicit-imports to replace * with the names used",Some(span)).with_source(module.source.clone());
            if level=="error" {return Err(diagnostic);}
            eprintln!("{level}: {diagnostic}");
        }
    }
    Ok(())
}
