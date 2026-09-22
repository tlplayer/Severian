//! Compare wildcard expansion with the automatic build import plan, without
//! type checking or rewriting sources. Pass a package manifest and source root.
use severian_driver::{
    config::{Catalog, Manifest},
    Compiler,
};
use severian_target::TargetSpec;
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: import_counts PACKAGE/package.json ENTRY.sev".into());
    }
    let manifest = Manifest::load(Path::new(&args[1]), &Catalog::load()?)?;
    let compiler = Compiler::new(TargetSpec::host())?.with_packages(manifest.module_graph(false));
    let graph = compiler.resolve_test_graph(Path::new(&args[2]))?;
    let start = std::time::Instant::now();
    let plan = severian_semantic::import_plan(&graph)?;
    let elapsed = start.elapsed();
    // The complete index remains available to tools that explicitly inspect
    // the whole public surface. Ordinary builds do not construct this index.
    let broad = severian_semantic::import_index(&graph)?;
    println!("modules={} wildcard_export_entries={} narrowed_export_entries={} imported_bindings={} name_requests={} narrowing_seconds={:.6}",
        graph.modules.len(), broad.exports.values().map(|e|e.len()).sum::<usize>(),
        plan.export_entries, plan.imported_bindings, plan.requested_names, elapsed.as_secs_f64());
    Ok(())
}
