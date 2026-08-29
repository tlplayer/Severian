mod analyze;
mod architecture;
mod baseline;
mod coverage;
mod graph;
mod model;
mod mutation;
mod options;
mod report;
mod repository;
mod source;

use std::env;
use std::process;

fn main() {
    if let Err(error) = run() {
        eprintln!("health error: {error}");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let options = options::parse(env::args().skip(1).collect())?;
    let root = repository::discover(options.root.clone())?;
    let changed = options
        .changed
        .as_deref()
        .map(|base| repository::changed_paths(&root, base))
        .transpose()?;
    let mut findings = analyze::run(&root, changed.as_ref())?;
    if let Some(path) = options.coverage.as_deref() {
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        let changed_lines = options
            .changed
            .as_deref()
            .map(|base| repository::changed_lines(&root, base))
            .transpose()?;
        findings.extend(coverage::analyze(&root, &path, changed_lines.as_ref())?);
    }
    if let Some(path) = options.mutation_report.as_deref() {
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        findings.extend(mutation::analyze(&root, &path)?);
    }
    let baseline_path = if options.baseline.is_absolute() {
        options.baseline.clone()
    } else {
        root.join(&options.baseline)
    };
    let fingerprints = baseline::load(&baseline_path)?;
    baseline::apply(&mut findings, &fingerprints);
    if let Some(path) = &options.write_baseline {
        let path = if path.is_absolute() {
            path.clone()
        } else {
            root.join(path)
        };
        baseline::write(&path, &findings)?;
        eprintln!(
            "wrote {} finding fingerprints to {}",
            findings.len(),
            path.display()
        );
    }
    print!("{}", report::render(&findings, options.format));
    let failed = findings
        .iter()
        .any(|finding| finding.fails_gate(options.deny_warnings));
    if failed {
        process::exit(1);
    }
    Ok(())
}
