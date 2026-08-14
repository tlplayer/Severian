use severian_package::BuildPolicy;
use std::path::Path;

pub(crate) fn enforce(policy: &BuildPolicy) -> Result<(), String> {
    analyze(policy, false)
}

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let mut input = Path::new(".");
    let mut graph = false;
    let mut has_input = false;
    for argument in args {
        match argument.as_str() {
            "--graph" => graph = true,
            value if !value.starts_with('-') && !has_input => {
                input = Path::new(value);
                has_input = true;
            }
            value => {
                return Err(format!(
                    "unknown architecture option `{value}`; expected `sev architecture [path] [--graph]`"
                ))
            }
        }
    }
    let policy = BuildPolicy::for_input(input).map_err(|error| error.to_string())?;
    if graph {
        emit_graph(&policy)
    } else {
        analyze(&policy, true)
    }
}

fn analyze(policy: &BuildPolicy, summary: bool) -> Result<(), String> {
    let dependencies = severian_driver::architecture::analyze_dependencies(policy)?;
    if summary {
        print_summary(&dependencies);
    }
    let mut errors = render_dependency_findings(&dependencies);
    errors += render_file_findings(policy)?;
    if errors == 0 {
        Ok(())
    } else {
        Err(format!(
            "architecture gate rejected {errors} architectural violation(s)"
        ))
    }
}

fn emit_graph(policy: &BuildPolicy) -> Result<(), String> {
    let dependencies = severian_driver::architecture::analyze_dependencies(policy)?;
    print!("{}", dependencies.to_dot());
    let errors = dependencies
        .findings
        .iter()
        .filter(|finding| finding.severity == "error")
        .count()
        + severian_driver::architecture::check_file_budgets(policy)?
            .iter()
            .filter(|finding| finding.severity == "error")
            .count();
    if errors == 0 {
        Ok(())
    } else {
        Err(format!(
            "architecture graph contains {errors} violation(s); run `sev architecture` for diagnostics"
        ))
    }
}

fn print_summary(analysis: &severian_driver::architecture::DependencyAnalysis) {
    println!("Architecture");
    println!("  Packages:   {}", analysis.nodes.len());
    println!("  Edges:      {}", analysis.dependencies.len());
    println!(
        "  Dependency violations: {}",
        analysis
            .findings
            .iter()
            .filter(|finding| finding.severity == "error")
            .count()
    );
    let high_fan_out = analysis
        .stats
        .iter()
        .filter(|stat| stat.fan_out > 0)
        .take(5)
        .collect::<Vec<_>>();
    if high_fan_out.is_empty() {
        return;
    }
    println!("  High fan-out:");
    for stat in high_fan_out {
        println!(
            "    {:<28} out {:>2}  in {:>2}",
            analysis.nodes[stat.node].path.display(),
            stat.fan_out,
            stat.fan_in
        );
    }
}

fn render_dependency_findings(
    analysis: &severian_driver::architecture::DependencyAnalysis,
) -> usize {
    let mut errors = 0;
    for finding in &analysis.findings {
        println!(
            "{}[{}] {}{}\n  {}",
            finding.severity,
            finding.code,
            finding.manifest.display(),
            finding
                .line
                .map_or(String::new(), |line| format!(":{line}")),
            finding.message
        );
        errors += usize::from(finding.severity == "error");
    }
    errors
}

fn render_file_findings(policy: &BuildPolicy) -> Result<usize, String> {
    let mut errors = 0;
    for finding in severian_driver::architecture::check_file_budgets(policy)? {
        println!(
            "{}[architecture::file_size] {}\n  {}\n  limit: {} lines{}",
            finding.severity,
            finding.path.display(),
            finding.message,
            finding.limit,
            finding
                .exception_reason
                .as_deref()
                .map_or(String::new(), |reason| format!("\n  exception: {reason}"))
        );
        errors += usize::from(finding.severity == "error");
    }
    Ok(errors)
}
