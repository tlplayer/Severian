use super::apply;
use super::discover;
use super::model::{Mutation, MutationResult, MutationStatus};
use super::report;
use crate::test_runner;
use severian_driver::Compiler;
use severian_modules::ModuleGraph;
use std::path::{Path, PathBuf};

struct Candidate {
    targets: Vec<Target>,
    mutation: Mutation,
}

struct Target {
    root: PathBuf,
    graph: ModuleGraph,
    mutation: Mutation,
}

pub(crate) fn run(
    compiler: &Compiler,
    sources: &[PathBuf],
    output_root: &Path,
) -> Result<(), String> {
    println!("running baseline tests...");
    test_runner::run(compiler, sources, &output_root.join("baseline"))
        .map_err(|error| format!("baseline tests failed; mutation testing was not run: {error}"))?;

    let mut candidates = Vec::new();
    for root in sources {
        let graph = compiler
            .resolve_test_graph(root)
            .map_err(|error| error.to_string())?;
        for mutation in discover::discover(&graph)? {
            candidates.push(Candidate {
                mutation: mutation.clone(),
                targets: vec![Target {
                    root: root.clone(),
                    graph: graph.clone(),
                    mutation,
                }],
            });
        }
    }
    candidates.sort_by(|left, right| {
        left.mutation
            .file
            .cmp(&right.mutation.file)
            .then_with(|| left.mutation.span.start.cmp(&right.mutation.span.start))
            .then_with(|| left.mutation.span.end.cmp(&right.mutation.span.end))
            .then_with(|| left.mutation.kind.cmp(&right.mutation.kind))
            .then_with(|| left.mutation.replacement.cmp(&right.mutation.replacement))
    });
    let mut grouped: Vec<Candidate> = Vec::new();
    for mut candidate in candidates {
        if let Some(existing) = grouped
            .last_mut()
            .filter(|existing| same_mutation(&existing.mutation, &candidate.mutation))
        {
            existing.targets.append(&mut candidate.targets);
        } else {
            grouped.push(candidate);
        }
    }
    let mut candidates = grouped;
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.mutation.id = index + 1;
    }

    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut applied = 0usize;
        let mut compile_failures = 0usize;
        let mut timeout_failures = 0usize;
        let mut failures = 0usize;
        for (target_index, target) in candidate.targets.into_iter().enumerate() {
            let mut graph = target.graph;
            if !apply::apply(&mut graph, &target.mutation) {
                continue;
            }
            applied += 1;
            let directory = output_root
                .join(format!("mutant-{}", candidate.mutation.id))
                .join(format!("root-{target_index}"));
            let summary = test_runner::run_graph(compiler, &target.root, graph, &directory)?;
            compile_failures += summary.compile_failures;
            timeout_failures += summary.timeout_failures;
            failures += summary.failed;
        }
        let status = if applied == 0 {
            MutationStatus::Skipped
        } else if compile_failures != 0 {
            MutationStatus::CompileKilled
        } else if timeout_failures != 0 {
            MutationStatus::TimeoutKilled
        } else if failures != 0 {
            MutationStatus::Killed
        } else {
            MutationStatus::Survived
        };
        results.push(MutationResult {
            mutation: candidate.mutation,
            status,
        });
    }
    report::print(&results);
    let survived = results
        .iter()
        .filter(|result| result.status == MutationStatus::Survived)
        .count();
    if survived == 0 {
        Ok(())
    } else if survived == 1 {
        Err("1 mutation survived".into())
    } else {
        Err(format!("{survived} mutations survived"))
    }
}

fn same_mutation(left: &Mutation, right: &Mutation) -> bool {
    left.file == right.file
        && left.span.start == right.span.start
        && left.span.end == right.span.end
        && left.kind == right.kind
        && left.original == right.original
        && left.replacement == right.replacement
}
