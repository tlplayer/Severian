use super::model::{MutationResult, MutationStatus};
use severian_source::SourceFile;

pub(crate) fn print(results: &[MutationResult]) {
    println!("\nmutation testing\n");
    let width = results.len().to_string().len().max(3);
    for result in results {
        let location = SourceFile::load(&result.mutation.file)
            .ok()
            .and_then(|source| source.location(result.mutation.span.start));
        let file = std::env::current_dir()
            .ok()
            .and_then(|root| {
                result
                    .mutation
                    .file
                    .strip_prefix(root)
                    .ok()
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| result.mutation.file.clone());
        let location = location.map_or_else(
            || file.display().to_string(),
            |location| format!("{}:{}:{}", file.display(), location.line, location.column),
        );
        println!(
            "M{:0width$} {:<8} {:<28} {} -> {}",
            result.mutation.id,
            result.status.label(),
            location,
            result.mutation.original,
            result.mutation.replacement,
            width = width,
        );
    }

    let killed = results
        .iter()
        .filter(|result| result.status.is_killed())
        .count();
    let survived = results
        .iter()
        .filter(|result| result.status == MutationStatus::Survived)
        .count();
    let skipped = results
        .iter()
        .filter(|result| result.status == MutationStatus::Skipped)
        .count();
    let denominator = killed + survived;
    let score = if denominator == 0 {
        100.0
    } else {
        (killed as f64 / denominator as f64) * 100.0
    };
    println!("\nmutation result:");
    println!("    {} mutations", results.len());
    println!("    {killed} killed");
    println!("    {survived} survived");
    if skipped != 0 {
        println!("    {skipped} skipped");
    }
    println!("\nmutation score: {score:.1}%");
}
