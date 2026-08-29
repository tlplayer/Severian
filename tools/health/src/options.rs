use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    Human,
    Json,
    Sarif,
}

#[derive(Debug)]
pub struct Options {
    pub root: Option<PathBuf>,
    pub changed: Option<String>,
    pub format: Format,
    pub baseline: PathBuf,
    pub write_baseline: Option<PathBuf>,
    pub coverage: Option<PathBuf>,
    pub mutation_report: Option<PathBuf>,
    pub deny_warnings: bool,
    pub all_targets: bool,
    pub all_features: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            root: None,
            changed: None,
            format: Format::Human,
            baseline: "tools/health/baseline.toml".into(),
            write_baseline: None,
            coverage: None,
            mutation_report: None,
            deny_warnings: false,
            all_targets: false,
            all_features: false,
        }
    }
}

pub fn parse(mut arguments: Vec<String>) -> Result<Options, String> {
    if arguments.first().map(String::as_str) != Some("health") {
        return Err(usage());
    }
    arguments.remove(0);
    let mut options = Options::default();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--changed" => {
                index += 1;
                options.changed = Some(
                    arguments
                        .get(index)
                        .ok_or("--changed requires a Git base")?
                        .clone(),
                );
            }
            "--format" => {
                index += 1;
                options.format = match arguments.get(index).map(String::as_str) {
                    Some("human") => Format::Human,
                    Some("json") => Format::Json,
                    Some("sarif") => Format::Sarif,
                    _ => return Err("--format must be human, json, or sarif".into()),
                };
            }
            "--root" => {
                index += 1;
                options.root = Some(PathBuf::from(
                    arguments.get(index).ok_or("--root requires a path")?,
                ));
            }
            "--baseline" => {
                index += 1;
                options.baseline =
                    PathBuf::from(arguments.get(index).ok_or("--baseline requires a path")?);
            }
            "--write-baseline" => {
                index += 1;
                options.write_baseline = Some(PathBuf::from(
                    arguments
                        .get(index)
                        .ok_or("--write-baseline requires a path")?,
                ));
            }
            "--coverage" => {
                index += 1;
                options.coverage = Some(PathBuf::from(
                    arguments.get(index).ok_or("--coverage requires a path")?,
                ));
            }
            "--mutation-report" => {
                index += 1;
                options.mutation_report = Some(PathBuf::from(
                    arguments
                        .get(index)
                        .ok_or("--mutation-report requires a path")?,
                ));
            }
            "--deny-warnings" => options.deny_warnings = true,
            "--all-targets" => options.all_targets = true,
            "--all-features" => options.all_features = true,
            "-h" | "--help" => return Err(usage()),
            unknown => return Err(format!("unknown health option `{unknown}`\n{}", usage())),
        }
        index += 1;
    }
    Ok(options)
}

fn usage() -> String {
    "usage: cargo xtask health [--changed GIT_BASE] [--coverage LLVM_COV_JSON] [--mutation-report CARGO_MUTANTS_JSON] [--all-targets] [--all-features] [--format human|json|sarif] [--baseline PATH] [--write-baseline PATH] [--deny-warnings]".into()
}
