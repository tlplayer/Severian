use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;

type Costs = BTreeMap<String, [u64; 6]>;

fn accumulate(rows: &mut Costs, folded: &str, metric: usize) -> Result<(), String> {
    for line in folded.lines().filter(|line| !line.is_empty()) {
        let (stack, amount) = line.rsplit_once(' ').ok_or("invalid folded stack")?;
        let amount = amount.parse::<u64>().map_err(|_| "invalid folded cost")?;
        let frames = stack
            .split(';')
            .filter(|frame| !frame.is_empty())
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        for (index, frame) in frames.iter().enumerate().rev() {
            if !seen.insert(*frame) {
                continue;
            }
            let costs = rows.entry((*frame).to_owned()).or_default();
            costs[metric * 2 + 1] = costs[metric * 2 + 1]
                .checked_add(amount)
                .ok_or("profile cost overflow")?;
            if index + 1 == frames.len() {
                costs[metric * 2] = costs[metric * 2]
                    .checked_add(amount)
                    .ok_or("profile cost overflow")?;
            }
        }
    }
    Ok(())
}

pub(super) fn source_hints(report: &str) -> String {
    let mut output = String::new();
    let mut seen = BTreeSet::new();
    let mut function = "";
    let mut selected = false;
    let mut cost = "";
    for line in report.lines().map(str::trim) {
        if line.is_empty() {
            selected = false;
            cost = "";
        } else if line.contains(" calls to allocation functions ") {
            cost = line;
        }
        if let Some(location) = line.strip_prefix("at ") {
            if selected {
                continue;
            }
            let Some((path, number)) = location.rsplit_once(':') else {
                continue;
            };
            if !path.ends_with(".sev") {
                continue;
            }
            selected = true;
            let Ok(number) = number.parse::<usize>() else {
                continue;
            };
            if number == 0 || !seen.insert(location) {
                continue;
            }
            let Ok(source) = fs::read_to_string(path) else {
                continue;
            };
            let Some(text) = source.lines().nth(number - 1) else {
                continue;
            };
            let display = function
                .strip_prefix("__sev_fn_")
                .and_then(|name| name.split_once('_'))
                .map_or(function, |(_, name)| name);
            let advice = if text.contains(".characters()") {
                "Decodes a string into a character list; reuse decoded input where its lifetime permits."
            } else {
                "Source line from native debug information; inspect allocations and ownership here."
            };
            output.push_str(&format!("Allocation hotspot: {display}\n  --> {location}\n   |\n{number} | {text}\n   | {}\n   = {cost}\n   = {advice}\n\n", "^".repeat(text.chars().count())));
            if seen.len() >= 10 {
                break;
            }
        } else if !line.is_empty() && !line.starts_with("in ") {
            function = line;
        }
    }
    output
}

pub(super) fn memory(
    directory: &Path,
    trace: &Path,
    name: &str,
    report: &str,
) -> Result<(), String> {
    let hints = source_hints(report);
    fs::write(directory.join(format!("{name}.hints.txt")), &hints).map_err(|e| e.to_string())?;
    eprint!("{hints}");
    let mut rows = Costs::new();
    for (metric, kind) in ["allocations", "peak", "leaked"].iter().enumerate() {
        let folded = directory.join(format!("{name}.{kind}.folded"));
        if metric != 0 {
            let result = Command::new("heaptrack_print")
                .arg("-f")
                .arg(trace)
                .args([
                    "--merge-backtraces",
                    "0",
                    "--print-peaks",
                    "0",
                    "--print-allocators",
                    "0",
                    "--print-temporary",
                    "0",
                    "--flamegraph-cost-type",
                    kind,
                    "--print-flamegraph",
                ])
                .arg(&folded)
                .output()
                .map_err(|e| e.to_string())?;
            fs::write(
                directory.join(format!("{name}.{kind}.stderr")),
                &result.stderr,
            )
            .map_err(|e| e.to_string())?;
            if !result.status.success() {
                return Err(format!("heaptrack {kind} analysis failed"));
            }
        }
        let input = BufReader::new(fs::File::open(folded).map_err(|e| e.to_string())?);
        for line in input.lines() {
            accumulate(&mut rows, &line.map_err(|e| e.to_string())?, metric)?;
        }
    }
    let mut ranked = rows.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right.1[1]
            .cmp(&left.1[1])
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut output = String::from("function\tself_allocations\tinclusive_allocations\tself_peak_bytes\tinclusive_peak_bytes\tself_retained_bytes\tinclusive_retained_bytes\n");
    for (name, costs) in ranked {
        output.push_str(&name.replace('\t', " "));
        for cost in costs {
            output.push_str(&format!("\t{cost}"));
        }
        output.push('\n');
    }
    let path = directory.join(format!("{name}.functions.tsv"));
    fs::write(&path, output).map_err(|e| e.to_string())?;
    eprintln!("Functions: {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recursion_is_counted_once_and_self_cost_uses_the_leaf() {
        let mut rows = Costs::new();
        accumulate(
            &mut rows,
            "root;recursive;recursive;allocate; 7\nroot;allocate; 3\n",
            0,
        )
        .unwrap();
        accumulate(&mut rows, "root;allocate; 32\n", 1).unwrap();
        assert_eq!(rows["root"], [0, 10, 0, 32, 0, 0]);
        assert_eq!(rows["recursive"][1], 7);
        assert_eq!(rows["allocate"], [10, 10, 32, 32, 0, 0]);
        assert!(accumulate(&mut rows, "bad trace", 0).is_err());
    }
}
