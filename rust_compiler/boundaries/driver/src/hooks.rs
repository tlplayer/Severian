//! The bootstrap uses the same with/without record protocol as source hooks.
use std::fs::OpenOptions;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use serde_json::{json, Value};

static NEXT: AtomicU64 = AtomicU64::new(1);
static FAILED: AtomicBool = AtomicBool::new(false);
thread_local! { static THREAD: u64 = NEXT.fetch_add(1, Ordering::Relaxed); }

fn emit(event: Value) {
    let Some(path) = std::env::var_os("SEVERIAN_HOOK_RECORDS") else { return; };
    let result = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.lock()?;
        writeln!(file, "{event}")
    })();
    if result.is_err() { FAILED.store(true, Ordering::Relaxed); }
}

pub struct Scope { active: Option<(u64, u64, Instant)> }
impl Scope {
    #[track_caller]
    pub fn enter(function: &str) -> Self {
        if std::env::var_os("SEVERIAN_HOOK_RECORDS").is_none() { return Self { active: None }; }
        let source = std::panic::Location::caller();
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let thread = THREAD.with(|thread| *thread);
        emit(json!({"event":"with", "pid":std::process::id(), "thread":thread, "id":id,
            "function":function, "source_id":0, "start":0, "end":0,
            "file":source.file(), "line":source.line(), "column":source.column(),
            "allocation_calls":null, "allocated_bytes":null, "process_live_bytes":null}));
        Self { active: Some((id, thread, Instant::now())) }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some((id, thread, started)) = self.active {
            let elapsed = started.elapsed().as_secs_f64();
            emit(json!({"event":"without", "pid":std::process::id(), "thread":thread, "id":id,
                "duration_seconds":elapsed, "error":std::thread::panicking(),
                "allocation_calls":null, "allocated_bytes":null, "process_live_bytes":null}));
        }
    }
}

#[derive(Default)]
struct Cost { calls: u64, errors: u64, incomplete: u64, inclusive: f64, own: f64 }

pub fn report(input: &std::path::Path, output: &std::path::Path) -> Result<(), String> {
    use std::collections::BTreeMap;
    if FAILED.load(Ordering::Relaxed) { return Err("could not write hook records".into()); }
    let file = std::fs::File::open(input).map_err(|error| error.to_string())?;
    let mut costs = BTreeMap::<(String, String, u64, u64), Cost>::new();
    let mut stacks = BTreeMap::<(u64, u64), Vec<(u64, (String, String, u64, u64), f64)>>::new();
    for line in std::io::BufReader::new(file).lines() {
        let event: Value = serde_json::from_str(&line.map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
        let number = |key: &str| event[key].as_u64().ok_or_else(|| format!("invalid hook {key}"));
        let stack = stacks.entry((number("pid")?, number("thread")?)).or_default();
        let id = number("id")?;
        match event["event"].as_str() {
            Some("with") => {
                let key = (event["function"].as_str().ok_or("missing hook function")?.into(),
                    event["file"].as_str().unwrap_or("").into(), number("line")?, number("column")?);
                costs.entry(key.clone()).or_default().calls += 1;
                stack.push((id, key, 0.0));
            }
            Some("without") => {
                let (entered, key, children) = stack.pop().ok_or("unmatched hook exit")?;
                if entered != id { return Err("non-nested hook exit".into()); }
                let elapsed = event["duration_seconds"].as_f64().filter(|value| value.is_finite() && *value >= 0.0).ok_or("invalid hook duration")?;
                let cost = costs.get_mut(&key).unwrap();
                cost.inclusive += elapsed; cost.own += (elapsed - children).max(0.0);
                cost.errors += u64::from(event["error"].as_bool().ok_or("invalid hook error")?);
                if let Some(parent) = stack.last_mut() { parent.2 += elapsed; }
            }
            _ => return Err("unknown hook event".into()),
        }
    }
    for stack in stacks.values() { for (_, key, _) in stack { costs.get_mut(key).unwrap().incomplete += 1; } }
    let mut costs = costs.into_iter().collect::<Vec<_>>();
    costs.sort_by(|a, b| b.1.inclusive.total_cmp(&a.1.inclusive).then_with(|| a.0.cmp(&b.0)));
    let mut text = "rank\tfunction\tfile\tline\tcolumn\tcalls\terrors\tincomplete\tself_seconds\tinclusive_seconds\n".to_owned();
    for (index, ((name, file, line, column), cost)) in costs.iter().enumerate() {
        use std::fmt::Write;
        writeln!(text, "{}\t{}\t{}\t{line}\t{column}\t{}\t{}\t{}\t{:.9}\t{:.9}",
            index + 1, json!(name), json!(file), cost.calls, cost.errors, cost.incomplete, cost.own, cost.inclusive).unwrap();
    }
    std::fs::write(output, text).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_calls_and_incomplete_scopes_remain_distinct() {
        let directory = std::env::temp_dir().join(format!("sev-hook-report-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&directory).unwrap();
        let entry = |id| json!({"event":"with", "pid":1, "thread":1, "id":id,
            "function":"recursive", "file":"test.sev", "line":3, "column":1});
        let leave = |id, time| json!({"event":"without", "pid":1, "thread":1, "id":id,
            "duration_seconds":time, "error":false});
        let events = [entry(1), entry(2), leave(2, 0.25), leave(1, 1.0), entry(3)];
        let input = directory.join("hooks.jsonl");
        let output = directory.join("functions.tsv");
        std::fs::write(&input, events.iter().map(|event| format!("{event}\n")).collect::<String>()).unwrap();
        report(&input, &output).unwrap();
        let table = std::fs::read_to_string(&output).unwrap();
        assert!(table.contains("\t3\t0\t1\t1.000000000\t1.250000000"));
        std::fs::write(&input, format!("{}\n", leave(7, 1.0))).unwrap();
        assert!(report(&input, &output).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
