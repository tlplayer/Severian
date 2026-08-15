use std::{
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn run_transition(initial: &str) -> std::process::Output {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-enum-transition-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("main.sev");
    std::fs::write(
        &source,
        format!(
            r#"enum Download:
    Pending -> Connecting
    Connecting -> Complete
    Complete

def selected() -> Download:
    return {initial}

def main():
    state := selected()
    state = Connecting
"#
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();
    std::fs::remove_dir_all(root).unwrap();
    output
}

#[test]
fn dynamic_enum_transitions_are_checked_at_runtime() {
    let valid = run_transition("Pending");
    assert!(
        valid.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&valid.stdout),
        String::from_utf8_lossy(&valid.stderr)
    );

    let invalid = run_transition("Complete");
    assert!(!invalid.status.success());
    let stderr = String::from_utf8_lossy(&invalid.stderr);
    assert!(stderr.contains("E000213"), "{stderr}");
    assert!(
        stderr.contains("Complete") && stderr.contains("Connecting"),
        "{stderr}"
    );
}

#[test]
fn typestate_transition_rebinding_executes_the_new_specialization() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-typestate-transition-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("main.sev");
    std::fs::write(
        &source,
        r#"enum SocketState:
    Closed -> Connected
    Connected -> Closed

class Socket[State]:
    descriptor: int

    def connect() -> Socket[Connected] with { State == Closed }:
        return Socket[Connected](descriptor)

    def send(data: string) -> int with { State == Connected }:
        return size(data)

def main():
    socket := Socket[Closed](7)
    socket = socket.connect()
    assert(socket.send("hello") == 5)
"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();
    std::fs::remove_dir_all(root).unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
