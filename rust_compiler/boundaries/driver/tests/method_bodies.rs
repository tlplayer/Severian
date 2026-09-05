use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn run(source: &str, expected_stdout: &str) {
    run_with_modules(source, &[], expected_stdout);
}

fn run_with_modules(source: &str, modules: &[(&str, &str)], expected_stdout: &str) {
    let root = std::env::temp_dir().join(format!(
        "severian-method-body-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    for (name, contents) in modules {
        fs::write(root.join(name), contents).unwrap();
    }
    let path = root.join("main.sev");
    fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "source: {}\nstdout:\n{}\nstderr:\n{}",
        path.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), expected_stdout);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn method_mutation_and_return_both_take_effect() {
    run(
        r#"class Counter:
    value: int
    def bump() -> int:
        value += 1
        return value
def main():
    counter := Counter(1)
    result = counter.bump()
    assert(result == 2)
    assert(counter.value == 2)
    counter.bump()
    assert(counter.value == 3)
    print("mutated")
"#,
        "mutated\n",
    );
}

#[test]
fn method_unit_return_resumes_its_caller() {
    run(
        r#"class Counter:
    value: int
    def stop():
        return
def main():
    counter := Counter(1)
    counter.stop()
    print("resumed")
"#,
        "resumed\n",
    );
}

#[test]
fn method_locals_loops_and_early_returns_are_preserved() {
    run(
        r#"class Counter:
    value: int
    def advance(limit: int) -> int:
        count := 0
        while count < limit:
            count += 1
            if count == 2:
                continue
            value += count
            if value > 10:
                return value
        result = value + count
        return result
def main():
    counter := Counter(1)
    assert(counter.advance(3) == 8)
    assert(counter.value == 5)
    assert(counter.advance(5) == 13)
    assert(counter.value == 13)
    print("flow")
"#,
        "flow\n",
    );
}

#[test]
fn borrowed_record_mutation_through_free_callable_reaches_caller() {
    run(
        r#"class Counter:
    value: int
def bump(counter: Counter):
    counter.value = counter.value + 1
def main():
    counter := Counter(1)
    bump(counter)
    assert(counter.value == 2)
    print("borrowed")
"#,
        "borrowed\n",
    );
}

#[test]
fn method_forward_and_recursive_calls_share_receiver() {
    run(
        r#"class Counter:
    value: int
    def add(count: int) -> int:
        if count == 0:
            return value
        bump()
        return add(count - 1)
    def bump():
        value += 1
def main():
    counter := Counter(1)
    assert(counter.add(3) == 4)
    assert(counter.value == 4)
    print("recursive")
"#,
        "recursive\n",
    );
}

#[test]
fn parameter_rebinding_does_not_replace_callers_record() {
    run(
        r#"class Counter:
    value: int
def replace(counter: Counter):
    counter = Counter(99)
    counter.value = 100
    assert(counter.value == 100)
def main():
    counter := Counter(1)
    replace(counter)
    assert(counter.value == 1)
    print("local")
"#,
        "local\n",
    );
}

#[test]
fn method_receiver_and_keyword_arguments_evaluate_once_in_source_order() {
    run(
        r#"class Counter:
    value: int
    def combine(first: int, second: int) -> int:
        value += first + second
        return first * 10 + second
def receiver() -> Counter:
    print("receiver")
    return Counter(0)
def argument(value: int) -> int:
    print(value)
    return value
def main():
    result = receiver().combine(second=argument(2), first=argument(1))
    assert(result == 12)
"#,
        "receiver\n2\n1\n",
    );
}

#[test]
fn nested_method_receiver_mutates_the_original_field() {
    run(
        r#"class Counter:
    value: int
    def bump() -> int:
        value += 1
        return value
class Holder:
    counter: Counter
def main():
    holder := Holder(Counter(1))
    assert(holder.counter.bump() == 2)
    assert(holder.counter.value == 2)
    print("nested")
"#,
        "nested\n",
    );
}

#[test]
fn conditional_parameter_rebinding_keeps_reference_until_rebound() {
    run(
        r#"class Counter:
    value: int
def update(counter: Counter, replace: bool):
    if replace:
        counter = Counter(99)
    counter.value = counter.value + 1
def main():
    counter := Counter(1)
    update(counter, false)
    assert(counter.value == 2)
    update(counter, true)
    assert(counter.value == 2)
    print("conditional")
"#,
        "conditional\n",
    );
}

#[test]
fn interner_method_keeps_loop_returns_collection_mutations_and_fresh_ids() {
    run(
        r#"class Interner:
    values: list[int] = []
    next: int = 0
    def intern(value: int) -> int:
        index := 0
        for known in values:
            if known == value:
                return index
            index += 1
        result = int(len(values))
        values.append(value)
        return result
    def fresh() -> int:
        value = next
        next += 1
        return intern(value)
def main():
    interner := Interner()
    assert(interner.fresh() == 0)
    assert(interner.fresh() == 1)
    assert(interner.intern(0) == 0)
    assert(len(interner.values) == 2)
    assert(interner.next == 2)
    print("interned")
"#,
        "interned\n",
    );
}

#[test]
fn fallible_unit_method_can_fall_through_after_mutation() {
    run(
        r#"class Counter:
    value: int
    def bump() -> unit | Error:
        value += 1
def main():
    counter := Counter(1)
    counter.bump()
    assert(counter.value == 2)
    print("success")
"#,
        "success\n",
    );
}

#[test]
fn method_self_retains_access_to_private_fields_and_helpers() {
    run(
        r#"class Counter:
    __value: int
    def __bump() -> int:
        self.__value = self.__value + 1
        return self.__value
    def bump() -> int:
        return self.__bump()
def main():
    counter := Counter(1)
    assert(counter.bump() == 2)
    assert(counter.bump() == 3)
    print("private")
"#,
        "private\n",
    );
}

#[test]
fn method_receiver_inside_enum_payload_uses_the_active_payload_storage() {
    run(
        r#"class Counter:
    value: int
    def bump() -> int:
        value += 1
        return value
enum Item:
    Number(value: int)
    CounterValue(counter: Counter)
def bump(item: Item) -> int:
    match item:
        case Number:
            return value
        case CounterValue:
            return counter.bump()
def main():
    item := Item.CounterValue(Counter(1))
    assert(bump(item) == 2)
    print("payload")
"#,
        "payload\n",
    );
}

#[test]
fn recursive_method_returning_a_field_value_does_not_consume_its_receiver() {
    run(
        r#"class Value:
    index: int
class Store:
    values: list[Value]
    def resolve(value: Value) -> Value:
        if value.index == 0:
            return value
        return resolve(values[0])
def main():
    store := Store([Value(0)])
    assert(store.resolve(Value(1)).index == 0)
    assert(store.resolve(Value(2)).index == 0)
    print("resolved")
"#,
        "resolved\n",
    );
}

#[test]
fn callable_body_scopes_retain_imported_constants() {
    run_with_modules(
        r#"import "settings.sev" as settings
class Counter:
    value: int
    def bump() -> int:
        value += settings.step
        return value
def step() -> int:
    return settings.step
def main():
    counter := Counter(1)
    assert(step() == 2)
    assert(counter.bump() == 3)
    print("imported")
"#,
        &[("settings.sev", "step: int = 2\n")],
        "imported\n",
    );
}
