use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn test_source(source: &str) {
    let root = std::env::temp_dir().join(format!(
        "severian-conversions-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("conversions.sev");
    fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("test")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\nstdout:\n{}\nstderr:\n{}",
        path.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("0 failed"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_policy_controls_approximate_assignments_calls_casts_and_operators() {
    test_source(
        r#"
test with compiler "conversion permission is scoped to each case":
    import package
    package.config.set("lossless.conversion", true)
    reject:
        big_number: i64 = 10 ** 10
        small_representation: f8e4m3fn = big_number
    reject:
        def take(value: float) -> float:
            return value
        value: i32 = 7
        converted = take(value)
    reject:
        value: i32 = 7
        converted = float(value)
    reject:
        value: i32 = 7
        converted = value as float
    reject:
        value: int = 4
        root = value ** .5
    accept:
        value: i32 = 7
        widened: i64 = value
    package.config.set("lossless.conversion", false)
    accept:
        big_number: i64 = 10 ** 10
        small_representation: f8e4m3fn = big_number
    accept:
        def take(value: float) -> float:
            return value
        value: i32 = 7
        converted = take(value)
        constructed = float(value)
        cast = value as float
        root = value ** .5
    package.config.set("lossless.conversion", true)
    reject:
        value: i32 = 7
        converted: float = value
"#,
    );
}

#[test]
fn approximate_and_checked_conversions_keep_their_native_guards() {
    test_source(
        r#"
def take(value: i8) -> i8:
    return value
class Counter:
    value: int
    def next() -> float:
        value += 1
        return 7.5
test:
    assert(i8(127.0) == 127)
    assert(i8(-128.0) == -128)
    assert(i8(7.5) == 7)
    assert(u8(255.0) == 255)
    throws(i8(128.0))
    throws(i8(127.5))
    assert(i64(f32(7.5)) == 7)
    assert(i128(f32(7.5)) == 7)
    throws(i8(-129.0))
    throws(u8(-1.0))
    throws(u8(256.0))
    throws(int(9223372036854775808.0))
    assert(int(-9223372036854775808.0) == -9223372036854775808)
    throws(int(0.0 / 0.0))
    throws(int(1.0 / 0.0))
    throws(int(-1.0 / 0.0))
    value: float = 128.0
    throws(take(value))
    throws(value as i8)
    wide: i16 = 128
    throws(take(wide))
    throws(wide as i8)
    counter := Counter(0)
    assert(take(counter.next()) == 7)
    assert(counter.value == 1)
"#,
    );
}

#[test]
fn mixed_power_uses_floating_signatures_without_changing_integer_exponents() {
    test_source(
        r#"
test:
    integer = 2 ** 2
    root = integer ** .5
    floating = 4.0 ** 2
    assert(integer == 4)
    assert(root == 2.0)
    assert(floating == 16.0)
    exponent: float = .5
    assert(integer ** exponent == 2.0)
    assert(i32(2) ** u32(5) == 32)
    assert(i8(1) ** u32(4294967295) == i8(1))
    assert(i8(2) ** u32(256) == i8(0))
    assert(f32(4.0) ** f32(.5) == f32(2.0))
"#,
    );
}

#[test]
fn overloads_rank_original_argument_types_before_inserted_guards() {
    test_source(
        r#"
def exact(value: i32) -> int:
    return 1
def exact(value: i64) -> int:
    return 2
def exact(value: float) -> int:
    return 3
def wide(value: i64) -> int:
    return 2
def wide(value: float) -> int:
    return 3
def floating_only(value: float) -> float:
    return value
def guarded(value: i8) -> int:
    return 1
def guarded(value: float) -> int:
    return 2
test:
    value: i32 = 7
    assert(exact(value) == 1)
    assert(wide(value) == 2)
    assert(floating_only(value) == 7.0)
    assert(guarded(7.5) == 2)
    ratio: float = 7.5
    assert(guarded(ratio) == 2)
    assert(guarded(i8(ratio)) == 1)
"#,
    );
}
