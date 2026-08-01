fn add(a: i64, b: i64) -> i64 { a + b }
fn apply(op: fn(i64, i64) -> i64, left: i64, right: i64) -> i64 { op(left, right) }
fn main() { println!("{}", apply(add, 20, 22)); }
