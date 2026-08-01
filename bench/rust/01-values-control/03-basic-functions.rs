fn add(a: i64, b: i64) -> i64 { a + b }
fn main() { println!("{}", if add(10, 32) > 40 { "large" } else { "small" }); }
