fn scale(value: f64, factor: f64) -> f64 { value * factor }
fn describe(label: &str, value: f64) -> String { format!("{label}: {value}") }
fn main() { println!("{}", describe("width", scale(12.0, 2.0))); }
