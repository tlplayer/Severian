enum Value<'a> { Text(&'a str), Int(i64), Float(f64) }
fn to_float(value: Value<'_>) -> f64 {
    match value {
        Value::Text(value) => value.parse().unwrap(),
        Value::Int(value) => value as f64,
        Value::Float(value) => value,
    }
}
fn main() {
    let _ = (to_float(Value::Int(4)), to_float(Value::Float(4.5)));
    println!("{}", to_float(Value::Text("4.5")));
}
