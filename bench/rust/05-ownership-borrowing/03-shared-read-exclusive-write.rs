fn total(values: &[i64]) -> i64 { values.iter().sum() }
fn push_value(values: &mut Vec<i64>, value: i64) { values.push(value); }
fn main() { let mut values = vec![1, 2, 3]; println!("{}", total(&values)); push_value(&mut values, 4); println!("{}", total(&values)); }
