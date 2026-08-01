use std::collections::HashMap;
fn sum_values<K, V>(values: &HashMap<K, V>) -> V where V: Copy + Default + std::iter::Sum<V> { values.values().copied().sum() }
fn main() { let counts = HashMap::from([("first", 34), ("second", 8)]); println!("{}", sum_values(&counts)); }
