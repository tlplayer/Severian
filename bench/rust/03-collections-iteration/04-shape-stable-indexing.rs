fn increment(values: &mut [i64]) { for value in values { *value += 1; } }
fn main() {
    let mut values = vec![10, 20, 30];
    increment(&mut values);
    println!("{values:?}");
}
