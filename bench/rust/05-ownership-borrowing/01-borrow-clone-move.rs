fn sum(values: &[i64]) -> i64 { values.iter().sum() }
fn main() {
    let numbers = vec![1, 2, 3, 4];
    let copied = numbers.clone();
    let owned = copied;
    println!("{}\n{}", sum(&numbers), sum(&owned));
}
