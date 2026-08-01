fn largest<T: Ord + Copy>(values: &[T]) -> Option<T> { values.iter().copied().max() }
fn main() { match largest(&[3, 9, 2]) { Some(value) => println!("present({value})"), None => println!("absent") } }
