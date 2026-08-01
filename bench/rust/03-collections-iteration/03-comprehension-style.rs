fn main() {
    let values = [1, 2, 3, 4];
    let doubled: Vec<_> = values.iter().map(|value| value * 2).collect();
    let evens: Vec<_> = doubled.into_iter().filter(|value| value % 2 == 0).collect();
    println!("{evens:?}");
}
