fn main() {
    let values = [4, -2, 7, 1];
    let summary = [
        *values.iter().min().unwrap(),
        *values.iter().max().unwrap(),
        values.iter().sum(),
    ];
    let weighted_total: i32 = [5, 10, 20]
        .iter()
        .enumerate()
        .map(|(index, value)| index as i32 * value)
        .sum();
    let dot_product: i32 = [1, 2, 3]
        .iter()
        .zip([4, 5])
        .map(|(left, right)| left * right)
        .sum();
    let until_negative: i32 = [2, 0, 5, -1, 100]
        .iter()
        .take_while(|value| **value >= 0)
        .sum();

    println!("{}", "alpha,beta,gamma".split(',').collect::<Vec<_>>().join(" | "));
    println!("{summary:?}");
    println!("{weighted_total}");
    println!("{dot_product}");
    println!("{until_negative}");
}
