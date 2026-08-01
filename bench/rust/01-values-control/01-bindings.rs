fn main() {
    let name = "Ada";
    let mut score = 41;
    score += 1;
    println!("{}", score == 42 && !name.is_empty());
}
