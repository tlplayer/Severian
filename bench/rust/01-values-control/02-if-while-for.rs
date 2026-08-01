fn main() {
    let mut count = 0;
    while count < 3 { println!("{count}"); count += 1; }
    for value in 0..4 { println!("{}", if value % 2 == 0 { "even" } else { "odd" }); }
}
