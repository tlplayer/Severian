struct X { value: i64 }
impl X {
    fn from_sum(x: i64, y: i64) -> Self { Self { value: x + y } }
    fn new(x: i64) -> Self { Self { value: x } }
}
fn main() { println!("{}\n{}", X::from_sum(20, 22).value, X::new(42).value); }
