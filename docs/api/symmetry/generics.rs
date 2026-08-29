fn affine<T>(value: T, scale: T, bias: T) -> T
where
    T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy,
{
    value * scale + bias
}

fn main() {
    println!("{}", affine(4_i64, 3, 2));
    println!("{}", affine(10_i64, 4, 2));
}
