struct Point { x: f64, y: f64 }
impl Point { fn translated(&self, dx: f64, dy: f64) -> Point { Point { x: self.x + dx, y: self.y + dy } } }
fn main() { let next = Point { x: 1.0, y: 2.0 }.translated(3.0, 4.0); println!("{}\n{}", next.x, next.y); }
