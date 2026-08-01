trait Drawable { fn draw(&self); }
struct Point { x: f64, y: f64 }
impl Point { fn magnitude(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() } }
impl Drawable for Point { fn draw(&self) { println!("point {} {}", self.x, self.y); } }
fn main() { let point = Point { x: 3.0, y: 4.0 }; point.draw(); println!("{}", point.magnitude()); }
