struct Point { x: i64, y: i64 }
fn describe(point: Point) {
    match point { Point { x: 0, y: 0 } => println!("origin"), Point { x, y: 0 } => println!("x axis\n{x}"), Point { x, y } => println!("{}", x + y) }
}
fn main() {
    describe(Point { x: 0, y: 0 });
    describe(Point { x: 7, y: 0 });
    describe(Point { x: 3, y: 4 });
}
