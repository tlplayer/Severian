trait Named { fn name(&self) -> &str; }
trait Drawable { fn draw(&self); }
struct Button { label: String }
impl Named for Button { fn name(&self) -> &str { &self.label } }
impl Drawable for Button { fn draw(&self) { println!("{}", self.label); } }
fn render(item: &dyn Drawable) { item.draw(); }
fn main() { let button = Button { label: "Save".into() }; let _ = button.name(); render(&button); }
