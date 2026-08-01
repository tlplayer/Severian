fn find_name(id: i64) -> Option<&'static str> { if id == 1 { Some("ada") } else { None } }
fn main() { match find_name(1) { Some(value) if !value.is_empty() => println!("{value}"), Some(_) => println!("blank"), None => println!("missing") } }
