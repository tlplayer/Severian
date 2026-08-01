fn parse_count(text: &str) -> Result<i64, String> { if text.is_empty() { Err("empty count".into()) } else { text.parse().map_err(|error| format!("{error}")) } }
fn main() { match parse_count("42") { Ok(count) => println!("{count}"), Err(reason) => println!("{reason}") } }
