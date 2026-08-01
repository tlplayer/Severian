use std::io;
fn load(_path: &str) -> Result<String, io::Error> { Ok("settings".into()) }
fn main() { match load("settings.toml") { Ok(data) => println!("{data}"), Err(error) => println!("{error}") } }
