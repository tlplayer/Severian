use std::collections::{HashMap, HashSet};
fn main() {
    let mut counts = HashMap::from([("red", 2), ("blue", 3)]);
    let seen = HashSet::from(["red", "green"]);
    *counts.get_mut("red").unwrap() += 1;
    if seen.contains("green") { println!("{}", counts["red"]); }
}
