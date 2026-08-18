pub(super) fn print() -> Result<(), String> {
    let errors = severian_diagnostics::explain::all()
        .into_iter()
        .filter(|explanation| {
            explanation.code.len() == 7
                && explanation.code.starts_with('E')
                && explanation.code.as_bytes()[1..]
                    .iter()
                    .all(u8::is_ascii_digit)
        })
        .collect::<Vec<_>>();
    println!("Severian compiler errors ({} registered)", errors.len());
    println!("Use `sev explain EXXXXXX` for causes, examples, and fixes.\n");
    for explanation in errors {
        println!("{:<8} {}", explanation.code, explanation.title);
    }
    Ok(())
}
