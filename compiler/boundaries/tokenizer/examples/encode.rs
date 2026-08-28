use std::process::ExitCode;
use tokenizers::Tokenizer;

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(path) = arguments.next() else {
        eprintln!("usage: encode <tokenizer.json> <text>");
        return ExitCode::FAILURE;
    };
    let Some(text) = arguments.next() else {
        eprintln!("usage: encode <tokenizer.json> <text>");
        return ExitCode::FAILURE;
    };
    let tokenizer = match Tokenizer::from_file(path) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("could not load tokenizer: {error}");
            return ExitCode::FAILURE;
        }
    };
    let encoding = match tokenizer.encode(text, false) {
        Ok(encoding) => encoding,
        Err(error) => {
            eprintln!("could not encode text: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("{:?}", encoding.get_ids());
    ExitCode::SUCCESS
}
