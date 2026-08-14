use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::analyze;

fn analyze_error(source: &str) -> String {
    let module = parse(&lex(source).unwrap()).unwrap();
    analyze(&module).unwrap_err().message
}

#[test]
fn suggests_an_unambiguous_named_argument_typo() {
    let message = analyze_error(
        "def load(name: string, device: string):\n    return\n\ndef main():\n    load(\"Qwen\", devcie = \"cuda\")\n",
    );
    assert_eq!(
        message,
        "E000204: unknown argument `devcie`; did you mean `device`?"
    );
}

#[test]
fn enum_switches_must_cover_every_variant() {
    let message = analyze_error(
        "enum Status:\n    Ready\n    Failed\n    Waiting\n\ndef run(status: Status):\n    switch status:\n        Ready:\n            return\n        Failed:\n            return\n",
    );
    assert_eq!(
        message,
        "E000206: non-exhaustive switch on `Status`; missing `Status.Waiting`"
    );
}

#[test]
fn exhaustive_enum_switches_remain_valid() {
    let source = "enum Status:\n    Ready\n    Failed\n\ndef run(status: Status):\n    switch status:\n        Ready:\n            return\n        Failed:\n            return\n";
    let module = parse(&lex(source).unwrap()).unwrap();
    analyze(&module).unwrap();
}

#[test]
fn constant_zero_divisors_fail_before_lowering() {
    let message = analyze_error("def main():\n    print(100 / 0)\n");
    assert_eq!(
        message,
        "E000502: division by zero is known at compile time"
    );
}
