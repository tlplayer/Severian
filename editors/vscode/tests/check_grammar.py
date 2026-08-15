#!/usr/bin/env python3
from pathlib import Path
import json
import re
import sys

repo_root = Path(__file__).resolve().parents[3]
lexer_path = repo_root / "compiler" / "lexer" / "src" / "lib.rs"
grammar_path = repo_root / "editors" / "vscode" / "syntaxes" / "severian.tmLanguage.json"

lexer = lexer_path.read_text()
grammar_text = grammar_path.read_text()
json.loads(grammar_text)

keyword_block_match = re.search(
    r"let kind = match value \{(?P<body>.*?)_ => TokenKind::Identifier",
    lexer,
    flags=re.S,
)
if not keyword_block_match:
    print("could not locate lexer keyword table", file=sys.stderr)
    sys.exit(1)

keywords = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*=>\s*TokenKind::', keyword_block_match.group("body")))
missing = sorted(keyword for keyword in keywords if keyword not in grammar_text)

required_syntax = {
    "formatted strings": 'f\\"',
    "power operator": r"\\*\\*",
    "decorators": "@",
    "cross operator": "^",
    "address-of": "&",
    "integ tests": "integ",
}
missing_features = [name for name, marker in required_syntax.items() if marker not in grammar_text]

if missing or missing_features:
    if missing:
        print("lexer keywords missing from VS Code grammar:")
        for keyword in missing:
            print(f"  - {keyword}")
    if missing_features:
        print("required syntax missing from VS Code grammar:")
        for name in missing_features:
            print(f"  - {name}")
    sys.exit(1)

print(f"VS Code grammar JSON is valid and covers {len(keywords)} lexer keywords.")
