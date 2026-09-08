# Source lexer

`Lx` denotes a lexeme: matched source text and its span. `To` denotes a token:
the classified lexical object handed to the parser. `L` remains Literal and
`Y` remains a syntactic/resolved Symbol. These are generic parameter conventions,
not reserved parameter names.

```text
characters → Lx → To → parser → Y / L / Ex / S / D / P / ... → G → CFG
                                                        → O / I / B
```

For `total += 1_024`, the lexemes retain `total`, `+=`, and `1_024`. The tokens
classify them as identifier, symbol, and integer. The integer token stores only `1_024`;
the parser normalizes that spelling to `1024` when constructing `L(IntegerLiteral)`. The symbol token carries the registered
syntax used to resolve `Y` and `G:AddAssign`. A token is not itself a `Y`.

## Executable layers

`src/lexeme.sev` defines the `LexemeTerm` contract and concrete `Lexeme` value
(`text`, `span`). `src/token/mod.sev` defines `TokenTerm` (`lexeme`, `span`) and
the concrete parser `Token`. `TokenKind` is a payload-free classification:

```text
Identifier Integer Float Character String FormattedString Symbol
Newline Indent Dedent Eof
```

All punctuation uses `Symbol`, including `@`, `:`, `:=`, `?=`, `=`, `->`,
brackets, braces, parentheses, `|`, `,`, and `.`. The bootstrap list contains
spellings, not token-kind assignments. The parser checks a symbol's spelling
through `token_is_symbol`/`parser_expect_symbol`; new value operators still
resolve through registered syntax rather than additional parser branches.

Ordinary identifiers, numeric tokens, and symbols store no second text value.
`token_text` reads their lexeme, while `token_number` normalizes numeric spelling
on demand at literal construction (or syntax-metadata evaluation). The optional
`Token.value` holds decoded string/character text or a macro replacement. Escape
validation remains lexical for now; character-to-codepoint conversion happens
when the parser constructs a literal. Full string interpretation in `L` remains
a possible follow-up.

`checked_token[To: TokenTerm](token: To) -> To` validates lexical origin while
preserving the concrete token type. `token_from_lexeme` constructs parser tokens
through that generic entry. The two spans agree, including source identity.

Lexer-created tokens retain exact matched text, including literal quotes,
escapes, separators, and indentation. EOF and EOF dedents have empty lexemes
and zero-width spans. Macro replacement tokens retain the original token's
lexeme as their source origin while changing the classified payload; the macro
EOF has an empty lexeme at the end of the definition. The lexeme is
never reconstructed from classification or a normalized value.

`src/rule.sev` defines `LexemeInput`, `LexemeMatch`, and `LexicalRule`. A rule
is the scanner that produces lexemes/tokens, rather than an `Lx` value itself.
A rule's `scan` method returns either a match result or a diagnostic. The
single result preserves decoded literal values and committed-prefix errors
without scanning a string or number a second time. A result has:

- `matched = false`: the rule does not recognize this prefix.
- `matched = true`, `kind = None`: consume trivia without emitting a token.
- `matched = true`, a token kind: consume input and emit that classification.

A match may additionally carry decoded text for a string or character. Numeric
and identifier matches carry no text payload.

`end` is an exclusive character offset. The source compiler currently uses
character offsets for spans and diagnostic rendering. `lex[Rule: LexicalRule]` in
`src/lexer.sev` validates progress and bounds for every successful match.

`IdentifierLexeme`, `NumericLexeme`, `StringLexeme`, `SymbolLexeme`, and
`TriviaLexeme` live in their corresponding source modules. Integer and float
forms share the numeric recognizer. Indentation helpers live in
`src/indentation.sev`; the driver maintains the indent stack, line boundaries,
and EOF dedents because these are stream state rather than isolated matches.

`StandardLexemes` composes the rules. Trivia has first priority. Symbols use
longest match, with a leading fractional number taking priority over `.`.
Strings, numbers, and identifiers follow. Word symbols attach grammar metadata
after consuming a complete identifier, so `fuse` does not split `fuse_count`.
Token construction attaches syntax metadata centrally, including the full
operator environment needed when parsing a formatted string's expressions.

`src/symbol.sev` contains the sole symbol matcher. `src/syntax/mod.sev` discovers
and resolves imported source descriptors. Registering a new `G` symbol adds
its spelling to that matcher; no lexical method is needed per operator.

## Bootstrap boundary and remaining work

`scan_bootstrap` uses only fixed declaration delimiters and the built-in
lexical classes. Unknown punctuation becomes individual discovery tokens; no
imported value operator is active in this pass. Module discovery traverses
imports, resolves their descriptors, then `scan_registered` scans the affected
source with the complete symbol environment. Bootstrap still shares the
literal implementations, including formatted and block strings, so it can
walk existing compiler definitions without another independent scanner.

The Rust seed compiles these Severian rules into the source compiler. It does
not yet support a heterogeneous `list[LexicalRule]`. The rule composition is
therefore statically specialized. The generic dispatcher lives with the
concrete rule imports because the seed also cannot specialize an imported
generic for a caller-only record type. This limitation must be fixed before
arbitrary external rules can use the shared generic entry point.

Importing arbitrary lexical-rule implementations into an already-built compiler is
**not implemented**. It requires discovery and validation of lexical contracts,
a compiler-time execution or precompiled-package mechanism for rule bodies,
and a registry that retains definition identity. Keep that work separate from
`G` metadata discovery: reading a rule declaration alone does not execute it.
A future registry must specify rule priority, reject ambiguous matches, retain
progress checks, and preserve lexical context through interpolation and imports.

The later acceptance gate is a package-defined lexical form that changes
behavior when only its Severian rule body changes, with the compiler binary
unchanged. The operator acceptance gate exercises that property for `G`/`Y` spellings.
`test_imported_punctuation_keeps_frontend_and_binary_unchanged` imports `|>` and
then `|~>` from a source package, executes both directly and in interpolation,
and verifies unchanged lexer/parser sources and compiler binary. `TokenKind`
is covered by the same source snapshot.

## Validation

From the repository root:

```sh
target/debug/sev test sev_compiler/frontend/lexer
target/debug/sev test docs/examples/01-types/04-generics/26-lexeme-generic.sev
target/debug/sev test docs/examples/01-types/04-generics/27-token-generic.sev
cd sev_compiler
../target/debug/sev build
cd ..
python3 tests/sev_compiler/migration.py Gate3SourceSyntax
python3 tests/sev_compiler/source_contracts.py
python3 tests/sev_compiler/diagnostics.py
```
