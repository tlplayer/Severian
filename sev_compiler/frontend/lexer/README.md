# Source lexer

`Lx` denotes a lexeme rule, before `Y` (symbol identity) and `G` (grammar
behavior). `L` continues to mean Literal. These are generic parameter
conventions, not reserved parameter names.

```text
characters → Lx → Y / literal / identifier / structural token
               → G → O / CFG semantics → I → B
```

## Executable layers

`src/lexeme.sev` defines `LexemeInput`, `LexemeMatch`, and the `Lexeme` trait.
A rule's `scan` method returns either a match result or a diagnostic. The
single result preserves decoded literal values and committed-prefix errors
without scanning a string or number a second time. A result has:

- `matched = false`: the rule does not recognize this prefix.
- `matched = true`, `kind = None`: consume trivia without emitting a token.
- `matched = true`, a token kind: consume input and emit that compiler term.

`end` is an exclusive character offset. The source compiler currently uses
character offsets for spans and diagnostic rendering. `lex[Lx: Lexeme]` in
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
not yet support a heterogeneous `list[Lexeme]`. The rule composition is
therefore statically specialized. The generic dispatcher lives with the
concrete rule imports because the seed also cannot specialize an imported
generic for a caller-only record type. This limitation must be fixed before
arbitrary external rules can use the shared generic entry point.

Importing arbitrary `Lx` implementations into an already-built compiler is
**not implemented**. It requires discovery and validation of lexical contracts,
a compiler-time execution or precompiled-package mechanism for rule bodies,
and a registry that retains definition identity. Keep that work separate from
`G` metadata discovery: reading a rule declaration alone does not execute it.
A future registry must specify rule priority, reject ambiguous matches, retain
progress checks, and preserve lexical context through interpolation and imports.

The later acceptance gate is a package-defined lexical form that changes
behavior when only its Severian rule body changes, with the compiler binary
unchanged. The current operator acceptance gate already exercises that property
for `G`/`Y` spellings.

## Validation

From the repository root:

```sh
target/debug/sev test sev_compiler/frontend/lexer
target/debug/sev test docs/examples/01-types/04-generics/26-lexeme-generic.sev
cd sev_compiler
../target/debug/sev build
cd ..
python3 tests/sev_compiler/migration.py Gate3SourceSyntax
python3 tests/sev_compiler/source_contracts.py
python3 tests/sev_compiler/diagnostics.py
```
