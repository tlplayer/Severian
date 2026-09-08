# Source lexer

`Lx` (Lexeme) preserves matched source text and its span. `To` (Token) classifies that
source for the parser. `L` remains Literal and `Y` remains a syntactic/resolved
Symbol. Generic lexical APIs use `Lx` and `To` with their specific contracts; the
letters are conventions, not reserved parameter names.

```text
characters → Lx → To → parser → Y / L / Ex / O / X → G → CFG
                                                 → O / B
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
`string(token)` uses Token's `operator <=>(self) -> string` to read their lexeme, while `token_number` normalizes numeric spelling
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
is the scanner that produces lexemes/tokens, rather than a `Lexeme` value itself.
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
walk existing compiler definitions without another independent scanner. Literal
diagnostics are deferred until configured scanning, so a new raw literal may
contain text that would be invalid inside a bootstrap string.

The Rust seed compiles the built-in rules into the source compiler. Imported
concrete, stateless `LexicalRule` classes are loaded separately: the module
loader parses their definitions with the standard language, and `runtime.sev`
executes their universal source bodies. The lexer never imports the parser.
Definitions are masked out of the application pass while preserving character
offsets; they do not become runtime application classes.

## Imported lexical rules

A relative source import can contribute a rule without changing the compiler
binary. Its contract is the same `scan(input: LexemeInput, start: int)` used by
the built-in lexical classes. `Lexeme` still denotes the resulting source lexeme,
not the rule implementation.

```sev
class SigilInteger: LexicalRule:
    def scan(input: LexemeInput, start: int) -> LexemeMatch:
        characters = input.characters
        if characters[start] != "~":
            return LexemeMatch(start, None, false)
        cursor := start + 1
        while cursor < len(characters) and is_digit(characters[cursor]):
            cursor += 1
        if cursor == start + 1:
            return LexemeMatch(start, None, false)
        return LexemeMatch(cursor, TokenKind.Integer, value=lexeme_text(input, start + 1, cursor))
```

Importing that file makes `~42` an integer token with source lexeme `~42` and
literal text `42`. Editing the method body changes tokenization on the next
compilation, without rebuilding either compiler. A symbol result resolves its
normalized spelling through the existing `Y`/`G` registry. Formatted-string
interpolation retains the imported lexical environment. Run an importing subject
with the source compiler (the Rust seed does not load these extensions):

```sh
sev_compiler/target/host/dev/bin/sev_compiler test subject.sev --sysroot .
```

Whitespace, comments, and structural events remain owned by the standard
scanner. At other positions imported rules run before built-in literal and
symbol rules. The longest imported match wins; an optional constant
`priority: int = 1` resolves equal lengths. Equal length and priority is an
ambiguity error, independent of import order. If no imported rule matches,
scanning falls back to the standard rules. Every successful match must advance within the source bounds.

The source evaluator supports integer/text/boolean expressions, character
indexing, local bindings, conditions, while loops, return/break/continue, assertions,
and concrete helper methods on the rule. It exposes `len`, `string`,
`is_digit`, `lexeme_text`, `lexeme_span`, source diagnostics, and text methods `characters`, `starts_with`,
`ends_with`, `replace`, and `join`. Unsupported operations produce diagnostics;
there is no filesystem, process, or CFG capability. Each invocation has a
10,000-expression budget and a 64-call depth limit. Generic/stateful imported
rule objects and arbitrary application/library calls are not yet supported;
built-in rule dispatch continues to use `lex[Rule: LexicalRule]`.

`tests/sev_compiler/lexical_rules.py` exercises imported numeric and string
forms, body changes, normalized symbols, interpolation, priorities, progress,
and execution limits. Its migration check verifies unchanged lexer/parser
sources and compiler binary. The corresponding operator gate in
`source_contracts.py` imports `|>` and `|~>` without changing those files.

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
python3 tests/sev_compiler/lexical_rules.py
python3 tests/sev_compiler/diagnostics.py
```
