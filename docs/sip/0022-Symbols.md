SIP-0000: Lexer

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix
| Term or interface | Kind | Meaning |
|---|---|---|
| `.sev` | File format | Severian source text. |
| SourceFile | Class | Original source text, path, and source identity supplied by Source. |
| SourceId | Class | Identifies one source snapshot. |
| Span | Class | Source identity and half-open Unicode scalar offsets `[start, end)`. |
| SourceWindow | Class | A bounded view of a source snapshot used for recognition. |
| Lexeme | Class | Exact matched source text and its span, before interpretation. |
| Token | Class | Classified parser input, including identifiers, structural boundaries, and end-of-file. |
| LexicalRule | Proposed trait | Declares how a spelling or structural form is recognized within a source window. |
| LexicalMatch | Proposed class | A recognized span, its applicable syntax declarations, and any captured source windows. |
| `lex(input, syntax)` | Proposed function | Applies the available syntax rules to a source window and returns recognized input or a diagnostic. |
| `input` | Parameter | Source window currently being examined. |
| `syntax` | Parameter | Syntax declarations available in the current scope. |
| `cursor` | Local variable | Current Unicode scalar offset within the source window. |
| `G` | Compiler term | Grammar defining valid arrangements and captures. |
| `W` | Compiler term | With-clause operations, including acceptance constraints. |
| `B` | Compiler term | Block, with its declared structure and scope behavior. |
| `S` | Compiler term | Sentence operating within a block. |
| `T` | Compiler term | Type contract required by a declaration or capture. |
| `L` | Compiler term | Literal spelling representing a direct value. |
| `Y` | Compiler term | Symbol role connecting lexemes and tokens to declared meaning. Recognition alone does not establish a resolved symbol identity. |
| Operator | Semantic term | An operation represented by syntax, such as `+`, `==` `not` |
| Identifier | Lexical term | A name spelling whose declaration and role may still require resolution. |
| Capture | Recognition term | A named source window retained for further recognition or resolution. |
| Greedy recognition | Recognition rule | Consume the longest valid spelling permitted by the active grammar and current window. |
| Trivia | Lexical term | Comments and spacing that do not independently express an operation. Indentation can remain structural. |

## Context

The lexer converts source text into tokens using the token forms declared by syntax.
```
SourceFile → lexer → TokenStream → parser → HIR
Syntax ->
```
Syntax owns the spellings and recognition requirements. The lexer supplies source windows, compares matches, preserves source locations, and advances through the input.
For example, syntax declares that "=" can produce an EQUAL token. The parser determines what that token means in its surrounding grammar.
A token identifies a recognized source form. Recognizing "int" does not construct an integer type, and recognizing "if" does not construct a conditional block.

## Goals
1. Syntax has the SOT for symbols ("=" -> <EQUAL>)
2. Symbols can be shared between syntax and the lexer
3. Symbols can be are not automatically linked with types, the Y of "int" should be housed in the int class whereas EOF or `[` should be their own declaration symbol in symbols not declarations in primitives. This should allow lexization and potential tokenization matching.

## Symbols

Keywords (some are implemented on classes, others on global scope grammars)
grammar
sentence
operator
def
class
trait
enum
extend
import
from
as
with
if
elif
else
for
while
match
case
try
catch
return
throw
break
continue
yield
defer
unsafe
async
await
select
lambda
and
or
not
in
is
test
accept
reject

| Spelling | Token label |
|---|---|
| `=` | `EQUAL` |
| `:=` | `COLON_EQUAL` |
| `?=` | `QUESTION_EQUAL` |
| `==` | `EQUAL_EQUAL` |
| `!=` | `NOT_EQUAL` |
| `<` | `LESS` |
| `<=` | `LESS_EQUAL` |
| `>` | `GREATER` |
| `>=` | `GREATER_EQUAL` |
| `+` | `PLUS` |
| `-` | `MINUS` |
| `*` | `STAR` |
| `/` | `SLASH` |
| `//` | `FLOOR_DIVIDE` |
| `%` | `PERCENT` |
| `**` | `POWER` |
| `+=` | `PLUS_EQUAL` |
| `-=` | `MINUS_EQUAL` |
| `*=` | `STAR_EQUAL` |
| `/=` | `SLASH_EQUAL` |
| `//=` | `FLOOR_DIVIDE_EQUAL` |
| `%=` | `PERCENT_EQUAL` |
| `&` | `AMPERSAND` |
| `\|` | `PIPE` |
| `^` | `CARET` |
| `!` | `BANG` |
| `<<` | `LEFT_SHIFT` |
| `>>` | `RIGHT_SHIFT` |
| `&=` | `AMPERSAND_EQUAL` |
| `\|=` | `PIPE_EQUAL` |
| `^=` | `CARET_EQUAL` |
| `<<=` | `LEFT_SHIFT_EQUAL` |
| `>>=` | `RIGHT_SHIFT_EQUAL` |
| `->` | `ARROW` |
| `~>` | `APPROX_ARROW` |
| `<=>` | `CONVERSION` |
| `(` | `LEFT_PAREN` |
| `)` | `RIGHT_PAREN` |
| `[` | `LEFT_BRACKET` |
| `]` | `RIGHT_BRACKET` |
| `{` | `LEFT_BRACE` |
| `}` | `RIGHT_BRACE` |
| `:` | `COLON` |
| `,` | `COMMA` |
| `.` | `DOT` |
| `..` | `RANGE` |
| `...` | `ELLIPSIS` |
| `@` | `AT` |

Quoted and structural forms:
| Form | Token label |
|---|---|
| `"..."` | `STRING` |
| `'...'` | `CHARACTER` |
| `"""..."""` | `BLOCK_STRING` |
| `'''...'''` | `BLOCK_STRING` |
| `f"..."` | `FORMATTED_STRING` |
| `f"""..."""` | `FORMATTED_BLOCK_STRING` |
| `# ...` | `COMMENT` |
| Leading whitespace | `INDENT` |
| Other horizontal whitespace | `SPACE` |
| Line ending | `NEWLINE` |
| End of source | `EOF` |

## Structure

### Data model
```sev
trait Y:
    token_forms: list[TokenForm] = []
    lexeme: Lexeme | None = None
    token: TokenTerm | None = None
    definition: DefId | None = None
    scope: ScopeId | None = None
    type_id: TypeId | None = None
```


syntax.tokens() returns Y's with unique/multiple token matches


Classes and implementers can have multiple symbol (Y) implementations

```
#example
class int: Y + G
    #grammar keyword, label of token
    grammar LEFT_SHIFT["<<"]() -> Y:
        return Y()

    grammar RIGHT_SHIFT["<<"]() -> Y:
        return Y()

#Syntax should register those automatically so syntax.tokens() gives the Y's tokens. and syntax.grammars() gives the grammars for parsers and or converting to standard code etc. 

```
### Recognition
The lexer applies token forms at the current cursor:
1. Give each recognizer an independent, bounded source window.
2. Collect successful matches.
3. Reject matches with invalid source identities, spans, or consumption.
4. Retain matches with the greatest ending offset.
5. Produce a token containing the original lexeme and surviving form identities.
6. Advance the cursor to that ending offset.
If no form matches, return a diagnostic at the current source position.
When several forms recognize the same longest range, retain their classifications and identities. The parser resolves their applicability to a grammar.

### Interface

```
lexer(source: SourceFile) → TokenStream | diagnostics

recognize(window: SourceWindow, form: TokenForm)
    → LexicalMatch | no match | diagnostic

TokenStream.print() → string
```

## Problem(s)


## Examples

## Testing

The lexer should resolve the following to the token stream

```
    # -> INDENT Y


## Todo
