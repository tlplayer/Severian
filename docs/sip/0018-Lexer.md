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

The lexer takes source's output and syntax and greedily attempts to convert the raw string to the corresponding:

- blocks (classes, traits, enums, if, else, try, catch, elif, )
- sentences (for x in 0..10: while x with x := 0: etc.)
- operators (+,-,=,<=, >=, ->, etc.) 
- type
- variables (a)
- literals (0,0.0, -1)

## Goals
1. syntax is the SOT for spellings, symbol arrangement. Lexer just processes in the order that they are declared based on trait implementation/syntax master list for resolution as a stop gap
2. If a new syntax occurs the definition of that in the syntax/other filess should not break the lexer/parser. Tokens/lexemes are defined on the Y generic implementers nothing else should be needed.

Why? it simplifies addition/troubleshooting. Root causing lexer/parser errors is straightforward. Forces grammar/sentence structure to be refined without rellying on magic functions to extract/remove certain elements from the language. 
Literal/complex symbol orientation for example class X with a constructor form X{} should just require an implementation of Y for that arrangement of symbols not modifying the lexer/parser for a simplfied language. 

## Structure

| File | Responsibility |
|---|---|
| `lexer.sev` | Public entry point. Pulls declarations from the **syntax master list**, processes them in the declared order specified by the stopgap, and coordinates recognition. |
| `window.sev` | Bounded source views, cursor movement, captured regions, and original source spans. |
| `recognize.sev` | Applies the supplied syntax declarations to source windows and returns matches, captures, or diagnostics. |

```sev
trait Lexer:
    def lex(input: SourceFile) -> list[LexicalMatch] | Diagnostic
```

```sev
trait Window:
    source: SourceFile
    start: u32
    end: u32
    cursor: u32

    def peek(distance: u32 = 0) -> string | None
    def advance(count: u32) -> unit | Diagnostic
    def finished() -> bool

    def capture(start: u32, end: u32) -> Window | Diagnostic
    def span() -> Span
    def text() -> string
```

```sev
trait Recognizer:
    def recognize(
        input: Window,
        syntax: Y
    ) -> LexicalMatch | None | Diagnostic
```


```
class LexicalMatch:
    syntax: Y
    lexeme: Lexeme
    token: TokenTerm | None
    captures: list[LexicalCapture]

class LexicalCapture:
    name: string
    window: Window
```

## Problem(s)

1. Greedy consumption and tokenization
```sev

def foo(a):
    b = 2 + a

def main():
    a := 1
    b = 1
    foo(a)
    if b != 1:
        throw Error("variables mangled b should be 1 because foo has no context of main.b vs foo.b")
```

The order in which the lexer processes this source is

1. get the blocks

main

foo

main.if

2. process inside of the blocks

for foo:

resolve a

resolve b, 2 as a literal int, +, and a as a


for main:
resolve a, :=, 1, b, =, 1, function call with a (handled by the function block parser), 

inside if parse out the throw operator, Error() class constructor sentece etc. 



2. mismatch between token and meaning
```sev
#resolves to an int 
#
a:int = 0
#a:int is a string here and the lexer should tokenize b as a string type with literal value a:int in characters by the "" literal binding that string should implement and lexer should pull on
b = "a:int"
```

3. The diagnostic (syntax.error.diagnostic) contains source, and compiler step information for pointing where in code the lexer failed to process.



## Examples

### Conditional block
```sev
if 1 == 0.0:
    return false
```
Recognition establishes:
- The conditional header and its body window.
- A condition containing 1, ==, and 0.0.
- A body containing the return sentence and false.
The integer and floating-point spellings retain distinct literal contracts. The applicable equality operation is resolved using its operand contracts.

### Loop with an initializer
```sev
while count < 3 with count := 0:
    count += 1
```
The loop grammar captures the condition, initializer, and body. Operator recognition preserves <, :=, and += as complete spellings.
The initializer’s scope and its relationship to the condition are established by the loop declaration’s contract.

### Type and variable names
```
point: Point = Point(1, 2)
```
The sentence grammar establishes a binding name, a type capture, and an initializer capture. Both occurrences of Point preserve their spelling and span; their roles follow from the captures and subsequent resolution.

### Operator boundaries
```
a >= b
a > = b #syntax error
```
The first line contains the contiguous spelling >=. The second contains two separate spellings. Whitespace cannot be discarded in a way that merges them.

### Quoted contents

```sev
message = "if ready: # still text"
The apparent keyword, colon, and comment marker remain inside the string’s quoted region.
```

## Testing
| Case | Required result |
|---|---|
| Empty source | Successful completion with an end-of-file boundary. |
| Nested blocks | Correct parent and child windows with original spans. |
| Sentence captures | Header, expression, initializer, and body captures remain distinct. |
| Overlapping operator spellings | Longest valid contiguous spelling is recognized. |
| Whitespace between operators | Separate spellings remain separate. |
| Quoted punctuation | Block and comment markers inside strings remain string contents. |
| Continued expressions | Internal newlines respect the enclosing grammar. |
| Identifier boundaries | A keyword prefix inside a longer identifier does not split the identifier. |
| Literal forms | Declared integer, floating-point, string, and character spellings retain their contracts. |
| Signed forms | Unary and binary uses follow the active grammar. |
| Unicode and removed trivia | Every reported span maps to the exact original text. |
| Ambiguous declarations | Ambiguity is reported independently of declaration order. |
| Deferred constraints | Unknown conditions remain pending until their required captures are resolved. |
| Selected implementation failure | The failure is reported without trying another implementation. |
| Malformed input | Unterminated regions and unknown spellings produce bounded diagnostics. |
| Recognition progress | Empty matches cannot create an infinite loop. |
| Syntax extension | A supplied declaration enables a new form without modifying lexer dispatch code. |

