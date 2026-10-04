# SIP-0017: Parser

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table


| Record | Contents |
|---|---|
| `BlockGraph` | Source units, blocks, sentences, semantic terms, bindings, uses, constraints, and dependency edges. |
| `Block` | Identity, source span, implementing `B`, enclosing scope, and ordered references to its contents. |
| `Sentence` | Owning block, source span, grammar declaration identity, captures, and references to its semantic results. |
| Semantic term | A declaration, expression, operation, type, literal value, or other syntax-defined result retaining its concrete contract. |
| Binding/use | A declaration’s identity and scope, or a reference awaiting resolution. |
| Dependency | Consumer → provider, with the originating use/span explaining why it exists. |
| `Submodule` | A resolved group of blocks, its interface, and dependencies on other submodules. |
| HIR result | The graph plus its submodule partition and exported interfaces. |

## Context

Once the lexer has tokenized the source file according to syntax. The parser then attempts to group by the grammar defined. 
First grouping blocks, then sentences, then atomic operations like a = b, a = 1+1, etc. So we don't miss parse items. 

## Structure

Blocks and terms live in indexed collections and reference one another by ID. This preserves sharing, cycles, and cross-file references. Ordered block contents preserve sentence order. The grammar’s implementing declaration supplies behavior; adding a new block form should not require another compiler-owned IfBlock-style variant.
There are three relationships we must keep distinct:
- Containment: where a sentence or nested block belongs.
- Scope: where a name is visible.
- Dependency: which declaration or implementation a consumer requires.
[SIP-0013](/home/tplayer/Documents/Severian/docs/sip/0013-Submodule.md) specifies strongly connected components for submodules. We should apply that to semantic dependencies. Mutually recursive declarations belong together; merely sharing a file or enclosing class does not force them together. A callable’s internal execution blocks remain attached to that callable during partitioning.

## Structure

### Data Models

```

```

## Problem(s)

For the parser → HIR boundary, successful output should guarantee:
- Valid, source-qualified identities and exact source spans.
- Explicit ownership and ordering for every sentence and block.
- Captures and syntax-defined results retained.
- Unresolved names, types, or constraints represented explicitly.

## Examples

## Testing


