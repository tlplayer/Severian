# SIP-0021: HIR Graph

## Testing

These integration tests exercise source and syntax → tokenizer → parser → HIR through the package build pipeline. Each stage consumes the preceding stage's actual output; tests must not substitute prebuilt tokens, block graphs, or dependency edges.

### 1. Nested blocks, end to end

```sev
if true:
    if false:
        true
```

Use the real syntax registry. Assert exact token spellings and source spans, global/parent/child block relationships, sentence order, Boolean captures, and ownership. HIR must keep internal execution blocks with their owning submodule and place it in the requested module while retaining the shared blocks.

### 2. Dependencies determine submodules

```sev
def first():
    second()

def second():
    first()

def independent():
    return
```

Run the same pipeline without executing these functions. Assert that HIR derives `first → second` and `second → first` from the parsed references, groups the recursive pair into one submodule, and keeps `independent` in a separate submodule. Both belong to the requested module. Do not manually insert dependency edges.

### 3. Ambiguous syntax fails before construction

Supply two type-owned grammars accepting the same spelling. Assert that tokenization retains both candidates, parsing reports ambiguity at the correct source span, neither construction body runs, and the pipeline returns no successful HIR program. Repeat with registration order reversed and require the same outcome.
