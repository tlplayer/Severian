SIP-0000: Title

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

## Context

Aliasing allows for competing similar verbage that should convey the same operation to reuse the same name and functionality. 
It's very useful to avoid desyncing and retyping it also allows backwards compatibility easily by allowing renaming/hotswaps of implementation without renaming hundreds/thousands of files and allow gradual rollout of fixes.

## Goal(s)

1. Aliasing is simple: functions, classes, traits, variables, imports, and types are easy to alias using as 
2. Aliasing does not cause any issues and flags mismatches and duplicate aliases
3. Aliasing allows hiding conflicting names and renaming them to something more appropriate
## Examples
| Form | Meaning |
|---|---|
| `import "file.sev" as functions` | Bind the imported namespace as `functions` |
| `import foo as bar from "file.sev"` | Introduce `bar` into this scope |
| `defer as suffix` | Add `suffix` as another spelling of `defer`; retain both |
```
class X:
    def yards(feet):
        return 

# Aliasing means they are the same
defer as suffix

defer foo()
suffix foo()
# Both select the same grammar and operation.
# Both defer execution. 
```
## Testing

## Problem(s)

## Responsibilities

## Data Models

## API

## Structure




