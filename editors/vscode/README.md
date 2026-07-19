# Severian for VSCodium and VS Code

This extension provides immediate editor support for `.sev` files:

- the `severian` language ID;
- TextMate syntax scopes for declarations, types, ownership operations,
  control flow, concurrency, literals, calls, comments, and operators;
- comment toggling, bracket pairing, indentation, and folding.

## Run locally

Launch VSCodium with this unpacked extension during development:

```bash
codium --extensionDevelopmentPath="$PWD/editors/vscode" "$PWD"
```

Then open a `.sev` file. The language selector in the lower-right corner should
show **Severian**. Use **Developer: Inspect Editor Tokens and Scopes** to inspect
the scopes selected by the grammar.

This layer intentionally performs lexical highlighting only. Compiler-backed
diagnostics, symbol resolution, ownership classifications, go-to-definition,
and precise semantic tokens belong in the future Severian language server.
