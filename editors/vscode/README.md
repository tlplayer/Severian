# Severian for VSCodium and VS Code

This extension provides lexical editor support for `.sev` files:

- the `severian` language ID;
- TextMate highlighting for declarations, decorators, native declarations, types, ownership operations, control flow, concurrency, tests, literals, calls, members, comments, and operators;
- comment toggling, bracket pairing, indentation, and folding.

The grammar is intended to track `compiler/lexer/src/lib.rs`. Run the grammar check from the repository root after changing language tokens:

```bash
python3 editors/vscode/tests/check_grammar.py
```

## Install locally

From the Severian repository root:

```bash
cd editors/vscode
./scripts/install-local.sh
```

Then run **Developer: Reload Window** and open a `.sev` file. The language selector in the lower-right corner should show **Severian**.

For extension development without packaging:

```bash
code --extensionDevelopmentPath="$PWD/editors/vscode" "$PWD"
# or
codium --extensionDevelopmentPath="$PWD/editors/vscode" "$PWD"
```

Use **Developer: Inspect Editor Tokens and Scopes** to inspect the TextMate scopes selected by the grammar.

This layer intentionally performs lexical highlighting only. Compiler-backed diagnostics, symbol resolution, ownership classifications, go-to-definition, completion, and precise semantic tokens belong in the Severian language server.
