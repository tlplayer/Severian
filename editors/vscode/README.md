# Severian for VSCodium and VS Code

This extension provides lexical editor support for `.sev` files:

- the `severian` language ID;
- TextMate highlighting for declarations, decorators, native declarations, types, ownership operations, control flow, concurrency, tests, literals, calls, members, comments, and operators;
- comment toggling, bracket pairing, indentation, and folding.

It is publicly distributable under the same Severian license as the compiler.
The complete terms are bundled in [`LICENSE.md`](LICENSE.md); this extension is
not offered under an alternate license.

The grammar is intended to track `compiler/lexer/src/lib.rs`. Run the grammar check from the repository root after changing language tokens:

```bash
python3 editors/vscode/tests/check_grammar.py
```

Or, from this directory after `npm install`:

```bash
npm run check
npm run package
```

## Install locally

No Marketplace account, publishing token, Node.js, or npm is needed for local
highlighting. From the Severian repository root, run:

```bash
./editors/vscode/scripts/install-local.sh
```

The command detects VSCodium or VS Code and links this checkout directly into
its local extensions directory. Repository grammar changes are therefore
available after reloading the editor; no VSIX rebuild is required.

Then run **Developer: Reload Window** and open a `.sev` file. The language
selector in the lower-right corner should show **Severian**.

For extension development without packaging:

```bash
code --extensionDevelopmentPath="$PWD/editors/vscode" "$PWD"
# or
codium --extensionDevelopmentPath="$PWD/editors/vscode" "$PWD"
```

Use **Developer: Inspect Editor Tokens and Scopes** to inspect the TextMate scopes selected by the grammar.

This layer intentionally performs lexical highlighting only. Compiler-backed diagnostics, symbol resolution, ownership classifications, go-to-definition, completion, and precise semantic tokens belong in the Severian language server.

## Publish publicly

The manifest is configured for the public `severian` Marketplace publisher.
The publisher owner must authenticate once and then publish:

```bash
npx @vscode/vsce login severian
npm run publish
```

For CI publication, configure the repository secret `VSCE_PAT`, run the
`vscode-extension` workflow manually, and select its `publish` input. Pushes
and pull requests only build the public VSIX artifact; they never publish it.
Increment `version` before publishing a new release.
