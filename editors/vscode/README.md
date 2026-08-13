# Severian for VSCodium and VS Code

This extension provides editor support for `.sev` files:

- the `severian` language ID;
- TextMate highlighting for declarations, decorators, native declarations, types, ownership operations, control flow, concurrency, tests, literals, calls, members, comments, and operators;
- comment toggling, bracket pairing, indentation, and folding;
- one-click Run, Build, Test, Profile, and Debug actions, plus **Run With…**
  and **Build With…** menus, backed directly by
  `sev run`, `sev build`, `sev test`, `sev test --profile`, and `sev debug`;
- contract failure diagnostics with source jumping and captured `vars`;
- native green/red coverage gutters backed by `sev coverage`.

## CLI-backed actions

The editor title exposes **Run**, **Build**, **Test**, **Profile**, and **Debug**.
**Run With…** groups the execution, test, profile, coverage, and debug modes;
**Build With…** selects native, verified, LLVM IR, assembly, or XLA StableHLO builds.
These actions are deliberately thin wrappers around the standard CLI; they do
not maintain a separate editor execution model. Debug runs in a terminal so
LLDB or GDB remains interactive. Set `severian.executable` when `sev` is not on
`PATH`, or `severian.target` to override the active file or nearest project.

## Coverage gutters

Open a `.sev` file and run **Severian: Run Coverage** from the Command Palette
or use the run button in the editor title. The extension runs `sev coverage`,
then marks executable lines in the gutter:

- blue means the line was reached;
- red means the line was not reached.

Hover a mark for its statement-region count. The status bar shows line
coverage for the active file; hover it for line, region, branch, and function
percentages. Existing reports under `target/coverage` load automatically, and
**Severian: Load Coverage Gutters** refreshes them manually.

By default the command covers the nearest project containing `package.toml`,
falling back to the workspace root. Set `severian.coverage.target` when the
coverage root is different (for example, `scripts_sev`), or
`severian.coverage.executable` when `sev` is not on `PATH`.

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
installation. From the Severian repository root, run:

```bash
./editors/vscode/scripts/install-local.sh
```

The command detects VSCodium or VS Code and links this checkout directly into
its local extensions directory. Extension changes are therefore available
after reloading the editor; no VSIX rebuild is required.

Then run **Developer: Reload Window** and open a `.sev` file. The language
selector in the lower-right corner should show **Severian**.

For extension development without packaging:

```bash
code --extensionDevelopmentPath="$PWD/editors/vscode" "$PWD"
# or
codium --extensionDevelopmentPath="$PWD/editors/vscode" "$PWD"
```

Use **Developer: Inspect Editor Tokens and Scopes** to inspect the TextMate scopes selected by the grammar.

Compiler-backed diagnostics, symbol resolution, ownership classifications,
go-to-definition, completion, and precise semantic tokens belong in the
Severian language server. Coverage is implemented directly because the
compiler already emits stable source spans and hit IDs.

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
