'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { makeImportsExplicit } = require('../explicit-imports');
function harness(text) {
  const applied = [];
  const vscode = {
    Uri: { file: path => ({ fsPath: path }) },
    Range: class { constructor(start, end) { Object.assign(this, { start, end }); } },
    WorkspaceEdit: class { constructor() { this.edits = []; } replace(uri, range, after) { this.edits.push({ uri, range, after }); } },
    workspace: {
      saveAll: async () => true,
      getConfiguration: () => ({ get: () => 'sev' }),
      openTextDocument: async uri => ({ uri, getText: () => text, positionAt: offset => offset }),
      applyEdit: async edit => { applied.push(edit); return true; },
    },
    window: { showInformationMessage: async () => {} },
  };
  return { vscode, applied };
}
test('uses the compiler JSON plan and applies an undoable workspace edit', async () => {
  const h = harness('import * from "lib.sev"\n');
  await makeImportsExplicit(h.vscode, { target: '.', cwd: '/workspace' }, { appendLine() {} }, async (executable, args) => {
    assert.equal(executable, 'sev');
    assert.deepEqual(args, ['--lint=json', '.']);
    return { stdout: JSON.stringify({ files: [{ path: '/workspace/main.sev', before: 'import * from "lib.sev"\n', after: 'from "lib.sev" import used\n' }], notes: [] }) };
  });
  assert.equal(h.applied[0].edits[0].after, 'from "lib.sev" import used\n');
});
test('rejects stale source before applying any edits', async () => {
  const h = harness('changed buffer');
  await assert.rejects(makeImportsExplicit(h.vscode, { target: '.', cwd: '/workspace' }, { appendLine() {} }, async () => ({ stdout: JSON.stringify({ files: [{ path: '/workspace/main.sev', before: 'old source', after: 'new source' }], notes: [] }) })), /changed during import analysis/);
  assert.equal(h.applied.length, 0);
});
