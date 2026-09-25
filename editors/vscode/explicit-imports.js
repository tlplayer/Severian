'use strict';

const { execFile } = require('child_process');
const { promisify } = require('util');
const execute = promisify(execFile);

async function makeImportsExplicit(vscode, run, output, invoke = execute) {
  if (!await vscode.workspace.saveAll(false)) return false;
  const executable = vscode.workspace.getConfiguration('severian', run.scope).get('executable', 'sev').trim() || 'sev';
  const { stdout } = await invoke(executable, ['--lint=json', run.target],
    { cwd: run.cwd, maxBuffer: 64 * 1024 * 1024 });
  const plan = JSON.parse(stdout);
  const edits = new vscode.WorkspaceEdit();
  for (const file of plan.files) {
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(file.path));
    if (document.getText() !== file.before) throw new Error(`${file.path} changed during import analysis; run the command again.`);
    edits.replace(document.uri, new vscode.Range(document.positionAt(0), document.positionAt(file.before.length)), file.after);
  }
  for (const note of plan.notes) output.appendLine(note);
  if (!plan.files.length) {
    await vscode.window.showInformationMessage('No wildcard imports could be converted. See the Severian output for details.');
    return false;
  }
  const applied = await vscode.workspace.applyEdit(edits);
  if (applied) await vscode.window.showInformationMessage(`Made imports explicit in ${plan.files.length} files. Changes can be undone in the editor.`);
  return applied;
}
module.exports = { makeImportsExplicit };
