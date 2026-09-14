'use strict';

const fs = require('fs/promises');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');
const { Worker } = require('worker_threads');
const { sourceSnapshot } = require('./source-index');
const { SemanticIndex } = require('./semantic-data');
const execute = promisify(execFile);

function mergeSnapshots(source, compiled) {
  const compiler = new SemanticIndex(compiled);
  const key = span => `${path.resolve(span.path)}:${span.start}`;
  const declarations = new Map([...compiler.definitions.values()].map(item => [key(compiler.selection(item) || item.source), item]));
  const ids = new Map(source.definitions.map(item => [item.id, declarations.get(key(item.selection))?.id || item.id]));
  const definitions = new Map(source.definitions.map(item => [ids.get(item.id), { ...item, id: ids.get(item.id) }]));
  for (const item of compiled.definitions) definitions.set(item.id, { ...definitions.get(item.id), ...item });
  const references = new Map(source.references.map(item => [key(item.source), {
    ...item, symbol: ids.get(item.symbol) || item.symbol, caller: ids.get(item.caller) || item.caller,
  }]));
  const calls = new Set(compiler.calls());
  for (const item of compiled.references) {
    const definition = compiler.definitions.get(item.symbol);
    const selection = definition && compiler.selection(item, definition.name);
    if (selection) references.set(key(selection), { ...item, selection, kind: calls.has(item) ? 'call' : 'read' });
  }
  const relationships = [...source.relationships.map(edge => ({ ...edge, from: ids.get(edge.from) || edge.from, to: ids.get(edge.to) || edge.to })), ...compiled.relationships || []];
  return { ...compiled, analysis: 'mixed',
    sources: [...new Map([...source.sources, ...compiled.sources].map(item => [path.resolve(item.path), item])).values()],
    definitions: [...definitions.values()], references: [...references.values()],
    relationships: [...new Map(relationships.map(edge => [`${edge.from}:${edge.to}:${edge.kind}`, edge])).values()],
  };
}

function createSemanticWorkspace(vscode, context, options = {}) {
  const run = options.execute || execute;
  const read = options.readFile || (file => fs.readFile(file, 'utf8'));
  const output = vscode.window.createOutputChannel('Severian Navigation');
  const snapshots = new Map();
  let generation = 0;
  let recovery;
  const workers = new Set();
  const invalidate = () => {
    generation++; recovery = undefined; snapshots.clear();
    for (const worker of workers) void worker.terminate();
  };
  const buildSnapshot = options.buildSnapshot || ((sources, manifests) => new Promise((resolve, reject) => {
    // Parsing a large workspace must not block the editor extension host.
    const worker = new Worker(require.resolve('./source-index'), { workerData: { sources, manifests } });
    workers.add(worker);
    worker.once('message', resolve);
    worker.once('error', reject);
    worker.once('exit', () => { workers.delete(worker); resolve(undefined); });
  }));
  const watcher = vscode.workspace.createFileSystemWatcher('**/*.{sev,json}');
  context.subscriptions.push(output, watcher, { dispose: invalidate },
    watcher.onDidCreate(invalidate), watcher.onDidChange(invalidate), watcher.onDidDelete(invalidate),
    vscode.workspace.onDidChangeTextDocument(event => { if (event.document.languageId === 'severian') invalidate(); }),
    vscode.workspace.onDidSaveTextDocument(document => { if (document.languageId === 'severian') invalidate(); }),
    vscode.workspace.onDidCloseTextDocument(document => { if (document.languageId === 'severian') invalidate(); }),
    vscode.workspace.onDidChangeWorkspaceFolders(invalidate),
    vscode.workspace.onDidChangeConfiguration(event => { if (event.affectsConfiguration('severian')) invalidate(); }),
  );

  async function sources(document) {
    if (!recovery) recovery = (async () => {
      const uris = await vscode.workspace.findFiles('**/{*.sev,package.json}', '**/{.git,node_modules,package.pkg,target,.venv}/**');
      const documents = new Map(vscode.workspace.textDocuments.filter(item => item.languageId === 'severian')
        .map(item => [item.uri.fsPath, item.getText()]));
      documents.set(document.uri.fsPath, document.getText());
      const files = [...new Set([...uris.map(uri => uri.fsPath), ...documents.keys()])];
      const loaded = [];
      // Bound open files without truncating the workspace index.
      for (let i = 0; i < files.length; i += 32) {
        const batch = await Promise.all(files.slice(i, i + 32).map(async file => {
          try { return { path: file, text: documents.has(file) ? documents.get(file) : await read(file) }; }
          catch (error) { output.appendLine(`Cannot index ${file}: ${error.message}`); return undefined; }
        }));
        loaded.push(...batch.filter(Boolean));
      }
      const manifests = [];
      for (const item of loaded.filter(item => item.path.endsWith('/package.json'))) {
        try {
          const json = item.text.replace(/"(?:\\.|[^"\\])*"|\/\/[^\n]*|\/\*[\s\S]*?\*\//gu,
            value => value.startsWith('/') ? '' : value);
          manifests.push({ path: item.path, manifest: JSON.parse(json) });
        } catch (error) { output.appendLine(`Cannot read package manifest ${item.path}: ${error.message}`); }
      }
      return buildSnapshot(loaded.filter(item => item.path.endsWith('.sev')), manifests);
    })().catch(error => { recovery = undefined; throw error; });
    return recovery;
  }

  async function snapshot(document) {
    const file = document.uri.fsPath;
    const cached = snapshots.get(file);
    if (cached?.version === document.version) return cached.promise;
    const requestGeneration = generation;
    const version = document.version;
    const promise = (async () => {
      const source = await sources(document);
      if (!source || requestGeneration !== generation) return undefined;
      let result = source;
      const dirty = vscode.workspace.textDocuments.some(item => item.languageId === 'severian' && item.isDirty);
      if (!dirty) {
        const executable = vscode.workspace.getConfiguration('severian', document.uri).get('executable', 'sev');
        try {
          const compiled = await run(executable, ['check', file, '--emit', 'editor'], {
            cwd: path.dirname(file), maxBuffer: 64 * 1024 * 1024, timeout: 15000,
          });
          result = mergeSnapshots(source, JSON.parse(compiled.stdout));
        } catch (error) {
          output.appendLine(`Using source navigation for ${file}. Compiler analysis failed: ${error.stderr || error.message}`);
        }
      }
      if (requestGeneration !== generation || version !== document.version) return undefined;
      return new SemanticIndex(result);
    })().catch(error => {
      if (snapshots.get(file)?.promise === promise) snapshots.delete(file);
      throw error;
    });
    snapshots.set(file, { version, promise });
    return promise;
  }
  return { snapshot, generation: () => generation };
}

module.exports = { createSemanticWorkspace, mergeSnapshots };
