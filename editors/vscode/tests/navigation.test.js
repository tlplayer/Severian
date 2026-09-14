'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { sourceSnapshot } = require('../source-index');
const { SemanticIndex } = require('../semantic-data');
const { registerSemantic } = require('../semantic');
const { mergeSnapshots } = require('../semantic-workspace');

const files = {
  '/workspace/lib.sev': 'trait Drawable:\n    def draw()\n\nclass Point: Drawable\n    def draw():\n        pass\n\ndef doubled(value: int) -> int:\n    return value * 2\n',
  '/workspace/main.sev': 'import doubled, Point from "./lib.sev"\n\ndef main():\n    value = doubled(3)\n    point = Point()\n    point.draw()\n    callback = doubled\n    # doubled(value)\n    print("😀 doubled(value)")\n\ndef other(value: int):\n    return value\n',
  '/workspace/unrelated.sev': 'def doubled():\n    pass\n',
};

function indexOf(contents = files, manifests) {
  return new SemanticIndex(sourceSnapshot(Object.entries(contents).map(([path, text]) => ({ path, text })), manifests));
}
const at = (index, file, text, needle, occurrence = 0) => {
  let offset = -1;
  for (let i = 0; i <= occurrence; i++) offset = text.indexOf(needle, offset + 1);
  assert.notEqual(offset, -1, needle);
  return index.at(file, Array.from(text.slice(0, offset)).length);
};

test('recovery resolves imports and receiver methods without guessing global names', () => {
  const index = indexOf();
  const main = files['/workspace/main.sev'];
  const doubled = at(index, '/workspace/main.sev', main, 'doubled(3)');
  assert.equal(doubled.source.path, '/workspace/lib.sev');
  assert.equal(index.usages(doubled.id).length, 3); // import, call, function value
  assert.equal(index.calls().filter(item => item.symbol === doubled.id).length, 1);
  assert.deepEqual(index.callers(doubled.id).map(item => item.name), ['main']);
  const draw = at(index, '/workspace/main.sev', main, 'draw()');
  assert.equal(index.declaration(draw).start.line, 4);
  const unrelated = at(index, '/workspace/unrelated.sev', files['/workspace/unrelated.sev'], 'doubled');
  assert.equal(index.usages(unrelated.id).length, 0);
  assert.equal(at(index, '/workspace/main.sev', main, 'doubled(value)'), undefined);
  assert.equal(at(index, '/workspace/main.sev', main, 'doubled(value)', 1), undefined);
});

test('recovery tracks trait implementations, type hierarchy, extensions and annotations', () => {
  const file = '/workspace/types.sev';
  const text = 'trait Drawable:\n    def draw()\nclass Point: Drawable\n    def draw():\n        pass\nextend Point:\n    def reset():\n        pass\ndef render(value: Point):\n    value.draw()\n    value.reset()\n';
  const index = indexOf({ [file]: text });
  const trait = at(index, file, text, 'Drawable');
  const point = at(index, file, text, 'Point');
  const requirement = at(index, file, text, 'draw');
  assert.equal(requirement.abstract, true);
  assert.deepEqual(index.implementations(trait.id).map(item => item.name), ['Point']);
  assert.deepEqual(index.supertypes(point.id).map(item => item.name), ['Drawable']);
  assert.deepEqual(index.implementations(requirement.id).map(item => index.declaration(item).start.line), [3]);
  assert.equal(at(index, file, text, 'draw()', 2).id, index.implementations(requirement.id)[0].id);
  assert.equal(at(index, file, text, 'reset()', 1).id, at(index, file, text, 'reset()').id);
});

test('rename uses exact Unicode identifiers, respects shadowed parameters and ignores literals', () => {
  const file = '/workspace/unicode.sev';
  const text = '# 😀\ndef first(値: int):\n    return 値\ndef second(値: int):\n    print("😀 値", 値)\n';
  const index = indexOf({ [file]: text });
  const parameter = at(index, file, text, '値: int', 1);
  assert.deepEqual(index.rename(parameter.id).map(item => [item.start.line, item.start.character, item.end.character]), [
    [3, 11, 12], [4, 18, 19],
  ]);
  assert.equal(index.at(file, Array.from(text.slice(0, text.indexOf('return'))).length), undefined);
});

test('package dependencies and re-exports resolve through their declared library entry', () => {
  const index = indexOf({
    '/workspace/app/src/main.sev': 'import math\ndef main():\n    math.double(3)\n',
    '/workspace/math/lib.sev': 'import * from "./double.sev"\n',
    '/workspace/math/double.sev': 'def double(value: int):\n    return value\n',
  }, [
    { path: '/workspace/app/package.json', manifest: { dependencies: { math: { path: '../math' } } } },
    { path: '/workspace/math/package.json', manifest: { lib: { path: 'lib.sev' } } },
  ]);
  const double = [...index.definitions.values()].find(item => item.name === 'double');
  assert.equal(index.callers(double.id)[0].name, 'main');
});

test('compiler identities override recovery data and keep references from importing files', () => {
  const recovery = sourceSnapshot(Object.entries(files).map(([path, text]) => ({ path, text })));
  const doubled = recovery.definitions.find(item => item.name === 'doubled' && item.source.path === '/workspace/lib.sev');
  const compiled = { schema_version: 1, sources: recovery.sources.filter(item => item.path === '/workspace/lib.sev'),
    definitions: [{ ...doubled, id: 'resolved:123', type: 'doubled(value: int) -> int' }], references: [], relationships: [] };
  const merged = new SemanticIndex(mergeSnapshots(recovery, compiled));
  assert.equal(merged.callers('resolved:123')[0].name, 'main');
  assert.equal(merged.usages('resolved:123').length, 3);
  assert.equal([...merged.definitions.values()].filter(item => item.source.path === '/workspace/lib.sev' && item.name === 'doubled').length, 1);
});

function harness(contents = files, execute = async () => { throw new Error('compiler unavailable'); }) {
  const providers = {};
  const events = {};
  const messages = [];
  const disposable = { dispose() {} };
  const listen = name => callback => { (events[name] ||= []).push(callback); return disposable; };
  const emit = (name, event) => { for (const callback of events[name] || []) callback(event); };
  class Position { constructor(line, character) { Object.assign(this, { line, character }); } }
  class Range {
    constructor(line, character, endLine, endCharacter) {
      this.start = new Position(line, character); this.end = new Position(endLine, endCharacter);
    }
  }
  class HierarchyItem {
    constructor(kind, name, detail, uri, range, selectionRange) { Object.assign(this, { kind, name, detail, uri, range, selectionRange }); }
  }
  const documents = new Map(Object.entries(contents).map(([file, text]) => [file, {
    uri: { fsPath: file, scheme: 'file' }, languageId: 'severian', version: 1, isDirty: false, text,
    getText() { return this.text; },
    lineAt(line) { return { text: this.text.split('\n')[line] }; },
    offsetAt(position) { return this.text.split('\n').slice(0, position.line).reduce((length, line) => length + line.length + 1, 0) + position.character; },
  }]));
  let executions = 0;
  const vscode = {
    Position, Range, Uri: { file: fsPath => ({ fsPath, scheme: 'file' }) },
    Location: class { constructor(uri, range) { Object.assign(this, { uri, range }); } },
    CallHierarchyItem: HierarchyItem, TypeHierarchyItem: HierarchyItem,
    CallHierarchyIncomingCall: class { constructor(from, fromRanges) { Object.assign(this, { from, fromRanges }); } },
    CallHierarchyOutgoingCall: class { constructor(to, fromRanges) { Object.assign(this, { to, fromRanges }); } },
    WorkspaceEdit: class { constructor() { this.edits = []; } replace(uri, range, newText) { this.edits.push({ uri, range, newText }); } },
    DocumentHighlight: class { constructor(range, kind) { Object.assign(this, { range, kind }); } },
    CodeAction: class { constructor(title, kind) { Object.assign(this, { title, kind }); } },
    CodeActionKind: { Refactor: { value: 'refactor' } },
    DocumentHighlightKind: { Text: 0 }, SymbolKind: { Function: 11, Class: 4 },
    SemanticTokensLegend: class {},
    window: { createOutputChannel: () => ({ ...disposable, appendLine: message => messages.push(message) }) },
    workspace: {
      textDocuments: [...documents.values()], findFiles: async () => [...documents.values()].map(item => item.uri),
      createFileSystemWatcher: () => ({ ...disposable, onDidCreate: listen('create'), onDidChange: listen('disk'), onDidDelete: listen('delete') }),
      getConfiguration: () => ({ get: (key, fallback) => fallback }),
      onDidChangeTextDocument: listen('change'), onDidSaveTextDocument: listen('save'),
      onDidCloseTextDocument: listen('close'), onDidChangeWorkspaceFolders: listen('folders'), onDidChangeConfiguration: listen('configuration'),
      openTextDocument: async uri => { throw new Error(`Hierarchy must retain its index instead of reopening ${uri.fsPath}`); },
    },
    languages: { createDiagnosticCollection: () => ({ ...disposable, clear() {}, set() {} }) },
  };
  for (const name of ['Definition', 'Implementation', 'CallHierarchy', 'Rename', 'TypeHierarchy', 'Reference', 'DocumentHighlight', 'CodeActions', 'Hover', 'InlayHints', 'DocumentSemanticTokens']) {
    vscode.languages[`register${name}Provider`] = (selector, provider) => { providers[name] = provider; return disposable; };
  }
  const context = { subscriptions: [] };
  registerSemantic(vscode, context, { buildSnapshot: sourceSnapshot,
    execute: async (...args) => { executions++; return execute(...args); } });
  const select = (file, needle) => {
    const document = documents.get(file);
    const offset = document.text.indexOf(needle);
    assert.notEqual(offset, -1);
    const prefix = document.text.slice(0, offset).split('\n');
    return [document, new Position(prefix.length - 1, prefix.at(-1).length)];
  };
  return { providers, documents, select, emit, messages, executions: () => executions };
}

test('editor commands recover from compiler failure and share a cached workspace snapshot', async () => {
  const h = harness();
  const selection = h.select('/workspace/main.sev', 'doubled(3)');
  const definition = await h.providers.Definition.provideDefinition(...selection);
  assert.equal(definition.uri.fsPath, '/workspace/lib.sev');
  assert.deepEqual([definition.range.start.line, definition.range.start.character, definition.range.end.character], [7, 4, 11]);
  assert.equal((await h.providers.Implementation.provideImplementation(...selection)).length, 1);
  assert.equal((await h.providers.Reference.provideReferences(...selection, { includeDeclaration: true })).length, 4);
  const highlights = await h.providers.DocumentHighlight.provideDocumentHighlights(...selection);
  assert.deepEqual(highlights.map(item => item.range.start.line), [0, 3, 6]);
  const prepared = await h.providers.Rename.prepareRename(...selection);
  assert.equal(prepared.placeholder, 'doubled');
  assert.equal(prepared.range.start.line, 3);
  const edit = await h.providers.Rename.provideRenameEdits(...selection, 'twice');
  assert.equal(edit.edits.length, 4);
  await assert.rejects(h.providers.Rename.provideRenameEdits(...selection, 'return'), /keyword/);
  assert.equal(h.executions(), 1);
  assert.match(h.messages[0], /Compiler analysis failed/);
  const actions = await h.providers.CodeActions.provideCodeActions(selection[0], { start: selection[1] }, {});
  assert.equal(actions[0].command.command, 'editor.action.rename');
});

test('incoming and outgoing hierarchy keep cross-file callers and exclude function values', async () => {
  const h = harness();
  const doubled = await h.providers.CallHierarchy.prepareCallHierarchy(...h.select('/workspace/main.sev', 'doubled(3)'));
  assert.equal(doubled.range.start.character, 0);
  assert.equal(doubled.selectionRange.start.character, 4);
  const incoming = await h.providers.CallHierarchy.provideCallHierarchyIncomingCalls(doubled);
  assert.deepEqual(incoming.map(item => item.from.name), ['main']);
  assert.equal(incoming[0].fromRanges.length, 1);
  assert.equal(incoming[0].fromRanges[0].start.line, 3);
  const outgoing = await h.providers.CallHierarchy.provideCallHierarchyOutgoingCalls(incoming[0].from);
  assert.deepEqual(outgoing.map(item => item.to.name).sort(), ['doubled', 'draw']);
  assert.equal(h.executions(), 1);
  h.emit('disk');
  assert.deepEqual(await h.providers.CallHierarchy.provideCallHierarchyIncomingCalls(doubled), []);
});

test('type hierarchy and implementation commands navigate trait requirements', async () => {
  const h = harness();
  const trait = await h.providers.TypeHierarchy.prepareTypeHierarchy(...h.select('/workspace/lib.sev', 'Drawable'));
  const subtypes = await h.providers.TypeHierarchy.provideTypeHierarchySubtypes(trait);
  assert.deepEqual(subtypes.map(item => item.name), ['Point']);
  assert.deepEqual((await h.providers.TypeHierarchy.provideTypeHierarchySupertypes(subtypes[0])).map(item => item.name), ['Drawable']);
  const implementations = await h.providers.Implementation.provideImplementation(...h.select('/workspace/lib.sev', 'draw()'));
  assert.equal(implementations[0].range.start.line, 4);
});

test('unsaved text in another document invalidates navigation and skips stale compiler output', async () => {
  const h = harness();
  const selection = h.select('/workspace/main.sev', 'doubled(3)');
  await h.providers.Definition.provideDefinition(...selection);
  const library = h.documents.get('/workspace/lib.sev');
  library.text = '\n' + library.text;
  library.version++;
  library.isDirty = true;
  h.emit('change', { document: library });
  const definition = await h.providers.Definition.provideDefinition(...selection);
  assert.equal(definition.range.start.line, 8);
  assert.equal(h.executions(), 1);
});

test('in-flight compiler results are discarded after an edit', async () => {
  let release;
  let started;
  const ready = new Promise(resolve => { started = resolve; });
  const h = harness(files, () => { started(); return new Promise(resolve => { release = resolve; }); });
  const selection = h.select('/workspace/main.sev', 'doubled(3)');
  const result = h.providers.Definition.provideDefinition(...selection);
  await ready;
  selection[0].version++;
  selection[0].isDirty = true;
  h.emit('change', { document: selection[0] });
  release({ stdout: JSON.stringify(sourceSnapshot(Object.entries(files).map(([path, text]) => ({ path, text })))) });
  assert.equal(await result, undefined);
  assert.equal((await h.providers.Definition.provideDefinition(...selection)).uri.fsPath, '/workspace/lib.sev');
});
