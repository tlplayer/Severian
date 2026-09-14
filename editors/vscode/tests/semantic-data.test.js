'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { SemanticIndex, position } = require('../semantic-data');

test('resolved identity distinguishes shadowed bindings and handles Unicode spans', () => {
  const text = '# 😀\ndef doubled(value):\n    return value * 2\ndoubled(3)\n';
  const characters = Array.from(text);
  const start = characters.join('').indexOf('def') - 1;
  const use = characters.join('').lastIndexOf('doubled') - 1;
  const source = { path: '/tmp/example.sev', start, end: use - 1 };
  const index = new SemanticIndex({ schema_version: 1,
    sources: [{ path: source.path, text }], diagnostics: [],
    definitions: [{ id: 'fn:1', name: 'doubled', kind: 'function', source },
      { id: 'fn:2', name: 'doubled', kind: 'function', source: { ...source, path: '/tmp/other.sev' } }],
    references: [{ symbol: 'fn:1', caller: 'fn:2', source: { path: source.path, start: use, end: use + 10 } }],
    relationships: [{ from: 'fn:1', to: 'fn:2', kind: 'implements' }],
  });
  assert.deepEqual(position(text, 3), { line: 0, character: 4 });
  assert.equal(index.at(source.path, use + 2).id, 'fn:1');
  assert.equal(index.usages('fn:1').length, 1);
  assert.equal(index.usages('fn:2').length, 0);
  assert.deepEqual(index.callers('fn:1').map(item => item.id), ['fn:2']);
  assert.deepEqual(index.implementations('fn:2').map(item => item.id), ['fn:1']);
  assert.deepEqual(index.implementations('fn:1'), []);
  assert.deepEqual(index.supertypes('fn:1').map(item => item.id), ['fn:2']);
  assert.deepEqual(index.rename('fn:1'), [
    { file: source.path, start: { line: 1, character: 4 }, end: { line: 1, character: 11 } },
    { file: source.path, start: { line: 3, character: 0 }, end: { line: 3, character: 7 } },
  ]);
  assert.equal(index.location(source).start.line, 1);
});

test('malformed editor snapshots fail instead of silently presenting empty results', () => {
  assert.throws(() => new SemanticIndex({ schema_version: 2 }), /unsupported/);
});
