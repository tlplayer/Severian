'use strict';

const path = require('path');
const { tokenize } = require('./source-index');

// Compiler spans count Unicode scalars. VS Code and JavaScript count UTF-16.
function position(text, offset) {
  const prefix = Array.from(text).slice(0, offset).join('');
  const lines = prefix.split('\n');
  return { line: lines.length - 1, character: lines.at(-1).length };
}

class SemanticIndex {
  constructor(document) {
    if (document.schema_version !== 1 || !Array.isArray(document.definitions)
        || !Array.isArray(document.references) || !Array.isArray(document.sources)) {
      throw new Error('unsupported Severian editor snapshot');
    }
    this.sources = new Map(document.sources.map(source => [path.resolve(source.path), source.text]));
    this.definitions = new Map(document.definitions.map(definition => [definition.id, definition]));
    this.references = document.references;
    this.diagnostics = document.diagnostics || [];
    this.relationships = document.relationships || [];
    this.analysis = document.analysis || 'compiler';
    this.selections = new Map();
  }

  location(span) {
    const file = path.resolve(span.path);
    const text = this.sources.get(file);
    if (text === undefined) throw new Error(`snapshot omitted source ${file}`);
    return { file, start: position(text, span.start), end: position(text, span.end) };
  }

  at(file, offset) {
    return this.occurrence(file, offset)?.definition;
  }

  occurrence(file, offset) {
    const matches = [];
    for (const item of [...this.references, ...this.definitions.values()]) {
      if (path.resolve(item.source.path) !== path.resolve(file)) continue;
      const definition = item.symbol ? this.definitions.get(item.symbol) : item;
      if (!definition) continue;
      const selection = this.selection(item, definition.name);
      if (selection && selection.start <= offset && offset <= selection.end) matches.push({ definition, selection });
    }
    return matches.sort((a, b) => Number(offset === a.selection.end) - Number(offset === b.selection.end)
      || (a.selection.end - a.selection.start) - (b.selection.end - b.selection.start))[0];
  }

  selection(item, name = item.name) {
    if (item.selection) return item.selection;
    if (this.selections.has(item)) return this.selections.get(item);
    const span = item.source;
    const text = this.sources.get(path.resolve(span.path));
    if (text === undefined) return undefined;
    const selected = Array.from(text).slice(span.start, span.end).join('');
    const token = tokenize(selected).find(token => token.kind === 'identifier' && token.value === name.split('.').at(-1));
    const result = token && { path: span.path, start: span.start + token.start, end: span.start + token.end };
    this.selections.set(item, result);
    return result;
  }

  declaration(definition) {
    return this.location(this.selection(definition) || definition.source);
  }

  identifier(span, name) {
    const file = path.resolve(span.path);
    const text = this.sources.get(file);
    if (text === undefined) throw new Error(`snapshot omitted source ${file}`);
    const scalars = Array.from(text);
    const selected = scalars.slice(span.start, span.end).join('');
    const simple = name.split('.').at(-1);
    const escaped = simple.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const match = new RegExp(`(?<![\\p{L}\\p{N}_])${escaped}(?![\\p{L}\\p{N}_])`, 'u').exec(selected);
    if (!match) return undefined;
    const start = span.start + Array.from(selected.slice(0, match.index)).length;
    return this.location({ path: file, start, end: start + Array.from(simple).length });
  }

  rename(id) {
    const definition = this.definitions.get(id);
    if (!definition) return [];
    const items = [definition, ...this.references.filter(item => item.symbol === id)];
    const edits = items.map(item => this.selection(item, definition.name)).filter(Boolean).map(span => this.location(span));
    return [...new Map(edits.map(edit => [`${edit.file}:${edit.start.line}:${edit.start.character}`, edit])).values()];
  }

  supertypes(id) {
    return this.relationships.filter(edge => edge.from === id && edge.kind === 'implements')
      .map(edge => this.definitions.get(edge.to)).filter(Boolean);
  }

  usages(id, includeDeclaration = false) {
    const definition = this.definitions.get(id);
    if (!definition) return [];
    const spans = this.references.filter(reference => reference.symbol === id)
      .map(reference => this.selection(reference, definition.name)).filter(Boolean);
    if (includeDeclaration) spans.push(this.selection(definition) || definition.source);
    const unique = new Map(spans.map(span => [`${span.path}:${span.start}:${span.end}`, span]));
    return [...unique.values()].map(span => this.location(span));
  }

  callers(id) {
    return [...new Set(this.calls().filter(reference => reference.symbol === id).map(reference => reference.caller))]
      .map(caller => this.definitions.get(caller)).filter(Boolean);
  }

  implementations(id) {
    return this.relationships.filter(edge => edge.to === id && edge.kind === 'implements')
      .map(edge => this.definitions.get(edge.from)).filter(Boolean);
  }

  calls() {
    return this.references.filter(reference => {
      if (reference.kind) return reference.kind === 'call';
      const definition = this.definitions.get(reference.symbol);
      if (definition?.kind !== 'function') return false;
      const selection = this.selection(reference, definition.name);
      if (!selection) return false;
      const suffix = Array.from(this.sources.get(path.resolve(selection.path))).slice(selection.end, reference.source.end).join('');
      return /^\s*(?:\[[^\]]*\]\s*)?\(/u.test(suffix);
    });
  }
}

module.exports = { SemanticIndex, position };
