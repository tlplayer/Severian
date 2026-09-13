'use strict';

const path = require('path');

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
  }

  location(span) {
    const file = path.resolve(span.path);
    const text = this.sources.get(file);
    if (text === undefined) throw new Error(`snapshot omitted source ${file}`);
    return { file, start: position(text, span.start), end: position(text, span.end) };
  }

  at(file, offset) {
    const matches = this.references.filter(reference => path.resolve(reference.source.path) === path.resolve(file)
      && reference.source.start <= offset && offset < reference.source.end);
    matches.sort((a, b) => (a.source.end - a.source.start) - (b.source.end - b.source.start));
    if (matches.length) return this.definitions.get(matches[0].symbol);
    return [...this.definitions.values()].filter(definition => path.resolve(definition.source.path) === path.resolve(file)
      && definition.source.start <= offset && offset < definition.source.end)
      .sort((a, b) => (a.source.end - a.source.start) - (b.source.end - b.source.start))[0];
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
    const spans = [definition.source, ...this.references.filter(item => item.symbol === id).map(item => item.source)];
    const edits = spans.map(span => this.identifier(span, definition.name)).filter(Boolean);
    return [...new Map(edits.map(edit => [`${edit.file}:${edit.start.line}:${edit.start.character}`, edit])).values()];
  }

  supertypes(id) {
    return this.relationships.filter(edge => edge.from === id && edge.kind === 'implements')
      .map(edge => this.definitions.get(edge.to)).filter(Boolean);
  }

  usages(id, includeDeclaration = false) {
    const spans = this.references.filter(reference => reference.symbol === id).map(reference => reference.source);
    if (includeDeclaration && this.definitions.has(id)) spans.push(this.definitions.get(id).source);
    const unique = new Map(spans.map(span => [`${span.path}:${span.start}:${span.end}`, span]));
    return [...unique.values()].map(span => this.location(span));
  }

  callers(id) {
    return [...new Set(this.references.filter(reference => reference.symbol === id).map(reference => reference.caller))]
      .map(caller => this.definitions.get(caller)).filter(Boolean);
  }

  implementations(id) {
    return this.relationships.filter(edge => edge.to === id && edge.kind === 'implements')
      .map(edge => this.definitions.get(edge.from)).filter(Boolean);
  }
}

module.exports = { SemanticIndex, position };
