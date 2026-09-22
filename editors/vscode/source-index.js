'use strict';

const path = require('path');

const keywords = new Set(('def class trait enum extend import from as return break continue if else elif while for in '
  + 'match case test with true false None and or not is pass operator native async await borrow move view clone drop').split(' '));

// This recovery index deliberately resolves lexical scopes and explicit imports only.
// It never guesses a workspace symbol by name. All offsets are Unicode scalars.
function tokenize(text) {
  const chars = Array.from(text);
  const tokens = [];
  let line = 0;
  let indent = 0;
  let atStart = true;
  for (let i = 0; i < chars.length;) {
    const start = i;
    const ch = chars[i];
    if (ch === '\n') { line++; indent = 0; atStart = true; i++; continue; }
    if (/\s/u.test(ch)) { if (atStart) indent += ch === '\t' ? 4 : 1; i++; continue; }
    atStart = false;
    if (ch === '#') { while (i < chars.length && chars[i] !== '\n') i++; continue; }
    const tokenLine = line;
    let kind = 'punctuation';
    if (ch === '"' || ch === "'") {
      kind = 'string';
      const width = chars[i + 1] === ch && chars[i + 2] === ch ? 3 : 1;
      i += width;
      while (i < chars.length) {
        if (chars[i] === '\\') { if (chars[i + 1] === '\n') line++; i += 2; continue; }
        if (chars.slice(i, i + width).join('') === ch.repeat(width)) { i += width; break; }
        if (chars[i] === '\n') line++;
        i++;
      }
    } else if (/[\p{L}_]/u.test(ch)) {
      kind = 'identifier';
      while (i < chars.length && /[\p{L}\p{N}_]/u.test(chars[i])) i++;
    } else {
      i++;
      if ([':=', '?=', '->', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/='].includes(ch + chars[i])) i++;
    }
    tokens.push({ value: chars.slice(start, i).join(''), kind, start, end: i, line: tokenLine, indent });
  }
  return tokens;
}

function sourceSnapshot(sources, manifests = []) {
  const definitions = [];
  const references = [];
  const relationships = [];
  const modules = new Map();
  const packages = new Map(manifests.map(item => [path.dirname(path.resolve(item.path)), item.manifest]));
  const span = (file, token) => ({ path: file, start: token.start, end: token.end });

  for (const source of sources) {
    const file = path.resolve(source.path);
    const tokens = tokenize(source.text);
    const root = { names: new Map(), parent: undefined, kind: 'module', file };
    const scopes = [{ scope: root, indent: -1, definition: undefined }];
    const declarations = new Set();
    const ignored = new Set();
    const imports = [];
    const owners = [];
    const module = { file, root, tokens, declarations, ignored, imports, owners };
    modules.set(file, module);
    const add = (token, kind, scope, extra = {}) => {
      const definition = { id: `${file}:${token.start}`, name: token.value, kind,
        source: span(file, token), selection: span(file, token), type: token.value,
        documentation: '', scope, ...extra };
      definitions.push(definition);
      const named = scope.names.get(token.value) || [];
      named.push(definition);
      scope.names.set(token.value, named);
      declarations.add(token);
      return definition;
    };
    let bracketDepth = 0;
    const lines = [];
    for (const token of tokens) {
      if (!lines.length || (token.line !== lines.at(-1).at(-1).line && bracketDepth === 0)) lines.push([]);
      lines.at(-1).push(token);
      if ('([{'.includes(token.value)) bracketDepth++;
      if (')]}'.includes(token.value)) bracketDepth = Math.max(0, bracketDepth - 1);
    }
    for (const line of lines) {
      const first = line[0];
      while (scopes.length > 1 && first.indent <= scopes.at(-1).indent) {
        const closed = scopes.pop();
        if (closed.definition) closed.definition.source.end = first.start;
      }
      const scope = scopes.at(-1).scope;
      for (const token of line) token.scope = scope;
      if (first.value === 'import' || first.value === 'from') {
        // Normalize source-first lists to the same representation as import N from X.
        const at = line.findIndex(token => token.value === 'import');
        imports.push(first.value === 'from'
          ? [line[at], ...line.slice(at + 1).filter(token => !['{', '}'].includes(token.value)), line[0], line[1]]
          : line);
        for (const token of line) ignored.add(token);
        continue;
      }
      const declaration = ['def', 'class', 'trait', 'enum', 'extend'].includes(first.value) ? first.value : undefined;
      if (declaration && line[1]?.kind === 'identifier') {
        const name = line[1];
        const kind = declaration === 'def' ? 'function' : 'class';
        const child = { names: new Map(), parent: scope, kind, file };
        const definition = declaration === 'extend' ? undefined : add(name, kind, scope,
          { flavor: declaration, owner: scope.owner, abstract: declaration === 'trait'
            || (declaration === 'def' && scope.owner?.flavor === 'trait' && line.at(-1).value !== ':') });
        if (definition) {
          definition.type = Array.from(source.text).slice(first.start, line.at(-1).end).join('');
          definition.source = { path: file, start: first.start, end: line.at(-1).end };
          definition.members = child.names;
          child.definition = definition;
        }
        child.owner = kind === 'class' ? definition : scope.owner;
        if (kind === 'class') {
          const colon = line.findIndex(token => token.value === ':');
          if (definition) definition.bases = colon >= 0 ? line.slice(colon + 1).filter(token => token.kind === 'identifier') : [];
          owners.push({ child, name, extension: declaration === 'extend' });
        } else {
          child.caller = definition.id;
          const open = line.findIndex(token => token.value === '(');
          let depth = 0;
          for (let i = open + 1; open >= 0 && i < line.length; i++) {
            const token = line[i];
            if (token.value === ')' && depth === 0) break;
            if (depth === 0 && token.kind === 'identifier' && (i === open + 1 || line[i - 1].value === ',')) {
              add(token, 'parameter', child, { annotation: line[i + 1]?.value === ':' ? line[i + 2]?.value : undefined });
            }
            if ('([{'.includes(token.value)) depth++;
            if (')]}'.includes(token.value)) depth--;
          }
        }
        for (const token of line) token.scope = child;
        // A bodyless trait requirement must not own following declarations.
        scopes.push({ scope: child, indent: first.indent, definition });
        continue;
      }
      if (first.kind === 'identifier' && !keywords.has(first.value)
          && [':', '=', ':=', '?='].includes(line[1]?.value)) {
        const existing = scope.names.get(first.value);
        if (!existing) {
          const assignment = line.findIndex(token => ['=', ':=', '?='].includes(token.value));
          add(first, 'variable', scope, {
            annotation: line[1].value === ':' ? line[2]?.value : undefined,
            initializer: assignment >= 0 && line[assignment + 2]?.value === '(' ? line[assignment + 1].value : undefined,
            owner: scope.kind === 'class' ? scope.owner : undefined,
          });
        }
      }
      if (first.value === 'test') {
        for (const token of line) ignored.add(token);
        scopes.push({ scope: { names: new Map(), parent: scope, kind: 'test', file }, indent: first.indent });
      }
    }
    for (const entry of scopes) if (entry.definition) entry.definition.source.end = Array.from(source.text).length;
  }

  const unique = values => values?.length === 1 ? values[0] : undefined;
  function importTarget(module, target) {
    const raw = target.kind === 'string' ? target.value.slice(1, -1) : target.value;
    let base = path.resolve(path.dirname(module.file), raw);
    if (target.kind !== 'string') {
      for (let dir = path.dirname(module.file);;) {
        const manifest = packages.get(dir);
        if (manifest) {
          const dependency = { ...manifest.dependencies, ...manifest['dev-dependencies'] }[raw];
          if (dependency?.path) {
            const root = path.resolve(dir, dependency.path);
            base = path.join(root, packages.get(root)?.lib?.path || 'src/lib.sev');
          }
          break;
        }
        const parent = path.dirname(dir);
        if (parent === dir) break;
        dir = parent;
      }
    }
    return modules.get(base) || modules.get(base + '.sev') || modules.get(path.join(base, 'src/lib.sev'));
  }
  function exported(module, name, seen) {
    return module.root.names.has(name) ? unique(module.root.names.get(name)) : imported(module, name, seen);
  }
  const imported = (module, name, seen = new Set()) => {
    if (!module || seen.has(module.file)) return undefined;
    seen = new Set(seen).add(module.file);
    const matches = [];
    for (const line of module.imports) {
      const from = line.findIndex(token => token.value === 'from');
      const target = from >= 0 ? line[from + 1] : line[1];
      if (!target) continue;
      const other = importTarget(module, target);
      if (!other) continue;
      if (from < 0 || line[1]?.value === '*') {
        const as = line.findIndex(token => token.value === 'as');
        const alias = as >= 0 ? line[as + 1]?.value : undefined;
        if ((alias || (from < 0 ? target.value : undefined)) === name) matches.push({ module: other });
        else if (!alias && from >= 0) {
          const definition = exported(other, name, seen);
          if (definition) matches.push(definition);
        }
      } else {
        for (let i = 1; i < from; i++) {
          const token = line[i];
          if (token.kind !== 'identifier') continue;
          const alias = line[i + 1]?.value === 'as' ? line[i + 2]?.value : undefined;
          if ((alias || token.value) === name) {
            const definition = exported(other, token.value, seen);
            if (definition) matches.push(definition);
          }
          if (alias) i += 2;
        }
      }
    }
    return unique(matches);
  };
  function resolve(scope, name) {
    if (name === 'self' || name === 'Self') {
      for (let current = scope; current; current = current.parent) if (current.owner) return current.owner;
    }
    for (let current = scope; current; current = current.parent) {
      if (current.names.has(name)) return unique(current.names.get(name));
    }
    return imported(modules.get(scope.file), name);
  }
  for (const module of modules.values()) {
    for (const owner of module.owners.filter(item => item.extension)) {
      const target = resolve(owner.child.parent, owner.name.value);
      if (target?.kind !== 'class') continue;
      owner.child.owner = target;
      for (const [name, members] of owner.child.names) {
        for (const member of members) member.owner = target;
        target.members.set(name, [...(target.members.get(name) || []), ...members]);
      }
    }
  }
  for (const definition of definitions.filter(item => item.kind === 'class')) {
    for (const base of definition.bases || []) {
      const target = resolve(definition.scope, base.value);
      if (target?.kind === 'class') relationships.push({ from: definition.id, to: target.id, kind: 'implements' });
    }
  }
  const byId = new Map(definitions.map(item => [item.id, item]));
  function member(owner, name, seen = new Set()) {
    if (!owner || seen.has(owner.id)) return undefined;
    seen.add(owner.id);
    if (owner.members?.has(name)) return unique(owner.members.get(name));
    const inherited = relationships.filter(edge => edge.from === owner.id)
      .map(edge => member(byId.get(edge.to), name, new Set(seen))).filter(Boolean);
    return unique([...new Set(inherited)]);
  }
  for (const edge of [...relationships]) {
    const owner = byId.get(edge.from);
    const contract = byId.get(edge.to);
    for (const [name, requirements] of contract.members) {
      const implementation = member(owner, name);
      if (implementation && implementation.kind === 'function') {
        for (const requirement of requirements) if (implementation.id !== requirement.id) {
          relationships.push({ from: implementation.id, to: requirement.id, kind: 'implements' });
        }
      }
    }
  }
  for (const module of modules.values()) {
    for (const line of module.imports) {
      const from = line.findIndex(token => token.value === 'from');
      if (from < 0) continue;
      const other = line[from + 1] && importTarget(module, line[from + 1]);
      if (!other) continue;
      for (let i = 1; i < from; i++) {
        const token = line[i];
        if (token.kind !== 'identifier') continue;
        const definition = exported(other, token.value, new Set());
        if (definition?.id) {
          references.push({ symbol: definition.id, caller: '', source: span(module.file, token), kind: 'read' });
          if (line[i + 1]?.value === 'as' && line[i + 2]) {
            references.push({ symbol: definition.id, caller: '', source: span(module.file, line[i + 2]), kind: 'read' });
          }
        }
        if (line[i + 1]?.value === 'as') i += 2;
      }
    }
    const resolved = new Map();
    for (let i = 0; i < module.tokens.length; i++) {
      const token = module.tokens[i];
      if (token.kind !== 'identifier' || keywords.has(token.value) || module.ignored.has(token)) continue;
      let definition;
      if (module.tokens[i - 1]?.value === '.') {
        let owner = resolved.get(module.tokens[i - 2]);
        if (owner?.module) definition = exported(owner.module, token.value, new Set());
        else {
          if (owner && owner.kind !== 'class') owner = resolve(owner.scope, owner.annotation || owner.initializer);
          definition = member(owner, token.value);
        }
      } else definition = resolve(token.scope, token.value);
      resolved.set(token, definition);
      if (!definition?.id || module.declarations.has(token)) continue;
      let caller = '';
      for (let scope = token.scope; scope; scope = scope.parent) if (scope.caller) { caller = scope.caller; break; }
      references.push({ symbol: definition.id, caller, source: span(module.file, token),
        kind: module.tokens[i + 1]?.value === '(' && definition.kind === 'function' ? 'call' : 'read' });
    }
  }
  return { schema_version: 1, sources, references, relationships, analysis: 'source',
    definitions: definitions.map(({ scope, members, bases, owner, ...definition }) => definition) };
}

module.exports = { sourceSnapshot, tokenize, keywords };

const { isMainThread, parentPort, workerData } = require('worker_threads');
if (!isMainThread && workerData?.sources) {
  parentPort.postMessage(sourceSnapshot(workerData.sources, workerData.manifests));
}
