'use strict';

const { execFile } = require('child_process');
const { promisify } = require('util');
const path = require('path');
const { SemanticIndex } = require('./semantic-data');
const execute = promisify(execFile);

function registerSemantic(vscode, context) {
  const snapshots = new Map();
  let generation = 0;
  const diagnostics = vscode.languages.createDiagnosticCollection('severian-quality');
  const selector = { language: 'severian', scheme: 'file' };
  const range = location => new vscode.Range(location.start.line, location.start.character, location.end.line, location.end.character);
  const location = value => new vscode.Location(vscode.Uri.file(value.file), range(value));

  async function snapshot(document) {
    // Never serve analysis of saved text for unsaved edits.
    if (document.isDirty) return undefined;
    const cached = snapshots.get(document.uri.fsPath);
    if (cached && cached.version === document.version) return cached.promise;
    const executable = vscode.workspace.getConfiguration('severian', document.uri).get('executable', 'sev');
    const version = document.version;
    const requestGeneration = generation;
    const promise = execute(executable, ['check', document.uri.fsPath, '--emit', 'editor'], {
      cwd: path.dirname(document.uri.fsPath), maxBuffer: 64 * 1024 * 1024, timeout: 120000,
    }).then(result => {
      if (document.version !== version || document.isDirty || generation !== requestGeneration) return undefined;
      return new SemanticIndex(JSON.parse(result.stdout));
    }).catch(error => {
      snapshots.delete(document.uri.fsPath);
      throw error;
    });
    snapshots.set(document.uri.fsPath, { version, promise });
    return promise;
  }

  async function selected(document, position) {
    const index = await snapshot(document);
    const offset = Array.from(document.getText().slice(0, document.offsetAt(position))).length;
    return { index, definition: index && index.at(document.uri.fsPath, offset) };
  }

  const definitionProvider = { async provideDefinition(document, position) {
    const { index, definition } = await selected(document, position);
    return definition && location(index.location(definition.source));
  } };
  const tokenTypes = ['function', 'parameter', 'variable', 'class'];
  const tokenModifiers = ['deprecated'];
  function hierarchyItem(index, definition) {
    const located = index.location(definition.source);
    return new vscode.CallHierarchyItem(vscode.SymbolKind.Function, definition.name, definition.type,
      vscode.Uri.file(located.file), range(located), range(located));
  }
  context.subscriptions.push(diagnostics,
    vscode.languages.registerDefinitionProvider(selector, definitionProvider),
    vscode.languages.registerImplementationProvider(selector, { async provideImplementation(document, position) {
      const { index, definition } = await selected(document, position);
      return definition ? index.implementations(definition.id).map(item => location(index.location(item.source))) : [];
    } }),
    vscode.languages.registerCallHierarchyProvider(selector, {
      async prepareCallHierarchy(document, position) {
        const { index, definition } = await selected(document, position);
        return definition?.kind === 'function' ? hierarchyItem(index, definition) : undefined;
      },
      async provideCallHierarchyIncomingCalls(item) {
        const document = await vscode.workspace.openTextDocument(item.uri);
        const { index, definition } = await selected(document, item.range.start);
        if (!definition) return [];
        return index.callers(definition.id).map(caller => new vscode.CallHierarchyIncomingCall(hierarchyItem(index, caller),
          index.references.filter(reference => reference.symbol === definition.id && reference.caller === caller.id)
            .map(reference => range(index.location(reference.source)))));
      },
      async provideCallHierarchyOutgoingCalls(item) {
        const document = await vscode.workspace.openTextDocument(item.uri);
        const { index, definition } = await selected(document, item.range.start);
        if (!definition) return [];
        const ids = [...new Set(index.references.filter(reference => reference.caller === definition.id).map(reference => reference.symbol))];
        return ids.map(id => index.definitions.get(id)).filter(target => target?.kind === 'function')
          .map(target => new vscode.CallHierarchyOutgoingCall(hierarchyItem(index, target),
            index.references.filter(reference => reference.caller === definition.id && reference.symbol === target.id)
              .map(reference => range(index.location(reference.source)))));
      },
    }),
    vscode.languages.registerRenameProvider(selector, { async provideRenameEdits(document, position, newName) {
      if (!/^[\p{L}_][\p{L}\p{N}_]*$/u.test(newName)) throw new Error('Use a Severian identifier.');
      const { index, definition } = await selected(document, position);
      if (!definition) return undefined;
      const edits = new vscode.WorkspaceEdit();
      for (const item of index.rename(definition.id)) edits.replace(vscode.Uri.file(item.file), range(item), newName);
      return edits;
    } }),
    vscode.languages.registerTypeHierarchyProvider(selector, {
      async prepareTypeHierarchy(document, position) {
        const { index, definition } = await selected(document, position);
        if (definition?.kind !== 'class') return undefined;
        const located = index.location(definition.source);
        return new vscode.TypeHierarchyItem(vscode.SymbolKind.Class, definition.name, definition.type,
          vscode.Uri.file(located.file), range(located), range(located));
      },
      async provideTypeHierarchySupertypes(item) { return typeRelations(item, false); },
      async provideTypeHierarchySubtypes(item) { return typeRelations(item, true); },
    }),
    vscode.languages.registerReferenceProvider(selector, { async provideReferences(document, position, options) {
      const { index, definition } = await selected(document, position);
      return definition ? index.usages(definition.id, options.includeDeclaration).map(location) : [];
    } }),
    vscode.languages.registerHoverProvider(selector, { async provideHover(document, position) {
      const { definition } = await selected(document, position);
      if (!definition) return undefined;
      const text = new vscode.MarkdownString();
      text.appendCodeblock(definition.type, 'severian');
      text.appendText('\n' + definition.documentation);
      return new vscode.Hover(text);
    } }),
    vscode.languages.registerInlayHintsProvider(selector, { async provideInlayHints(document, requestedRange) {
      const index = await snapshot(document);
      if (!index) return [];
      return [...index.definitions.values()].filter(item => item.kind === 'variable' && path.resolve(item.source.path) === document.uri.fsPath)
        .map(item => {
          const located = index.location(item.source);
          const pos = new vscode.Position(located.start.line, located.start.character + item.name.length);
          return new vscode.InlayHint(pos, ': ' + item.type, vscode.InlayHintKind.Type);
        }).filter(hint => requestedRange.contains(hint.position));
    } }),
    vscode.languages.registerDocumentSemanticTokensProvider(selector, { async provideDocumentSemanticTokens(document) {
      const index = await snapshot(document);
      const builder = new vscode.SemanticTokensBuilder(new vscode.SemanticTokensLegend(tokenTypes, tokenModifiers));
      if (index) {
        const tokens = [...index.definitions.values()].filter(item => path.resolve(item.source.path) === document.uri.fsPath && tokenTypes.includes(item.kind));
        for (const item of tokens) {
          const located = index.location(item.source);
          const line = document.lineAt(located.start.line).text;
          const column = line.indexOf(item.name.split('.').at(-1), located.start.character);
          if (column >= 0) builder.push(located.start.line, column, item.name.split('.').at(-1).length, tokenTypes.indexOf(item.kind), /@deprecated\b/u.test(item.documentation) ? 1 : 0);
        }
      }
      return builder.build();
    } }, new vscode.SemanticTokensLegend(tokenTypes, tokenModifiers)),
    vscode.workspace.onDidChangeTextDocument(() => { generation += 1; snapshots.clear(); diagnostics.clear(); }),
    vscode.workspace.onDidSaveTextDocument(document => {
      snapshots.clear();
      if (document.languageId !== 'severian') return;
      void publishLint(document);
    }),
  );

  async function typeRelations(item, subtypes) {
    const document = await vscode.workspace.openTextDocument(item.uri);
    const { index, definition } = await selected(document, item.range.start);
    if (!definition) return [];
    return (subtypes ? index.implementations(definition.id) : index.supertypes(definition.id))
      .filter(target => target.kind === 'class').map(target => {
        const located = index.location(target.source);
        return new vscode.TypeHierarchyItem(vscode.SymbolKind.Class, target.name, target.type,
          vscode.Uri.file(located.file), range(located), range(located));
      });
  }

  async function publishLint(document) {
    const requestedGeneration = generation;
    const executable = vscode.workspace.getConfiguration('severian', document.uri).get('executable', 'sev');
    try {
      const result = await execute(executable, ['lint', path.dirname(document.uri.fsPath)], { timeout: 120000, maxBuffer: 16 * 1024 * 1024 });
      const grouped = new Map();
      for (const line of result.stdout.trim().split('\n').filter(Boolean)) {
        const finding = JSON.parse(line);
        const uri = vscode.Uri.file(finding.file);
        const severity = { error: 0, warning: 1, info: 2, hint: 3 }[finding.severity];
        const source = await vscode.workspace.openTextDocument(uri);
        const utf16 = (line, column) => Array.from(source.lineAt(line - 1).text).slice(0, column - 1).join('').length;
        const diagnostic = new vscode.Diagnostic(new vscode.Range(finding.line - 1, utf16(finding.line, finding.column), finding.end_line - 1, utf16(finding.end_line, finding.end_column)), finding.message + '\n' + finding.remediation, severity);
        diagnostic.code = finding.rule;
        diagnostic.source = 'Severian';
        if (['L0004', 'L0005'].includes(finding.rule)) diagnostic.tags = [vscode.DiagnosticTag.Unnecessary];
        const group = grouped.get(uri.fsPath) || { uri, values: [] };
        group.values.push(diagnostic);
        grouped.set(uri.fsPath, group);
      }
      if (requestedGeneration !== generation || document.isDirty) return;
      diagnostics.clear();
      for (const group of grouped.values()) diagnostics.set(group.uri, group.values);
    } catch (error) {
      console.error('Severian lint:', error.message);
    }
  }
}

module.exports = { registerSemantic };
