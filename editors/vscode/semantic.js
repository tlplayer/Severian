'use strict';

const { execFile } = require('child_process');
const { promisify } = require('util');
const path = require('path');
const { createSemanticWorkspace } = require('./semantic-workspace');
const { keywords } = require('./source-index');
const execute = promisify(execFile);

function registerSemantic(vscode, context, options = {}) {
  const workspace = createSemanticWorkspace(vscode, context, options);
  const snapshot = workspace.snapshot;
  const hierarchy = new WeakMap();
  const diagnostics = vscode.languages.createDiagnosticCollection('severian-quality');
  const selector = { language: 'severian', scheme: 'file' };
  const range = location => new vscode.Range(location.start.line, location.start.character, location.end.line, location.end.character);
  const location = value => new vscode.Location(vscode.Uri.file(value.file), range(value));

  async function selected(document, position) {
    const index = await snapshot(document);
    const offset = Array.from(document.getText().slice(0, document.offsetAt(position))).length;
    return { index, ...index?.occurrence(document.uri.fsPath, offset) };
  }

  const definitionProvider = { async provideDefinition(document, position) {
    const { index, definition } = await selected(document, position);
    return definition && location(index.declaration(definition));
  } };
  const tokenTypes = ['function', 'parameter', 'variable', 'class'];
  const tokenModifiers = ['deprecated'];
  function hierarchyItem(index, definition, type = false) {
    const located = index.location(definition.source);
    const Item = type ? vscode.TypeHierarchyItem : vscode.CallHierarchyItem;
    const item = new Item(type ? vscode.SymbolKind.Class : vscode.SymbolKind.Function, definition.name, definition.type,
      vscode.Uri.file(located.file), range(located), range(index.declaration(definition)));
    hierarchy.set(item, { index, definition, generation: workspace.generation() });
    return item;
  }
  function hierarchySelection(item) {
    const selected = hierarchy.get(item);
    return selected?.generation === workspace.generation() ? selected : {};
  }
  function callRanges(index, caller, target) {
    return index.calls().filter(reference => reference.caller === caller.id && reference.symbol === target.id
      && path.resolve(reference.source.path) === path.resolve(caller.source.path))
      .map(reference => range(index.location(index.selection(reference, target.name) || reference.source)));
  }
  context.subscriptions.push(diagnostics,
    vscode.languages.registerDefinitionProvider(selector, definitionProvider),
    vscode.languages.registerImplementationProvider(selector, { async provideImplementation(document, position) {
      const { index, definition } = await selected(document, position);
      if (!definition) return [];
      const implementations = index.implementations(definition.id);
      return (implementations.length ? implementations : definition.abstract ? [] : [definition])
        .map(item => location(index.declaration(item)));
    } }),
    vscode.languages.registerCallHierarchyProvider(selector, {
      async prepareCallHierarchy(document, position) {
        const { index, definition } = await selected(document, position);
        return definition?.kind === 'function' ? hierarchyItem(index, definition) : undefined;
      },
      async provideCallHierarchyIncomingCalls(item) {
        const { index, definition } = hierarchySelection(item);
        if (!definition) return [];
        return index.callers(definition.id).map(caller => new vscode.CallHierarchyIncomingCall(hierarchyItem(index, caller),
          callRanges(index, caller, definition)));
      },
      async provideCallHierarchyOutgoingCalls(item) {
        const { index, definition } = hierarchySelection(item);
        if (!definition) return [];
        const ids = [...new Set(index.calls().filter(reference => reference.caller === definition.id).map(reference => reference.symbol))];
        return ids.map(id => index.definitions.get(id)).filter(target => target?.kind === 'function')
          .map(target => new vscode.CallHierarchyOutgoingCall(hierarchyItem(index, target),
            callRanges(index, definition, target)));
      },
    }),
    vscode.languages.registerRenameProvider(selector, {
      async prepareRename(document, position) {
        const { index, definition, selection } = await selected(document, position);
        if (!definition) throw new Error('Select a resolved Severian identifier to rename.');
        return { range: range(index.location(selection)), placeholder: definition.name.split('.').at(-1) };
      },
      async provideRenameEdits(document, position, newName) {
      if (!/^[\p{L}_][\p{L}\p{N}_]*$/u.test(newName) || keywords.has(newName)) throw new Error('Use a Severian identifier that is not a keyword.');
      const { index, definition } = await selected(document, position);
      if (!definition) return undefined;
      const edits = new vscode.WorkspaceEdit();
      for (const item of index.rename(definition.id)) edits.replace(vscode.Uri.file(item.file), range(item), newName);
      return edits;
    } }),
    // Older supported editor versions do not expose the type hierarchy API.
    ...(vscode.languages.registerTypeHierarchyProvider ? [vscode.languages.registerTypeHierarchyProvider(selector, {
      async prepareTypeHierarchy(document, position) {
        const { index, definition } = await selected(document, position);
        if (definition?.kind !== 'class') return undefined;
        return hierarchyItem(index, definition, true);
      },
      async provideTypeHierarchySupertypes(item) { return typeRelations(item, false); },
      async provideTypeHierarchySubtypes(item) { return typeRelations(item, true); },
    })] : []),
    vscode.languages.registerReferenceProvider(selector, { async provideReferences(document, position, options) {
      const { index, definition } = await selected(document, position);
      return definition ? index.usages(definition.id, options.includeDeclaration).map(location) : [];
    } }),
    vscode.languages.registerDocumentHighlightProvider(selector, { async provideDocumentHighlights(document, position) {
      const { index, definition } = await selected(document, position);
      return definition ? index.usages(definition.id, true).filter(item => item.file === document.uri.fsPath)
        .map(item => new vscode.DocumentHighlight(range(item), vscode.DocumentHighlightKind.Text)) : [];
    } }),
    vscode.languages.registerCodeActionsProvider(selector, { async provideCodeActions(document, requestedRange, context) {
      if (context.only && !context.only.contains(vscode.CodeActionKind.Refactor)) return [];
      const { definition } = await selected(document, requestedRange.start);
      if (!definition) return [];
      const action = new vscode.CodeAction('Rename symbol', vscode.CodeActionKind.Refactor);
      action.command = { command: 'editor.action.rename', title: action.title,
        arguments: [document.uri, requestedRange.start] };
      return [action];
    } }, { providedCodeActionKinds: [vscode.CodeActionKind.Refactor] }),
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
      if (!index || index.analysis === 'source') return [];
      return [...index.definitions.values()].filter(item => item.kind === 'variable' && path.resolve(item.source.path) === document.uri.fsPath)
        .map(item => {
          const located = index.declaration(item);
          const pos = new vscode.Position(located.end.line, located.end.character);
          return new vscode.InlayHint(pos, ': ' + item.type, vscode.InlayHintKind.Type);
        }).filter(hint => requestedRange.contains(hint.position));
    } }),
    vscode.languages.registerDocumentSemanticTokensProvider(selector, { async provideDocumentSemanticTokens(document) {
      const index = await snapshot(document);
      const builder = new vscode.SemanticTokensBuilder(new vscode.SemanticTokensLegend(tokenTypes, tokenModifiers));
      if (index) {
        const tokens = [...index.definitions.values()].filter(item => path.resolve(item.source.path) === document.uri.fsPath && tokenTypes.includes(item.kind));
        for (const item of tokens) {
          const located = index.declaration(item);
          if (located.start.line === located.end.line) builder.push(located.start.line, located.start.character,
            located.end.character - located.start.character, tokenTypes.indexOf(item.kind), /@deprecated\b/u.test(item.documentation) ? 1 : 0);
        }
      }
      return builder.build();
    } }, new vscode.SemanticTokensLegend(tokenTypes, tokenModifiers)),
    vscode.workspace.onDidChangeTextDocument(event => { if (event.document.languageId === 'severian') diagnostics.clear(); }),
    vscode.workspace.onDidSaveTextDocument(document => {
      if (document.languageId !== 'severian') return;
      void publishLint(document);
    }),
  );

  async function typeRelations(item, subtypes) {
    const { index, definition } = hierarchySelection(item);
    if (!definition) return [];
    return (subtypes ? index.implementations(definition.id) : index.supertypes(definition.id))
      .filter(target => target.kind === 'class').map(target => hierarchyItem(index, target, true));
  }

  async function publishLint(document) {
    const requestedGeneration = workspace.generation();
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
      if (requestedGeneration !== workspace.generation() || document.isDirty) return;
      diagnostics.clear();
      for (const group of grouped.values()) diagnostics.set(group.uri, group.values);
    } catch (error) {
      console.error('Severian lint:', error.message);
    }
  }
}

module.exports = { registerSemantic };
