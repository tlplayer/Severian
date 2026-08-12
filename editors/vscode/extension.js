'use strict';

const childProcess = require('child_process');
const fs = require('fs');
const path = require('path');
const vscode = require('vscode');
const {
  buildFileCoverage,
  normalizeFilePath,
  parseCoverageMap,
  parseHits,
} = require('./coverage-data');

const decoder = new TextDecoder('utf-8');

function activate(context) {
  const output = vscode.window.createOutputChannel('Severian Coverage');
  const coveredDecoration = vscode.window.createTextEditorDecorationType({
    gutterIconPath: context.asAbsolutePath('images/coverage-covered.svg'),
    gutterIconSize: 'contain',
    overviewRulerColor: new vscode.ThemeColor('testing.iconPassed'),
    overviewRulerLane: vscode.OverviewRulerLane.Left,
    isWholeLine: true,
  });
  const uncoveredDecoration = vscode.window.createTextEditorDecorationType({
    gutterIconPath: context.asAbsolutePath('images/coverage-uncovered.svg'),
    gutterIconSize: 'contain',
    overviewRulerColor: new vscode.ThemeColor('testing.iconFailed'),
    overviewRulerLane: vscode.OverviewRulerLane.Left,
    isWholeLine: true,
  });
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  status.command = 'severian.coverage.run';
  status.name = 'Severian Coverage';

  let files = new Map();
  let refreshSerial = 0;
  let refreshTimer;
  let running = false;

  function lineDecoration(line) {
    const names = [...line.functions].sort();
    const regionSummary = `${line.coveredRegions}/${line.ids.size} statement region${line.ids.size === 1 ? '' : 's'} covered`;
    const functionSummary = names.length > 0 ? ` in ${names.join(', ')}` : '';
    return {
      range: new vscode.Range(line.line, 0, line.line, 0),
      hoverMessage: `${line.covered ? 'Covered' : 'Uncovered'}: ${regionSummary}${functionSummary}`,
    };
  }

  function renderEditor(editor) {
    if (editor.document.languageId !== 'severian' || editor.document.uri.scheme !== 'file') {
      editor.setDecorations(coveredDecoration, []);
      editor.setDecorations(uncoveredDecoration, []);
      return;
    }
    const coverage = files.get(normalizeFilePath(editor.document.uri.fsPath));
    const covered = [];
    const uncovered = [];
    if (coverage) {
      for (const line of coverage.lines.values()) {
        (line.covered ? covered : uncovered).push(lineDecoration(line));
      }
    }
    editor.setDecorations(coveredDecoration, covered);
    editor.setDecorations(uncoveredDecoration, uncovered);
  }

  function render() {
    for (const editor of vscode.window.visibleTextEditors) {
      renderEditor(editor);
    }

    const editor = vscode.window.activeTextEditor;
    const coverage =
      editor && editor.document.uri.scheme === 'file'
        ? files.get(normalizeFilePath(editor.document.uri.fsPath))
        : undefined;
    if (!editor || editor.document.languageId !== 'severian' || !coverage) {
      status.hide();
      return;
    }
    const metrics = coverage.metrics;
    status.text = `$(beaker) ${metrics.lines.percent.toFixed(1)}% coverage`;
    status.tooltip = [
      `Lines: ${formatMetric(metrics.lines)}`,
      `Statement regions: ${formatMetric(metrics.regions)}`,
      `Branches: ${formatMetric(metrics.branches)}`,
      `Functions: ${formatMetric(metrics.functions)}`,
      '',
      'Click to run Severian coverage.',
    ].join('\n');
    status.show();
  }

  async function loadCoverage(showResult = false) {
    const serial = ++refreshSerial;
    const mapUris = await vscode.workspace.findFiles(
      '**/target/coverage/coverage-map.json',
      '**/{.git,node_modules}/**',
      200,
    );
    const allRegions = [];
    const allHits = new Set();
    const failures = [];

    await Promise.all(
      mapUris.map(async (mapUri) => {
        try {
          const mapContents = decoder.decode(await vscode.workspace.fs.readFile(mapUri));
          allRegions.push(...parseCoverageMap(mapContents));
          const directory = vscode.Uri.joinPath(mapUri, '..');
          const consolidated = vscode.Uri.joinPath(directory, 'coverage.hits');
          try {
            addHits(allHits, decoder.decode(await vscode.workspace.fs.readFile(consolidated)));
          } catch (error) {
            if (!isMissingFile(error)) {
              throw error;
            }
            const entries = await vscode.workspace.fs.readDirectory(directory);
            const legacyHitFiles = entries
              .filter(([name, type]) => type === vscode.FileType.File && name.endsWith('.hits'))
              .map(([name]) => vscode.Uri.joinPath(directory, name));
            const hitContents = await Promise.all(
              legacyHitFiles.map((uri) => vscode.workspace.fs.readFile(uri)),
            );
            for (const contents of hitContents) {
              addHits(allHits, decoder.decode(contents));
            }
          }
        } catch (error) {
          failures.push(`${mapUri.fsPath}: ${messageFor(error)}`);
        }
      }),
    );

    if (serial !== refreshSerial) {
      return;
    }
    files = buildFileCoverage(allRegions, allHits);
    render();

    if (failures.length > 0) {
      output.appendLine(`Could not load ${failures.length} coverage report(s):`);
      failures.forEach((failure) => output.appendLine(`  ${failure}`));
      output.show(true);
    }
    if (showResult) {
      if (mapUris.length === 0) {
        void vscode.window.showWarningMessage(
          'No Severian coverage report was found. Run “Severian: Run Coverage” first.',
        );
      } else {
        void vscode.window.showInformationMessage(
          `Loaded Severian coverage for ${files.size} file${files.size === 1 ? '' : 's'}.`,
        );
      }
    }
  }

  async function runCoverage() {
    if (running) {
      void vscode.window.showInformationMessage('Severian coverage is already running.');
      return;
    }
    const run = resolveCoverageRun();
    if (!run) {
      void vscode.window.showErrorMessage('Open a .sev file inside a workspace before running coverage.');
      return;
    }

    const configuration = vscode.workspace.getConfiguration('severian.coverage', run.scope);
    const executable = configuration.get('executable', 'sev').trim() || 'sev';
    output.clear();
    output.appendLine(`$ ${executable} coverage ${run.target}`);
    output.show(true);
    running = true;
    try {
      const exitCode = await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'Running Severian coverage',
          cancellable: true,
        },
        (_progress, token) => spawnCoverage(executable, run.target, run.cwd, token, output),
      );
      await loadCoverage(false);
      if (exitCode === 0) {
        void vscode.window.showInformationMessage('Severian coverage gutters updated.');
      } else if (exitCode !== undefined) {
        void vscode.window.showErrorMessage(
          `Severian coverage exited with code ${exitCode}. See the Severian Coverage output.`,
        );
      }
    } catch (error) {
      void vscode.window.showErrorMessage(`Could not run Severian coverage: ${messageFor(error)}`);
    } finally {
      running = false;
    }
  }

  function clearCoverage() {
    refreshSerial += 1;
    files = new Map();
    render();
  }

  function scheduleRefresh() {
    if (!vscode.workspace.getConfiguration('severian.coverage').get('autoLoad', true)) {
      return;
    }
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => void loadCoverage(false), 250);
  }

  const watcher = vscode.workspace.createFileSystemWatcher('**/target/coverage/{coverage-map.json,*.hits}');
  watcher.onDidCreate(scheduleRefresh);
  watcher.onDidChange(scheduleRefresh);
  watcher.onDidDelete(scheduleRefresh);

  context.subscriptions.push(
    output,
    coveredDecoration,
    uncoveredDecoration,
    status,
    watcher,
    vscode.commands.registerCommand('severian.coverage.run', runCoverage),
    vscode.commands.registerCommand('severian.coverage.load', () => loadCoverage(true)),
    vscode.commands.registerCommand('severian.coverage.clear', clearCoverage),
    vscode.window.onDidChangeVisibleTextEditors(render),
    vscode.window.onDidChangeActiveTextEditor(render),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration('severian.coverage.autoLoad')) {
        const autoLoad = vscode.workspace.getConfiguration('severian.coverage').get('autoLoad', true);
        if (autoLoad) {
          void loadCoverage(false);
        } else {
          clearCoverage();
        }
      }
    }),
    { dispose: () => clearTimeout(refreshTimer) },
  );

  if (vscode.workspace.getConfiguration('severian.coverage').get('autoLoad', true)) {
    void loadCoverage(false);
  }
}

function addHits(target, contents) {
  for (const id of parseHits(contents)) {
    target.add(id);
  }
}

function formatMetric(metric) {
  return `${metric.percent.toFixed(1)}% (${metric.covered}/${metric.count})`;
}

function isMissingFile(error) {
  return error && (error.code === 'FileNotFound' || error.code === 'ENOENT');
}

function messageFor(error) {
  return error instanceof Error ? error.message : String(error);
}

function resolveCoverageRun() {
  const editor = vscode.window.activeTextEditor;
  const scope = editor && editor.document.uri;
  const folder = scope ? vscode.workspace.getWorkspaceFolder(scope) : undefined;
  const workspaceFolder = folder || (vscode.workspace.workspaceFolders || [])[0];
  if (!workspaceFolder || workspaceFolder.uri.scheme !== 'file') {
    return undefined;
  }

  const configuration = vscode.workspace.getConfiguration('severian.coverage', scope);
  const configuredTarget = configuration.get('target', '').trim();
  if (configuredTarget) {
    return {
      scope,
      cwd: workspaceFolder.uri.fsPath,
      target: path.isAbsolute(configuredTarget)
        ? configuredTarget
        : path.join(workspaceFolder.uri.fsPath, configuredTarget),
    };
  }

  let target = workspaceFolder.uri.fsPath;
  if (editor && editor.document.languageId === 'severian' && editor.document.uri.scheme === 'file') {
    target = nearestPackageRoot(editor.document.uri.fsPath, workspaceFolder.uri.fsPath) || target;
  }
  return { scope, cwd: workspaceFolder.uri.fsPath, target };
}

function nearestPackageRoot(source, workspaceRoot) {
  const boundary = path.resolve(workspaceRoot);
  let current = path.dirname(path.resolve(source));
  while (current === boundary || current.startsWith(`${boundary}${path.sep}`)) {
    if (fs.existsSync(path.join(current, 'package.toml'))) {
      return current;
    }
    if (current === boundary) {
      break;
    }
    current = path.dirname(current);
  }
  return undefined;
}

function spawnCoverage(executable, target, cwd, token, output) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const child = childProcess.spawn(executable, ['coverage', target], {
      cwd,
      env: process.env,
      shell: false,
    });
    child.stdout.on('data', (data) => output.append(data.toString()));
    child.stderr.on('data', (data) => output.append(data.toString()));
    child.on('error', (error) => {
      settled = true;
      reject(error);
    });
    child.on('close', (code) => {
      if (!settled) {
        settled = true;
        resolve(code === null ? undefined : code);
      }
    });
    token.onCancellationRequested(() => {
      if (!settled) {
        output.appendLine('\nCoverage run cancelled.');
        child.kill();
      }
    });
  });
}

function deactivate() {}

module.exports = { activate, deactivate };
