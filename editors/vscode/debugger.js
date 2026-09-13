'use strict';

const { execFile } = require('child_process');
const { promisify } = require('util');
const execute = promisify(execFile);

function registerDebugger(vscode, context) {
  context.subscriptions.push(
    vscode.debug.registerDebugAdapterDescriptorFactory('severian', {
      createDebugAdapterDescriptor(session) {
        const debuggerPath = vscode.workspace.getConfiguration('severian').get('debugger', 'gdb');
        return new vscode.DebugAdapterExecutable(debuggerPath, ['--interpreter=dap'], { cwd: session.configuration.cwd });
      },
    }),
    vscode.debug.registerDebugConfigurationProvider('severian', {
      async resolveDebugConfiguration(folder, configuration) {
        if (configuration.program) return configuration;
        const executable = vscode.workspace.getConfiguration('severian').get('executable', 'sev');
        const target = configuration.target || folder?.uri.fsPath;
        if (!target) throw new Error('Select a Severian source or package to debug.');
        const result = await execute(executable, ['build', target, '--build-profile', 'dev'], {
          cwd: configuration.cwd || folder?.uri.fsPath, timeout: 300000, maxBuffer: 16 * 1024 * 1024,
        });
        const artifacts = result.stdout.trim().split('\n').filter(Boolean);
        if (artifacts.length !== 1) throw new Error('Debugging requires one binary target; set program in launch.json.');
        return { ...configuration, program: artifacts[0].replace(/^built /, ''), stopAtBeginningOfMainSubprogram: true };
      },
    }),
  );
}

module.exports = { registerDebugger };
