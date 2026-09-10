#!/usr/bin/env python3
"""Build and install the source compiler with a Rust recovery command."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run(*args, capture=False, cwd=ROOT):
    result = subprocess.run(list(map(str, args)), cwd=cwd, check=True,
                            text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def update_checkout():
    # Never stash, reset, or overwrite local work on the user's behalf.
    if run('git', 'status', '--porcelain', '--untracked-files=normal', capture=True):
        raise RuntimeError('checkout has local changes; commit them before updating, '
                           'or use update --local to build the current checkout')
    upstream = run('git', 'rev-parse', '--abbrev-ref', '--symbolic-full-name',
                   '@{upstream}', capture=True)
    branch = run('git', 'symbolic-ref', '--short', 'HEAD', capture=True)
    remote = run('git', 'config', f'branch.{branch}.remote', capture=True)
    if remote == '.':
        raise RuntimeError('the current branch has no remote upstream')
    run('git', 'fetch', remote)
    run('git', 'merge', '--ff-only', upstream)


def install(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('sev_rust', 'sev'):
        destination = directory / name
        source = ROOT / 'bin' / name
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.exists() or destination.is_symlink():
            # Preserve the old installation exactly once, including symlinks.
            backup = directory / (name + '.before-source-default')
            if not backup.exists() and not backup.is_symlink():
                if destination.is_symlink():
                    backup.symlink_to(os.readlink(destination))
                else:
                    shutil.copy2(destination, backup)
        temporary = directory / ('.' + name + f'.install-{os.getpid()}')
        try:
            temporary.symlink_to(source)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true',
                        help='build the current checkout without fetching upstream')
    parser.add_argument('--install-dir', type=Path,
                        default=Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo')) / 'bin')
    parser.add_argument('--no-install', action='store_true', help='build and verify only')
    args = parser.parse_args()
    cache = ROOT / 'package.pkg'
    cache.mkdir(exist_ok=True)
    with (cache / 'compiler-update.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another compiler update is running') from None
        if not args.local:
            update_checkout()
        # Always invoke concrete artifacts, so changing the default cannot
        # accidentally make the source compiler bootstrap itself.
        run('cargo', 'build', '--release', '--target-dir', cache, '-p', 'severian-driver', '--bin', 'sev')
        seed = cache / 'release' / 'sev'
        compiler = ROOT / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler'
        # Keep the previous working source compiler if build or smoke tests fail.
        with tempfile.TemporaryDirectory(prefix='sev-update-') as temporary:
            previous = Path(temporary) / 'sev_compiler'
            if compiler.exists():
                shutil.copy2(compiler, previous)
            try:
                run(seed, 'build', ROOT / 'sev_compiler', '--bin', 'sev_compiler')
                run(seed, '--version')
                run(compiler, '--help')
                smoke = Path(temporary) / 'smoke.sev'
                smoke.write_text('assert(20 + 22 == 42)\n')
                run(compiler, smoke, '--sysroot', ROOT)
            except BaseException:
                if previous.exists():
                    shutil.copy2(previous, compiler)
                raise
        if not args.no_install:
            install(args.install_dir.expanduser().resolve())
        revision = run('git', 'rev-parse', '--short', 'HEAD', capture=True)
        print(f'Compiler updated ({revision}): sev = source compiler; sev_rust = Rust backup.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Compiler update failed: {error}', file=sys.stderr)
        sys.exit(1)
