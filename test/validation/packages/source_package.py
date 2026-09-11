#!/usr/bin/env python3
"""Black-box package lifecycle through the runnable Severian compiler."""
import os
from pathlib import Path
import subprocess
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[3]
COMPILER = Path(os.environ.get("SEVERIAN_SOURCE_COMPILER", ROOT / "sev_compiler/package.pkg/host/dev/bin/sev_compiler"))

def main():
    with tempfile.TemporaryDirectory(prefix="sev-source-package-") as temporary:
        root = Path(temporary)
        env = {**os.environ, "SEVERIAN_REGISTRY": str(root / "registry"), "SEVERIAN_SYSROOT": str(ROOT)}
        def sev(*args, cwd=root, succeeds=True):
            result = subprocess.run([str(COMPILER), *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
            if (result.returncode == 0) != succeeds:
                raise AssertionError(f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
            return result
        sev("new", "geometry", "--lib")
        sev("publish", cwd=root / "geometry")
        release = root / "registry/packages/geometry/0.1.0/geometry-0.1.0.pkg"
        assert release.read_bytes().startswith(b"SEVPKG\0\x02")
        sev("new", "application")
        app = root / "application"
        sev("add", "geometry@0", "--alias", "shapes", cwd=app)
        (app / "src/main.sev").write_text("import shapes\ndef main():\n    print(shapes.answer())\n")
        sev("check", cwd=app)
        sev("build", cwd=app)
        assert sev("run", cwd=app).stdout.strip() == "42"
        assert (app / "package.pkg/host/dev/bin/application").is_file()
        assert not (app / "target").exists()
        # Generated source must not enter resolution or a published source tree.
        (app / "package.pkg/generated.sev").write_text("invalid generated source")
        (app / "target").mkdir()
        (app / "target/keep.sev").write_text("def preserved() -> int:\n    return 7\n")
        assert "shapes = geometry@0.1.0" in sev("tree", cwd=app).stdout
        before = (app / "package.toml").read_bytes(), (app / "sev.lock").read_bytes()
        sev("add", "missing@9", cwd=app, succeeds=False)
        assert before == ((app / "package.toml").read_bytes(), (app / "sev.lock").read_bytes())
        (app / "src/main.sev").write_text("test \"answer\":\n    assert(42 == 42)\ndef main():\n    print(42)\n")
        sev("test", cwd=app)
        sev("publish", cwd=app)
        published_source = root / "registry/packages/application/0.1.0/source"
        assert not (published_source / "package.pkg").exists()
        assert (published_source / "target/keep.sev").is_file()
        assert sev("run", "application@0.1.0").stdout.strip() == "42"
        sev("install", "application@0.1.0", "-o", root / "bin")
        assert subprocess.check_output([root / "bin/application"], text=True).strip() == "42"
        # A damaged release is an integrity failure, not permission to rebuild
        # or silently replace the published artifact.
        binary = root / "registry/packages/application/0.1.0/artifacts/host/release/bin/application"
        original_binary = binary.read_bytes()
        binary.unlink()
        rejected = sev("run", "application@0.1.0", succeeds=False)
        assert "PackageIntegrityError" in rejected.stderr
        binary.write_bytes(original_binary)
        binary.chmod(0o755)
        assert sev("run", "application@0.1.0").stdout.strip() == "42"
        # A transitive dependency is usable only through its declared alias edge.
        sev("new", "matrix", "--lib")
        matrix = root / "matrix"
        sev("add", "geometry@0", cwd=matrix)
        (matrix / "src/lib.sev").write_text("import geometry\ndef answer() -> int:\n    return geometry.answer() + 1\n")
        sev("publish", cwd=matrix)
        sev("new", "service", "--lib")
        service = root / "service"
        sev("add", "matrix@0", cwd=service)
        (service / "src/lib.sev").write_text("import matrix\ndef answer() -> int:\n    return matrix.answer() * 2\n")
        sev("publish", cwd=service)
        # Removing a working checkout never affects its published snapshot.
        shutil.rmtree(root / "geometry")
        sev("new", "transitive")
        transitive = root / "transitive"
        sev("add", "service@0", cwd=transitive)
        (transitive / "src/main.sev").write_text("import service\ndef main():\n    print(service.answer())\n")
        assert sev("run", cwd=transitive).stdout.strip() == "86"
        (transitive / "src/main.sev").write_text("import matrix\ndef main():\n    print(matrix.answer())\n")
        rejected = sev("check", cwd=transitive, succeeds=False)
        assert "undeclared dependency import" in rejected.stderr and "matrix" in rejected.stderr, rejected.stderr
        sev("clean", cwd=app)
        assert not (app / "package.pkg").exists()
        assert (app / "target/keep.sev").is_file()
        assert (app / "src/main.sev").exists()
    print("source package lifecycle: passed")

if __name__ == "__main__":
    main()
