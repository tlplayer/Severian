from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


QUALITY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALITY))

from check_bootstrap_mirror import check_mirror  # noqa: E402


class BootstrapMirrorTests(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "compiler/source/src").mkdir(parents=True)
        (root / "compiler/source/Cargo.toml").write_text("[package]\nname='host'\n")
        (root / "compiler/source/src/lib.rs").write_text("pub struct Source;\n")
        (root / "sev_compiler/source/src").mkdir(parents=True)
        (root / "sev_compiler/source/package.toml").write_text(
            "[package]\nname='bootstrap'\n[lib]\npath='src/lib.sev'\n"
        )
        (root / "sev_compiler/source/src/lib.sev").write_text("trait Source:\n    pass\n")
        (root / "docs/examples").mkdir(parents=True)
        (root / "test/validation/examples").mkdir(parents=True)
        (root / "sev_compiler/bootstrap.toml").write_text(
            "[mirror]\n"
            "require-directory-parity=true\n"
            "require-file-parity=true\n"
            "require-package-parity=true\n"
            "[acceptance]\n"
            "canonical-examples='../docs/examples'\n"
            "validation-package='../test/validation/examples'\n"
            "allow-skips=false\n"
            "allow-source-rewrites=false\n"
            "[[stage]]\ncompiler='sev0'\nvalidation='sev0 test test/validation/examples'\n"
            "[[stage]]\ncompiler='sev1'\nvalidation='sev1 test test/validation/examples'\n"
            "[[stage]]\ncompiler='sev2'\nvalidation='sev2 test test/validation/examples'\n"
            "[[stage]]\ncompiler='sev3'\nvalidation='sev3 test test/validation/examples'\n"
        )
        return temporary, root

    def test_accepts_complete_directory_source_and_package_parity(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        failures, packages, sources = check_mirror(root)
        self.assertEqual(failures, [])
        self.assertEqual(packages, 1)
        self.assertEqual(sources, 1)

    def test_rejects_a_missing_source_counterpart(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        (root / "sev_compiler/source/src/lib.sev").unlink()
        failures, _, _ = check_mirror(root)
        self.assertTrue(any("missing source mirror" in failure for failure in failures))

    def test_rejects_a_missing_directory_counterpart(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        (root / "compiler/source/src/model").mkdir()
        failures, _, _ = check_mirror(root)
        self.assertTrue(any("missing mirrored directory" in failure for failure in failures))

    def test_rejects_an_incomplete_stage_sequence(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        contract = root / "sev_compiler/bootstrap.toml"
        contract.write_text(contract.read_text().replace("compiler='sev3'", "compiler='sev4'"))
        failures, _, _ = check_mirror(root)
        self.assertTrue(any("sev0 through sev3" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
