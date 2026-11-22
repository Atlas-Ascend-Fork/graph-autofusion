#!/usr/bin/env python3
"""Tests for the build-time Ascend target profile resource generator."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


GENERATOR = (
    Path(__file__).resolve().parents[3]
    / "target"
    / "Ascend"
    / "Profiles"
    / "generate_builtin_profiles.py"
)
REAL_PROFILE_ROOT = (
    Path(__file__).resolve().parents[3] / "target" / "Ascend" / "Profiles"
)


def parse_manifest(manifest: Path) -> list[tuple[str, list[str]]]:
    groups = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        canonical, alias_list = line.split(":", 1)
        groups.append((canonical, alias_list.split("|")))
    return groups


class BuiltinTargetProfilesGeneratorTest(unittest.TestCase):
    def run_generator(self, manifest_text: str, profiles: dict[str, str]):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        profile_dir = root / "profiles"
        profile_dir.mkdir()
        for platform, contents in profiles.items():
            (profile_dir / f"{platform}.ini").write_text(contents, encoding="utf-8")
        manifest = root / "manifest.txt"
        manifest.write_text(manifest_text, encoding="utf-8")
        output = root / "BuiltinTargetProfiles.inc"
        result = subprocess.run(
            [sys.executable, GENERATOR, manifest, profile_dir, output],
            check=False,
            capture_output=True,
            text=True,
        )
        return result, output

    def test_embeds_canonical_ini_and_aliases(self):
        result, output = self.run_generator(
            "AscendTest1:AscendTest1|AscendTest2\n",
            {"AscendTest1": '[version]\nSoC_version=AscendTest1\nvalue="quoted"\n'},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        generated = output.read_text(encoding="utf-8")
        self.assertIn('"AscendTest1,AscendTest2"', generated)
        self.assertIn('"SoC_version=AscendTest1\\n"', generated)
        self.assertIn('"value=\\"quoted\\"\\n"', generated)

    def test_rejects_duplicate_alias(self):
        result, _ = self.run_generator(
            "AscendTest1:AscendTest1\nAscendTest2:AscendTest2|AscendTest1\n",
            {"AscendTest1": "first", "AscendTest2": "second"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate or empty platform alias", result.stderr)

    def test_rejects_missing_canonical_ini(self):
        result, _ = self.run_generator("AscendMissing:AscendMissing\n", {})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing canonical INI", result.stderr)

    def test_rejects_invalid_platform_name_and_alias_order(self):
        result, _ = self.run_generator("AscendTest1|AscendTest1\n", {})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected canonical:alias|alias", result.stderr)

        result, _ = self.run_generator("../AscendTest:../AscendTest\n", {})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid platform name", result.stderr)

        result, _ = self.run_generator(
            "AscendTest1:AscendTest2|AscendTest1\n",
            {"AscendTest1": "profile"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical platform must be the first alias", result.stderr)

    def test_platform_name_length_boundary(self):
        maximum_length_name = "A" * 128
        result, _ = self.run_generator(
            f"{maximum_length_name}:{maximum_length_name}\n",
            {maximum_length_name: "profile"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        oversized_name = "A" * 129
        result, _ = self.run_generator(
            f"{oversized_name}:{oversized_name}\n",
            {oversized_name: "profile"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid platform name", result.stderr)

    def test_real_manifest_aliases_reference_canonical_ini(self):
        manifest = REAL_PROFILE_ROOT / "manifest.txt"
        profile_dir = REAL_PROFILE_ROOT / "platform_config"
        groups = parse_manifest(manifest)
        seen_aliases = set()

        for canonical, aliases in groups:
            self.assertGreater(len(aliases), 0)
            self.assertEqual(aliases[0], canonical)
            self.assertTrue((profile_dir / f"{canonical}.ini").is_file(), canonical)
            for alias in aliases:
                self.assertNotIn(alias, seen_aliases)
                seen_aliases.add(alias)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "BuiltinTargetProfiles.inc"
            result = subprocess.run(
                [sys.executable, GENERATOR, manifest, profile_dir, output],
                check=False,
                capture_output=True,
                text=True,
            )
            generated = output.read_text(encoding="utf-8") if output.exists() else ""

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(len(seen_aliases), 0)
        for _, aliases in groups:
            self.assertIn(f'"{",".join(aliases)}"', generated)

    def test_real_profile_checksums_match_audit_record(self):
        checksums = {}
        checksum_file = REAL_PROFILE_ROOT / "SHA256SUMS"
        for raw_line in checksum_file.read_text(encoding="utf-8").splitlines():
            digest, relative_path = raw_line.split("  ", 1)
            self.assertEqual(len(digest), 64)
            int(digest, 16)
            self.assertNotIn(relative_path, checksums)
            checksums[relative_path] = digest

        profile_paths = sorted((REAL_PROFILE_ROOT / "platform_config").glob("*.ini"))
        expected_paths = {
            path.relative_to(REAL_PROFILE_ROOT).as_posix() for path in profile_paths
        }
        self.assertEqual(set(checksums), expected_paths)
        for relative_path, expected_digest in checksums.items():
            actual_digest = hashlib.sha256(
                (REAL_PROFILE_ROOT / relative_path).read_bytes()
            ).hexdigest()
            self.assertEqual(actual_digest, expected_digest, relative_path)


if __name__ == "__main__":
    unittest.main()
