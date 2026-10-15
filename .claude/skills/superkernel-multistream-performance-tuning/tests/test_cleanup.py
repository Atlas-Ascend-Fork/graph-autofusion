# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_cleanup  # noqa: E402


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.isolation = self.root / "isolated"
        self.incumbent = self.root / "incumbent"
        self.quarantine = self.root / "quarantine"
        (self.isolation / "source-trial").mkdir(parents=True)
        (self.isolation / "cache-trial").mkdir()
        (self.isolation / "artifacts").mkdir()
        self.incumbent.mkdir()
        (self.isolation / "source-trial" / "model.py").write_text("candidate = True\n")
        (self.isolation / "cache-trial" / "kernel.bin").write_bytes(b"cache")
        (self.isolation / "artifacts" / "result.json").write_text("{}\n")
        (self.incumbent / "model.py").write_text("incumbent = True\n")

    def tearDown(self):
        self.temporary.cleanup()

    def _draft(self, terminal_state="rejected"):
        return {
            "schema_version": multistream_cleanup.PLAN_SCHEMA,
            "cleanup_id": "cleanup-MS-R1",
            "trial_id": "MS-R1",
            "terminal_state": terminal_state,
            "isolation_root": str(self.isolation),
            "incumbent_root": str(self.incumbent),
            "quarantine_root": str(self.quarantine),
            "targets": ["source-trial", "cache-trial"],
            "preserved_paths": ["artifacts"],
        }

    def test_every_terminal_state_is_supported_and_incumbent_is_unchanged(self):
        for terminal_state in sorted(multistream_cleanup.TERMINAL_STATES):
            with self.subTest(terminal_state=terminal_state):
                if not (self.isolation / "source-trial").exists():
                    (self.isolation / "source-trial").mkdir()
                    (self.isolation / "source-trial" / "model.py").write_text(
                        "candidate = True\n"
                    )
                    (self.isolation / "cache-trial").mkdir()
                    (self.isolation / "cache-trial" / "kernel.bin").write_bytes(
                        b"cache"
                    )
                draft = self._draft(terminal_state)
                draft["cleanup_id"] = f"cleanup-{terminal_state}"
                plan = multistream_cleanup.freeze_plan(draft)
                receipt = multistream_cleanup.run(
                    plan, self.root / f"{terminal_state}.json"
                )
                self.assertEqual(receipt["terminal_state"], terminal_state)
                self.assertTrue(
                    (self.isolation / "artifacts" / "result.json").is_file()
                )
                self.assertEqual(
                    (self.incumbent / "model.py").read_text(), "incumbent = True\n"
                )

    def test_run_is_idempotent(self):
        plan = multistream_cleanup.freeze_plan(self._draft())
        receipt_path = self.root / "receipt.json"
        first = multistream_cleanup.run(plan, receipt_path)
        second = multistream_cleanup.run(plan, receipt_path)
        self.assertEqual(first, second)
        self.assertTrue(multistream_cleanup.validate_receipt(second, plan)["valid"])

    def test_retry_recovers_after_atomic_move_failure(self):
        plan = multistream_cleanup.freeze_plan(self._draft())
        receipt_path = self.root / "receipt.json"
        real_replace = multistream_cleanup.os.replace
        calls = 0

        def fail_second_source_move(source, destination):
            nonlocal calls
            if Path(source).name in {"source-trial", "cache-trial"}:
                calls += 1
                if calls == 2:
                    raise OSError("injected move failure")
            return real_replace(source, destination)

        with mock.patch.object(
            multistream_cleanup.os, "replace", side_effect=fail_second_source_move
        ):
            with self.assertRaisesRegex(OSError, "injected"):
                multistream_cleanup.run(plan, receipt_path)
        self.assertFalse(receipt_path.exists())
        receipt = multistream_cleanup.run(plan, receipt_path)
        self.assertTrue(multistream_cleanup.validate_receipt(receipt, plan)["valid"])

    def test_rejects_incumbent_change_and_overlapping_roots(self):
        plan = multistream_cleanup.freeze_plan(self._draft())
        (self.incumbent / "model.py").write_text("tampered = True\n")
        with self.assertRaisesRegex(ValueError, "incumbent snapshot"):
            multistream_cleanup.run(plan, self.root / "receipt.json")
        draft = self._draft()
        draft["incumbent_root"] = str(self.isolation / "artifacts")
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            multistream_cleanup.freeze_plan(draft)

    def test_rejects_symlink_and_target_preserve_overlap(self):
        link = self.isolation / "linked"
        link.symlink_to(self.incumbent, target_is_directory=True)
        draft = self._draft()
        draft["targets"] = ["linked"]
        with self.assertRaisesRegex(ValueError, "symlink"):
            multistream_cleanup.freeze_plan(draft)
        draft = self._draft()
        draft["preserved_paths"] = ["source-trial/model.py"]
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            multistream_cleanup.freeze_plan(draft)


if __name__ == "__main__":
    unittest.main()
