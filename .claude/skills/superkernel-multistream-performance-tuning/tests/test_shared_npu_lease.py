# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


MULTISTREAM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MULTISTREAM_ROOT.parent
sys.path.insert(0, str(MULTISTREAM_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "superkernel-runtime-common" / "scripts"))

import device_lease_runner  # noqa: E402
import multistream_runner  # noqa: E402
import shared_npu_lease_inventory  # noqa: E402


class SharedNpuLeaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.leases = self.root / "leases"
        self.workspace.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_parent_wrapper_injects_canonical_child_markers(self):
        output = self.root / "child-environment.json"
        program = (
            "import json,os,pathlib; "
            f"pathlib.Path({str(output)!r}).write_text(json.dumps({{k: os.environ[k] for k in "
            "['SUPERKERNEL_DEVICE_LEASE_HELD','SUPERKERNEL_DEVICE_IDS','SUPERKERNEL_DEVICE_LEASE_ROOT']}))"
        )
        result = device_lease_runner.run_command(
            [sys.executable, "-c", program],
            lease_root=self.leases,
            device_ids=[1, 0],
            timeout_seconds=2,
            cwd=self.workspace,
            manifest_out=self.root / "manifest.json",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["lease_mode"], "acquired")
        self.assertEqual(
            json.loads(output.read_text()),
            {
                "SUPERKERNEL_DEVICE_LEASE_HELD": "1",
                "SUPERKERNEL_DEVICE_IDS": "0,1",
                "SUPERKERNEL_DEVICE_LEASE_ROOT": str(self.leases.resolve()),
            },
        )

    def test_same_device_parent_commands_do_not_overlap(self):
        ready = self.root / "ready"
        first_result = {}

        def first():
            program = f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(0.3)"
            first_result.update(
                device_lease_runner.run_command(
                    [sys.executable, "-c", program],
                    lease_root=self.leases,
                    device_ids=[0],
                    timeout_seconds=2,
                    cwd=self.workspace,
                    manifest_out=self.root / "first.json",
                )
            )

        thread = threading.Thread(target=first)
        thread.start()
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        second = device_lease_runner.run_command(
            ["/bin/true"],
            lease_root=self.leases,
            device_ids=[0],
            timeout_seconds=1,
            lease_timeout_seconds=0.05,
            cwd=self.workspace,
            manifest_out=self.root / "second.json",
        )
        thread.join()
        self.assertEqual(first_result["status"], "passed")
        self.assertEqual(second["status"], "blocked")

    def test_matching_parent_is_inherited_and_conflicts_are_rejected(self):
        markers = multistream_runner.lease_environment({}, [0], self.leases)
        with mock.patch.dict(os.environ, markers, clear=False):
            with multistream_runner.shared_device_leases(
                [0], self.leases, 0.01
            ) as mode:
                self.assertEqual(mode, "inherited")
        conflict = {**markers, "SUPERKERNEL_DEVICE_IDS": "1"}
        with self.assertRaisesRegex(ValueError, "marker mismatch"):
            multistream_runner.parent_lease_held([0], self.leases, conflict)

    def test_inventory_covers_all_launches_and_blocks_unknown_call(self):
        report = shared_npu_lease_inventory.build_inventory(REPOSITORY_ROOT)
        self.assertEqual(report["status"], "passed", report["blockers"])
        self.assertTrue(
            shared_npu_lease_inventory.validate_inventory(report, REPOSITORY_ROOT)[
                "valid"
            ]
        )
        original = shared_npu_lease_inventory._discover_calls

        def unknown(root):
            calls, sources = original(root)
            calls.append(
                {
                    "path": "new_launcher.py",
                    "function": "run",
                    "call": "subprocess.run",
                    "line": 1,
                }
            )
            return calls, sources

        with mock.patch.object(
            shared_npu_lease_inventory, "_discover_calls", side_effect=unknown
        ):
            blocked = shared_npu_lease_inventory.build_inventory(REPOSITORY_ROOT)
        self.assertEqual(blocked["status"], "blocked")
        self.assertTrue(
            any("unregistered process launch" in item for item in blocked["blockers"])
        )


if __name__ == "__main__":
    unittest.main()
