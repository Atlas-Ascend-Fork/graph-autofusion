# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_four_profile  # noqa: E402
import multistream_runner  # noqa: E402


class FourProfileTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.artifacts = self.root / "artifacts"
        self.leases = self.root / "leases"
        for path in (self.workspace, self.artifacts, self.leases):
            path.mkdir()
        (self.workspace / "run").mkdir()
        self.helper = self.workspace / "profile_helper.py"
        self.helper.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "mode, path, role = sys.argv[1:]\n"
            "path = Path(path)\n"
            "if mode == 'write':\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text(json.dumps({'role': role}) + '\\n')\n"
            "elif json.loads(path.read_text())['role'] != role:\n"
            "    raise SystemExit(2)\n"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _draft(self):
        programs = [
            {
                "path": str(Path(sys.executable).resolve()),
                "file_fingerprint": multistream_runner.file_fingerprint(
                    Path(sys.executable).resolve()
                ),
            },
            {
                "path": str(self.helper.resolve()),
                "file_fingerprint": multistream_runner.file_fingerprint(self.helper),
            },
        ]
        profiles = []
        for role in multistream_four_profile.ROLES:
            output = self.artifacts / "profiles" / role / "profile.json"
            profiles.append(
                {
                    "role": role,
                    "argv": [
                        sys.executable,
                        str(self.helper),
                        "write",
                        str(output),
                        role,
                    ],
                    "validator_argv": [
                        sys.executable,
                        str(self.helper),
                        "validate",
                        str(output),
                        role,
                    ],
                    "cwd": "run",
                    "timeout_seconds": 2,
                    "validator_timeout_seconds": 2,
                    "environment_overrides": {},
                    "program_files": programs,
                    "required_artifacts": [f"profiles/{role}/profile.json"],
                    "manifest": f"profiles/{role}/manifest.json",
                }
            )
        return {
            "schema_version": multistream_four_profile.PLAN_SCHEMA,
            "plan_id": "four-profile-1",
            "trial_id": "trial-1",
            "request_fingerprint": "request-fp-1",
            "workspace_root": str(self.workspace),
            "artifact_root": str(self.artifacts),
            "lease_root": str(self.leases),
            "device_ids": [0],
            "environment": {"PATH": os.environ.get("PATH", "")},
            "profiles": profiles,
            "summary": "four-profile-summary.json",
        }

    def _lease_environment(self):
        return {
            "SUPERKERNEL_DEVICE_LEASE_HELD": "1",
            "SUPERKERNEL_DEVICE_IDS": "0",
            "SUPERKERNEL_DEVICE_LEASE_ROOT": str(self.leases.resolve()),
        }

    def test_runs_all_roles_and_builds_replayable_summary(self):
        plan = multistream_four_profile.freeze_plan(self._draft())
        with mock.patch.dict(os.environ, self._lease_environment(), clear=False):
            summary = multistream_four_profile.run_plan(plan)
            replay = multistream_four_profile.run_plan(plan)
        self.assertEqual(set(summary["roles"]), set(multistream_four_profile.ROLES))
        self.assertEqual(summary, replay)
        self.assertEqual(
            summary["comparisons"]["candidate_fusion"],
            ["candidate_sk_off", "candidate_sk_on"],
        )
        validated = multistream_four_profile.validate_summary(
            self.artifacts / "four-profile-summary.json", plan, self.artifacts
        )
        self.assertTrue(validated["complete"])

    def test_refuses_to_run_without_parent_lease_marker(self):
        plan = multistream_four_profile.freeze_plan(self._draft())
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "lacks matching parent lease marker"
            ):
                multistream_four_profile.run_plan(plan)

    def test_profile_failure_stops_before_later_roles(self):
        draft = self._draft()
        draft["profiles"][1]["argv"] = [sys.executable, "-c", "raise SystemExit(3)"]
        plan = multistream_four_profile.freeze_plan(draft)
        with mock.patch.dict(os.environ, self._lease_environment(), clear=False):
            with self.assertRaisesRegex(ValueError, "incumbent_sk_on"):
                multistream_four_profile.run_plan(plan)
        self.assertTrue(
            (self.artifacts / "profiles/incumbent_sk_on/manifest.json").is_file()
        )
        self.assertFalse(
            (self.artifacts / "profiles/candidate_sk_off/manifest.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
