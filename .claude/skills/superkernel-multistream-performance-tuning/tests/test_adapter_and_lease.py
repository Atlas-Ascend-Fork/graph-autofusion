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


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
sys.path.insert(0, str(SKILL_ROOT.parent / "superkernel-runtime-common" / "scripts"))

import device_lease_runner  # noqa: E402
import multistream_adapter_generator  # noqa: E402
import multistream_plan_compiler  # noqa: E402
import multistream_runner  # noqa: E402


class AdapterAndLeaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.artifacts = self.root / "artifacts"
        self.leases = self.root / "leases"
        self.workspace.mkdir()
        self.artifacts.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _spec(self):
        def phase(inputs):
            return {
                "command_argv_template": ["/bin/true", "{candidate_config}"],
                "cwd_template": ".",
                "timeout_seconds": 30,
                "validator_timeout_seconds": 10,
                "environment_overrides": {},
                "program_file_templates": ["/bin/true"],
                "output_templates": [],
                "validator_inputs": inputs,
            }

        return {
            "schema_version": multistream_adapter_generator.SPEC_SCHEMA,
            "adapter_id": "model-a-v1",
            "workspace_root": str(self.workspace),
            "artifact_root": str(self.artifacts),
            "lease_root": str(self.leases),
            "environment": {"PATH": "/usr/bin"},
            "device_ids": [0],
            "lease_timeout_seconds": 30,
            "expected_ranks": 8,
            "warmup": 8,
            "phases": {
                "correctness": phase({"run_root": "trials/{trial_id}/correctness"}),
                "profile": phase(
                    {
                        "baseline_manifest": "baseline/manifest.json",
                        "candidate_manifest": "trials/{trial_id}/profile/manifest.json",
                        "trace_analysis": "trials/{trial_id}/trace-analysis.json",
                    }
                ),
                "analysis": phase(
                    {"analysis_result": "trials/{trial_id}/analysis.json"}
                ),
                "clean3": phase(
                    {
                        "baseline_root": "baseline/clean",
                        "candidate_root": "trials/{trial_id}/clean3",
                        "candidate_name": "{trial_id}",
                    }
                ),
                "clean5": phase(
                    {
                        "baseline_root": "baseline/clean",
                        "candidate_root": "trials/{trial_id}/clean5",
                        "candidate_name": "{trial_id}",
                    }
                ),
            },
        }

    def test_generator_builds_frozen_standard_adapter(self):
        adapter = multistream_adapter_generator.generate(self._spec())
        validated = multistream_plan_compiler.validate_adapter(adapter)
        self.assertEqual(validated["adapter_id"], "model-a-v1")
        self.assertEqual(
            [item["state_after"] for item in validated["phases"]],
            list(multistream_runner.PHASE_STATES),
        )
        self.assertIn(
            "multistream_evidence.py",
            validated["phases"][0]["validator_argv_template"][1],
        )
        self.assertIn(
            "--trace-analysis", validated["phases"][1]["validator_argv_template"]
        )
        self.assertEqual(
            validated["phases"][3]["validator_exit_actions"]["10"], "reject"
        )

    def test_parent_wrapper_uses_same_device_lease_protocol(self):
        manifest = self.root / "parent-command.json"
        with multistream_runner.device_leases([0], self.leases, 1):
            result = device_lease_runner.run_command(
                ["/bin/true"],
                lease_root=self.leases,
                device_ids=[0],
                timeout_seconds=0.05,
                cwd=self.workspace,
                manifest_out=manifest,
            )
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(manifest.is_file())

        passed = device_lease_runner.run_command(
            ["/bin/true"],
            lease_root=self.leases,
            device_ids=[0],
            timeout_seconds=1,
            cwd=self.workspace,
            manifest_out=self.root / "parent-command-passed.json",
        )
        self.assertEqual(passed["status"], "passed")


if __name__ == "__main__":
    unittest.main()
