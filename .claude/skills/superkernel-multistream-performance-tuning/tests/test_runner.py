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
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_execution  # noqa: E402
import multistream_event_stage_dispatch  # noqa: E402
import multistream_runner  # noqa: E402
import multistream_source_transform  # noqa: E402


class MultistreamRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.artifacts = self.root / "artifacts"
        self.leases = self.root / "leases"
        self.workspace.mkdir()
        self.artifacts.mkdir()
        self.action = self.artifacts / "action.json"
        self.action.write_text(
            json.dumps(
                {
                    "schema_version": multistream_execution.ACTION_SCHEMA,
                    "trial_id": "MS-O1",
                }
            )
            + "\n"
        )
        state = multistream_execution.initialize_state(
            "MS-O1", "sha256:request", self.action
        )
        state = multistream_execution.advance_state(state, "materialized")
        state = multistream_execution.advance_state(
            state, "diff_verified", "action.json"
        )
        self.state_path = self.artifacts / "state.json"
        self.state_path.write_text(json.dumps(state) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def _draft(self, *, failing_state=None, timeout_state=None, reject_state=None):
        phases = []
        for state in multistream_runner.PHASE_STATES:
            artifact = f"outputs/{state}.json"
            artifact_path = self.artifacts / artifact
            if timeout_state == state:
                command = [sys.executable, "-c", "import time; time.sleep(2)"]
                timeout = 0.05
            else:
                command = [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(artifact_path)!r}).parent.mkdir(parents=True, exist_ok=True); Path({str(artifact_path)!r}).write_text('ok')",
                ]
                timeout = 2
            validator = (
                [sys.executable, "-c", "raise SystemExit(7)"]
                if failing_state == state
                else [sys.executable, "-c", "raise SystemExit(10)"]
                if reject_state == state
                else [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; raise SystemExit(0 if Path({str(artifact_path)!r}).is_file() else 8)",
                ]
            )
            phases.append(
                {
                    "phase_id": state,
                    "state_after": state,
                    "argv": command,
                    "validator_argv": validator,
                    "cwd": "run",
                    "timeout_seconds": timeout,
                    "validator_timeout_seconds": 2,
                    "requires_device": state in multistream_runner.DEVICE_PHASE_STATES,
                    "environment_overrides": {},
                    "validator_exit_actions": (
                        {"0": "pass", "10": "reject"}
                        if state in {"clean3_passed", "clean5_passed"}
                        else {"0": "pass", "20": "block"}
                        if state == "analysis_validated"
                        else {"0": "pass"}
                    ),
                    "program_files": [
                        {
                            "path": str(Path(sys.executable).resolve()),
                            "file_fingerprint": multistream_runner.file_fingerprint(
                                Path(sys.executable).resolve()
                            ),
                        }
                    ],
                    "required_artifacts": [artifact],
                    "manifest": f"phases/{state}.json",
                }
            )
        (self.workspace / "run").mkdir(exist_ok=True)
        return {
            "schema_version": multistream_runner.PLAN_SCHEMA,
            "trial_id": "MS-O1",
            "request_fingerprint": "sha256:request",
            "workspace_root": str(self.workspace.resolve()),
            "artifact_root": str(self.artifacts.resolve()),
            "lease_root": str(self.leases.resolve()),
            "environment": {"PATH": os.environ.get("PATH", "")},
            "device_ids": [0],
            "lease_timeout_seconds": 0.15,
            "phases": phases,
        }

    def test_freeze_and_run_full_plan(self):
        plan = multistream_runner.freeze_plan(self._draft())

        result = multistream_runner.run_plan(
            plan, self.state_path, self.workspace, self.artifacts, self.leases
        )

        self.assertEqual(result["state"], "clean5_passed")
        state = json.loads(self.state_path.read_text())
        self.assertEqual(state["state"], "clean5_passed")
        for phase in plan["phases"]:
            manifest = json.loads((self.artifacts / phase["manifest"]).read_text())
            self.assertEqual(manifest["status"], "passed")
            if phase["requires_device"]:
                self.assertEqual(
                    manifest["environment"]["SUPERKERNEL_DEVICE_LEASE_HELD"], "1"
                )
                self.assertEqual(manifest["environment"]["SUPERKERNEL_DEVICE_IDS"], "0")
            else:
                self.assertNotIn(
                    "SUPERKERNEL_DEVICE_LEASE_HELD", manifest["environment"]
                )
            multistream_runner.validate_phase_manifest(
                manifest, plan, phase, self.artifacts
            )

    def test_completed_manifest_recovers_state_without_rerunning(self):
        plan = multistream_runner.freeze_plan(self._draft())
        first = plan["phases"][0]
        manifest = multistream_runner._run_phase(
            plan, first, self.workspace, self.artifacts, self.leases
        )
        self.assertEqual(manifest["status"], "passed")

        multistream_runner.run_plan(
            plan, self.state_path, self.workspace, self.artifacts, self.leases
        )

        state = json.loads(self.state_path.read_text())
        correctness_events = [
            event for event in state["events"] if event["state"] == "correctness_passed"
        ]
        self.assertEqual(len(correctness_events), 1)

    def test_reorder_plan_blocks_profile_until_dispatch_evidence_exists(self):
        draft = self._draft()
        draft["pre_profile_evidence"] = {
            "kind": "dispatch_order",
            "path": "dispatch/order-evidence.json",
        }
        plan = multistream_runner.freeze_plan(draft)

        with self.assertRaisesRegex(
            ValueError, "blocked until dispatch_order_evidence"
        ):
            multistream_runner.run_plan(
                plan, self.state_path, self.workspace, self.artifacts, self.leases
            )

        self.assertEqual(
            json.loads(self.state_path.read_text())["state"], "correctness_passed"
        )
        self.assertFalse((self.artifacts / "outputs/profile_collected.json").exists())

    def _event_stage_gate(self, candidate_ordinal):
        action = {
            "schema_version": multistream_source_transform.ACTION_MANIFEST_SCHEMA,
            "trial_id": "MS-O1",
            "action_id": "event-1",
            "expected_dispatch_change": {
                "kind": "event_notify_earlier",
                "logical_id": "aux.ready",
            },
        }
        self.action.write_text(json.dumps(action) + "\n")
        state = multistream_execution.initialize_state(
            "MS-O1", "sha256:request", self.action
        )
        state = multistream_execution.advance_state(state, "materialized")
        state = multistream_execution.advance_state(
            state, "diff_verified", "action.json"
        )
        self.state_path.write_text(json.dumps(state) + "\n")

        def row(logical_id, kind, role, ordinal):
            return {
                "logical_id": logical_id,
                "kind": kind,
                "stream_role": role,
                "dispatch_ordinal": ordinal,
            }

        occurrences = [
            {
                "alignment_id": f"decode-{index}",
                "baseline": [
                    row("main", "stage", "main", 0),
                    row("aux.ready", "event_notify", "aux", 3),
                ],
                "candidate": [
                    row("main", "stage", "main", 0),
                    row("aux.ready", "event_notify", "aux", candidate_ordinal),
                ],
            }
            for index in range(3)
        ]
        evidence = multistream_event_stage_dispatch.build(
            "MS-O1", "sha256:request", action, occurrences
        )
        evidence_path = self.artifacts / "dispatch/event-stage.json"
        evidence_path.parent.mkdir()
        evidence_path.write_text(json.dumps(evidence) + "\n")
        draft = self._draft()
        draft["pre_profile_evidence"] = {
            "kind": "event_stage_dispatch",
            "path": "dispatch/event-stage.json",
        }
        return multistream_runner.freeze_plan(draft)

    def test_event_stage_plan_runs_only_after_effective_dispatch_evidence(self):
        plan = self._event_stage_gate(candidate_ordinal=1)
        result = multistream_runner.run_plan(
            plan, self.state_path, self.workspace, self.artifacts, self.leases
        )
        self.assertEqual(result["state"], "clean5_passed")

    def test_event_stage_plan_blocks_profile_when_action_did_not_move(self):
        plan = self._event_stage_gate(candidate_ordinal=3)
        with self.assertRaisesRegex(ValueError, "not effective"):
            multistream_runner.run_plan(
                plan, self.state_path, self.workspace, self.artifacts, self.leases
            )
        self.assertEqual(
            json.loads(self.state_path.read_text())["state"], "correctness_passed"
        )

    def test_validator_failure_seals_logs_and_marks_failed(self):
        plan = multistream_runner.freeze_plan(
            self._draft(failing_state="correctness_passed")
        )

        with self.assertRaisesRegex(RuntimeError, "validator_fail"):
            multistream_runner.run_plan(
                plan, self.state_path, self.workspace, self.artifacts, self.leases
            )

        state = json.loads(self.state_path.read_text())
        self.assertEqual(state["state"], "failed")
        manifest = json.loads(
            (self.artifacts / "phases/correctness_passed.json").read_text()
        )
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["validator"]["return_code"], 7)
        self.assertIn(
            "outputs/correctness_passed.json",
            {item["path"] for item in manifest["sealed_files"]},
        )

    def test_clean3_no_gain_marks_rejected_without_running_clean5(self):
        plan = multistream_runner.freeze_plan(self._draft(reject_state="clean3_passed"))

        with self.assertRaisesRegex(RuntimeError, "validator_reject"):
            multistream_runner.run_plan(
                plan, self.state_path, self.workspace, self.artifacts, self.leases
            )

        self.assertEqual(json.loads(self.state_path.read_text())["state"], "rejected")
        self.assertFalse((self.artifacts / "phases/clean5_passed.json").exists())

    def test_timeout_terminates_process_group_and_marks_failed(self):
        plan = multistream_runner.freeze_plan(
            self._draft(timeout_state="correctness_passed")
        )

        with self.assertRaisesRegex(RuntimeError, "command_timeout"):
            multistream_runner.run_plan(
                plan, self.state_path, self.workspace, self.artifacts, self.leases
            )

        manifest = json.loads(
            (self.artifacts / "phases/correctness_passed.json").read_text()
        )
        self.assertTrue(manifest["command"]["timed_out"])
        self.assertEqual(json.loads(self.state_path.read_text())["state"], "failed")

    def test_process_start_failure_is_audited(self):
        draft = self._draft()
        non_executable = self.root / "not-executable"
        non_executable.write_text("not a program\n")
        draft["phases"][0]["argv"] = [str(non_executable)]
        draft["phases"][0]["program_files"].append(
            {
                "path": str(non_executable),
                "file_fingerprint": multistream_runner.file_fingerprint(non_executable),
            }
        )
        plan = multistream_runner.freeze_plan(draft)

        with self.assertRaisesRegex(RuntimeError, "process_start_failed"):
            multistream_runner.run_plan(
                plan, self.state_path, self.workspace, self.artifacts, self.leases
            )

        manifest = json.loads(
            (self.artifacts / "phases/correctness_passed.json").read_text()
        )
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(json.loads(self.state_path.read_text())["state"], "failed")

    def test_device_lease_contention_marks_blocked(self):
        plan = multistream_runner.freeze_plan(self._draft())

        with multistream_runner.device_leases([0], self.leases, 1):
            with self.assertRaisesRegex(RuntimeError, "device lease"):
                multistream_runner.run_plan(
                    plan, self.state_path, self.workspace, self.artifacts, self.leases
                )

        self.assertEqual(json.loads(self.state_path.read_text())["state"], "blocked")

    def test_plan_rejects_shell_string_and_secret_environment(self):
        draft = self._draft()
        draft["phases"][0]["argv"] = "python run.py"
        with self.assertRaisesRegex(ValueError, "argv array"):
            multistream_runner.freeze_plan(draft)

        draft = self._draft()
        draft["environment"]["API_TOKEN"] = "secret"
        with self.assertRaisesRegex(ValueError, "secret-like"):
            multistream_runner.freeze_plan(draft)

        draft = self._draft()
        draft["unsigned_extension"] = True
        with self.assertRaisesRegex(ValueError, "contain exactly"):
            multistream_runner.freeze_plan(draft)

    def test_partial_logs_without_manifest_fail_closed(self):
        plan = multistream_runner.freeze_plan(self._draft())
        log = (
            self.artifacts
            / multistream_runner._phase_log_paths(plan["phases"][0])["command_stdout"]
        )
        log.parent.mkdir(parents=True)
        log.write_text("partial")

        with self.assertRaisesRegex(ValueError, "pre-existing outputs"):
            multistream_runner.run_plan(
                plan, self.state_path, self.workspace, self.artifacts, self.leases
            )

        self.assertEqual(
            json.loads(self.state_path.read_text())["state"], "diff_verified"
        )

    def test_runtime_roots_must_match_the_frozen_plan(self):
        plan = multistream_runner.freeze_plan(self._draft())
        alternate = self.root / "alternate-leases"

        with self.assertRaisesRegex(ValueError, "runtime lease_root differs"):
            multistream_runner.run_plan(
                plan, self.state_path, self.workspace, self.artifacts, alternate
            )


if __name__ == "__main__":
    unittest.main()
