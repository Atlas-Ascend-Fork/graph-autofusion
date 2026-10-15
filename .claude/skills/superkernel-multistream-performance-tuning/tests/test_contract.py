# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
ADAPTATION_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "superkernel-runtime-common" / "scripts"
)
sys.path.insert(0, str(ADAPTATION_SCRIPTS))

import multistream_contract  # noqa: E402
import multistream_execution  # noqa: E402
import multistream_runner  # noqa: E402
import multistream_plan_compiler  # noqa: E402
import multistream_evidence  # noqa: E402
import multistream_trace_analysis  # noqa: E402
import multistream_candidate_planner  # noqa: E402
import bootstrap_derived_family  # noqa: E402
import derived_family_lifecycle  # noqa: E402


class MultistreamContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in (
            "incumbent-clean.json",
            "incumbent-analysis.json",
            "baseline-manifest.json",
            "candidate-manifest.json",
            "projection.json",
            "environment.json",
            "option-history.json",
            "accepted-option.json",
            "selected-config.json",
            "selected-config.yaml",
            "source-manifest.json",
            "action-manifest.json",
            "execution-state.json",
            "selected-analysis.json",
            "selected-clean.json",
        ):
            (self.root / relative).write_text("{}\n")
        (self.root / "source").mkdir()
        (self.root / "source" / "model.py").write_text("def run():\n    return 1\n")
        (self.root / "semantic-evidence-source.txt").write_text("fixture\n")
        self.source_file_fingerprint = multistream_execution.file_fingerprint(
            self.root / "source" / "model.py"
        )
        self.source_fingerprint = multistream_execution.content_fingerprint(
            {
                "files": [
                    {
                        "path": "source/model.py",
                        "file_fingerprint": self.source_file_fingerprint,
                    }
                ]
            }
        )
        self.request = self._request()
        self._write_analysis()
        (self.root / "selected-config.yaml").write_text("option: 1\n")

    def tearDown(self):
        self.temporary.cleanup()

    def _request(self):
        return {
            "schema_version": "superkernel-multistream-request-v1",
            "request_id": "S3-MS-REQUEST-1",
            "parent_experiment_id": "S3",
            "incumbent": {
                "candidate_name": "S3-BASE",
                "source_revision": "source-r1",
                "source_fingerprint": self.source_fingerprint,
                "config_fingerprint": "config-fp-1",
                "control_fingerprint": "control-fp-1",
                "workload_fingerprint": "workload-fp-1",
                "clean_performance_summary": "incumbent-clean.json",
                "profiling_analysis_result": "incumbent-analysis.json",
                "clean_run_count": 5,
                "stable": True,
            },
            "artifacts": {
                "baseline_collection_manifest": "baseline-manifest.json",
                "candidate_collection_manifest": "candidate-manifest.json",
                "projected_trace_mapping": "projection.json",
                "environment_evidence": "environment.json",
                "option_history": "option-history.json",
            },
            "targets": [
                {
                    "range_id": "range-cv-1",
                    "graph_occurrence_fingerprint": "occurrence-fp-1",
                    "net_effect": "beneficial",
                    "parallelism_effect": "degraded",
                    "optimization_status": "opportunity",
                }
            ],
            "requested_change_kinds": ["option"],
            "isolation": {
                "source_worktree": "isolated/source",
                "experiment_root": "isolated/experiments",
                "config_root": "isolated/configs",
                "cache_root": "isolated/cache",
                "immutable_incumbent": True,
                "dedicated_source_worktree": True,
                "dedicated_experiment_root": True,
                "dedicated_config_root": True,
                "dedicated_cache_namespace": True,
            },
            "authorization": {
                "run_inference": True,
                "edit_isolated_worktree": True,
            },
            "execution": {
                "command_argv": ["python3", "run_model.py", "--config", "{config}"],
                "correctness_method": "compare token output with frozen reference",
                "expected_ranks": 8,
                "warmup": 8,
            },
            "budget": {
                "max_trials": 4,
                "max_recollections_per_trial": 3,
                "timeout_seconds": 1800,
            },
        }

    def _write_analysis(self, *, source_exact=False):
        decision = {
            "range_id": "range-cv-1",
            "graph_occurrence_fingerprint": "occurrence-fp-1",
            "classification": "beneficial",
            "mapping_method": (
                "source_scope_map" if source_exact else "kernel_projection_structural"
            ),
            "mapping_confidence": (
                "exact" if source_exact else "exact_projected_trace"
            ),
            "candidate_occurrence_count": 3,
            "original": {"interval_us": {"count": 3}},
            "boundary": (
                {
                    "source_file": "model.py",
                    "start_offset": 10,
                    "end_offset": 20,
                }
                if source_exact
                else {}
            ),
        }
        analysis = {
            "schema_version": "1.2",
            "candidate_name": self.request["incumbent"]["candidate_name"],
            "source_revision": self.request["incumbent"]["source_revision"],
            "candidate_config_fingerprint": self.request["incumbent"][
                "config_fingerprint"
            ],
            "control_fingerprint": self.request["incumbent"]["control_fingerprint"],
            "workload_fingerprint": self.request["incumbent"]["workload_fingerprint"],
            "source_scope_mapping": {
                "protocol": "source_scope_map_v2",
                "status": "exact" if source_exact else "not_requested",
            },
            "per_sk_decisions": [decision],
        }
        analysis["analysis_content_fingerprint"] = (
            multistream_contract._analysis_fingerprint(analysis)
        )
        (self.root / "incumbent-analysis.json").write_text(json.dumps(analysis) + "\n")

    def _write_source_map(self):
        source_map = {
            "schema_version": "2.1",
            "protocol": "source_scope_map_v2",
            "provenance": {
                "source_revision": self.request["incumbent"]["source_revision"],
            },
            "source_ranges": [
                {
                    "sk_occurrence_fingerprint": "occurrence-fp-1",
                    "relation": "exact_cover",
                    "boundary": {
                        "source_file": "model.py",
                        "start_offset": 10,
                        "end_offset": 20,
                    },
                }
            ],
        }
        source_map["provenance"]["source_scope_map_content_fingerprint"] = (
            multistream_contract._source_map_fingerprint(source_map)
        )
        (self.root / "source-map.json").write_text(json.dumps(source_map) + "\n")

    def _fallback(self, status="blocked"):
        return multistream_contract.build_fallback(
            self.request,
            self.root,
            status,
            "多流调优证据不足，保持原始候选。",
        )

    def _accepted_option_result(self):
        fingerprint = multistream_contract.content_fingerprint(self.request)
        config_fingerprint = multistream_execution.content_fingerprint({"option": 1})
        action = {
            "schema_version": multistream_execution.ACTION_SCHEMA,
            "trial_id": "MS-O1",
            "change_kind": "option",
            "output_content_fingerprint": config_fingerprint,
            "only_change": {
                "json_pointer": (
                    "/model_config/custom_params/super_kernel_optimize_options/"
                    "auto_op_parallel"
                ),
                "before": 0,
                "after": 1,
            },
            "changed_pointers": [
                "/model_config/custom_params/super_kernel_optimize_options/auto_op_parallel"
            ],
            "single_change_verified": True,
        }
        (self.root / "action-manifest.json").write_text(json.dumps(action) + "\n")
        adapter = multistream_plan_compiler.freeze_adapter(self._model_adapter_draft())
        (self.root / "model-adapter.json").write_text(json.dumps(adapter) + "\n")
        plan, compilation = multistream_plan_compiler.compile_plan(
            self.request,
            adapter,
            action,
            artifact_root=self.root,
            candidate_config="selected-config.yaml",
        )
        (self.root / "execution-plan.json").write_text(json.dumps(plan) + "\n")
        (self.root / "plan-compilation.json").write_text(json.dumps(compilation) + "\n")
        (self.root / "workspace").mkdir(exist_ok=True)
        for phase in plan["phases"]:
            artifact_path = self.root / phase["required_artifacts"][0]
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            kind = multistream_evidence.STATE_KINDS[phase["state_after"]]
            semantic_evidence = multistream_evidence._finish(
                kind,
                phase["state_after"],
                "MS-O1",
                fingerprint,
                {"fixture": True},
                multistream_evidence._source_records(
                    self.root, [self.root / "semantic-evidence-source.txt"]
                ),
                {"fixture": True},
                "pass",
            )
            artifact_path.write_text(json.dumps(semantic_evidence) + "\n")
            sealed = [
                {
                    "path": phase["required_artifacts"][0],
                    "size_bytes": artifact_path.stat().st_size,
                    "fingerprint": multistream_runner.file_fingerprint(artifact_path),
                }
            ]
            for relative in multistream_runner._phase_log_paths(phase).values():
                log_path = self.root / relative
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("")
                sealed.append(
                    {
                        "path": relative,
                        "size_bytes": 0,
                        "fingerprint": multistream_runner.file_fingerprint(log_path),
                    }
                )
            phase_manifest = {
                "schema_version": multistream_runner.PHASE_SCHEMA,
                "trial_id": "MS-O1",
                "request_fingerprint": fingerprint,
                "plan_fingerprint": plan["plan_fingerprint"],
                "phase_id": phase["phase_id"],
                "state_after": phase["state_after"],
                "status": "passed",
                "reason": None,
                "cwd": phase["cwd"],
                "workspace_root": plan["workspace_root"],
                "artifact_root": plan["artifact_root"],
                "lease_root": plan["lease_root"],
                "environment": multistream_runner._phase_environment(plan, phase),
                "device_ids": (plan["device_ids"] if phase["requires_device"] else []),
                "command": {
                    "argv": phase["argv"],
                    "pid": 101,
                    "started_at": "2026-08-28T00:00:00Z",
                    "finished_at": "2026-08-28T00:00:01Z",
                    "return_code": 0,
                    "timed_out": False,
                },
                "validator": {
                    "argv": phase["validator_argv"],
                    "pid": 102,
                    "started_at": "2026-08-28T00:00:01Z",
                    "finished_at": "2026-08-28T00:00:02Z",
                    "return_code": 0,
                    "timed_out": False,
                },
                "sealed_files": sealed,
                "completed_at": "2026-08-28T00:00:02Z",
            }
            (self.root / phase["manifest"]).write_text(
                json.dumps(phase_manifest) + "\n"
            )
        state = multistream_execution.initialize_state(
            "MS-O1", fingerprint, self.root / "action-manifest.json"
        )
        for target in multistream_execution.ORDERED_STATES[1:]:
            phase = next(
                (item for item in plan["phases"] if item["state_after"] == target),
                None,
            )
            state = multistream_execution.advance_state(
                state,
                target,
                (
                    None
                    if target == "materialized"
                    else "action-manifest.json"
                    if target == "diff_verified"
                    else phase["manifest"]
                    if phase is not None
                    else "accepted-option.json"
                ),
            )
        (self.root / "execution-state.json").write_text(json.dumps(state) + "\n")
        source_manifest = {
            "schema_version": "superkernel-source-snapshot-manifest-v1",
            "source_revision": "source-r1",
            "source_fingerprint": self.source_fingerprint,
            "files": [
                {
                    "path": "source/model.py",
                    "file_fingerprint": self.source_file_fingerprint,
                }
            ],
        }
        (self.root / "source-manifest.json").write_text(
            json.dumps(source_manifest) + "\n"
        )
        return {
            "schema_version": "superkernel-multistream-result-v2",
            "request_id": self.request["request_id"],
            "parent_experiment_id": self.request["parent_experiment_id"],
            "request_fingerprint": fingerprint,
            "status": "accepted",
            "incumbent_unchanged": False,
            "selected_candidate": {
                "trial_id": "MS-O1",
                "candidate_name": "S3-MBASE",
                "derived_experiment_id": "S3M",
                "source_revision": "source-r1",
                "source_fingerprint": self.source_fingerprint,
                "config_fingerprint": config_fingerprint,
                "control_fingerprint": "control-fp-1",
                "workload_fingerprint": "workload-fp-1",
                "config_manifest": "action-manifest.json",
                "config_snapshot": "selected-config.yaml",
                "source_manifest": "source-manifest.json",
                "profiling_analysis_result": "selected-analysis.json",
                "clean_performance_summary": "selected-clean.json",
            },
            "trials": [
                {
                    "trial_id": "MS-O1",
                    "target_range_id": "range-cv-1",
                    "change_kind": "option",
                    "decision": "accepted",
                    "only_change": {
                        "json_pointer": (
                            "/model_config/custom_params/super_kernel_optimize_options/"
                            "auto_op_parallel"
                        ),
                        "before": 0,
                        "after": 1,
                        "accepted_evidence": "accepted-option.json",
                    },
                    "action_manifest": "action-manifest.json",
                    "execution_state": "execution-state.json",
                    "execution_plan": "execution-plan.json",
                    "model_adapter": "model-adapter.json",
                    "plan_compilation": "plan-compilation.json",
                    "gates": {
                        "correctness": "passed",
                        "profiling": "passed",
                        "clean_performance": "passed",
                    },
                    "clean_metrics": {
                        "run_count": 5,
                        "stable": True,
                        "incremental_mean_gain": True,
                        "positive_median_direction": True,
                        "p90_non_regression": True,
                        "stddev_non_regression": True,
                    },
                    "mechanism_status": "unproven",
                }
            ],
            "multistream_findings": [
                {
                    "range_id": "range-cv-1",
                    "net_effect": "beneficial",
                    "parallelism_effect": "unknown",
                    "optimization_status": "validated",
                }
            ],
            "blockers": [],
            "fallback": {
                "candidate_name": "S3-BASE",
                "source_revision": "source-r1",
                "source_fingerprint": self.source_fingerprint,
                "config_fingerprint": "config-fp-1",
                "control_fingerprint": "control-fp-1",
                "workload_fingerprint": "workload-fp-1",
                "reason_zh": "若上层复核失败，保持原始候选。",
            },
        }

    def _model_adapter_draft(self):
        phases = []
        for state in multistream_runner.PHASE_STATES:
            common_validator = [
                "{python_executable}",
                "{skill_root}/scripts/multistream_evidence.py",
            ]
            common_options = [
                "--artifact-root",
                "{artifact_root}",
                "--state-after",
                state,
                "--trial-id",
                "{trial_id}",
                "--request-fingerprint",
                "{request_fingerprint}",
                "--out",
                "{artifact_root}/phase-artifacts/" + state + ".json",
            ]
            if state == "correctness_passed":
                validator = (
                    common_validator
                    + ["correctness"]
                    + common_options
                    + [
                        "--run-root",
                        "runs/correctness",
                        "--expected-ranks",
                        "1",
                    ]
                )
            elif state == "profile_collected":
                validator = (
                    common_validator
                    + ["profile"]
                    + common_options
                    + [
                        "--baseline-manifest",
                        "profiles/baseline/collection-manifest.json",
                        "--candidate-manifest",
                        "profiles/candidate/collection-manifest.json",
                    ]
                )
            elif state == "analysis_validated":
                validator = (
                    common_validator
                    + ["analysis"]
                    + common_options
                    + [
                        "--analysis-result",
                        "analysis/result.json",
                    ]
                )
            else:
                expected_runs = "3" if state == "clean3_passed" else "5"
                validator = (
                    common_validator
                    + ["clean"]
                    + common_options
                    + [
                        "--baseline-root",
                        "runs/baseline",
                        "--candidate-root",
                        "runs/candidate",
                        "--candidate-name",
                        "{trial_id}",
                        "--expected-ranks",
                        "1",
                        "--warmup",
                        "0",
                        "--expected-runs",
                        expected_runs,
                    ]
                )
            phases.append(
                {
                    "state_after": state,
                    "phase_id_template": state,
                    "argv_template": [
                        "/bin/true",
                        "{candidate_config}",
                        "{artifact_root}/{trial_id}/" + state,
                    ],
                    "validator_argv_template": validator,
                    "cwd_template": "workspace",
                    "timeout_seconds": 10,
                    "validator_timeout_seconds": 10,
                    "environment_overrides": {},
                    "validator_exit_actions": (
                        {"0": "pass", "10": "reject"}
                        if state in {"clean3_passed", "clean5_passed"}
                        else {"0": "pass", "20": "block"}
                        if state == "analysis_validated"
                        else {"0": "pass"}
                    ),
                    "program_file_templates": ["/bin/true"],
                    "required_artifact_templates": [
                        "phase-artifacts/" + state + ".json"
                    ],
                    "manifest_template": ("phase-manifests/" + state + ".json"),
                }
            )
        return {
            "schema_version": multistream_plan_compiler.ADAPTER_SCHEMA,
            "adapter_id": "deepseek-test-v1",
            "workspace_root": str((self.root / "workspace").resolve()),
            "artifact_root": str(self.root.resolve()),
            "lease_root": str((self.root / "leases").resolve()),
            "environment": {},
            "device_ids": [0],
            "lease_timeout_seconds": 10,
            "phases": phases,
        }

    def _refresh_compilation_for_action(self, action):
        path = self.root / "plan-compilation.json"
        compilation = json.loads(path.read_text())
        compilation["action_manifest_fingerprint"] = (
            multistream_plan_compiler.content_fingerprint(action)
        )
        compilation.pop("compilation_fingerprint")
        compilation["compilation_fingerprint"] = (
            multistream_plan_compiler.content_fingerprint(compilation)
        )
        path.write_text(json.dumps(compilation) + "\n")

    def _trace_capture(self, *, overflow=False, candidate_mode="serialized"):
        raw = self.root / "short-trace.json"
        raw.write_text('{"traceEvents": []}\n')

        def occurrences(mode):
            result = []
            for step in range(3):
                vector_start = {
                    "baseline": 5.0,
                    "serialized": 10.0,
                    "improved": 0.0,
                }[mode]
                result.append(
                    {
                        "alignment_id": f"decode-{step}",
                        "step_id": step,
                        "events": [
                            {
                                "child_origin_identity": "node-cube",
                                "lane_id": "aic-0",
                                "stream_id": 1,
                                "accelerator_core": "MIX_AIC",
                                "block_num": 22,
                                "mix_block_num": 0,
                                "start_us": 0.0,
                                "duration_us": 10.0,
                                "event_kind": "kernel",
                            },
                            {
                                "child_origin_identity": "node-vector",
                                "lane_id": "aiv-0",
                                "stream_id": 2,
                                "accelerator_core": "MIX_AIV",
                                "block_num": 32,
                                "mix_block_num": 0,
                                "start_us": vector_start,
                                "duration_us": 6.0,
                                "event_kind": "kernel",
                            },
                        ],
                    }
                )
            return result

        capture = {
            "schema_version": multistream_trace_analysis.CAPTURE_SCHEMA,
            "capture_id": "capture-1",
            "trial_id": "MS-O1",
            "request_fingerprint": multistream_contract.content_fingerprint(
                self.request
            ),
            "overflow_detected": overflow,
            "source_files": [
                {
                    "path": "short-trace.json",
                    "size_bytes": raw.stat().st_size,
                    "file_fingerprint": multistream_trace_analysis.file_fingerprint(
                        raw
                    ),
                }
            ],
            "targets": [
                {
                    "range_id": "range-cv-1",
                    "graph_occurrence_fingerprint": "occurrence-fp-1",
                    "parent_identity": {
                        "device_id": 0,
                        "model_id": 48,
                        "parent_sk_id": 732,
                    },
                    "baseline_occurrences": occurrences("baseline"),
                    "candidate_occurrences": occurrences(candidate_mode),
                }
            ],
        }
        capture["capture_fingerprint"] = multistream_trace_analysis.fingerprint(capture)
        path = self.root / "short-trace-capture.json"
        path.write_text(json.dumps(capture) + "\n")
        return path

    def test_option_only_request_does_not_require_source_map(self):
        summary = multistream_contract.validate_request(self.request, self.root)
        self.assertTrue(summary["valid"])
        self.assertFalse(summary["source_scope_map_available"])

    def test_short_trace_aligns_three_occurrences_and_preserves_net_effect(self):
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        analysis = multistream_trace_analysis.analyze(
            request_path, self._trace_capture(), self.root
        )
        self.assertEqual(analysis["targets"][0]["aligned_occurrence_count"], 3)
        self.assertEqual(analysis["targets"][0]["parallelism_effect"], "degraded")
        self.assertEqual(analysis["targets"][0]["net_effect"], "beneficial")
        self.assertEqual(analysis["targets"][0]["optimization_status"], "opportunity")
        self.assertTrue(analysis["targets"][0]["latent_opportunity"])
        self.assertTrue(analysis["targets"][0]["actionable_opportunity"])
        self.assertEqual(
            analysis["targets"][0]["opportunity_kind"],
            "beneficial_with_degraded_parallelism",
        )
        self.assertEqual(
            analysis["targets"][0]["diagnostic_decomposition"][
                "lost_cube_vector_overlap_us"
            ],
            5.0,
        )
        self.assertEqual(
            analysis["targets"][0]["diagnostic_decomposition"][
                "cube_vector_overlap_retention_ratio"
            ],
            0.0,
        )
        output = self.root / "trace-analysis.json"
        output.write_text(json.dumps(analysis) + "\n")
        replay = multistream_trace_analysis.validate_analysis(
            output, request_path, self.root, expected_trial_id="MS-O1"
        )
        self.assertTrue(replay["valid"])

    def test_short_trace_overflow_blocks_parallelism_claim(self):
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        analysis = multistream_trace_analysis.analyze(
            request_path, self._trace_capture(overflow=True), self.root
        )
        self.assertEqual(analysis["targets"][0]["parallelism_effect"], "unknown")
        self.assertIn("range-cv-1:trace_overflow", analysis["blockers"])

    def test_lost_mix_overlap_is_degraded_but_not_actionable(self):
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        capture_path = self._trace_capture()
        capture = json.loads(capture_path.read_text())
        for occurrence in (
            capture["targets"][0]["baseline_occurrences"]
            + capture["targets"][0]["candidate_occurrences"]
        ):
            occurrence["events"][0]["mix_block_num"] = 22
        capture.pop("capture_fingerprint")
        capture["capture_fingerprint"] = multistream_trace_analysis.fingerprint(capture)
        capture_path.write_text(json.dumps(capture) + "\n")

        analysis = multistream_trace_analysis.analyze(
            request_path, capture_path, self.root
        )

        target = analysis["targets"][0]
        self.assertEqual(target["parallelism_effect"], "degraded")
        self.assertEqual(target["optimization_status"], "blocked")
        self.assertTrue(target["latent_opportunity"])
        self.assertFalse(target["actionable_opportunity"])
        self.assertEqual(
            target["actionability_blocker"],
            "degraded_overlap_not_resource_complementary",
        )

    def test_short_trace_rejects_changed_child_identity_set(self):
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        capture_path = self._trace_capture()
        capture = json.loads(capture_path.read_text())
        for occurrence in capture["targets"][0]["candidate_occurrences"]:
            occurrence["events"][1]["child_origin_identity"] = "node-vector-other"
        capture.pop("capture_fingerprint")
        capture["capture_fingerprint"] = multistream_trace_analysis.fingerprint(capture)
        capture_path.write_text(json.dumps(capture) + "\n")

        with self.assertRaisesRegex(ValueError, "child origin identity set differs"):
            multistream_trace_analysis.analyze(request_path, capture_path, self.root)

    def test_multistream_without_baseline_overlap_completes_as_no_action(self):
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        capture_path = self._trace_capture()
        capture = json.loads(capture_path.read_text())
        for occurrence in capture["targets"][0]["baseline_occurrences"]:
            occurrence["events"][1]["start_us"] = 10.0
        capture.pop("capture_fingerprint")
        capture["capture_fingerprint"] = multistream_trace_analysis.fingerprint(capture)
        capture_path.write_text(json.dumps(capture) + "\n")

        target = multistream_trace_analysis.analyze(
            request_path, capture_path, self.root
        )["targets"][0]

        self.assertEqual(target["parallelism_effect"], "preserved")
        self.assertEqual(
            target["parallelism_basis"], "baseline_cross_stream_overlap_absent"
        )
        self.assertEqual(target["optimization_status"], "no_action")

    def test_lost_cube_vector_overlap_is_not_masked_by_new_mix_overlap(self):
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        capture_path = self._trace_capture()
        capture = json.loads(capture_path.read_text())
        for side in ("baseline_occurrences", "candidate_occurrences"):
            for occurrence in capture["targets"][0][side]:
                occurrence["events"].append(
                    {
                        "child_origin_identity": "node-mix",
                        "lane_id": "mix-0",
                        "stream_id": 3,
                        "accelerator_core": "MIX_AIC",
                        "block_num": 22,
                        "mix_block_num": 22,
                        "start_us": 20.0 if side == "baseline_occurrences" else 0.0,
                        "duration_us": 10.0,
                        "event_kind": "kernel",
                    }
                )
        capture.pop("capture_fingerprint")
        capture["capture_fingerprint"] = multistream_trace_analysis.fingerprint(capture)
        capture_path.write_text(json.dumps(capture) + "\n")

        target = multistream_trace_analysis.analyze(
            request_path, capture_path, self.root
        )["targets"][0]

        self.assertEqual(target["parallelism_effect"], "degraded")
        self.assertEqual(target["parallelism_basis"], "cube_vector_overlap_loss")
        self.assertEqual(target["optimization_status"], "opportunity")

    def test_validated_mechanism_requires_bound_improved_trace(self):
        result = self._accepted_option_result()
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        capture_path = self._trace_capture(candidate_mode="improved")
        trace = multistream_trace_analysis.analyze(
            request_path, capture_path, self.root
        )
        trace_path = self.root / "accepted-trace-analysis.json"
        trace_path.write_text(json.dumps(trace) + "\n")
        trial = result["trials"][0]
        trial["mechanism_status"] = "validated"
        trial["scheduling_evidence"] = {
            "trace_artifact": capture_path.name,
            "binding_artifact": trace_path.name,
            "overflow_detected": False,
            "aligned_occurrence_count": 3,
        }
        result["multistream_findings"][0]["parallelism_effect"] = "improved"
        result["multistream_findings"][0]["optimization_status"] = "validated"
        summary = multistream_contract.validate_result(self.request, result, self.root)
        self.assertEqual(summary["status"], "accepted")

    def test_candidate_planner_deduplicates_stage_o_and_freezes_budget(self):
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        trace = multistream_trace_analysis.analyze(
            request_path, self._trace_capture(), self.root
        )
        trace_path = self.root / "trace-analysis.json"
        trace_path.write_text(json.dumps(trace) + "\n")
        history = {
            "schema_version": multistream_candidate_planner.HISTORY_SCHEMA,
            "settled_trials": [
                {
                    "json_pointer": "/options/auto_op_parallel",
                    "after": 1,
                    "status": "no_gain",
                }
            ],
        }
        (self.root / "option-history.json").write_text(json.dumps(history) + "\n")
        catalog = {
            "schema_version": multistream_candidate_planner.CATALOG_SCHEMA,
            "options": [
                {
                    "option_name": "auto_op_parallel",
                    "json_pointer": "/options/auto_op_parallel",
                    "before": 0,
                    "accepted_values": [1, 2],
                    "accepted_evidence": "accepted-option.json",
                    "applies_to": ["*"],
                    "risk": "low",
                    "priority": 20,
                }
            ],
            "source_actions": [],
        }
        catalog["catalog_fingerprint"] = multistream_candidate_planner.fingerprint(
            catalog
        )
        catalog_path = self.root / "action-catalog.json"
        catalog_path.write_text(json.dumps(catalog) + "\n")
        matrix = multistream_candidate_planner.plan(
            request_path, trace_path, catalog_path, self.root
        )
        self.assertEqual(len(matrix["candidates"]), 1)
        self.assertEqual(matrix["candidates"][0]["action"]["after"], 2)
        self.assertTrue(matrix["candidates"][0]["selected_for_execution"])
        self.assertEqual(
            matrix["deduplicated_candidates"][0]["reason"],
            "identical_global_option_value_settled_by_stage_o",
        )
        matrix_path = self.root / "candidate-matrix.json"
        matrix_path.write_text(json.dumps(matrix) + "\n")
        self.assertTrue(
            multistream_candidate_planner.validate_matrix(
                matrix_path, request_path, self.root
            )["valid"]
        )

    def test_candidate_planner_does_not_plan_blocked_mix_overlap_loss(self):
        request_path = self.root / "multistream-request.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        capture_path = self._trace_capture()
        capture = json.loads(capture_path.read_text())
        for occurrence in (
            capture["targets"][0]["baseline_occurrences"]
            + capture["targets"][0]["candidate_occurrences"]
        ):
            occurrence["events"][0]["mix_block_num"] = 22
        capture.pop("capture_fingerprint")
        capture["capture_fingerprint"] = multistream_trace_analysis.fingerprint(capture)
        capture_path.write_text(json.dumps(capture) + "\n")
        trace = multistream_trace_analysis.analyze(
            request_path, capture_path, self.root
        )
        trace_path = self.root / "trace-analysis.json"
        trace_path.write_text(json.dumps(trace) + "\n")
        (self.root / "option-history.json").write_text(
            json.dumps(
                {
                    "schema_version": multistream_candidate_planner.HISTORY_SCHEMA,
                    "settled_trials": [],
                }
            )
            + "\n"
        )
        catalog = {
            "schema_version": multistream_candidate_planner.CATALOG_SCHEMA,
            "options": [
                {
                    "option_name": "auto_op_parallel",
                    "json_pointer": "/options/auto_op_parallel",
                    "before": 0,
                    "accepted_values": [1],
                    "accepted_evidence": "accepted-option.json",
                    "applies_to": ["*"],
                    "risk": "low",
                    "priority": 20,
                }
            ],
            "source_actions": [],
        }
        catalog["catalog_fingerprint"] = multistream_candidate_planner.fingerprint(
            catalog
        )
        catalog_path = self.root / "action-catalog.json"
        catalog_path.write_text(json.dumps(catalog) + "\n")

        matrix = multistream_candidate_planner.plan(
            request_path, trace_path, catalog_path, self.root
        )

        self.assertEqual(matrix["candidates"], [])
        self.assertEqual(matrix["budget"]["planned_trials"], 0)

    def test_model_adapter_compiles_a_trial_bound_frozen_plan(self):
        self._accepted_option_result()
        adapter = multistream_plan_compiler.freeze_adapter(self._model_adapter_draft())
        action = json.loads((self.root / "action-manifest.json").read_text())

        plan, compilation = multistream_plan_compiler.compile_plan(
            self.request,
            adapter,
            action,
            artifact_root=self.root,
            candidate_config="selected-config.yaml",
        )

        self.assertEqual(plan["trial_id"], "MS-O1")
        self.assertEqual(
            plan["phases"][0]["argv"][1],
            str((self.root / "selected-config.yaml").resolve()),
        )
        self.assertEqual(
            compilation["adapter_fingerprint"], adapter["adapter_fingerprint"]
        )
        self.assertEqual(compilation["plan_fingerprint"], plan["plan_fingerprint"])
        multistream_runner.validate_plan(plan)

    def test_model_adapter_rejects_tampering_and_unknown_placeholders(self):
        adapter = multistream_plan_compiler.freeze_adapter(self._model_adapter_draft())
        adapter["device_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "adapter_fingerprint mismatch"):
            multistream_plan_compiler.validate_adapter(adapter)

        draft = self._model_adapter_draft()
        draft["phases"][0]["argv_template"].append("{unknown_value}")
        adapter = multistream_plan_compiler.freeze_adapter(draft)
        self._accepted_option_result()
        action = json.loads((self.root / "action-manifest.json").read_text())
        with self.assertRaisesRegex(ValueError, "unknown placeholders"):
            multistream_plan_compiler.compile_plan(
                self.request,
                adapter,
                action,
                artifact_root=self.root,
                candidate_config="selected-config.yaml",
            )

        draft = self._model_adapter_draft()
        draft["phases"][0]["validator_argv_template"] = ["/bin/true"]
        adapter = multistream_plan_compiler.freeze_adapter(draft)
        with self.assertRaisesRegex(ValueError, "standard semantic evidence"):
            multistream_plan_compiler.compile_plan(
                self.request,
                adapter,
                action,
                artifact_root=self.root,
                candidate_config="selected-config.yaml",
            )

    def test_scope_change_requires_source_map(self):
        self.request["requested_change_kinds"] = ["scope_split"]
        with self.assertRaisesRegex(ValueError, "requires artifacts.source_scope_map"):
            multistream_contract.validate_request(self.request, self.root)

    def test_fallback_is_bound_to_unchanged_incumbent(self):
        result = self._fallback()
        summary = multistream_contract.validate_result(self.request, result, self.root)
        self.assertEqual(summary["status"], "blocked")
        self.assertTrue(summary["incumbent_unchanged"])
        self.assertEqual(summary["fallback"]["candidate_name"], "S3-BASE")

    def test_nonaccepted_result_cannot_select_candidate(self):
        result = self._fallback("no_gain")
        result["selected_candidate"] = self._accepted_option_result()[
            "selected_candidate"
        ]
        with self.assertRaisesRegex(ValueError, "selected_candidate=null"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_accepts_complete_incremental_option_trial_without_source_map(self):
        result = self._accepted_option_result()
        summary = multistream_contract.validate_result(self.request, result, self.root)
        self.assertEqual(summary["status"], "accepted")
        self.assertEqual(summary["accepted_trial_id"], "MS-O1")

    def test_old_v1_result_is_rejected_after_execution_evidence_upgrade(self):
        result = self._accepted_option_result()
        result["schema_version"] = "superkernel-multistream-result-v1"
        with self.assertRaisesRegex(ValueError, "result-v2"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_selected_config_fingerprint_is_recomputed_from_snapshot(self):
        result = self._accepted_option_result()
        (self.root / "selected-config.yaml").write_text("option: 2\n")
        with self.assertRaisesRegex(ValueError, "config_fingerprint"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_source_snapshot_rejects_file_tampering(self):
        result = self._accepted_option_result()
        (self.root / "source" / "model.py").write_text("def run():\n    return 9\n")
        with self.assertRaisesRegex(ValueError, "file fingerprint mismatch"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_accepted_trial_replays_phase_artifact_fingerprints(self):
        result = self._accepted_option_result()
        (self.root / "phase-artifacts/correctness_passed.json").write_text(
            '{"tampered": true}\n'
        )
        with self.assertRaisesRegex(ValueError, "sealed file fingerprint mismatch"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_accepted_trial_rejects_semantic_source_tampering(self):
        result = self._accepted_option_result()
        (self.root / "semantic-evidence-source.txt").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "semantic evidence source changed"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_accepted_trial_rejects_execution_plan_tampering(self):
        result = self._accepted_option_result()
        plan_path = self.root / result["trials"][0]["execution_plan"]
        plan = json.loads(plan_path.read_text())
        plan["lease_timeout_seconds"] = 11
        plan_path.write_text(json.dumps(plan) + "\n")
        with self.assertRaisesRegex(ValueError, "plan_fingerprint mismatch"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_coordinated_plan_rewrite_cannot_bypass_adapter_replay(self):
        result = self._accepted_option_result()
        plan_path = self.root / result["trials"][0]["execution_plan"]
        plan = json.loads(plan_path.read_text())
        plan["phases"][0]["argv"].append("--rewritten")
        plan["plan_fingerprint"] = multistream_runner.content_fingerprint(
            multistream_runner._plan_payload(plan)
        )
        plan_path.write_text(json.dumps(plan) + "\n")
        compilation_path = self.root / result["trials"][0]["plan_compilation"]
        compilation = json.loads(compilation_path.read_text())
        compilation["plan_fingerprint"] = plan["plan_fingerprint"]
        compilation.pop("compilation_fingerprint")
        compilation["compilation_fingerprint"] = (
            multistream_plan_compiler.content_fingerprint(compilation)
        )
        compilation_path.write_text(json.dumps(compilation) + "\n")

        with self.assertRaisesRegex(ValueError, "deterministic adapter compilation"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_local_mechanism_signal_cannot_replace_clean_gate(self):
        result = self._accepted_option_result()
        result["trials"][0]["clean_metrics"]["incremental_mean_gain"] = False
        with self.assertRaisesRegex(ValueError, "incremental_mean_gain=true"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_accepted_trial_rejects_extra_gate_fields(self):
        result = self._accepted_option_result()
        result["trials"][0]["gates"]["local_overlap"] = "passed"
        with self.assertRaisesRegex(ValueError, "gates fields must be"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_validated_mechanism_requires_bound_short_trace(self):
        result = self._accepted_option_result()
        result["trials"][0]["mechanism_status"] = "validated"
        with self.assertRaisesRegex(ValueError, "scheduling_evidence"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_request_requires_three_aligned_occurrences(self):
        analysis_path = self.root / "incumbent-analysis.json"
        analysis = json.loads(analysis_path.read_text())
        analysis["per_sk_decisions"][0]["candidate_occurrence_count"] = 2
        analysis["analysis_content_fingerprint"] = (
            multistream_contract._analysis_fingerprint(analysis)
        )
        analysis_path.write_text(json.dumps(analysis) + "\n")
        with self.assertRaisesRegex(ValueError, "must be >= 3"):
            multistream_contract.validate_request(self.request, self.root)

    def test_stale_request_fingerprint_is_rejected(self):
        result = self._accepted_option_result()
        result["request_fingerprint"] = "sha256:stale"
        with self.assertRaisesRegex(ValueError, "does not match request content"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_request_rejects_artifact_escape(self):
        request = copy.deepcopy(self.request)
        request["artifacts"]["option_history"] = "../outside.json"
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            multistream_contract.validate_request(request, self.root)

    def test_request_rejects_malformed_profiling_analysis(self):
        (self.root / "incumbent-analysis.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "schema_version 1.2"):
            multistream_contract.validate_request(self.request, self.root)

    def test_placeholder_source_map_cannot_authorize_source_action(self):
        (self.root / "source-map.json").write_text("{}\n")
        self.request["artifacts"]["source_scope_map"] = "source-map.json"
        self.request["requested_change_kinds"] = ["scope_split"]
        result = self._accepted_option_result()
        result["request_fingerprint"] = multistream_contract.content_fingerprint(
            self.request
        )
        trial = result["trials"][0]
        trial["change_kind"] = "scope_split"
        trial["only_change"] = {
            "source_scope_map": "source-map.json",
            "source_file": "model.py",
            "start_offset": 10,
            "end_offset": 20,
            "boundary_change": "split one dependency boundary",
        }
        with self.assertRaisesRegex(ValueError, "source_scope_map_v2"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_source_change_accepts_exact_request_map(self):
        self._write_source_map()
        self.request["artifacts"]["source_scope_map"] = "source-map.json"
        self.request["requested_change_kinds"] = ["range_exclusion"]
        self._write_analysis(source_exact=True)
        result = self._accepted_option_result()
        result["request_fingerprint"] = multistream_contract.content_fingerprint(
            self.request
        )
        trial = result["trials"][0]
        trial["change_kind"] = "range_exclusion"
        trial["only_change"] = {
            "source_scope_map": "source-map.json",
            "source_file": "model.py",
            "start_offset": 10,
            "end_offset": 20,
            "boundary_change": "exclude one exact range from fusion",
        }
        result["selected_candidate"]["source_revision"] = "source-r2"
        changed_source = self.root / "source" / "model-trial.py"
        changed_source.write_text("def run():\n    return 2\n")
        changed_file_fingerprint = multistream_execution.file_fingerprint(
            changed_source
        )
        changed_source_fingerprint = multistream_execution.content_fingerprint(
            {
                "files": [
                    {
                        "path": "source/model-trial.py",
                        "file_fingerprint": changed_file_fingerprint,
                    }
                ]
            }
        )
        result["selected_candidate"]["source_fingerprint"] = changed_source_fingerprint
        (self.root / "source-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "superkernel-source-snapshot-manifest-v1",
                    "source_revision": "source-r2",
                    "source_fingerprint": changed_source_fingerprint,
                    "files": [
                        {
                            "path": "source/model-trial.py",
                            "file_fingerprint": changed_file_fingerprint,
                        }
                    ],
                }
            )
            + "\n"
        )
        action = json.loads((self.root / "action-manifest.json").read_text())
        action.update(
            {
                "change_kind": "range_exclusion",
                "only_change": {
                    "source_file": "model.py",
                    "start_offset": 10,
                    "end_offset": 20,
                    "boundary_change": "exclude one exact range from fusion",
                },
                "source_adapter_validation": "passed",
            }
        )
        action.pop("changed_pointers")
        (self.root / "action-manifest.json").write_text(json.dumps(action) + "\n")
        self._refresh_compilation_for_action(action)
        summary = multistream_contract.validate_result(self.request, result, self.root)
        self.assertEqual(summary["status"], "accepted")

    def test_source_change_must_change_source_identity(self):
        self._write_source_map()
        self.request["artifacts"]["source_scope_map"] = "source-map.json"
        self.request["requested_change_kinds"] = ["range_exclusion"]
        self._write_analysis(source_exact=True)
        result = self._accepted_option_result()
        result["request_fingerprint"] = multistream_contract.content_fingerprint(
            self.request
        )
        trial = result["trials"][0]
        trial["change_kind"] = "range_exclusion"
        trial["only_change"] = {
            "source_scope_map": "source-map.json",
            "source_file": "model.py",
            "start_offset": 10,
            "end_offset": 20,
            "boundary_change": "exclude one exact range from fusion",
        }
        action = json.loads((self.root / "action-manifest.json").read_text())
        action.update(
            {
                "change_kind": "range_exclusion",
                "only_change": {
                    "source_file": "model.py",
                    "start_offset": 10,
                    "end_offset": 20,
                    "boundary_change": "exclude one exact range from fusion",
                },
                "source_adapter_validation": "passed",
            }
        )
        action.pop("changed_pointers")
        (self.root / "action-manifest.json").write_text(json.dumps(action) + "\n")
        self._refresh_compilation_for_action(action)
        with self.assertRaisesRegex(ValueError, "must change source revision"):
            multistream_contract.validate_result(self.request, result, self.root)

    def test_parent_bootstrap_registers_seed_without_merging_performance_ledger(self):
        result = self._accepted_option_result()
        request_path = self.root / "multistream-request.json"
        result_path = self.root / "multistream-result.json"
        incumbent_path = self.root / "current-incumbent.json"
        registry_path = self.root / "derived-family-registry.json"
        family_root = self.root / "families"
        request_path.write_text(json.dumps(self.request) + "\n")
        result_path.write_text(json.dumps(result) + "\n")
        incumbent_path.write_text(json.dumps(self.request["incumbent"]) + "\n")

        entry = bootstrap_derived_family.bootstrap(
            request_path,
            result_path,
            incumbent_path,
            registry_path,
            family_root,
        )

        self.assertEqual(entry["status"], "seed_registered")
        self.assertTrue(entry["fresh_base_required"])
        self.assertFalse(entry["ledger_merge_allowed"])
        self.assertTrue((family_root / "S3M" / "lineage.json").is_file())
        registry = json.loads(registry_path.read_text())
        self.assertEqual(registry["families"]["S3M"]["status"], "seed_registered")
        audit = family_root / "S3M" / "SEED" / "execution-audit"
        self.assertTrue((audit / "execution-plan.json").is_file())
        self.assertTrue((audit / "model-adapter.json").is_file())
        self.assertEqual(len(list((audit / "phase-manifests").glob("*.json"))), 5)

        repeated = bootstrap_derived_family.bootstrap(
            request_path,
            result_path,
            incumbent_path,
            registry_path,
            family_root,
        )
        self.assertEqual(repeated, entry)

    def test_parent_bootstrap_rejects_stale_current_incumbent(self):
        result = self._accepted_option_result()
        request_path = self.root / "multistream-request.json"
        result_path = self.root / "multistream-result.json"
        incumbent_path = self.root / "current-incumbent.json"
        request_path.write_text(json.dumps(self.request) + "\n")
        result_path.write_text(json.dumps(result) + "\n")
        stale = copy.deepcopy(self.request["incumbent"])
        stale["config_fingerprint"] = "newer-config"
        incumbent_path.write_text(json.dumps(stale) + "\n")

        with self.assertRaisesRegex(ValueError, "stale_result"):
            bootstrap_derived_family.bootstrap(
                request_path,
                result_path,
                incumbent_path,
                self.root / "derived-family-registry.json",
                self.root / "families",
            )

    def test_derived_family_runs_fresh_base_before_lifecycle_and_ledger_merge(self):
        result = self._accepted_option_result()
        request_path = self.root / "multistream-request.json"
        result_path = self.root / "multistream-result.json"
        incumbent_path = self.root / "current-incumbent.json"
        registry_path = self.root / "derived-family-registry.json"
        family_root = self.root / "families"
        request_path.write_text(json.dumps(self.request) + "\n")
        result_path.write_text(json.dumps(result) + "\n")
        incumbent_path.write_text(json.dumps(self.request["incumbent"]) + "\n")
        bootstrap_derived_family.bootstrap(
            request_path, result_path, incumbent_path, registry_path, family_root
        )
        family = family_root / "S3M"
        lineage = json.loads((family / "lineage.json").read_text())
        artifact_root = family / "fresh-base" / "artifacts"
        artifact_root.mkdir(parents=True)
        artifacts = {}
        source_files = []
        for name in sorted(derived_family_lifecycle.REQUIRED_ARTIFACTS):
            path = artifact_root / f"{name}.json"
            path.write_text("{}\n")
            relative = str(path.relative_to(family))
            artifacts[name] = relative
            source_files.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "file_fingerprint": derived_family_lifecycle.file_fingerprint(path),
                }
            )
        receipt = {
            "schema_version": derived_family_lifecycle.RECEIPT_SCHEMA,
            "derived_experiment_id": "S3M",
            "round_id": "S3M-BASE",
            "seed_identity": lineage["seed_identity"],
            "gates": {
                "correctness": "passed",
                "profiling": "passed",
                "analysis": "passed",
                "clean_performance": "passed",
            },
            "clean_run_count": 5,
            "stable": True,
            "artifacts": artifacts,
            "source_files": source_files,
        }
        receipt["receipt_fingerprint"] = derived_family_lifecycle.fingerprint(receipt)
        receipt_path = family / "fresh-base" / "receipt.json"
        command = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                f"Path({str(receipt_path)!r}).write_text({(json.dumps(receipt) + chr(10))!r})"
            ),
        ]
        draft = {
            "schema_version": derived_family_lifecycle.PLAN_SCHEMA,
            "family_id": "S3M",
            "round_id": "S3M-BASE",
            "family_root": str(family.resolve()),
            "lease_root": str((self.root / "shared-leases").resolve()),
            "device_ids": [0],
            "lease_timeout_seconds": 2,
            "environment": {"PATH": "/usr/bin"},
            "cwd": str(family.resolve()),
            "command_argv": command,
            "timeout_seconds": 2,
            "program_files": [
                {
                    "path": str(Path(sys.executable).resolve()),
                    "file_fingerprint": derived_family_lifecycle.file_fingerprint(
                        Path(sys.executable).resolve()
                    ),
                }
            ],
            "receipt": "fresh-base/receipt.json",
            "command_manifest": "fresh-base/command.json",
        }
        plan = derived_family_lifecycle.freeze_plan(draft)
        plan_path = family / "fresh-base-plan.json"
        plan_path.write_text(json.dumps(plan) + "\n")
        entry = derived_family_lifecycle.run_fresh_base(plan_path, registry_path)
        self.assertEqual(entry["status"], "fresh_base_completed")
        self.assertFalse(entry["ledger_merge_allowed"])
        with self.assertRaisesRegex(ValueError, "ordinary_lifecycle_completed"):
            derived_family_lifecycle.merge_ledger(
                registry_path,
                "S3M",
                self.root / "ledger.json",
                self.root / "merged-too-early.json",
            )

        ordinary = {
            "experiment_id": "S3M",
            "parent_experiment_id": "S3M",
            "rounds": [{"round_id": "S3M-BASE", "round_kind": "base"}],
        }
        ordinary_path = family / "ordinary-result.json"
        ordinary_path.write_text(json.dumps(ordinary) + "\n")
        with mock.patch.object(
            derived_family_lifecycle.experiment_ledger,
            "validate_experiment_result",
            return_value={"valid": True, "errors": []},
        ):
            entry = derived_family_lifecycle.complete_lifecycle(
                registry_path, "S3M", ordinary_path
            )
        self.assertEqual(entry["status"], "ordinary_lifecycle_completed")
        self.assertTrue(entry["ledger_merge_allowed"])
        with mock.patch.object(
            derived_family_lifecycle.experiment_ledger,
            "merge_experiment_result",
            return_value={"schema_version": 2, "experiments": {"S3M": ordinary}},
        ):
            entry = derived_family_lifecycle.merge_ledger(
                registry_path,
                "S3M",
                self.root / "ledger.json",
                self.root / "merged-ledger.json",
            )
        self.assertEqual(entry["status"], "ledger_merged")
        self.assertFalse(entry["ledger_merge_allowed"])


if __name__ == "__main__":
    unittest.main()
