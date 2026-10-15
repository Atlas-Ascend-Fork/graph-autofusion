# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TEST_DIR))

import multistream_critical_path  # noqa: E402
import multistream_critical_path_contract  # noqa: E402
import multistream_dependency_evidence  # noqa: E402
import multistream_event_stage_action  # noqa: E402
import multistream_event_stage_planner  # noqa: E402
import multistream_evidence  # noqa: E402
import multistream_join_validation  # noqa: E402
import multistream_source_transform  # noqa: E402
from test_critical_path import CriticalPathTest  # noqa: E402


class CriticalPathContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("source", "experiment", "config", "cache"):
            (self.root / name).mkdir()
        (self.root / "clean.json").write_text("{}\n")
        self.request = {
            "schema_version": multistream_critical_path_contract.REQUEST_SCHEMA,
            "request_id": "request-1",
            "parent_experiment_id": "parent-1",
            "incumbent": {
                "candidate_name": "incumbent",
                "source_revision": "rev-1",
                "source_fingerprint": "source-fp-1",
                "config_fingerprint": "config-fp-1",
                "control_fingerprint": "control-fp-1",
                "workload_fingerprint": "workload-fp-1",
                "clean_performance_summary": "clean.json",
                "clean_run_count": 5,
                "stable": True,
            },
            "artifacts": {
                "logical_graph": "graph.json",
                "critical_path_capture": "capture.json",
                "dependency_evidence": "dependency.json",
                "event_stage_action_catalog": "catalog.json",
                "event_stage_history": "history.json",
                "candidate_matrix": "matrix.json",
                "scope_derivatives": None,
            },
            "isolation": {
                "source_worktree": "source",
                "experiment_root": "experiment",
                "config_root": "config",
                "cache_root": "cache",
                "immutable_incumbent": True,
                "dedicated_roots": True,
            },
            "budget": {
                "max_trials": 3,
                "min_predicted_e2e_upper_bound": 0.03,
                "min_clean_gain_pct": 2.0,
            },
            "authorization": {
                "run_inference": True,
                "edit_isolated_worktree": True,
                "allowed_change_kinds": [
                    "event_edge_refinement",
                    "scope_event_derivative",
                    "stage_split",
                ],
            },
        }
        request_fp = multistream_critical_path_contract.fingerprint(self.request)
        fixture = CriticalPathTest()
        capture = fixture._capture()
        capture["request_fingerprint"] = request_fp
        capture["logical_graph"]["request_fingerprint"] = request_fp
        provider = self.root / "dependency_provider.py"
        provider.write_text("# reviewed provider\n")
        provider_input = self.root / "dependency_input.json"
        provider_input.write_text("{}\n")
        statements = [
            statement_id
            for stage in capture["logical_graph"]["stages"]
            for statement_id in stage["source_statement_ids"]
        ]
        fragments = {
            "schema_version": multistream_dependency_evidence.FRAGMENT_SET_SCHEMA,
            "fragment_set_id": "critical-path-dependencies-1",
            "request_fingerprint": request_fp,
            "range_id": "logical-graph-1",
            "graph_occurrence_fingerprint": "logical-occurrence-1",
            "statement_ids": statements,
            "required_dependency_kinds": sorted(
                multistream_dependency_evidence.HARD_DEPENDENCY_KINDS
            ),
            "providers": [
                {
                    "provider_id": "source-review-1",
                    "provider_api_version": multistream_dependency_evidence.PROVIDER_API_VERSION,
                    "provider_kind": "source_review",
                    "implementation": multistream_dependency_evidence.source_file_record(
                        provider, self.root
                    ),
                    "input_files": [
                        multistream_dependency_evidence.source_file_record(
                            provider_input, self.root
                        )
                    ],
                    "covered_dependency_kinds": sorted(
                        multistream_dependency_evidence.HARD_DEPENDENCY_KINDS
                    ),
                    "edges": [],
                    "blockers": [],
                }
            ],
        }
        fragments["fragment_set_fingerprint"] = (
            multistream_dependency_evidence.fingerprint(fragments)
        )
        fragments_path = self.root / "dependency-fragments.json"
        fragments_path.write_text(json.dumps(fragments) + "\n")
        dependency = multistream_dependency_evidence.build(fragments_path, self.root)
        (self.root / "dependency.json").write_text(json.dumps(dependency) + "\n")
        capture["logical_graph"]["dependency_evidence_fingerprint"] = dependency[
            "evidence_fingerprint"
        ]
        capture["logical_graph"]["graph_fingerprint"] = (
            multistream_critical_path_contract.graph_fingerprint(
                capture["logical_graph"]
            )
        )
        capture["capture_fingerprint"] = multistream_critical_path.fingerprint(
            {
                key: value
                for key, value in capture.items()
                if key != "capture_fingerprint"
            }
        )
        graph = capture["logical_graph"]
        analysis = multistream_critical_path.analyze(capture)
        catalog = multistream_event_stage_action.build(analysis, graph)
        history = {"settled_actions": []}
        matrix = multistream_event_stage_planner.plan(analysis, catalog, history, 3)
        for name, value in (
            ("graph.json", graph),
            ("capture.json", capture),
            ("catalog.json", catalog),
            ("history.json", history),
            ("matrix.json", matrix),
        ):
            (self.root / name).write_text(json.dumps(value) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_validates_complete_request_evidence_chain(self):
        result = multistream_critical_path_contract.validate_request(
            self.request, self.root
        )
        self.assertEqual(
            result["request_fingerprint"],
            multistream_critical_path_contract.fingerprint(self.request),
        )
        self.assertEqual(result["candidate_count"], 1)

    def test_no_candidate_fallback_preserves_incumbent(self):
        result = multistream_critical_path_contract.build_fallback(
            self.request, self.root, "no_gain", "关键路径证据未产生合法候选"
        )
        validated = multistream_critical_path_contract.validate_result(
            self.request, result, self.root
        )
        self.assertTrue(validated["incumbent_unchanged"])
        self.assertIsNone(result["selected_candidate"])

    def test_nonaccepted_result_cannot_select_candidate(self):
        result = multistream_critical_path_contract.build_fallback(
            self.request, self.root, "no_gain", "关键路径证据未产生合法候选"
        )
        result["selected_candidate"] = {"trial_id": "trial-1"}
        with self.assertRaisesRegex(ValueError, "selected_candidate"):
            multistream_critical_path_contract.validate_result(
                self.request, result, self.root
            )

    def test_request_rejects_tampered_candidate_matrix(self):
        matrix_path = self.root / "matrix.json"
        matrix = json.loads(matrix_path.read_text())
        matrix["budget"]["max_trials"] = 2
        matrix_path.write_text(json.dumps(matrix) + "\n")
        with self.assertRaisesRegex(ValueError, "matrix"):
            multistream_critical_path_contract.validate_request(self.request, self.root)

    def _executed_trial(
        self, *, clean_decision="pass", clean_state="clean5_passed", decision="accepted"
    ):
        request_fp = multistream_critical_path_contract.fingerprint(self.request)
        matrix = json.loads((self.root / "matrix.json").read_text())
        candidate_record = matrix["candidates"][0]
        graph = json.loads((self.root / "graph.json").read_text())
        candidate_input = self.root / "candidate-dependency-input.json"
        candidate_input.write_text(
            json.dumps({"candidate": candidate_record["action_id"]}) + "\n"
        )
        statements = [
            statement_id
            for stage in graph["stages"]
            for statement_id in stage["source_statement_ids"]
        ]
        post_fragments = {
            "schema_version": multistream_dependency_evidence.FRAGMENT_SET_SCHEMA,
            "fragment_set_id": "post-transform-dependencies-1",
            "request_fingerprint": request_fp,
            "range_id": "logical-graph-1",
            "graph_occurrence_fingerprint": "logical-occurrence-post-1",
            "statement_ids": statements,
            "required_dependency_kinds": sorted(
                multistream_dependency_evidence.HARD_DEPENDENCY_KINDS
            ),
            "providers": [
                {
                    "provider_id": "source-review-post-1",
                    "provider_api_version": multistream_dependency_evidence.PROVIDER_API_VERSION,
                    "provider_kind": "source_review",
                    "implementation": multistream_dependency_evidence.source_file_record(
                        self.root / "dependency_provider.py", self.root
                    ),
                    "input_files": [
                        multistream_dependency_evidence.source_file_record(
                            candidate_input, self.root
                        )
                    ],
                    "covered_dependency_kinds": sorted(
                        multistream_dependency_evidence.HARD_DEPENDENCY_KINDS
                    ),
                    "edges": [],
                    "blockers": [],
                }
            ],
        }
        post_fragments["fragment_set_fingerprint"] = (
            multistream_dependency_evidence.fingerprint(post_fragments)
        )
        post_fragments_path = self.root / "post-dependency-fragments.json"
        post_fragments_path.write_text(json.dumps(post_fragments) + "\n")
        post_dependency = multistream_dependency_evidence.build(
            post_fragments_path, self.root
        )
        (self.root / "post-dependency.json").write_text(
            json.dumps(post_dependency) + "\n"
        )
        action = {
            "schema_version": multistream_source_transform.ACTION_MANIFEST_SCHEMA,
            "trial_id": "trial-1",
            "change_kind": candidate_record["change_kind"],
            "action_id": candidate_record["action_id"],
            "single_change_verified": True,
            "multistream_only_verified": True,
            "dependency_evidence_fingerprint_before": graph[
                "dependency_evidence_fingerprint"
            ],
            "dependency_evidence_fingerprint_after": post_dependency[
                "evidence_fingerprint"
            ],
        }
        (self.root / "action.json").write_text(json.dumps(action) + "\n")
        fixture = CriticalPathTest()
        baseline_capture = fixture._capture()
        candidate_capture = fixture._capture(aux_start=0.0, downstream_start=18.0)
        for capture in (baseline_capture, candidate_capture):
            capture["request_fingerprint"] = request_fp
            capture["logical_graph"]["request_fingerprint"] = request_fp
            capture["logical_graph"]["graph_fingerprint"] = (
                multistream_critical_path_contract.graph_fingerprint(
                    capture["logical_graph"]
                )
            )
            capture["capture_fingerprint"] = multistream_critical_path.fingerprint(
                {
                    key: value
                    for key, value in capture.items()
                    if key != "capture_fingerprint"
                }
            )
        join = multistream_join_validation.compare(
            "trial-1",
            multistream_source_transform.fingerprint(action),
            multistream_critical_path.analyze(baseline_capture),
            multistream_critical_path.analyze(candidate_capture),
        )
        (self.root / "join.json").write_text(json.dumps(join) + "\n")
        source = self.root / "clean-source.log"
        source.write_text("clean evidence\n")
        clean = {
            "schema_version": multistream_evidence.EVIDENCE_SCHEMA,
            "evidence_kind": "clean",
            "state_after": clean_state,
            "trial_id": "trial-1",
            "request_fingerprint": request_fp,
            "inputs": {},
            "source_files": [
                {
                    "path": source.name,
                    "size_bytes": source.stat().st_size,
                    "file_fingerprint": multistream_evidence.file_fingerprint(source),
                }
            ],
            "semantic_result": {},
            "decision": clean_decision,
        }
        clean["evidence_fingerprint"] = multistream_evidence.content_fingerprint(clean)
        (self.root / "clean-evidence.json").write_text(json.dumps(clean) + "\n")
        return {
            "trial_id": "trial-1",
            "candidate_id": candidate_record["candidate_id"],
            "action_id": candidate_record["action_id"],
            "parent_action_id": candidate_record["parent_action_id"],
            "change_kind": candidate_record["change_kind"],
            "decision": decision,
            "action_manifest": "action.json",
            "join_validation": "join.json",
            "parent_clean_evidence": None,
            "post_dependency_evidence": "post-dependency.json",
            "clean_evidence": "clean-evidence.json",
            "clean_state": clean_state,
        }

    def test_accepted_result_requires_join_and_clean5(self):
        result = multistream_critical_path_contract.build_fallback(
            self.request, self.root, "no_gain", "初始化未选择候选"
        )
        result.update(
            {
                "status": "accepted",
                "incumbent_unchanged": False,
                "trials": [self._executed_trial()],
                "selected_candidate": {
                    "trial_id": "trial-1",
                    "candidate_name": "candidate",
                    "source_revision": "rev-2",
                    "source_fingerprint": "source-fp-2",
                    "config_fingerprint": "config-fp-1",
                    "control_fingerprint": "control-fp-1",
                    "workload_fingerprint": "workload-fp-1",
                },
            }
        )
        validated = multistream_critical_path_contract.validate_result(
            self.request, result, self.root
        )
        self.assertEqual(validated["status"], "accepted")

    def test_local_join_improvement_cannot_replace_clean_gate(self):
        result = multistream_critical_path_contract.build_fallback(
            self.request, self.root, "no_gain", "初始化未选择候选"
        )
        result.update(
            {
                "status": "accepted",
                "incumbent_unchanged": False,
                "trials": [
                    self._executed_trial(
                        clean_decision="reject", clean_state="clean3_passed"
                    )
                ],
                "selected_candidate": {"trial_id": "trial-1"},
            }
        )
        with self.assertRaisesRegex(ValueError, "clean5"):
            multistream_critical_path_contract.validate_result(
                self.request, result, self.root
            )


if __name__ == "__main__":
    unittest.main()
