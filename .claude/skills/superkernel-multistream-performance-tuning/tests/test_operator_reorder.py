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
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_execution  # noqa: E402
import multistream_candidate_planner  # noqa: E402
import multistream_contract  # noqa: E402
import multistream_evidence  # noqa: E402
import multistream_operator_order  # noqa: E402
import multistream_dependency_evidence  # noqa: E402


class OperatorOrderAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "model.py"
        self.source.write_text(
            "def run(x):\n"
            "    vector = vector_op(x)\n"
            "    cube = cube_op(x)\n"
            "    return cube + vector\n"
        )
        self.dependency_provider = self.root / "dependency_provider.py"
        self.dependency_provider.write_text("# reviewed source dependency provider\n")
        self.dependency_input = self.root / "dependency-input.json"
        self.dependency_input.write_text("{}\n")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _resource_fields(family):
        return {
            "CUBE": {
                "accelerator_core": "AI_CORE",
                "block_num": 16,
                "mix_block_num": 0,
            },
            "VECTOR": {
                "accelerator_core": "AI_VECTOR_CORE",
                "block_num": 32,
                "mix_block_num": 0,
            },
            "MIX": {
                "accelerator_core": "MIX_AIV",
                "block_num": 32,
                "mix_block_num": 16,
            },
        }[family]

    def _dependency_evidence(self, statement_ids, dependencies, request_fingerprint):
        fragments = {
            "schema_version": multistream_dependency_evidence.FRAGMENT_SET_SCHEMA,
            "fragment_set_id": "dependency-set-1",
            "request_fingerprint": request_fingerprint,
            "range_id": "range-1",
            "graph_occurrence_fingerprint": "occurrence-1",
            "statement_ids": statement_ids,
            "required_dependency_kinds": sorted(
                multistream_dependency_evidence.HARD_DEPENDENCY_KINDS
            ),
            "providers": [
                {
                    "provider_id": "source-review-1",
                    "provider_api_version": multistream_dependency_evidence.PROVIDER_API_VERSION,
                    "provider_kind": "source_review",
                    "implementation": multistream_dependency_evidence.source_file_record(
                        self.dependency_provider, self.root
                    ),
                    "input_files": [
                        multistream_dependency_evidence.source_file_record(
                            self.dependency_input, self.root
                        )
                    ],
                    "covered_dependency_kinds": sorted(
                        multistream_dependency_evidence.HARD_DEPENDENCY_KINDS
                    ),
                    "edges": [
                        {**edge, "evidence_locator": f"review.edges[{index}]"}
                        for index, edge in enumerate(dependencies)
                    ],
                    "blockers": [],
                }
            ],
        }
        fragments["fragment_set_fingerprint"] = (
            multistream_dependency_evidence.fingerprint(fragments)
        )
        fragments_path = self.root / "dependency-fragments.json"
        fragments_path.write_text(json.dumps(fragments) + "\n")
        evidence = multistream_dependency_evidence.build(fragments_path, self.root)
        evidence_path = self.root / "dependency-evidence.json"
        evidence_path.write_text(json.dumps(evidence) + "\n")
        return evidence_path.name

    def _capture(
        self,
        *,
        stream_ids=(1, 2),
        core_families=("CUBE", "VECTOR"),
        hard_dependencies=None,
        request_fingerprint="request-fp-1",
    ):
        hard_dependencies = hard_dependencies or []
        source = self.source.read_bytes()
        vector_start = source.index(b"    vector")
        cube_start = source.index(b"    cube")
        return {
            "schema_version": multistream_operator_order.CAPTURE_SCHEMA,
            "capture_id": "order-capture-1",
            "request_fingerprint": request_fingerprint,
            "source_files": [
                {
                    "path": "model.py",
                    "size_bytes": self.source.stat().st_size,
                    "file_fingerprint": multistream_operator_order.file_fingerprint(
                        self.source
                    ),
                }
            ],
            "targets": [
                {
                    "range_id": "range-1",
                    "graph_occurrence_fingerprint": "occurrence-1",
                    "source_file": "model.py",
                    "range_start_offset": vector_start,
                    "range_end_offset": source.index(b"    return"),
                    "statements": [
                        {
                            "statement_id": "vector",
                            "start_offset": vector_start,
                            "end_offset": cube_start,
                            "movable": True,
                            "side_effect_free": True,
                        },
                        {
                            "statement_id": "cube",
                            "start_offset": cube_start,
                            "end_offset": source.index(b"    return"),
                            "movable": True,
                            "side_effect_free": True,
                        },
                    ],
                    "hard_dependencies": hard_dependencies,
                    "dependency_evidence": self._dependency_evidence(
                        ["vector", "cube"], hard_dependencies, request_fingerprint
                    ),
                    "occurrences": [
                        {
                            "alignment_id": f"decode-{step}",
                            "sk_off_operators": [
                                {
                                    "operator_id": "cube-op",
                                    "statement_id": "cube",
                                    "stream_id": stream_ids[0],
                                    **self._resource_fields(core_families[0]),
                                    "start_us": 0.0,
                                    "duration_us": 10.0,
                                },
                                {
                                    "operator_id": "vector-op",
                                    "statement_id": "vector",
                                    "stream_id": stream_ids[1],
                                    **self._resource_fields(core_families[1]),
                                    "start_us": 5.0,
                                    "duration_us": 8.0,
                                },
                            ],
                            "sk_on_dispatch_order": ["vector-op", "cube-op"],
                        }
                        for step in range(3)
                    ],
                }
            ],
        }

    def _analyze(self, capture):
        capture["capture_fingerprint"] = multistream_operator_order.fingerprint(capture)
        path = self.root / "capture.json"
        path.write_text(json.dumps(capture) + "\n")
        return multistream_operator_order.analyze(
            path,
            self.root,
            expected_request_fingerprint=capture["request_fingerprint"],
        )

    def _route3_capture(self):
        self.source.write_text(
            "def run(x):\n"
            "    vector = vector_op(x)\n"
            "    cube = cube_op(x)\n"
            "    aux = aux_op(x)\n"
            "    return cube + vector + aux\n"
        )
        source = self.source.read_bytes()
        starts = {
            name: source.index(f"    {name}".encode())
            for name in ("vector", "cube", "aux")
        }
        end = source.index(b"    return")
        statements = []
        names = ["vector", "cube", "aux"]
        for index, name in enumerate(names):
            statements.append(
                {
                    "statement_id": name,
                    "start_offset": starts[name],
                    "end_offset": starts[names[index + 1]]
                    if index + 1 < len(names)
                    else end,
                    "movable": True,
                    "side_effect_free": True,
                }
            )
        capture = {
            "schema_version": multistream_operator_order.CAPTURE_SCHEMA,
            "capture_id": "order-capture-route3",
            "request_fingerprint": "request-fp-1",
            "source_files": [
                {
                    "path": "model.py",
                    "size_bytes": self.source.stat().st_size,
                    "file_fingerprint": multistream_operator_order.file_fingerprint(
                        self.source
                    ),
                }
            ],
            "targets": [
                {
                    "range_id": "range-1",
                    "graph_occurrence_fingerprint": "occurrence-1",
                    "source_file": "model.py",
                    "range_start_offset": starts["vector"],
                    "range_end_offset": end,
                    "statements": statements,
                    "hard_dependencies": [],
                    "dependency_evidence": self._dependency_evidence(
                        names, [], "request-fp-1"
                    ),
                    "occurrences": [
                        {
                            "alignment_id": f"decode-{step}",
                            "sk_off_operators": [
                                {
                                    "operator_id": "cube-op",
                                    "statement_id": "cube",
                                    "stream_id": 1,
                                    **self._resource_fields("CUBE"),
                                    "start_us": 0.0,
                                    "duration_us": 10.0,
                                },
                                {
                                    "operator_id": "aux-op",
                                    "statement_id": "aux",
                                    "stream_id": 3,
                                    **self._resource_fields("VECTOR"),
                                    "start_us": 0.0,
                                    "duration_us": 10.0,
                                },
                                {
                                    "operator_id": "vector-op",
                                    "statement_id": "vector",
                                    "stream_id": 2,
                                    **self._resource_fields("VECTOR"),
                                    "start_us": 5.0,
                                    "duration_us": 8.0,
                                },
                            ],
                            "sk_on_dispatch_order": ["vector-op", "cube-op", "aux-op"],
                        }
                        for step in range(3)
                    ],
                }
            ],
        }
        return capture

    def test_multistream_inversion_produces_minimal_route2_reorder(self):
        analysis = self._analyze(self._capture())
        target = analysis["targets"][0]
        self.assertTrue(target["multistream_reorder_authorized"])
        self.assertEqual(target["route2_order"], ["cube", "vector"])
        self.assertEqual(target["current_source_order"], ["vector", "cube"])
        self.assertEqual(
            target["stable_resource_complementary_pairs"], [["cube", "vector"]]
        )
        self.assertEqual(target["stable_dispatch_inversions"], [["cube", "vector"]])

    def test_csv_execution_order_wins_over_submicrosecond_start_reversal(self):
        capture = self._capture()
        for index, occurrence in enumerate(capture["targets"][0]["occurrences"]):
            cube, vector = occurrence["sk_off_operators"]
            cube["start_us"] = 5.0 + (0.25 if index % 2 else 0.0)
            vector["start_us"] = 5.0 if index % 2 else 5.25
        analysis = self._analyze(capture)
        target = analysis["targets"][0]
        self.assertTrue(target["multistream_reorder_authorized"])
        self.assertEqual(target["stable_execution_preferences"], [["cube", "vector"]])
        self.assertEqual(target["route2_order"], ["cube", "vector"])

    def test_mix_pair_never_authorizes_business_operator_reorder(self):
        analysis = self._analyze(self._capture(core_families=("MIX", "MIX")))
        target = analysis["targets"][0]
        self.assertFalse(target["multistream_reorder_authorized"])
        self.assertEqual(target["stable_resource_complementary_pairs"], [])
        self.assertIn("resource_complementary_pair_missing", target["blockers"])
        self.assertIn("mix_resource_pair_excluded", target["blockers"])

    def test_same_engine_pair_never_authorizes_business_operator_reorder(self):
        analysis = self._analyze(self._capture(core_families=("CUBE", "CUBE")))
        target = analysis["targets"][0]
        self.assertFalse(target["multistream_reorder_authorized"])
        self.assertEqual(target["stable_resource_complementary_pairs"], [])
        self.assertIn("resource_complementary_pair_missing", target["blockers"])
        self.assertIn("same_engine_pair_excluded", target["blockers"])

    def test_single_stream_never_authorizes_business_operator_reorder(self):
        analysis = self._analyze(self._capture(stream_ids=(1, 1)))
        target = analysis["targets"][0]
        self.assertFalse(target["multistream_reorder_authorized"])
        self.assertIn("direct_multistream_evidence_missing", target["blockers"])
        self.assertIsNone(target["route2_order"])

    def test_hard_dependency_cannot_be_reversed(self):
        analysis = self._analyze(
            self._capture(
                hard_dependencies=[
                    {"before": "vector", "after": "cube", "kind": "DATA"}
                ]
            )
        )
        target = analysis["targets"][0]
        self.assertFalse(target["multistream_reorder_authorized"])
        self.assertIn(
            "stable_preference_conflicts_with_hard_dependency", target["blockers"]
        )

    def test_incomplete_dependency_provider_coverage_blocks_analysis(self):
        capture = self._capture()
        fragments_path = self.root / "dependency-fragments.json"
        fragments = json.loads(fragments_path.read_text())
        fragments["providers"][0]["covered_dependency_kinds"] = ["DATA"]
        fragments["fragment_set_fingerprint"] = (
            multistream_dependency_evidence.fingerprint(
                {
                    key: item
                    for key, item in fragments.items()
                    if key != "fragment_set_fingerprint"
                }
            )
        )
        fragments_path.write_text(json.dumps(fragments) + "\n")
        evidence = multistream_dependency_evidence.build(fragments_path, self.root)
        (self.root / "dependency-evidence.json").write_text(json.dumps(evidence) + "\n")
        capture["capture_fingerprint"] = multistream_operator_order.fingerprint(capture)
        path = self.root / "capture.json"
        path.write_text(json.dumps(capture) + "\n")
        with self.assertRaisesRegex(ValueError, "dependency evidence is incomplete"):
            multistream_operator_order.analyze(
                path, self.root, expected_request_fingerprint="request-fp-1"
            )

    def test_route3_search_is_bounded_and_keeps_hard_dependencies(self):
        orders = multistream_operator_order._bounded_orders(
            ["a", "b", "c"],
            {("a", "c")},
            ["a", "b", "c"],
            ["b", "a", "c"],
            maximum=4,
        )
        self.assertLessEqual(len(orders), 4)
        self.assertEqual(orders[0], ["a", "b", "c"])
        for order in orders:
            self.assertLess(order.index("a"), order.index("c"))

    def test_route3_requires_settled_route2_and_keeps_stable_preferences(self):
        analysis = self._analyze(self._route3_capture())
        target = analysis["targets"][0]
        self.assertEqual(target["route2_order"], ["cube", "vector", "aux"])
        self.assertEqual(
            target["route3_orders"],
            [["cube", "aux", "vector"]],
        )
        for order in target["route3_orders"]:
            for before, after in target["stable_dispatch_inversions"]:
                self.assertLess(order.index(before), order.index(after))

        analysis_path = self.root / "analysis.json"
        analysis_path.write_text(json.dumps(analysis) + "\n")
        with self.assertRaisesRegex(
            ValueError, "route3 requires route2 dispatch evidence"
        ):
            multistream_execution.materialize_operator_reorder(
                self.source,
                self.root / "route3.py",
                analysis_path,
                "range-1",
                target["route3_orders"][0],
                "MS-R3",
                self.root,
            )

    def test_route3_clean_evidence_must_be_stable_non_regressing_no_gain(self):
        raw = self.root / "clean.log"
        raw.write_text("sealed clean evidence\n")

        def write_evidence(name, mean_improvement, median_improvement):
            evidence = {
                "schema_version": multistream_evidence.EVIDENCE_SCHEMA,
                "evidence_kind": "clean",
                "state_after": "clean3_passed",
                "trial_id": "MS-R2",
                "request_fingerprint": "request-fp-1",
                "inputs": {},
                "source_files": [
                    {
                        "path": raw.name,
                        "size_bytes": raw.stat().st_size,
                        "file_fingerprint": multistream_operator_order.file_fingerprint(
                            raw
                        ),
                    }
                ],
                "semantic_result": {
                    "candidate": {
                        "option_trial_evaluation": {
                            "checks": {
                                "baseline_stable": True,
                                "baseline_min_runs": True,
                                "candidate_min_runs": True,
                                "mean_improvement": False,
                                "p90_no_material_regression": True,
                                "stddev_no_material_regression": True,
                            },
                            "mean_improvement_pct": mean_improvement,
                            "median_run_improvement_pct": median_improvement,
                        }
                    }
                },
                "decision": "reject",
            }
            evidence["evidence_fingerprint"] = multistream_evidence.content_fingerprint(
                evidence
            )
            path = self.root / name
            path.write_text(json.dumps(evidence) + "\n")
            return path

        valid = write_evidence("clean-neutral.json", 0.5, 0.1)
        result = multistream_operator_order.validate_route2_no_gain_evidence(
            valid,
            self.root,
            trial_id="MS-R2",
            request_fingerprint="request-fp-1",
        )
        self.assertEqual(result["decision"], "reject")

        regressed = write_evidence("clean-regressed.json", -0.5, -0.1)
        with self.assertRaisesRegex(ValueError, "mean or median clean regression"):
            multistream_operator_order.validate_route2_no_gain_evidence(
                regressed,
                self.root,
                trial_id="MS-R2",
                request_fingerprint="request-fp-1",
            )

    def test_materialize_reorder_moves_only_verified_statement_blocks(self):
        analysis = self._analyze(self._capture())
        analysis_path = self.root / "analysis.json"
        analysis_path.write_text(json.dumps(analysis) + "\n")
        output = self.root / "reordered.py"
        manifest = multistream_execution.materialize_operator_reorder(
            self.source,
            output,
            analysis_path,
            "range-1",
            ["cube", "vector"],
            "MS-R1",
            self.root,
        )
        self.assertLess(
            output.read_text().index("cube ="), output.read_text().index("vector =")
        )
        self.assertEqual(manifest["change_kind"], "dependency_safe_operator_reorder")
        self.assertTrue(manifest["multistream_only_verified"])
        self.assertEqual(manifest["resource_pair_policy"], "cube_vector_only")
        self.assertEqual(
            manifest["stable_resource_complementary_pairs"], [["cube", "vector"]]
        )
        self.assertEqual(manifest["only_change"]["before_order"], ["vector", "cube"])
        self.assertEqual(manifest["only_change"]["after_order"], ["cube", "vector"])

        manifest_path = self.root / "action.json"
        manifest_path.write_text(json.dumps(manifest) + "\n")
        dispatch_raw = self.root / "post-dispatch.log"
        dispatch_raw.write_text("sealed dispatch source\n")
        capture = {
            "schema_version": multistream_operator_order.DISPATCH_CAPTURE_SCHEMA,
            "trial_id": "MS-R1",
            "request_fingerprint": "request-fp-1",
            "action_manifest_fingerprint": multistream_operator_order.fingerprint(
                manifest
            ),
            "range_id": "range-1",
            "child_set_preserved": True,
            "stream_identity_complete": True,
            "occurrences": [
                {
                    "alignment_id": f"decode-{step}",
                    "observed_statement_order": ["cube", "vector"],
                    "stream_ids": [1, 2],
                }
                for step in range(3)
            ],
            "source_files": [
                {
                    "path": dispatch_raw.name,
                    "size_bytes": dispatch_raw.stat().st_size,
                    "file_fingerprint": multistream_operator_order.file_fingerprint(
                        dispatch_raw
                    ),
                }
            ],
        }
        capture["capture_fingerprint"] = multistream_operator_order.fingerprint(capture)
        capture_path = self.root / "post-dispatch.json"
        capture_path.write_text(json.dumps(capture) + "\n")
        evidence = multistream_operator_order.build_dispatch_evidence(
            manifest_path, capture_path, self.root
        )
        self.assertEqual(evidence["decision"], "pass")
        self.assertEqual(evidence["aligned_occurrence_count"], 3)

    def test_post_reorder_single_stream_dispatch_is_rejected(self):
        analysis = self._analyze(self._capture())
        analysis_path = self.root / "analysis.json"
        analysis_path.write_text(json.dumps(analysis) + "\n")
        manifest = multistream_execution.materialize_operator_reorder(
            self.source,
            self.root / "reordered.py",
            analysis_path,
            "range-1",
            ["cube", "vector"],
            "MS-R1",
            self.root,
        )
        manifest_path = self.root / "action.json"
        manifest_path.write_text(json.dumps(manifest) + "\n")
        raw = self.root / "dispatch.log"
        raw.write_text("dispatch\n")
        capture = {
            "schema_version": multistream_operator_order.DISPATCH_CAPTURE_SCHEMA,
            "trial_id": "MS-R1",
            "request_fingerprint": "request-fp-1",
            "action_manifest_fingerprint": multistream_operator_order.fingerprint(
                manifest
            ),
            "range_id": "range-1",
            "child_set_preserved": True,
            "stream_identity_complete": True,
            "occurrences": [
                {
                    "alignment_id": f"decode-{step}",
                    "observed_statement_order": ["cube", "vector"],
                    "stream_ids": [1, 1],
                }
                for step in range(3)
            ],
            "source_files": [
                {
                    "path": raw.name,
                    "size_bytes": raw.stat().st_size,
                    "file_fingerprint": multistream_operator_order.file_fingerprint(
                        raw
                    ),
                }
            ],
        }
        capture["capture_fingerprint"] = multistream_operator_order.fingerprint(capture)
        capture_path = self.root / "post-dispatch.json"
        capture_path.write_text(json.dumps(capture) + "\n")
        with self.assertRaisesRegex(ValueError, "is not multistream"):
            multistream_operator_order.build_dispatch_evidence(
                manifest_path, capture_path, self.root
            )

    def test_planner_prioritizes_route2_and_requires_degraded_trace(self):
        analysis = self._analyze(self._capture())
        analysis_path = self.root / "analysis.json"
        analysis_path.write_text(json.dumps(analysis) + "\n")
        request = {
            "request_id": "request-1",
            "artifacts": {
                "option_history": "history.json",
                "source_scope_map": "source-map.json",
            },
            "requested_change_kinds": ["dependency_safe_operator_reorder"],
            "budget": {"max_trials": 4},
        }
        request_path = self.root / "request.json"
        request_path.write_text(json.dumps(request) + "\n")
        (self.root / "history.json").write_text(
            json.dumps(
                {
                    "schema_version": multistream_candidate_planner.HISTORY_SCHEMA,
                    "settled_trials": [],
                }
            )
            + "\n"
        )
        (self.root / "source-map.json").write_text("{}\n")
        trace = {
            "targets": [
                {
                    "range_id": "range-1",
                    "net_effect": "beneficial",
                    "parallelism_effect": "degraded",
                }
            ]
        }
        trace_path = self.root / "trace.json"
        trace_path.write_text(json.dumps(trace) + "\n")
        catalog = {
            "schema_version": multistream_candidate_planner.CATALOG_SCHEMA,
            "options": [],
            "source_actions": [],
        }
        catalog["catalog_fingerprint"] = multistream_candidate_planner.fingerprint(
            catalog
        )
        catalog_path = self.root / "catalog.json"
        catalog_path.write_text(json.dumps(catalog) + "\n")
        summary = {
            "request_fingerprint": "request-fp-1",
            "target_range_ids": ["range-1"],
            "requested_change_kinds": ["dependency_safe_operator_reorder"],
            "analysis_targets": {
                "range-1": {
                    "graph_occurrence_fingerprint": "occurrence-1",
                    "mapping_method": "source_scope_map",
                    "mapping_confidence": "exact",
                    "boundary": {
                        "source_file": "model.py",
                        "start_offset": analysis["targets"][0]["range_start_offset"],
                        "end_offset": analysis["targets"][0]["range_end_offset"],
                    },
                }
            },
        }
        with (
            mock.patch.object(
                multistream_candidate_planner.multistream_contract,
                "validate_request",
                return_value=summary,
            ),
            mock.patch.object(
                multistream_candidate_planner.multistream_trace_analysis,
                "validate_analysis",
                return_value={"analysis_fingerprint": "trace-fp"},
            ),
        ):
            matrix = multistream_candidate_planner.plan(
                request_path, trace_path, catalog_path, self.root, analysis_path
            )
            candidate = matrix["candidates"][0]
            self.assertEqual(
                candidate["change_kind"], "dependency_safe_operator_reorder"
            )
            self.assertEqual(candidate["route"], "route2")
            self.assertTrue(candidate["selected_for_execution"])

            trace["targets"][0]["parallelism_effect"] = "preserved"
            trace_path.write_text(json.dumps(trace) + "\n")
            with self.assertRaisesRegex(ValueError, "requires degraded parallelism"):
                multistream_candidate_planner.plan(
                    request_path, trace_path, catalog_path, self.root, analysis_path
                )

    def test_request_contract_allows_reorder_only_with_direct_multistream_capture(self):
        for name in (
            "clean.json",
            "baseline-manifest.json",
            "candidate-manifest.json",
            "projection.json",
            "environment.json",
            "history.json",
            "source-map.json",
        ):
            (self.root / name).write_text("{}\n")
        source_bytes = self.source.read_bytes()
        start = source_bytes.index(b"    vector")
        end = source_bytes.index(b"    return")
        request = {
            "schema_version": multistream_contract.REQUEST_SCHEMA,
            "request_id": "request-1",
            "parent_experiment_id": "parent-1",
            "incumbent": {
                "candidate_name": "Sbest-BASE",
                "source_revision": "source-r1",
                "source_fingerprint": "source-fp-1",
                "config_fingerprint": "config-fp-1",
                "control_fingerprint": "control-fp-1",
                "workload_fingerprint": "workload-fp-1",
                "clean_performance_summary": "clean.json",
                "profiling_analysis_result": "profiling-analysis.json",
                "clean_run_count": 5,
                "stable": True,
            },
            "artifacts": {
                "baseline_collection_manifest": "baseline-manifest.json",
                "candidate_collection_manifest": "candidate-manifest.json",
                "projected_trace_mapping": "projection.json",
                "environment_evidence": "environment.json",
                "option_history": "history.json",
                "source_scope_map": "source-map.json",
                "operator_order_capture": "operator-order-capture.json",
            },
            "targets": [
                {
                    "range_id": "range-1",
                    "graph_occurrence_fingerprint": "occurrence-1",
                    "net_effect": "beneficial",
                    "parallelism_effect": "degraded",
                    "optimization_status": "opportunity",
                }
            ],
            "requested_change_kinds": ["dependency_safe_operator_reorder"],
            "isolation": {
                "source_worktree": "isolated/source",
                "experiment_root": "isolated/experiments",
                "config_root": "isolated/config",
                "cache_root": "isolated/cache",
                "immutable_incumbent": True,
                "dedicated_source_worktree": True,
                "dedicated_experiment_root": True,
                "dedicated_config_root": True,
                "dedicated_cache_namespace": True,
            },
            "authorization": {"run_inference": True, "edit_isolated_worktree": True},
            "execution": {
                "command_argv": ["python3", "run.py"],
                "correctness_method": "frozen token comparison",
                "expected_ranks": 8,
                "warmup": 8,
            },
            "budget": {
                "max_trials": 4,
                "max_recollections_per_trial": 3,
                "timeout_seconds": 1800,
            },
        }
        profiling = {
            "schema_version": "1.2",
            "candidate_name": "Sbest-BASE",
            "source_revision": "source-r1",
            "candidate_config_fingerprint": "config-fp-1",
            "control_fingerprint": "control-fp-1",
            "workload_fingerprint": "workload-fp-1",
            "source_scope_mapping": {
                "protocol": "source_scope_map_v2",
                "status": "exact",
            },
            "per_sk_decisions": [
                {
                    "range_id": "range-1",
                    "graph_occurrence_fingerprint": "occurrence-1",
                    "classification": "beneficial",
                    "mapping_method": "source_scope_map",
                    "mapping_confidence": "exact",
                    "candidate_occurrence_count": 3,
                    "original": {"interval_us": {"count": 3}},
                    "boundary": {
                        "source_file": "model.py",
                        "start_offset": start,
                        "end_offset": end,
                    },
                }
            ],
        }
        profiling["analysis_content_fingerprint"] = (
            multistream_contract._analysis_fingerprint(profiling)
        )
        (self.root / "profiling-analysis.json").write_text(json.dumps(profiling) + "\n")
        capture = self._capture(
            request_fingerprint=multistream_contract.content_fingerprint(request)
        )
        capture["capture_fingerprint"] = multistream_operator_order.fingerprint(capture)
        (self.root / "operator-order-capture.json").write_text(
            json.dumps(capture) + "\n"
        )
        summary = multistream_contract.validate_request(request, self.root)
        self.assertTrue(summary["operator_order_capture_available"])

        request["targets"][0]["parallelism_effect"] = "preserved"
        with self.assertRaisesRegex(
            ValueError, "allowed only for degraded multistream"
        ):
            multistream_contract.validate_request(request, self.root)


if __name__ == "__main__":
    unittest.main()
