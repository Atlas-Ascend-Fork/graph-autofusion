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
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_component_reorder  # noqa: E402


class ComponentReorderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "model.py"
        self.source.write_text(
            "class Model:\n"
            "    def mixed(self, value):\n"
            "        qbmm = value + 1\n"
            "        return qbmm\n\n"
            "    def moe(self, value):\n"
            "        combine = value * 2\n"
            "        return combine\n"
        )
        self.raw = self.source.read_bytes()
        self.map = self._source_map()
        self.capture = self._capture()

    def tearDown(self):
        self.temporary.cleanup()

    def _node_span(self, function, node_type):
        import ast

        tree = ast.parse(self.raw.decode())
        lines = [0]
        for index, value in enumerate(self.raw):
            if value == 10:
                lines.append(index + 1)
        for node in ast.walk(tree):
            if type(node).__name__ != node_type:
                continue
            if getattr(node, "name", None) != function and not (
                node_type == "Assign" and function in ast.unparse(node)
            ):
                continue
            start = lines[node.lineno - 1] + node.col_offset
            end = lines[node.end_lineno - 1] + node.end_col_offset
            return start, end
        self.fail(f"missing {function}/{node_type}")

    def _span(self, span_id, function, node_type, relation, operators):
        start, end = self._node_span(function, node_type)
        return {
            "span_id": span_id,
            "source_file": "model.py",
            "function_qualname": f"Model.{function}"
            if node_type != "Assign"
            else ("Model.mixed" if function == "qbmm" else "Model.moe"),
            "node_type": node_type,
            "start_offset": start,
            "end_offset": end,
            "before_fingerprint": multistream_component_reorder.bytes_fingerprint(
                self.raw[start:end]
            ),
            "relation": relation,
            "operator_ids": operators,
        }

    def _source_map(self):
        source_map = {
            "schema_version": multistream_component_reorder.SOURCE_MAP_SCHEMA,
            "source_revision": "source-r1",
            "target_set_id": "moe-qbmm-combine",
            "source_files": [
                {
                    "path": "model.py",
                    "file_fingerprint": multistream_component_reorder.file_fingerprint(
                        self.source
                    ),
                }
            ],
            "spans": [
                self._span("qbmm-call", "qbmm", "Assign", "operator_call", ["qbmm"]),
                self._span(
                    "combine-call", "combine", "Assign", "operator_call", ["combine"]
                ),
                self._span(
                    "mixed-transform", "mixed", "FunctionDef", "transform_region", []
                ),
                self._span(
                    "moe-transform", "moe", "FunctionDef", "transform_region", []
                ),
            ],
            "target_bindings": [
                {
                    "range_id": "moe-1",
                    "graph_occurrence_fingerprint": "occurrence-1",
                    "operator_spans": {"qbmm": "qbmm-call", "combine": "combine-call"},
                }
            ],
            "hard_dependencies": [],
            "dependency_coverage": {
                "covered_kinds": sorted(
                    multistream_component_reorder.HARD_DEPENDENCY_KINDS
                ),
                "complete": True,
            },
        }
        source_map["mapping_fingerprint"] = multistream_component_reorder.fingerprint(
            source_map
        )
        return source_map

    def _capture(self):
        capture = {
            "schema_version": multistream_component_reorder.CAPTURE_SCHEMA,
            "request_fingerprint": "request-fp-1",
            "target_set_id": self.map["target_set_id"],
            "source_map": "component-map.json",
            "source_map_fingerprint": self.map["mapping_fingerprint"],
            "component_policy": {
                "mix_statement_id": "qbmm",
                "complementary_statement_id": "combine",
                "mix_overlap_engine": "AIC",
                "complementary_engine": "AIV",
                "mix_same_engine": "AIV",
                "expected_component_counts": {
                    "qbmm": {"AIC": 1, "AIV": 1},
                    "combine": {"AIV": 1},
                },
                "before_order": ["qbmm", "combine"],
                "after_order": ["combine", "qbmm"],
            },
            "targets": [
                {
                    "range_id": "moe-1",
                    "occurrences": [
                        {
                            "alignment_id": f"decode-{index}",
                            "dispatch_order": ["qbmm", "combine"],
                            "operators": [
                                {
                                    "operator_id": "qbmm",
                                    "statement_id": "qbmm",
                                    "kernel_type": "MIX_1_1",
                                    "stream_id": 1,
                                    "components": [
                                        {
                                            "component_id": "qbmm-aic",
                                            "engine": "AIC",
                                            "start_us": 0,
                                            "duration_us": 2,
                                        },
                                        {
                                            "component_id": "qbmm-aiv",
                                            "engine": "AIV",
                                            "start_us": 0,
                                            "duration_us": 3,
                                        },
                                    ],
                                },
                                {
                                    "operator_id": "combine",
                                    "statement_id": "combine",
                                    "kernel_type": "AIV_ONLY",
                                    "stream_id": 2,
                                    "components": [
                                        {
                                            "component_id": "combine-aiv",
                                            "engine": "AIV",
                                            "start_us": 4,
                                            "duration_us": 9,
                                        },
                                    ],
                                },
                            ],
                        }
                        for index in range(3)
                    ],
                }
            ],
            "source_files": [
                {
                    "path": "model.py",
                    "file_fingerprint": multistream_component_reorder.file_fingerprint(
                        self.source
                    ),
                }
            ],
        }
        capture["capture_fingerprint"] = multistream_component_reorder.fingerprint(
            capture
        )
        return capture

    def _transform(self):
        spans = {item["span_id"]: item for item in self.map["spans"]}
        replacements = []
        for span_id, replacement in (
            (
                "mixed-transform",
                "def mixed(self, value):\n        qbmm = value + 1\n        return qbmm + 0",
            ),
            (
                "moe-transform",
                "def moe(self, value):\n        combine = value * 2\n        return combine + 0",
            ),
        ):
            span = spans[span_id]
            encoded = replacement.encode()
            replacements.append(
                {
                    "span_id": span_id,
                    "source_file": "model.py",
                    "start_offset": span["start_offset"],
                    "end_offset": span["end_offset"],
                    "before_fingerprint": span["before_fingerprint"],
                    "replacement": replacement,
                    "after_fingerprint": multistream_component_reorder.bytes_fingerprint(
                        encoded
                    ),
                }
            )
        replacement_plan_fingerprint = multistream_component_reorder.fingerprint(
            replacements
        )
        safety_proofs = {
            name: True for name in multistream_component_reorder.SAFETY_PROOFS
        }
        review = {
            "schema_version": multistream_component_reorder.REVIEW_SCHEMA,
            "review_policy_id": "reviewed-moe-component-v1",
            "allowed_change_kind": multistream_component_reorder.CHANGE_KIND,
            "source_revision": "source-r1",
            "target_set_id": self.map["target_set_id"],
            "source_map_fingerprint": self.map["mapping_fingerprint"],
            "component_capture_fingerprint": self.capture["capture_fingerprint"],
            "replacement_plan_fingerprint": replacement_plan_fingerprint,
            "approved_safety_proofs": safety_proofs,
        }
        review["review_fingerprint"] = multistream_component_reorder.fingerprint(review)
        review_path = self.root / "review.json"
        review_path.write_text(json.dumps(review) + "\n")
        sorted_replacements = sorted(
            replacements, key=lambda item: item["start_offset"]
        )
        cursor = 0
        chunks = []
        for replacement in sorted_replacements:
            chunks.append(self.raw[cursor : replacement["start_offset"]])
            chunks.append(replacement["replacement"].encode())
            cursor = replacement["end_offset"]
        chunks.append(self.raw[cursor:])
        projection = "sha256:" + "b" * 64
        dependency = "sha256:" + "c" * 64
        audit = {
            "schema_version": multistream_component_reorder.AUDIT_SCHEMA,
            "source_revision": "source-r1",
            "target_set_id": self.map["target_set_id"],
            "source_map_fingerprint": self.map["mapping_fingerprint"],
            "component_capture_fingerprint": self.capture["capture_fingerprint"],
            "replacement_plan_fingerprint": replacement_plan_fingerprint,
            "expected_output_fingerprint": multistream_component_reorder.bytes_fingerprint(
                b"".join(chunks)
            ),
            "single_stream_projection_fingerprint_before": projection,
            "single_stream_projection_fingerprint_after": projection,
            "dependency_contract_fingerprint_before": dependency,
            "dependency_contract_fingerprint_after": dependency,
            "safety_proofs": safety_proofs,
        }
        audit["audit_fingerprint"] = multistream_component_reorder.fingerprint(audit)
        audit_path = self.root / "audit.json"
        audit_path.write_text(json.dumps(audit) + "\n")
        transform = {
            "schema_version": multistream_component_reorder.TRANSFORM_SCHEMA,
            "trial_id": "MS-C1",
            "request_fingerprint": "request-fp-1",
            "target_set_id": self.map["target_set_id"],
            "source_map": "component-map.json",
            "source_map_fingerprint": self.map["mapping_fingerprint"],
            "component_capture": "component-capture.json",
            "component_capture_fingerprint": self.capture["capture_fingerprint"],
            "source_revision": "source-r1",
            "review_policy_id": "reviewed-moe-component-v1",
            "review_record": "review.json",
            "reviewer_fingerprint": multistream_component_reorder.file_fingerprint(
                review_path
            ),
            "single_stream_projection_fingerprint_before": projection,
            "single_stream_projection_fingerprint_after": projection,
            "dependency_contract_fingerprint_before": dependency,
            "dependency_contract_fingerprint_after": dependency,
            "post_transform_audit": "audit.json",
            "post_transform_audit_fingerprint": multistream_component_reorder.file_fingerprint(
                audit_path
            ),
            "safety_proofs": safety_proofs,
            "replacements": replacements,
        }
        transform["transform_fingerprint"] = multistream_component_reorder.fingerprint(
            transform
        )
        return transform

    def _write_inputs(self):
        (self.root / "component-map.json").write_text(json.dumps(self.map) + "\n")
        (self.root / "component-capture.json").write_text(
            json.dumps(self.capture) + "\n"
        )

    def test_validates_exact_multi_span_map_and_materializes_cross_function_hunks(self):
        self._write_inputs()
        source_map = multistream_component_reorder.validate_source_map(
            self.map, self.root
        )
        capture = multistream_component_reorder.validate_capture(
            self.capture, self.root, source_map
        )
        self.assertTrue(capture["authorized"])
        output = self.root / "candidate.py"
        manifest = multistream_component_reorder.materialize(
            self._transform(), source_map, self.capture, self.root, self.root, output
        )
        self.assertTrue(output.is_file())
        self.assertEqual(manifest["change_kind"], "component_overlap_reorder")
        self.assertTrue(manifest["component_aware_verified"])
        self.assertEqual(manifest["only_change"]["after_order"], ["combine", "qbmm"])

    def test_materialize_cli_keeps_candidate_and_manifest_outputs_distinct(self):
        self._write_inputs()
        transform_path = self.root / "transform.json"
        transform_path.write_text(json.dumps(self._transform()) + "\n")
        output = self.root / "candidate.py"
        manifest = self.root / "action-manifest.json"
        result = multistream_component_reorder.main(
            [
                "materialize",
                "--transform",
                str(transform_path),
                "--map",
                str(self.root / "component-map.json"),
                "--capture",
                str(self.root / "component-capture.json"),
                "--source-root",
                str(self.root),
                "--artifact-root",
                str(self.root),
                "--output",
                str(output),
                "--manifest-out",
                str(manifest),
            ]
        )
        self.assertEqual(result, 0)
        materialized = output.read_text()
        self.assertIn("def mixed(self, value):", materialized)
        self.assertIn("def moe(self, value):", materialized)
        self.assertEqual(
            json.loads(manifest.read_text())["materialized_source"], str(output)
        )

    def test_rejects_non_ast_exact_source_span(self):
        self.map["spans"][0]["start_offset"] += 1
        self.map["mapping_fingerprint"] = multistream_component_reorder.fingerprint(
            {
                key: value
                for key, value in self.map.items()
                if key != "mapping_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "exact AST node"):
            multistream_component_reorder.validate_source_map(self.map, self.root)

    def test_rejects_incomplete_mix_component_capture(self):
        self.capture["targets"][0]["occurrences"][0]["operators"][0]["components"] = [
            {
                "component_id": "qbmm-aic",
                "engine": "AIC",
                "start_us": 0,
                "duration_us": 2,
            },
        ]
        self.capture["capture_fingerprint"] = multistream_component_reorder.fingerprint(
            {
                key: value
                for key, value in self.capture.items()
                if key != "capture_fingerprint"
            }
        )
        source_map = multistream_component_reorder.validate_source_map(
            self.map, self.root
        )
        with self.assertRaisesRegex(ValueError, "complete AIC/AIV"):
            multistream_component_reorder.validate_capture(
                self.capture, self.root, source_map
            )

    def test_rejects_single_function_transform(self):
        transform = self._transform()
        transform["replacements"] = transform["replacements"][:1]
        transform["transform_fingerprint"] = multistream_component_reorder.fingerprint(
            {
                key: value
                for key, value in transform.items()
                if key != "transform_fingerprint"
            }
        )
        source_map = multistream_component_reorder.validate_source_map(
            self.map, self.root
        )
        with self.assertRaisesRegex(ValueError, "at least two replacement hunks"):
            multistream_component_reorder.validate_transform(
                transform, source_map, self.capture, self.root, self.root
            )

    def test_existing_worktree_copy_must_match_immutable_input(self):
        self._write_inputs()
        source_map = multistream_component_reorder.validate_source_map(
            self.map, self.root
        )
        output = self.root / "candidate.py"
        output.write_bytes(self.source.read_bytes())
        manifest = multistream_component_reorder.materialize(
            self._transform(), source_map, self.capture, self.root, self.root, output
        )
        self.assertEqual(
            manifest["input_source_fingerprint"],
            multistream_component_reorder.file_fingerprint(self.source),
        )

        stale = self.root / "stale.py"
        stale.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "differs from immutable input"):
            multistream_component_reorder.materialize(
                self._transform(), source_map, self.capture, self.root, self.root, stale
            )

    def test_dispatch_evidence_requires_all_target_occurrences_to_reverse_order(self):
        self._write_inputs()
        source_map = multistream_component_reorder.validate_source_map(
            self.map, self.root
        )
        action = multistream_component_reorder.materialize(
            self._transform(),
            source_map,
            self.capture,
            self.root,
            self.root,
            self.root / "candidate.py",
        )
        action_path = self.root / "action.json"
        action_path.write_text(json.dumps(action) + "\n")
        dispatch = {
            "schema_version": multistream_component_reorder.DISPATCH_CAPTURE_SCHEMA,
            "trial_id": "MS-C1",
            "request_fingerprint": "request-fp-1",
            "action_manifest_fingerprint": multistream_component_reorder.fingerprint(
                action
            ),
            "target_set_id": "moe-qbmm-combine",
            "target_range_ids": ["moe-1"],
            "child_set_preserved": True,
            "component_lanes_complete": True,
            "stream_identity_complete": True,
            "targets": [
                {
                    "range_id": "moe-1",
                    "occurrences": [
                        {
                            "alignment_id": f"decode-{index}",
                            "observed_statement_order": ["combine", "qbmm"],
                            "stream_ids": [2, 1],
                        }
                        for index in range(3)
                    ],
                }
            ],
            "source_files": [
                {
                    "path": "model.py",
                    "file_fingerprint": multistream_component_reorder.file_fingerprint(
                        self.source
                    ),
                }
            ],
        }
        dispatch["capture_fingerprint"] = multistream_component_reorder.fingerprint(
            dispatch
        )
        dispatch_path = self.root / "dispatch.json"
        dispatch_path.write_text(json.dumps(dispatch) + "\n")
        evidence = multistream_component_reorder.build_dispatch_evidence(
            action_path, dispatch_path, self.root
        )
        self.assertEqual(evidence["decision"], "pass")
        dispatch["targets"][0]["occurrences"][2]["observed_statement_order"] = [
            "qbmm",
            "combine",
        ]
        dispatch["capture_fingerprint"] = multistream_component_reorder.fingerprint(
            {
                key: value
                for key, value in dispatch.items()
                if key != "capture_fingerprint"
            }
        )
        dispatch_path.write_text(json.dumps(dispatch) + "\n")
        with self.assertRaisesRegex(ValueError, "does not match planned order"):
            multistream_component_reorder.build_dispatch_evidence(
                action_path, dispatch_path, self.root
            )


if __name__ == "__main__":
    unittest.main()
