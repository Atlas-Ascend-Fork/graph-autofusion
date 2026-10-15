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

import multistream_capture_producer  # noqa: E402
import multistream_dependency_evidence  # noqa: E402
import multistream_operator_order  # noqa: E402
import multistream_trace_analysis  # noqa: E402
import multistream_logical_graph  # noqa: E402
import multistream_critical_path  # noqa: E402


class CaptureProducerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plugins = self.root / "plugins"
        self.plugins.mkdir()
        self.source = self.root / "model.py"
        self.source.write_text(
            "def run(x):\n"
            "    vector = vector_op(x)\n"
            "    cube = cube_op(x)\n"
            "    return cube + vector\n"
        )
        self.payload = self.root / "payload.json"
        self.dependency_provider = self.root / "dependency_provider.py"
        self.dependency_provider.write_text("# reviewed provider\n")
        self.dependency_input = self.root / "dependency-input.json"
        self.dependency_input.write_text("{}\n")
        self.plugin = self.plugins / "fixture_plugin.py"
        self.plugin.write_text(
            "import json\n"
            "from pathlib import Path\n"
            "PLUGIN_API_VERSION = 'superkernel-multistream-capture-plugin-v1'\n"
            "PLUGIN_ID = 'fixture-plugin-v1'\n"
            "CAPABILITIES = ['logical_graph', 'critical_path', 'short_trace', 'operator_order', 'post_dispatch']\n"
            "def produce_capture(kind, context):\n"
            "    return json.loads(Path(context['inputs']['payload']).read_text())[kind]\n"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _operator_capture(self):
        content = self.source.read_bytes()
        vector = content.index(b"    vector")
        cube = content.index(b"    cube")
        end = content.index(b"    return")
        resource = {
            "cube": {
                "accelerator_core": "AI_CORE",
                "block_num": 16,
                "mix_block_num": 0,
            },
            "vector": {
                "accelerator_core": "AI_VECTOR_CORE",
                "block_num": 32,
                "mix_block_num": 0,
            },
        }
        fragments = {
            "schema_version": multistream_dependency_evidence.FRAGMENT_SET_SCHEMA,
            "fragment_set_id": "dependencies-1",
            "request_fingerprint": "request-fp-1",
            "range_id": "range-1",
            "graph_occurrence_fingerprint": "occurrence-1",
            "statement_ids": ["vector", "cube"],
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
        dependency_evidence = multistream_dependency_evidence.build(
            fragments_path, self.root
        )
        (self.root / "dependency-evidence.json").write_text(
            json.dumps(dependency_evidence) + "\n"
        )
        return {
            "schema_version": multistream_operator_order.CAPTURE_SCHEMA,
            "capture_id": "capture-1",
            "request_fingerprint": "request-fp-1",
            "source_files": [
                multistream_capture_producer.source_file_record(self.source, self.root)
            ],
            "targets": [
                {
                    "range_id": "range-1",
                    "graph_occurrence_fingerprint": "occurrence-1",
                    "source_file": "model.py",
                    "range_start_offset": vector,
                    "range_end_offset": end,
                    "statements": [
                        {
                            "statement_id": "vector",
                            "start_offset": vector,
                            "end_offset": cube,
                            "movable": True,
                            "side_effect_free": True,
                        },
                        {
                            "statement_id": "cube",
                            "start_offset": cube,
                            "end_offset": end,
                            "movable": True,
                            "side_effect_free": True,
                        },
                    ],
                    "hard_dependencies": [],
                    "dependency_evidence": "dependency-evidence.json",
                    "occurrences": [
                        {
                            "alignment_id": f"decode-{index}",
                            "sk_off_operators": [
                                {
                                    "operator_id": "cube-op",
                                    "statement_id": "cube",
                                    "stream_id": 1,
                                    **resource["cube"],
                                    "start_us": 0.0,
                                    "duration_us": 10.0,
                                },
                                {
                                    "operator_id": "vector-op",
                                    "statement_id": "vector",
                                    "stream_id": 2,
                                    **resource["vector"],
                                    "start_us": 5.0,
                                    "duration_us": 8.0,
                                },
                            ],
                            "sk_on_dispatch_order": ["vector-op", "cube-op"],
                        }
                        for index in range(3)
                    ],
                }
            ],
        }

    def _job(self, kind, validation):
        self.payload.write_text(json.dumps({kind: self._operator_capture()}) + "\n")
        return {
            "schema_version": multistream_capture_producer.JOB_SCHEMA,
            "job_id": "job-1",
            "capture_kind": kind,
            "artifact_root": str(self.root),
            "plugin": {
                "path": self.plugin.name,
                "file_fingerprint": multistream_capture_producer.file_fingerprint(
                    self.plugin
                ),
            },
            "inputs": {
                "payload": {
                    "path": self.payload.name,
                    "file_fingerprint": multistream_capture_producer.file_fingerprint(
                        self.payload
                    ),
                }
            },
            "parameters": {},
            "validation": validation,
            "capture_path": "capture.json",
            "receipt_path": "receipt.json",
        }

    def test_produces_and_conforms_operator_order_capture(self):
        job = self._job("operator_order", {"request_fingerprint": "request-fp-1"})
        job_path = self.root / "job.json"
        job_path.write_text(json.dumps(job) + "\n")
        receipt = multistream_capture_producer.produce(job_path, self.plugins)
        capture = json.loads((self.root / "capture.json").read_text())
        self.assertEqual(receipt["conformance"]["authorized_target_count"], 1)
        self.assertEqual(
            capture["capture_fingerprint"],
            multistream_capture_producer.fingerprint(
                {
                    key: value
                    for key, value in capture.items()
                    if key != "capture_fingerprint"
                }
            ),
        )
        self.assertEqual(receipt["plugin"]["plugin_id"], "fixture-plugin-v1")

    def test_rejects_changed_plugin_before_execution(self):
        job = self._job("operator_order", {"request_fingerprint": "request-fp-1"})
        self.plugin.write_text(self.plugin.read_text() + "# changed\n")
        job_path = self.root / "job.json"
        job_path.write_text(json.dumps(job) + "\n")
        with self.assertRaisesRegex(ValueError, "plugin fingerprint mismatch"):
            multistream_capture_producer.produce(job_path, self.plugins)
        self.assertFalse((self.root / "capture.json").exists())

    def test_short_trace_uses_existing_analyzer_as_conformance_gate(self):
        capture = {
            "schema_version": multistream_trace_analysis.CAPTURE_SCHEMA,
            "capture_id": "trace-1",
            "trial_id": "trial-1",
            "request_fingerprint": "request-fp-1",
            "overflow_detected": False,
            "source_files": [
                multistream_capture_producer.source_file_record(self.source, self.root)
            ],
            "targets": [],
        }
        job = self._job("operator_order", {"request_fingerprint": "request-fp-1"})
        self.payload.write_text(json.dumps({"short_trace": capture}) + "\n")
        job["capture_kind"] = "short_trace"
        job["validation"] = {"request": "request.json"}
        (self.root / "request.json").write_text("{}\n")
        job["inputs"]["payload"]["file_fingerprint"] = (
            multistream_capture_producer.file_fingerprint(self.payload)
        )
        job_path = self.root / "job.json"
        job_path.write_text(json.dumps(job) + "\n")
        with mock.patch(
            "multistream_capture_producer.multistream_trace_analysis.analyze",
            return_value={"targets": [{"range_id": "range-1"}], "blockers": []},
        ) as analyze:
            receipt = multistream_capture_producer.produce(job_path, self.plugins)
        analyze.assert_called_once()
        self.assertEqual(receipt["conformance"]["target_count"], 1)

    def test_logical_graph_capture_uses_generic_validator(self):
        graph = {
            "schema_version": multistream_logical_graph.SCHEMA,
            "graph_id": "graph-1",
            "request_fingerprint": "request-fp-1",
            "stages": [
                {
                    "stage_id": "a",
                    "stream_role": "main",
                    "stream_reliable": True,
                    "source_statement_ids": ["a"],
                    "movable": False,
                    "blocked_effects": [],
                },
                {
                    "stage_id": "b",
                    "stream_role": "aux",
                    "stream_reliable": True,
                    "source_statement_ids": ["b"],
                    "movable": True,
                    "blocked_effects": [],
                },
                {
                    "stage_id": "c",
                    "stream_role": "main",
                    "stream_reliable": True,
                    "source_statement_ids": ["c"],
                    "movable": False,
                    "blocked_effects": [],
                },
            ],
            "event_edges": [
                {
                    "event_edge_id": "b.ready",
                    "producer_stage_id": "b",
                    "consumer_stage_id": "c",
                    "reuse_scope": "per_iteration",
                    "reuse_proven_safe": True,
                }
            ],
            "forks": [
                {
                    "fork_id": "fork",
                    "source_stage_id": "a",
                    "branch_stage_ids": ["a", "b"],
                }
            ],
            "joins": [
                {
                    "join_id": "join",
                    "fork_id": "fork",
                    "branch_stage_ids": ["a", "b"],
                    "downstream_stage_id": "c",
                    "required_event_edge_ids": ["b.ready"],
                }
            ],
            "hard_dependencies": [
                {"before_stage_id": "a", "after_stage_id": "c", "kind": "DATA"},
                {"before_stage_id": "b", "after_stage_id": "c", "kind": "EVENT"},
            ],
            "dependency_evidence_fingerprint": "dependency-fp-1",
        }
        graph["graph_fingerprint"] = multistream_logical_graph.fingerprint(graph)
        job = self._job("logical_graph", {"request_fingerprint": "request-fp-1"})
        self.payload.write_text(json.dumps({"logical_graph": graph}) + "\n")
        job["inputs"]["payload"]["file_fingerprint"] = (
            multistream_capture_producer.file_fingerprint(self.payload)
        )
        path = self.root / "job.json"
        path.write_text(json.dumps(job) + "\n")
        receipt = multistream_capture_producer.produce(path, self.plugins)
        self.assertEqual(receipt["conformance"]["join_count"], 1)

    def test_critical_path_capture_uses_generic_analyzer(self):
        capture = {
            "schema_version": multistream_critical_path.CAPTURE_SCHEMA,
            "capture_id": "critical-1",
            "request_fingerprint": "request-fp-1",
            "logical_graph": {},
            "clock_domain": "short-window-1",
            "timestamp_resolution_us": 0.01,
            "trace_overflow_detected": False,
            "occurrences": [],
        }
        job = self._job("critical_path", {"request_fingerprint": "request-fp-1"})
        self.payload.write_text(json.dumps({"critical_path": capture}) + "\n")
        job["inputs"]["payload"]["file_fingerprint"] = (
            multistream_capture_producer.file_fingerprint(self.payload)
        )
        path = self.root / "job.json"
        path.write_text(json.dumps(job) + "\n")
        with mock.patch(
            "multistream_capture_producer.multistream_critical_path.analyze",
            return_value={
                "request_fingerprint": "request-fp-1",
                "decision": "opportunity",
                "targets": [{}],
            },
        ) as analyze:
            receipt = multistream_capture_producer.produce(path, self.plugins)
        analyze.assert_called_once()
        self.assertEqual(receipt["conformance"]["decision"], "opportunity")

    def test_post_dispatch_capture_is_checked_against_action(self):
        action = {
            "trial_id": "trial-1",
            "change_kind": "dependency_safe_operator_reorder",
            "multistream_only_verified": True,
            "only_change": {
                "range_id": "range-1",
                "after_order": ["cube", "vector"],
            },
        }
        action_path = self.root / "action.json"
        action_path.write_text(json.dumps(action) + "\n")
        capture = {
            "schema_version": multistream_operator_order.DISPATCH_CAPTURE_SCHEMA,
            "trial_id": "trial-1",
            "request_fingerprint": "request-fp-1",
            "action_manifest_fingerprint": multistream_operator_order.fingerprint(
                action
            ),
            "range_id": "range-1",
            "child_set_preserved": True,
            "stream_identity_complete": True,
            "occurrences": [
                {
                    "alignment_id": f"decode-{index}",
                    "observed_statement_order": ["cube", "vector"],
                    "stream_ids": [1, 2],
                }
                for index in range(3)
            ],
            "source_files": [
                multistream_capture_producer.source_file_record(self.source, self.root)
            ],
        }
        job = self._job("operator_order", {"request_fingerprint": "request-fp-1"})
        self.payload.write_text(json.dumps({"post_dispatch": capture}) + "\n")
        job["capture_kind"] = "post_dispatch"
        job["validation"] = {"action_manifest": "action.json"}
        job["inputs"]["payload"]["file_fingerprint"] = (
            multistream_capture_producer.file_fingerprint(self.payload)
        )
        job_path = self.root / "job.json"
        job_path.write_text(json.dumps(job) + "\n")
        receipt = multistream_capture_producer.produce(job_path, self.plugins)
        self.assertEqual(receipt["conformance"]["decision"], "pass")
        self.assertEqual(receipt["conformance"]["aligned_occurrence_count"], 3)

    def test_csv_helpers_require_raw_columns_and_strict_numbers(self):
        csv_path = self.root / "kernel_details.csv"
        csv_path.write_text("Accelerator Core,Block Num,Mix Block Num\nAI_CORE,16,0\n")
        rows = multistream_capture_producer.read_csv_rows(
            csv_path, ("Accelerator Core", "Block Num", "Mix Block Num")
        )
        self.assertEqual(rows[0]["Accelerator Core"], "AI_CORE")
        self.assertEqual(
            multistream_capture_producer.parse_integer(rows[0]["Block Num"], "block"),
            16,
        )
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            multistream_capture_producer.read_csv_rows(csv_path, ("Stream Id",))


if __name__ == "__main__":
    unittest.main()
