# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_critical_path  # noqa: E402
import multistream_logical_graph  # noqa: E402


class CriticalPathTest(unittest.TestCase):
    def _graph(self):
        graph = {
            "schema_version": multistream_logical_graph.SCHEMA,
            "graph_id": "graph-1",
            "request_fingerprint": "request-fp-1",
            "stages": [
                {
                    "stage_id": "main.cube",
                    "stream_role": "main",
                    "stream_reliable": True,
                    "source_statement_ids": ["main-cube"],
                    "movable": False,
                    "blocked_effects": [],
                },
                {
                    "stage_id": "aux.vector",
                    "stream_role": "auxiliary",
                    "stream_reliable": True,
                    "source_statement_ids": ["aux-vector"],
                    "movable": True,
                    "blocked_effects": [],
                },
                {
                    "stage_id": "join.consumer",
                    "stream_role": "main",
                    "stream_reliable": True,
                    "source_statement_ids": ["join-consumer"],
                    "movable": False,
                    "blocked_effects": [],
                },
            ],
            "event_edges": [
                {
                    "event_edge_id": "aux.ready",
                    "producer_stage_id": "aux.vector",
                    "consumer_stage_id": "join.consumer",
                    "reuse_scope": "per_iteration",
                    "reuse_proven_safe": True,
                }
            ],
            "forks": [
                {
                    "fork_id": "decode.fork",
                    "source_stage_id": "main.cube",
                    "branch_stage_ids": ["main.cube", "aux.vector"],
                }
            ],
            "joins": [
                {
                    "join_id": "decode.join",
                    "fork_id": "decode.fork",
                    "branch_stage_ids": ["main.cube", "aux.vector"],
                    "downstream_stage_id": "join.consumer",
                    "required_event_edge_ids": ["aux.ready"],
                }
            ],
            "hard_dependencies": [
                {
                    "before_stage_id": "aux.vector",
                    "after_stage_id": "join.consumer",
                    "kind": "EVENT",
                },
                {
                    "before_stage_id": "main.cube",
                    "after_stage_id": "join.consumer",
                    "kind": "DATA",
                },
            ],
            "dependency_evidence_fingerprint": "dependency-fp-1",
        }
        graph["graph_fingerprint"] = multistream_logical_graph.fingerprint(graph)
        return graph

    @staticmethod
    def _stage(stage_id, stream_id, core, start, duration, ready, mix=0):
        return {
            "stage_id": stage_id,
            "stream_id": stream_id,
            "accelerator_core": core,
            "block_num": 16,
            "mix_block_num": mix,
            "ready_time_us": ready,
            "start_us": start,
            "duration_us": duration,
        }

    def _capture(
        self,
        count=3,
        *,
        aux_start=8.0,
        aux_ready=0.0,
        event_delay=0.0,
        downstream_start=20.0,
        step_latency=100.0,
        aux_core="AI_VECTOR_CORE",
        aux_mix=0,
    ):
        occurrences = []
        for index in range(count):
            shift = index * 100.0
            occurrences.append(
                {
                    "alignment_id": f"decode-{index}",
                    "join_id": "decode.join",
                    "fork_time_us": shift,
                    "step_latency_us": step_latency,
                    "timeline_complete": True,
                    "stages": [
                        self._stage("main.cube", 1, "AI_CORE", shift, 10.0, shift),
                        self._stage(
                            "aux.vector",
                            2,
                            aux_core,
                            shift + aux_start,
                            4.0,
                            shift + aux_ready,
                            aux_mix,
                        ),
                        self._stage(
                            "join.consumer",
                            1,
                            "AI_VECTOR_CORE",
                            shift + downstream_start,
                            2.0,
                            shift + downstream_start,
                        ),
                    ],
                    "events": [
                        {
                            "event_edge_id": "aux.ready",
                            "event_kind": "notify",
                            "stream_id": 2,
                            "time_us": shift + aux_start + 4.0 + event_delay,
                        },
                    ],
                }
            )
        capture = {
            "schema_version": multistream_critical_path.CAPTURE_SCHEMA,
            "capture_id": "capture-1",
            "request_fingerprint": "request-fp-1",
            "logical_graph": self._graph(),
            "clock_domain": "bound-short-window-1",
            "timestamp_resolution_us": 0.01,
            "trace_overflow_detected": False,
            "occurrences": occurrences,
        }
        capture["capture_fingerprint"] = multistream_critical_path.fingerprint(capture)
        return capture

    def test_graph_rejects_event_not_connected_to_join(self):
        graph = self._graph()
        graph["joins"][0]["required_event_edge_ids"] = ["missing"]
        graph["graph_fingerprint"] = multistream_logical_graph.fingerprint(
            {key: value for key, value in graph.items() if key != "graph_fingerprint"}
        )
        with self.assertRaisesRegex(ValueError, "unknown event"):
            multistream_logical_graph.validate(graph)

    def test_analyze_identifies_critical_branch_and_complementary_deferred_stage(self):
        result = multistream_critical_path.analyze(self._capture())
        self.assertEqual(result["decision"], "opportunity")
        target = result["targets"][0]
        self.assertEqual(target["critical_branch_stage_id"], "aux.vector")
        self.assertEqual(target["actionable_stage_ids"], ["aux.vector"])
        self.assertEqual(target["actionable_event_edge_ids"], [])
        self.assertGreater(target["recoverable_us"]["p50"], 0.0)
        self.assertEqual(target["occurrence_count"], 3)

    def test_delayed_notify_authorizes_event_without_stage_action(self):
        result = multistream_critical_path.analyze(
            self._capture(aux_start=0.0, event_delay=8.0)
        )
        self.assertEqual(result["decision"], "opportunity")
        target = result["targets"][0]
        self.assertEqual(target["actionable_event_edge_ids"], ["aux.ready"])
        self.assertEqual(target["actionable_stage_ids"], [])
        self.assertEqual(target["event_recoverable_us"]["aux.ready"]["p50"], 8.0)

    def test_already_hidden_noncritical_branch_does_not_authorize_action(self):
        result = multistream_critical_path.analyze(
            self._capture(aux_start=0.0, aux_ready=0.0)
        )
        self.assertEqual(result["decision"], "no_event_or_stage_candidate")
        self.assertEqual(result["targets"][0]["reason"], "no_deferred_critical_stage")

    def test_mix_stage_cannot_be_resource_complementary_candidate(self):
        result = multistream_critical_path.analyze(
            self._capture(aux_core="MIX_AIV", aux_mix=4)
        )
        self.assertEqual(result["decision"], "no_event_or_stage_candidate")
        self.assertEqual(
            result["targets"][0]["reason"], "critical_stage_not_cube_vector"
        )

    def test_fewer_than_three_occurrences_blocks_analysis(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            multistream_critical_path.analyze(self._capture(count=2))

    def test_trace_overflow_blocks_analysis(self):
        capture = self._capture()
        capture["trace_overflow_detected"] = True
        capture["capture_fingerprint"] = multistream_critical_path.fingerprint(
            {
                key: value
                for key, value in capture.items()
                if key != "capture_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "overflow"):
            multistream_critical_path.analyze(capture)

    def test_unreliable_critical_stream_does_not_authorize_action(self):
        capture = self._capture()
        capture["logical_graph"]["stages"][1]["stream_reliable"] = False
        capture["logical_graph"]["graph_fingerprint"] = (
            multistream_logical_graph.fingerprint(
                {
                    key: value
                    for key, value in capture["logical_graph"].items()
                    if key != "graph_fingerprint"
                }
            )
        )
        capture["capture_fingerprint"] = multistream_critical_path.fingerprint(
            {
                key: value
                for key, value in capture.items()
                if key != "capture_fingerprint"
            }
        )
        result = multistream_critical_path.analyze(capture)
        self.assertEqual(result["decision"], "no_event_or_stage_candidate")
        self.assertEqual(result["targets"][0]["reason"], "critical_stream_unreliable")

    def test_ambiguous_event_reuse_is_rejected(self):
        graph = self._graph()
        graph["event_edges"][0]["reuse_proven_safe"] = False
        graph["graph_fingerprint"] = multistream_logical_graph.fingerprint(
            {key: value for key, value in graph.items() if key != "graph_fingerprint"}
        )
        with self.assertRaisesRegex(ValueError, "ambiguous event reuse"):
            multistream_logical_graph.validate(graph)

    def test_predicted_e2e_bound_below_three_percent_blocks_trial(self):
        result = multistream_critical_path.analyze(self._capture(step_latency=300.0))
        self.assertEqual(result["decision"], "no_event_or_stage_candidate")
        self.assertEqual(
            result["targets"][0]["reason"], "predicted_e2e_bound_below_gate"
        )


if __name__ == "__main__":
    unittest.main()
