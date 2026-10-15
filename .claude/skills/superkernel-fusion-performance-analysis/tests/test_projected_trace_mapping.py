# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import projected_trace_mapping as mapping  # noqa: E402 - load sibling scripts after sys.path setup
import structural_association as association  # noqa: E402 - load sibling scripts after sys.path setup


class ProjectedTraceMappingTest(unittest.TestCase):
    @staticmethod
    def _origin_kernel(func_name):
        return {
            "version": "1.0",
            "modelId": 48,
            "deviceId": 0,
            "totalStreams": 1,
            "totalNodes": 1,
            "streams": [
                {
                    "streamIdxInGraph": 6,
                    "nodeCount": 1,
                    "nodes": [
                        {
                            "nodeId": 7,
                            "streamIdxInGraph": 6,
                            "nodeIdxInStream": 0,
                            "nodeType": "KERNEL",
                            "preNodeId": "none",
                            "nextNodeId": "none",
                            "taskInfo": [
                                {
                                    "taskType": "KERNEL",
                                    "streamId": 94,
                                    "taskId": 11,
                                    "kernelParams": {
                                        "funcName": func_name,
                                        "kernelType": "AI_VECTOR_CORE",
                                        "numBlocks": 1,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        }

    def test_scope_marker_names_are_not_globally_topology_only(self):
        for func_name in (
            "sk_scope_kernel_begin_dav",
            "sk_placeholder_kernel_dav",
            "sk_scope_kernel_end_dav",
            "sk_scope_kernel_begin_dav_2201",
            "sk_placeholder_kernel_dav_2201",
            "sk_scope_kernel_end_dav_2201",
        ):
            with self.subTest(func_name=func_name):
                graph = association._graph_from_sk_origin_value(
                    self._origin_kernel(func_name),
                    model_role="main_decode",
                )
                self.assertFalse(graph.nodes[0].topology_only)

    @staticmethod
    def _projection_node(key, ordinal, func_name, *, stream="role-a"):
        return SimpleNamespace(
            key=key,
            stream_key=stream,
            stream_ordinal=ordinal,
            task_kind="KERNEL",
            canonical_op=mapping.projection_op(func_name, "AI_VECTOR_CORE"),
            core_family="AI_VECTOR_CORE",
            topology_only=False,
            provenance=(("func_name", func_name),),
        )

    @classmethod
    def _sentinel_graph(cls, *, end_ordinal=105, extra_stream=False):
        nodes = [
            cls._projection_node("begin", 10, "sk_scope_kernel_begin_dav_2201"),
            cls._projection_node("p1", 11, "sk_placeholder_kernel_dav_2201"),
            cls._projection_node("p2", 12, "sk_placeholder_kernel_dav_2201"),
            cls._projection_node("add", 13, "Add"),
            cls._projection_node(
                "p3", end_ordinal - 2, "sk_placeholder_kernel_dav_2201"
            ),
            cls._projection_node(
                "p4", end_ordinal - 1, "sk_placeholder_kernel_dav_2201"
            ),
            cls._projection_node("end", end_ordinal, "sk_scope_kernel_end_dav_2201"),
        ]
        if extra_stream:
            nodes.append(cls._projection_node("add-b", 0, "Add", stream="role-b"))
        return SimpleNamespace(nodes=tuple(nodes))

    @classmethod
    def _repeated_sentinel_graph(cls):
        first = list(cls._sentinel_graph(end_ordinal=20).nodes)
        second = [
            cls._projection_node("begin-2", 21, "sk_scope_kernel_begin_dav_2201"),
            cls._projection_node("p1-2", 22, "sk_placeholder_kernel_dav_2201"),
            cls._projection_node("p2-2", 23, "sk_placeholder_kernel_dav_2201"),
            cls._projection_node("mul-2", 24, "Mul"),
            cls._projection_node("p3-2", 30, "sk_placeholder_kernel_dav_2201"),
            cls._projection_node("p4-2", 31, "sk_placeholder_kernel_dav_2201"),
            cls._projection_node("end-2", 32, "sk_scope_kernel_end_dav_2201"),
        ]
        return SimpleNamespace(nodes=tuple(first + second))

    @staticmethod
    def _baseline_steps(*, ambiguous=False):
        row = {
            "source_row": 2,
            "stream_id": 20,
            "task_id": 1,
            "name": "Add",
            "projected_op": "Add",
            "core_family": "AI_VECTOR_CORE",
            "start_us": Decimal("1"),
            "duration_us": Decimal("1"),
        }
        result = {}
        for step in (3, 4, 5):
            streams = {
                20: {
                    "signature": (("Add", "AI_VECTOR_CORE"),),
                    "rows": (row,),
                },
            }
            if ambiguous:
                streams[21] = {
                    "signature": (("Add", "AI_VECTOR_CORE"),),
                    "rows": ({**row, "stream_id": 21},),
                }
            result[step] = streams
        return result

    def _guarded_exclusion(self, graph, groups, baseline_steps):
        graph_set = SimpleNamespace(
            graphs=(SimpleNamespace(device_id=0, model_id="48", graph=graph),)
        )
        baseline_rows = [
            {"step_id": step, "name": "Add", "projected_op": "Add"}
            for step in (3, 4, 5)
        ]
        with (
            mock.patch.object(
                mapping, "_candidate_profiler_marker_evidence", return_value=(3, [])
            ),
            mock.patch.object(
                mapping, "_updated_graph_marker_evidence", return_value=(1, [])
            ),
        ):
            return mapping.guarded_sentinel_exclusion(
                graph,
                graph_set,
                groups,
                SimpleNamespace(),
                baseline_rows,
                baseline_steps,
                [3, 4, 5],
                device_id=0,
                model_id=48,
            )

    def test_guarded_scope_sentinel_exclusion_accepts_complete_evidence(self):
        graph = self._sentinel_graph()
        child = SimpleNamespace(node_key="add")
        groups = (SimpleNamespace(nodes=(child,), kernel_nodes=(child,)),)
        excluded, evidence = self._guarded_exclusion(
            graph, groups, self._baseline_steps()
        )
        self.assertEqual(evidence["status"], "accepted")
        self.assertEqual(excluded, {"begin", "p1", "p2", "p3", "p4", "end"})
        self.assertTrue(all(evidence["gates"].values()))

    def test_scope_sentinel_proposal_rejects_incomplete_or_noncontiguous_chain(self):
        incomplete = self._sentinel_graph()
        incomplete.nodes = tuple(node for node in incomplete.nodes if node.key != "end")
        self.assertIn(
            "scope_sentinel_chain_incomplete",
            mapping._sentinel_chain_proposal(incomplete)["blockers"],
        )
        noncontiguous = self._sentinel_graph()
        next(
            node for node in noncontiguous.nodes if node.key == "p4"
        ).stream_ordinal = 103
        self.assertIn(
            "scope_sentinel_chain_noncontiguous",
            mapping._sentinel_chain_proposal(noncontiguous)["blockers"],
        )

    def test_scope_sentinel_proposal_partitions_repeated_tag_by_occurrence(self):
        proposal = mapping._sentinel_chain_proposal(self._repeated_sentinel_graph())
        self.assertEqual(proposal["status"], "proposed")
        self.assertEqual(proposal["tags"], ["2201"])
        self.assertEqual(proposal["occurrence_count"], 2)
        self.assertEqual(
            [item["occurrence_ordinal"] for item in proposal["signature"]], [0, 1]
        )
        self.assertEqual(len(proposal["node_keys"]), 12)

    def test_scope_sentinel_proposal_requires_business_nodes_between_halves(self):
        graph = self._sentinel_graph(end_ordinal=15)
        proposal = mapping._sentinel_chain_proposal(graph)
        self.assertEqual(proposal["status"], "rejected")
        self.assertIn("scope_sentinel_business_interval_missing", proposal["blockers"])

    def test_scope_sentinel_proposal_accepts_balanced_multistream_layout(self):
        graph = SimpleNamespace(
            nodes=(
                self._projection_node(
                    "begin", 10, "sk_scope_kernel_begin_dav_2201", stream="role-a"
                ),
                self._projection_node(
                    "p1", 11, "sk_placeholder_kernel_dav_2201", stream="role-a"
                ),
                self._projection_node(
                    "p2", 12, "sk_placeholder_kernel_dav_2201", stream="role-a"
                ),
                self._projection_node("add", 13, "Add", stream="role-a"),
                self._projection_node(
                    "p3", 20, "sk_placeholder_kernel_dav_2201", stream="role-b"
                ),
                self._projection_node(
                    "p4", 21, "sk_placeholder_kernel_dav_2201", stream="role-b"
                ),
                self._projection_node(
                    "end", 22, "sk_scope_kernel_end_dav_2201", stream="role-b"
                ),
            )
        )
        proposal = mapping._sentinel_chain_proposal(graph)
        self.assertEqual(proposal["status"], "proposed")
        self.assertEqual(proposal["stream_roles"], ["role-a", "role-b"])
        self.assertEqual(proposal["occurrence_count"], 1)

    def test_guarded_scope_sentinel_exclusion_rejects_fused_reference(self):
        graph = self._sentinel_graph()
        marker = SimpleNamespace(node_key="begin")
        child = SimpleNamespace(node_key="add")
        groups = (SimpleNamespace(nodes=(marker, child), kernel_nodes=(child,)),)
        excluded, evidence = self._guarded_exclusion(
            graph, groups, self._baseline_steps()
        )
        self.assertFalse(excluded)
        self.assertEqual(evidence["status"], "rejected")
        self.assertIn("scope_sentinel_referenced_by_fused_group", evidence["blockers"])

    def test_guarded_scope_sentinel_exclusion_rejects_profiler_visibility(self):
        graph = self._sentinel_graph()
        child = SimpleNamespace(node_key="add")
        groups = (SimpleNamespace(nodes=(child,), kernel_nodes=(child,)),)
        graph_set = SimpleNamespace(
            graphs=(SimpleNamespace(device_id=0, model_id="48", graph=graph),)
        )
        with (
            mock.patch.object(
                mapping,
                "_candidate_profiler_marker_evidence",
                return_value=(3, ["sk_scope_kernel_begin_dav_2201"]),
            ),
            mock.patch.object(
                mapping, "_updated_graph_marker_evidence", return_value=(1, [])
            ),
        ):
            excluded, evidence = mapping.guarded_sentinel_exclusion(
                graph,
                graph_set,
                groups,
                SimpleNamespace(),
                [{"step_id": 3, "name": "Add", "projected_op": "Add"}],
                self._baseline_steps(),
                [3, 4, 5],
                device_id=0,
                model_id=48,
            )
        self.assertFalse(excluded)
        self.assertIn(
            "scope_sentinel_visible_in_candidate_profiler", evidence["blockers"]
        )

    def test_guarded_scope_sentinel_exclusion_allows_calibration_visibility(self):
        graph = self._sentinel_graph()
        child = SimpleNamespace(node_key="add")
        groups = (SimpleNamespace(nodes=(child,), kernel_nodes=(child,)),)
        graph_set = SimpleNamespace(
            graphs=(SimpleNamespace(device_id=0, model_id="48", graph=graph),)
        )
        manifest = SimpleNamespace(
            manifest_value={"config": {"marker_namespace": "skcal"}}
        )
        with (
            mock.patch.object(
                mapping,
                "_candidate_profiler_marker_evidence",
                return_value=(3, ["sk_scope_kernel_begin_dav_2201"]),
            ),
            mock.patch.object(
                mapping, "_updated_graph_marker_evidence", return_value=(1, [])
            ),
        ):
            excluded, evidence = mapping.guarded_sentinel_exclusion(
                graph,
                graph_set,
                groups,
                manifest,
                [{"step_id": 3, "name": "Add", "projected_op": "Add"}],
                self._baseline_steps(),
                [3, 4, 5],
                device_id=0,
                model_id=48,
            )
        self.assertEqual(evidence["status"], "accepted")
        self.assertEqual(evidence["marker_namespace"], "skcal")
        self.assertEqual(
            evidence["candidate_profiler_policy"],
            "calibration_visibility_allowed",
        )
        self.assertEqual(excluded, {"begin", "p1", "p2", "p3", "p4", "end"})

    def test_guarded_scope_sentinel_exclusion_rejects_ambiguous_projection(self):
        graph = self._sentinel_graph(extra_stream=True)
        groups = ()
        excluded, evidence = self._guarded_exclusion(
            graph, groups, self._baseline_steps(ambiguous=True)
        )
        self.assertFalse(excluded)
        self.assertIn(
            "scope_sentinel_post_exclusion_mapping_not_unique", evidence["blockers"]
        )

    def test_projection_op_removes_profiler_wrapper_but_preserves_op(self):
        self.assertEqual(
            mapping.projection_op(
                "aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3",
                "MIX_AIC",
            ),
            "QuantBatchMatmulV3",
        )
        self.assertEqual(
            mapping.projection_op("aiv_all_gather_bfloat16_t", "COMMUNICATION"),
            "COMMUNICATION",
        )

    def test_stream_assignment_requires_unique_injective_solution(self):
        source = {
            "role-a": (("Add", "VECTOR"),),
            "role-b": (("MatMul", "CUBE"),),
        }
        baseline = {
            10: {"signature": (("MatMul", "CUBE"),), "rows": ()},
            20: {"signature": (("Add", "VECTOR"),), "rows": ()},
        }
        assignment, blocker, count = mapping.solve_unique_stream_assignment(
            source, baseline
        )
        self.assertEqual(assignment, {"role-a": 20, "role-b": 10})
        self.assertIsNone(blocker)
        self.assertEqual(count, 1)

    def test_repeated_stream_signatures_are_ambiguous(self):
        signature = (("Add", "VECTOR"),)
        source = {"role-a": signature, "role-b": signature}
        baseline = {
            10: {"signature": signature, "rows": ()},
            20: {"signature": signature, "rows": ()},
        }
        assignment, blocker, count = mapping.solve_unique_stream_assignment(
            source, baseline
        )
        self.assertIsNone(assignment)
        self.assertEqual(blocker, "kernel_projection_stream_assignment_ambiguous")
        self.assertEqual(count, 2)

    def test_materialize_group_maps_origin_ordinal_in_every_step(self):
        child = SimpleNamespace(node_key="origin:add", canonical_op="Add")
        group = SimpleNamespace(
            blockers=(),
            kernel_nodes=(child,),
            device_id=0,
            model_id=48,
            sk_id=7,
            name="sk_7_none_start_Add_end_Add",
            fused_fingerprint="a" * 64,
        )
        candidate_rows = [
            SimpleNamespace(
                device_id=0,
                model_id=48,
                sk_id=7,
                step_id=step,
                start_us=Decimal(step * 100),
                duration_us=Decimal("3"),
            )
            for step in (3, 4, 5)
        ]

        def row(step):
            return {
                "source_row": step,
                "stream_id": 20,
                "task_id": step,
                "name": "aclnnAdd_AddAiCore_Add",
                "projected_op": "Add",
                "core_family": "VECTOR",
                "start_us": Decimal(step * 10),
                "duration_us": Decimal("2"),
            }

        baseline_steps = {
            step: {
                20: {
                    "signature": (("Add", "VECTOR"),),
                    "rows": (row(step),),
                }
            }
            for step in (3, 4, 5)
        }
        result = mapping.materialize_group(
            group,
            candidate_rows,
            {"origin:add": ("role-a", 0, ("Add", "VECTOR"))},
            baseline_steps,
            {step: {"role-a": 20} for step in (3, 4, 5)},
            [3, 4, 5],
        )
        self.assertEqual(result["status"], "exact")
        self.assertEqual(result["mapping_confidence"], "exact_projected_trace")
        self.assertEqual(len(result["baseline_occurrences"]), 3)
        self.assertEqual(
            result["baseline_occurrences"][0]["children"][0]["baseline_source_row"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
