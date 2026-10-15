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


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_sk_source_map import build_sk_source_map  # noqa: E402 - load sibling scripts after sys.path setup
from generate_source_unit_manifest import build_manifest  # noqa: E402 - load sibling scripts after sys.path setup
from marker_only_calibration import (  # noqa: E402 - load sibling scripts after sys.path setup
    build_marker_only_calibration,
    validate_marker_only_calibration,
)
from normalize_business_graph import normalize_origin_graph  # noqa: E402 - load sibling scripts after sys.path setup
from partition_calibration_scope_log import partition_scope_log, semantic_unit_matches  # noqa: E402 - load sibling scripts after sys.path setup
from project_calibration_graph import project_graphs  # noqa: E402 - load sibling scripts after sys.path setup
from source_calibration_common import atomic_write_json, canonical_sha256, file_sha256  # noqa: E402 - load sibling scripts after sys.path setup
from source_scope_map_v2 import (  # noqa: E402 - load sibling scripts after sys.path setup
    _content_fingerprint,
    _replay_calibration_projection,
    build_source_scope_map_v2,
    load_source_scope_map_v2,
)
from structural_association import graph_fingerprint, graph_from_sk_origin  # noqa: E402 - load sibling scripts after sys.path setup
from analyze_fusion_performance import _source_scope_entry_matches  # noqa: E402 - load sibling scripts after sys.path setup


def _graph(node_key, stream_role, *, collection):
    return {
        "protocol": "normalized_business_graph_v2",
        "collection_fingerprint": collection,
        "identity": {"device_id": 0, "model_id": "48"},
        "nodes": [
            {
                "node_key": node_key,
                "stream_role": stream_role,
                "stream_ordinal": 0,
                "canonical_op": "Compute",
                "core_family": "VECTOR",
                "observable_signature": {"dtype": "fp16", "rank": 2},
            }
        ],
        "edges": [],
    }


class SourceCalibrationTest(unittest.TestCase):
    def test_nested_outer_scope_suffix_uses_manifest_unit_id(self):
        self.assertEqual(
            semantic_unit_matches(
                "decode_skcal.block.2_skcal.unit.2.indexer.q_dynamic_quant_decode",
                {"indexer.q_dynamic_quant", "indexer.q"},
            ),
            [("2", "indexer.q_dynamic_quant")],
        )

    @staticmethod
    def _origin_graph(*func_names, model_id=48):
        nodes = []
        for index, func_name in enumerate(func_names):
            nodes.append(
                {
                    "nodeId": index + 1,
                    "streamIdxInGraph": 6,
                    "nodeIdxInStream": index,
                    "nodeType": "KERNEL",
                    "preNodeId": "none" if index == 0 else index,
                    "nextNodeId": "none" if index + 1 == len(func_names) else index + 2,
                    "taskInfo": [
                        {
                            "taskType": "KERNEL",
                            "streamId": 94,
                            "taskId": index + 10,
                            "kernelParams": {
                                "funcName": func_name,
                                "kernelType": "AI_VECTOR_CORE",
                                "numBlocks": 1,
                            },
                        }
                    ],
                }
            )
        return {
            "version": "1.0",
            "modelId": model_id,
            "deviceId": 0,
            "totalStreams": 1,
            "totalNodes": len(nodes),
            "streams": [
                {"streamIdxInGraph": 6, "nodeCount": len(nodes), "nodes": nodes}
            ],
        }

    def test_origin_graph_normalizer_reindexes_business_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sk_graph_origin.json"
            path.write_text(
                json.dumps(self._origin_graph("ComputeA", "ComputeB")), encoding="utf-8"
            )
            result = normalize_origin_graph(path, collection_fingerprint="a" * 64)
            self.assertEqual(result["protocol"], "normalized_business_graph_v2")
            self.assertEqual(
                [node["stream_ordinal"] for node in result["nodes"]], [0, 1]
            )
            self.assertEqual(
                [edge["kind"] for edge in result["edges"]], ["STREAM_ORDER"]
            )
            self.assertEqual(result["identity"], {"device_id": 0, "model_id": "48"})

    def test_origin_graph_normalizer_rejects_unguarded_scope_sentinel(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sk_graph_origin.json"
            path.write_text(
                json.dumps(
                    self._origin_graph("sk_scope_kernel_begin_dav_2201", "ComputeA")
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "accepted, complete"):
                normalize_origin_graph(path, collection_fingerprint="a" * 64)

    def test_origin_graph_normalizer_accepts_revalidated_sentinel_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sk_graph_origin.json"
            path.write_text(
                json.dumps(
                    self._origin_graph("sk_scope_kernel_begin_dav_2201", "ComputeA")
                ),
                encoding="utf-8",
            )
            graph = graph_from_sk_origin(path, model_role="candidate")
            trace = {
                "protocol": "kernel_projection_trace_v2",
                "identity": {"device_id": 0, "model_id": 48},
                "evidence": {
                    "profile_origin_graph_fingerprint": graph_fingerprint(graph)
                },
                "projection_exclusions": {
                    "protocol": "scope_sentinel_exclusion_v2",
                    "status": "accepted",
                    "excluded_node_keys": ["sk:n000000"],
                },
            }
            trace["mapping_fingerprint"] = canonical_sha256(trace)
            result = normalize_origin_graph(
                path,
                collection_fingerprint="a" * 64,
                projection_trace=trace,
            )
            self.assertEqual(len(result["nodes"]), 1)
            self.assertEqual(
                result["normalization_evidence"]["excluded_node_keys"],
                ["sk:n000000"],
            )

    def test_origin_graph_normalizer_matches_projection_base_model_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sk_graph_origin.json"
            path.write_text(
                json.dumps(
                    self._origin_graph(
                        "sk_scope_kernel_begin_dav_2201",
                        "ComputeA",
                        model_id="48_1",
                    )
                ),
                encoding="utf-8",
            )
            graph = graph_from_sk_origin(path, model_role="candidate")
            trace = {
                "protocol": "kernel_projection_trace_v2",
                "identity": {"device_id": 0, "model_id": 48},
                "evidence": {
                    "profile_origin_graph_fingerprint": graph_fingerprint(graph)
                },
                "projection_exclusions": {
                    "protocol": "scope_sentinel_exclusion_v2",
                    "status": "accepted",
                    "excluded_node_keys": ["sk:n000000"],
                },
            }
            trace["mapping_fingerprint"] = canonical_sha256(trace)

            result = normalize_origin_graph(
                path,
                collection_fingerprint="a" * 64,
                projection_trace=trace,
            )

            self.assertEqual(result["identity"], {"device_id": 0, "model_id": "48"})
            self.assertEqual(
                result["normalization_evidence"]["origin_model_id"], "48_1"
            )

    def test_scope_partitioner_only_assigns_continuation_break_trigger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scope_log = root / "sk_scope_split.log"
            scope_log.write_text(
                "\n".join(
                    [
                        "Scope split results begin: pass=InitialScopeSplitPass",
                        "Scope 7 (scopeId=7): 4 nodes, 1 streams, scopeNames=[skcal.block.2_skcal.unit.2.attn.quant]",
                        'BreakInfo: breakReason=UNFUSIBLE_NODE, triggerNode=11, triggerStream=6, detail="break"',
                        "[0] [nodeId:8, streamId:94] - KernelInfos{funcName:sk_scope_kernel_begin}",
                        "[1] [nodeId:10, streamId:94] - KernelInfos{funcName:DynamicQuant}",
                        "Scope 8 (scopeId=8): 3 nodes, 1 streams, scopeNames=[skcal.block.2_skcal.unit.2.attn.quant]",
                        'BreakInfo: breakReason=UNFUSIBLE_NODE, triggerNode=13, triggerStream=6, detail="break"',
                        "[0] [nodeId:12, streamId:94] - KernelInfos{funcName:DynamicQuantEpilog}",
                        "Scope 9 (scopeId=9): 1 nodes, 1 streams, scopeNames=[skcal.block.2]",
                        "[0] [nodeId:14, streamId:94] - EventNotify(eventId:1)",
                        "Scope split results end: pass=InitialScopeSplitPass",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            graph = {
                "nodes": [
                    {"node_key": "sk:n000010", "provenance": {"node_id": "10"}},
                    {"node_key": "sk:n000011", "provenance": {"node_id": "11"}},
                    {"node_key": "sk:n000012", "provenance": {"node_id": "12"}},
                    {"node_key": "sk:n000013", "provenance": {"node_id": "13"}},
                ]
            }
            manifest = {
                "manifest_fingerprint": "c" * 64,
                "model_adapter": {"fingerprint": "d" * 64},
                "block_templates": [{"block_template_id": "decoder.layer"}],
                "units": [
                    {
                        "block_template_id": "decoder.layer",
                        "unit_id": "attn.quant",
                    }
                ],
            }

            summary = partition_scope_log(
                scope_log, graph, manifest, [3, 4, 5], root / "partition"
            )
            assignments = json.loads(
                (root / "partition" / "unit-assignments.json").read_text(
                    encoding="utf-8"
                )
            )["assignments"]
            records = json.loads(
                (root / "partition" / "normalized-scope-records.json").read_text(
                    encoding="utf-8"
                )
            )["records"]

            assignment_artifact = json.loads(
                (root / "partition" / "unit-assignments.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                assignment_artifact["evidence_catalog"][
                    "fine-calibration-initial-scope-log"
                ]["relative_path"],
                "normalized-scope-records.json",
            )

            self.assertEqual(
                [item["calibration_node_key"] for item in assignments],
                ["sk:n000010", "sk:n000011", "sk:n000012"],
            )
            self.assertEqual(
                [item["evidence_kind"] for item in records],
                ["scope_member", "break_trigger_continuation", "scope_member"],
            )
            self.assertEqual(summary["ignored_break_trigger_business_node_count"], 1)
            self.assertEqual(summary["skipped_node_count"], 1)

    def test_stable_source_manifest_and_marker_only_calibration_restore_auto_source(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stable_root = root / "stable"
            calibration_root = root / "calibration"
            stable_root.mkdir()
            calibration_root.mkdir()
            (stable_root / "model.dsl").write_bytes(b"compute\n")
            (calibration_root / "model.dsl").write_bytes(b"BEGIN\ncompute\nEND\n")
            common = {
                "base_revision": "auto-base-rev",
                "calibration_revision": "auto-calibration-rev",
                "source_language": "example-dsl",
                "model_adapter": {
                    "name": "example-dsl-adapter",
                    "version": "1.0",
                    "fingerprint": "a" * 64,
                },
                "block_templates": [
                    {
                        "block_template_id": "block.execute",
                        "source_symbol": "Block.execute",
                        "instance_binding": "exact_scope_name",
                        "binding_fingerprint": "b" * 64,
                    }
                ],
            }
            stable_spec = {
                **common,
                "revision_role": "stable_source",
                "source_revision": "auto-base-rev",
                "units": [
                    {
                        "unit_id": "block.compute",
                        "block_template_id": "block.execute",
                        "source_symbol": "Block.execute",
                        "source_file": "model.dsl",
                        "start_offset": 0,
                        "end_offset": 8,
                        "normalized_syntax": {"kind": "call", "name": "compute"},
                    }
                ],
            }
            calibration_spec = {
                **common,
                "revision_role": "calibration",
                "source_revision": "auto-calibration-rev",
                "units": [
                    {
                        "unit_id": "block.compute",
                        "block_template_id": "block.execute",
                        "source_symbol": "Block.execute",
                        "source_file": "model.dsl",
                        "start_offset": 0,
                        "end_offset": 18,
                        "normalized_syntax": {"kind": "call", "name": "compute"},
                        "scope_name_template": "skcal.{block_instance_id}.compute",
                        "marker_operations": {
                            "begin": {
                                "operation_kind": "scope_begin",
                                "parser_node_id": "begin-0",
                                "start_offset": 0,
                                "end_offset": 5,
                                "insertion_start_offset": 0,
                                "insertion_end_offset": 6,
                            },
                            "end": {
                                "operation_kind": "scope_end",
                                "parser_node_id": "end-0",
                                "start_offset": 14,
                                "end_offset": 17,
                                "insertion_start_offset": 14,
                                "insertion_end_offset": 18,
                            },
                        },
                    }
                ],
            }
            stable_snapshot = root / "stable-snapshot"
            calibration_snapshot = root / "calibration-snapshot"
            stable_manifest, _ = build_manifest(
                stable_spec, stable_root, snapshot_root=stable_snapshot
            )
            calibration_manifest, _ = build_manifest(
                calibration_spec,
                calibration_root,
                snapshot_root=calibration_snapshot,
            )

            bridge = build_marker_only_calibration(
                stable_manifest=stable_manifest,
                calibration_manifest=calibration_manifest,
                stable_snapshot_root=stable_snapshot,
                calibration_snapshot_root=calibration_snapshot,
            )
            validated = validate_marker_only_calibration(
                bridge,
                stable_manifest=stable_manifest,
                calibration_manifest=calibration_manifest,
                stable_snapshot_root=stable_snapshot,
                calibration_snapshot_root=calibration_snapshot,
            )

            self.assertEqual(stable_manifest["revision_role"], "stable_source")
            self.assertNotIn("marker_operations", stable_manifest["units"][0])
            self.assertEqual(validated["status"], "exact")
            self.assertEqual(validated["unit_ids"], ["block.compute"])

            tampered = copy.deepcopy(bridge)
            tampered["files"][0]["stable_sha256"] = "f" * 64
            tampered["bridge_fingerprint"] = canonical_sha256(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "bridge_fingerprint"
                }
            )
            with self.assertRaisesRegex(ValueError, "stable source SHA"):
                validate_marker_only_calibration(
                    tampered,
                    stable_manifest=stable_manifest,
                    calibration_manifest=calibration_manifest,
                    stable_snapshot_root=stable_snapshot,
                    calibration_snapshot_root=calibration_snapshot,
                )

    def _artifacts(self, root, *, source_mode="stable_marker"):
        source = root / "worktree"
        source.mkdir()
        content = (
            b"compute\n" if source_mode == "stable_source" else b"BEGIN\ncompute\nEND\n"
        )
        (source / "model.dsl").write_bytes(content)
        adapter_output = {
            "revision_role": source_mode,
            "base_revision": "base-rev",
            "source_revision": "stable-rev",
            "calibration_revision": "calibration-rev",
            "source_language": "example-dsl",
            "model_adapter": {
                "name": "example-dsl-adapter",
                "version": "1.0",
                "fingerprint": "a" * 64,
            },
            "block_templates": [
                {
                    "block_template_id": "block.execute",
                    "source_symbol": "Block.execute",
                    "instance_binding": "exact_scope_name",
                    "binding_fingerprint": "b" * 64,
                }
            ],
            "units": [
                {
                    "unit_id": "block.compute",
                    "block_template_id": "block.execute",
                    "source_symbol": "Block.execute",
                    "source_file": "model.dsl",
                    "start_offset": 0,
                    "end_offset": len(content),
                    "normalized_syntax": {"kind": "scope", "body": ["compute"]},
                }
            ],
        }
        if source_mode == "stable_marker":
            adapter_output["stable_marker_revision"] = "stable-rev"
            adapter_output["units"][0].update(
                {
                    "scope_name_template": "skcal.{block_instance_id}.compute",
                    "marker_operations": {
                        "begin": {
                            "operation_kind": "scope_begin",
                            "parser_node_id": "begin-0",
                            "start_offset": 0,
                            "end_offset": 5,
                        },
                        "end": {
                            "operation_kind": "scope_end",
                            "parser_node_id": "end-0",
                            "start_offset": 14,
                            "end_offset": 17,
                        },
                    },
                }
            )
        bundle = root / "bundle"
        snapshot = bundle / "source" / "snapshot"
        source_manifest, _ = build_manifest(
            adapter_output, source, snapshot_root=snapshot
        )
        source_manifest_path = bundle / "source" / "source-unit-manifest.json"
        atomic_write_json(source_manifest_path, source_manifest)
        auto_evidence = {}
        if source_mode == "stable_source":
            calibration_source = root / "calibration-worktree"
            calibration_source.mkdir()
            (calibration_source / "model.dsl").write_bytes(b"BEGIN\ncompute\nEND\n")
            calibration_spec = copy.deepcopy(adapter_output)
            calibration_spec.update(
                {
                    "revision_role": "calibration",
                    "source_revision": "calibration-rev",
                }
            )
            calibration_spec["units"][0].update(
                {
                    "end_offset": 18,
                    "scope_name_template": "skcal.{block_instance_id}.compute",
                    "marker_operations": {
                        "begin": {
                            "operation_kind": "scope_begin",
                            "parser_node_id": "begin-0",
                            "start_offset": 0,
                            "end_offset": 5,
                            "insertion_start_offset": 0,
                            "insertion_end_offset": 6,
                        },
                        "end": {
                            "operation_kind": "scope_end",
                            "parser_node_id": "end-0",
                            "start_offset": 14,
                            "end_offset": 17,
                            "insertion_start_offset": 14,
                            "insertion_end_offset": 18,
                        },
                    },
                }
            )
            calibration_snapshot = bundle / "source" / "calibration-snapshot"
            calibration_manifest, _ = build_manifest(
                calibration_spec,
                calibration_source,
                snapshot_root=calibration_snapshot,
            )
            calibration_manifest_path = (
                bundle / "source" / "calibration-source-unit-manifest.json"
            )
            atomic_write_json(calibration_manifest_path, calibration_manifest)
            bridge = build_marker_only_calibration(
                stable_manifest=source_manifest,
                calibration_manifest=calibration_manifest,
                stable_snapshot_root=snapshot,
                calibration_snapshot_root=calibration_snapshot,
            )
            bridge_path = bundle / "mapping" / "marker-only-calibration.json"
            atomic_write_json(bridge_path, bridge)
            auto_evidence = {
                "calibration_source_manifest": calibration_manifest_path,
                "marker_only_calibration": bridge_path,
                "calibration_source_snapshot_root": calibration_snapshot,
            }

        original_graph = _graph(
            "original-1",
            "original-stream",
            collection=canonical_sha256("original-candidate"),
        )
        calibration_graph = _graph(
            "calibration-9",
            "calibration-stream",
            collection=canonical_sha256("calibration"),
        )
        runtime_validation = [
            {
                "step_id": step,
                "business_occurrence_count": 1,
                "assigned_occurrence_count": 1,
                "unscoped_occurrence_count": 0,
                "conflicting_assignment_count": 0,
                "parent_binding_exact": True,
                "unit_assignment_complete": True,
            }
            for step in (1, 2, 3)
        ]
        evidence_dir = bundle / "evidence"
        original_graph_path = evidence_dir / "original-graph.json"
        calibration_graph_path = evidence_dir / "calibration-graph.json"
        runtime_validation_path = evidence_dir / "runtime-validation.json"
        atomic_write_json(original_graph_path, original_graph)
        atomic_write_json(calibration_graph_path, calibration_graph)
        atomic_write_json(runtime_validation_path, runtime_validation)
        evidence_catalog = {
            "original_graph": {
                "relative_path": original_graph_path.relative_to(bundle).as_posix(),
                "sha256": file_sha256(original_graph_path),
            },
            "calibration_graph": {
                "relative_path": calibration_graph_path.relative_to(bundle).as_posix(),
                "sha256": file_sha256(calibration_graph_path),
            },
            "runtime_validation": {
                "relative_path": runtime_validation_path.relative_to(bundle).as_posix(),
                "sha256": file_sha256(runtime_validation_path),
            },
        }
        projection = project_graphs(
            original_graph,
            calibration_graph,
            runtime_validation=runtime_validation,
            evidence_catalog=evidence_catalog,
        )
        projection_path = bundle / "mapping" / "calibration-projection.json"
        atomic_write_json(projection_path, projection)
        block_inventory = {
            "schema_version": "1.0",
            "model_adapter_fingerprint": "a" * 64,
            "source_manifest_fingerprint": (
                calibration_manifest["manifest_fingerprint"]
                if source_mode == "stable_source"
                else source_manifest["manifest_fingerprint"]
            ),
            "instances": [
                {
                    "block_instance_id": "block-instance-A",
                    "block_template_id": "block.execute",
                    "runtime_scope_binding": {
                        "method": "exact_scope_name",
                        "value": "runtime.block.A",
                    },
                    "control_flow_path": "adapter-path:main",
                    "diagnostic_labels": {"display": "not-a-layer"},
                }
            ],
        }
        block_path = bundle / "mapping" / "block-instance-inventory.json"
        atomic_write_json(block_path, block_inventory)
        baseline_projection_fingerprint = "d" * 64
        graph_occurrence_fingerprint = "c" * 64
        fused_inventory = {
            "protocol": "original_fused_inventory_v1",
            "sk_groups": [
                {
                    "device_id": 0,
                    "model_id": "48",
                    "block_instance_id": "block-instance-A",
                    "source_scope": "runtime.block.A",
                    "candidate_source_scope": "decode",
                    "sk_occurrence_fingerprint": graph_occurrence_fingerprint,
                    "baseline_projection_fingerprint": baseline_projection_fingerprint,
                    "block_local_sk_ordinal": 0,
                    "ordered_child_ops": ["Compute"],
                    "original_child_node_keys": ["original-1"],
                    "evidence_records": [],
                }
            ],
        }
        fused_record = {
            "record_id": "fused-record-1",
            **{
                field: fused_inventory["sk_groups"][0][field]
                for field in (
                    "device_id",
                    "model_id",
                    "block_instance_id",
                    "candidate_source_scope",
                    "sk_occurrence_fingerprint",
                    "original_child_node_keys",
                )
            },
        }
        fused_record["record_fingerprint"] = canonical_sha256(fused_record)
        fused_evidence_path = evidence_dir / "fused-log-records.json"
        atomic_write_json(
            fused_evidence_path,
            {"protocol": "normalized_log_records_v1", "records": [fused_record]},
        )
        fused_inventory["evidence_catalog"] = {
            "fused-log": {
                "relative_path": fused_evidence_path.relative_to(bundle).as_posix(),
                "sha256": file_sha256(fused_evidence_path),
            }
        }
        fused_inventory["sk_groups"][0]["evidence_records"] = [
            {
                "artifact_id": "fused-log",
                "record_id": fused_record["record_id"],
                "record_fingerprint": fused_record["record_fingerprint"],
            }
        ]
        fused_path = bundle / "mapping" / "original-fused-inventory.json"
        atomic_write_json(fused_path, fused_inventory)
        assignments = {
            "protocol": "calibration_unit_assignments_v1",
            "assignments": [
                {
                    "calibration_node_key": "calibration-9",
                    "block_instance_id": "block-instance-A",
                    "unit_id": "block.compute",
                    "control_flow_path": "adapter-path:main",
                    "evidence_records": [],
                }
            ],
        }
        assignment_record = {
            "record_id": "assignment-record-1",
            **{
                field: assignments["assignments"][0][field]
                for field in ("calibration_node_key", "block_instance_id", "unit_id")
            },
        }
        assignment_record["record_fingerprint"] = canonical_sha256(assignment_record)
        assignment_evidence_path = evidence_dir / "assignment-log-records.json"
        atomic_write_json(
            assignment_evidence_path,
            {"protocol": "normalized_log_records_v1", "records": [assignment_record]},
        )
        assignments["evidence_catalog"] = {
            "assignment-log": {
                "relative_path": assignment_evidence_path.relative_to(
                    bundle
                ).as_posix(),
                "sha256": file_sha256(assignment_evidence_path),
            }
        }
        assignments["assignments"][0]["evidence_records"] = [
            {
                "artifact_id": "assignment-log",
                "record_id": assignment_record["record_id"],
                "record_fingerprint": assignment_record["record_fingerprint"],
            }
        ]
        assignments_path = bundle / "mapping" / "unit-assignments.json"
        atomic_write_json(assignments_path, assignments)
        sk_map = build_sk_source_map(
            projection,
            source_manifest,
            block_inventory,
            fused_inventory,
            assignments,
            calibration_source_manifest=(
                calibration_manifest if source_mode == "stable_source" else None
            ),
        )
        sk_map_path = bundle / "mapping" / "sk-source-map.json"
        atomic_write_json(sk_map_path, sk_map)
        collection_paths = {}
        for name in ("original-candidate", "calibration", "baseline"):
            path = bundle / name / "association-artifact-manifest.json"
            atomic_write_json(
                path, {"kind": name, "manifest_fingerprint": canonical_sha256(name)}
            )
            collection_paths[name] = path

        required = {
            "source_manifest": source_manifest_path,
            "sk_source_map": sk_map_path,
            "block_instance_inventory": block_path,
            "calibration_projection": projection_path,
            "original_fused_inventory": fused_path,
            "unit_assignments": assignments_path,
            "original_candidate_collection_manifest": collection_paths[
                "original-candidate"
            ],
            "calibration_collection_manifest": collection_paths["calibration"],
            "baseline_collection_manifest": collection_paths["baseline"],
            **{
                key: value
                for key, value in auto_evidence.items()
                if key != "calibration_source_snapshot_root"
            },
        }
        dag_artifacts = {
            **required,
            "source_snapshot_manifest": snapshot / "source-snapshot-manifest.json",
            "original_graph": original_graph_path,
            "calibration_graph": calibration_graph_path,
            "runtime_validation": runtime_validation_path,
            "fused_log_records": fused_evidence_path,
            "assignment_log_records": assignment_evidence_path,
        }
        if source_mode == "stable_source":
            dag_artifacts["calibration_source_snapshot_manifest"] = (
                auto_evidence["calibration_source_snapshot_root"]
                / "source-snapshot-manifest.json"
            )
        dag_path = bundle / "provenance" / "provenance-dag-manifest.json"
        dag_dependencies = {
            "source_manifest": ["source_snapshot_manifest"],
            "calibration_projection": [
                "original_graph",
                "calibration_graph",
                "runtime_validation",
            ],
            "original_fused_inventory": ["fused_log_records"],
            "unit_assignments": ["assignment_log_records"],
            "sk_source_map": [
                "source_manifest",
                "block_instance_inventory",
                "calibration_projection",
                "original_fused_inventory",
                "unit_assignments",
            ],
        }
        if source_mode == "stable_source":
            dag_dependencies.update(
                {
                    "calibration_source_manifest": [
                        "calibration_source_snapshot_manifest"
                    ],
                    "marker_only_calibration": [
                        "source_manifest",
                        "calibration_source_manifest",
                    ],
                    "sk_source_map": [
                        *dag_dependencies["sk_source_map"],
                        "marker_only_calibration",
                    ],
                }
            )
        dag = {
            "protocol": "source_mapping_provenance_dag_v1",
            "nodes": [
                {
                    "artifact_id": key,
                    "relative_path": path.relative_to(bundle).as_posix(),
                    "sha256": file_sha256(path),
                    "depends_on": dag_dependencies.get(key, []),
                }
                for key, path in dag_artifacts.items()
            ],
        }
        atomic_write_json(dag_path, dag)
        build_kwargs = {
            "artifact_root": bundle,
            "provenance_dag": dag_path,
            "source_snapshot_root": snapshot,
            **required,
        }
        if source_mode == "stable_source":
            build_kwargs.update(
                {
                    "stable_source_revision": "stable-rev",
                    "calibration_source_snapshot_root": auto_evidence[
                        "calibration_source_snapshot_root"
                    ],
                }
            )
        else:
            build_kwargs["stable_marker_revision"] = "stable-rev"
        scope_map = build_source_scope_map_v2(**build_kwargs)
        scope_map_path = bundle / "source-scope-map-v2.json"
        atomic_write_json(scope_map_path, scope_map)
        return {
            "bundle": bundle,
            "scope_map": scope_map_path,
            "candidate_manifest": collection_paths["original-candidate"],
            "source_file": snapshot / "model.dsl",
            "source_root": source,
            "graph_occurrence_fingerprint": graph_occurrence_fingerprint,
            "baseline_projection_fingerprint": baseline_projection_fingerprint,
            "marker_only_calibration": auto_evidence.get("marker_only_calibration"),
        }

    def test_non_python_manifest_and_v2_bundle_validate_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            result = load_source_scope_map_v2(
                artifacts["scope_map"],
                expected_source_revision="stable-rev",
                expected_candidate_manifest_sha256=file_sha256(
                    artifacts["candidate_manifest"]
                ),
                expected_source_root=artifacts["source_root"],
            )
            self.assertEqual(result["status"], "exact")
            source_range = result["lookup"][artifacts["graph_occurrence_fingerprint"]]
            self.assertEqual(source_range["boundary"]["source_file"], "model.dsl")
            self.assertEqual(source_range["baseline_projection_fingerprint"], "d" * 64)

    def test_automatic_source_map_uses_marker_only_calibration_and_stable_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary), source_mode="stable_source")
            result = load_source_scope_map_v2(
                artifacts["scope_map"],
                expected_source_revision="stable-rev",
                expected_candidate_manifest_sha256=file_sha256(
                    artifacts["candidate_manifest"]
                ),
                expected_source_root=artifacts["source_root"],
            )

            self.assertEqual(result["status"], "exact")
            self.assertEqual(result["source_revision_role"], "stable_source")
            source_range = result["lookup"][artifacts["graph_occurrence_fingerprint"]]
            self.assertEqual(
                source_range["boundary"],
                {
                    "start_op": "Compute",
                    "end_op": "Compute",
                    "source_file": "model.dsl",
                    "start_offset": 0,
                    "end_offset": 8,
                },
            )

            artifacts["marker_only_calibration"].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marker_only_calibration SHA"):
                load_source_scope_map_v2(
                    artifacts["scope_map"],
                    expected_source_revision="stable-rev",
                )

    def test_v2_bundle_rejects_snapshot_tampering_and_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            with self.assertRaisesRegex(ValueError, "source revision"):
                load_source_scope_map_v2(
                    artifacts["scope_map"], expected_source_revision="other-rev"
                )
            artifacts["source_file"].write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot file evidence mismatch"):
                load_source_scope_map_v2(
                    artifacts["scope_map"], expected_source_revision="stable-rev"
                )
            artifacts["source_file"].write_bytes(b"BEGIN\ncompute\nEND\n")
            (artifacts["source_root"] / "model.dsl").write_text(
                "current source changed", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "current source differs"):
                load_source_scope_map_v2(
                    artifacts["scope_map"],
                    expected_source_revision="stable-rev",
                    expected_source_root=artifacts["source_root"],
                )

    def test_v2_bundle_rejects_range_identity_tampering_after_resigning(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            value = json.loads(artifacts["scope_map"].read_text())
            value["source_ranges"][0]["source_scope"] = "forged.scope"
            value["provenance"]["source_scope_map_content_fingerprint"] = (
                _content_fingerprint(value)
            )
            atomic_write_json(artifacts["scope_map"], value)
            with self.assertRaisesRegex(ValueError, "source_scope mismatch"):
                load_source_scope_map_v2(
                    artifacts["scope_map"], expected_source_revision="stable-rev"
                )

    def test_projection_rejects_second_node_correspondence(self):
        original = {
            "protocol": "normalized_business_graph_v2",
            "collection_fingerprint": "original",
            "identity": {"device_id": 0, "model_id": "model-x"},
            "nodes": [
                {
                    "node_key": key,
                    "stream_role": "s",
                    "stream_ordinal": ordinal,
                    "canonical_op": "Same",
                    "core_family": "VECTOR",
                    "observable_signature": {},
                }
                for ordinal, key in enumerate(("a", "b"))
            ],
            "edges": [],
        }
        calibration = copy.deepcopy(original)
        calibration["collection_fingerprint"] = "calibration"
        calibration["nodes"][0]["node_key"] = "x"
        calibration["nodes"][1]["node_key"] = "y"
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            project_graphs(original, calibration)

    def test_projection_uses_proven_stream_chain_for_repeated_nodes(self):
        node_count = 120

        def graph(prefix, collection):
            keys = [f"{prefix}-{index}" for index in range(node_count)]
            return {
                "protocol": "normalized_business_graph_v2",
                "collection_fingerprint": collection,
                "identity": {"device_id": 0, "model_id": "48"},
                "nodes": [
                    {
                        "node_key": key,
                        "stream_role": f"{prefix}-stream",
                        "stream_ordinal": ordinal,
                        "canonical_op": "Repeated",
                        "core_family": "VECTOR",
                        "observable_signature": {},
                    }
                    for ordinal, key in enumerate(keys)
                ],
                "edges": [
                    {
                        "source": left,
                        "target": right,
                        "kind": "STREAM_ORDER",
                        "ports": {},
                    }
                    for left, right in zip(keys, keys[1:])
                ],
            }

        result = project_graphs(
            graph("original", "original"), graph("calibration", "calibration")
        )

        self.assertEqual(len(result["occurrences"]), node_count)
        self.assertEqual(
            sorted(
                (item["original"][1], item["calibration"][1])
                for item in result["occurrences"]
            ),
            [(index, index) for index in range(node_count)],
        )

    def _partial_assignment_projection(self, *, target_assigned):
        def graph(prefix, collection):
            keys = [f"{prefix}-target", f"{prefix}-outside"]
            return {
                "protocol": "normalized_business_graph_v2",
                "collection_fingerprint": collection,
                "identity": {"device_id": 0, "model_id": "48"},
                "nodes": [
                    {
                        "node_key": keys[0],
                        "stream_role": f"{prefix}-stream",
                        "stream_ordinal": 0,
                        "canonical_op": "Compute",
                        "core_family": "VECTOR",
                        "observable_signature": {},
                    },
                    {
                        "node_key": keys[1],
                        "stream_role": f"{prefix}-stream",
                        "stream_ordinal": 1,
                        "canonical_op": "OutsideUnit",
                        "core_family": "VECTOR",
                        "observable_signature": {},
                    },
                ],
                "edges": [
                    {
                        "source": keys[0],
                        "target": keys[1],
                        "kind": "STREAM_ORDER",
                        "ports": {},
                    }
                ],
            }

        exact = ["calibration-target"] if target_assigned else ["calibration-outside"]
        skipped = ["calibration-outside"] if target_assigned else ["calibration-target"]
        runtime_validation = [
            {
                "step_id": step,
                "business_occurrence_count": 2,
                "assigned_occurrence_count": 1,
                "unscoped_occurrence_count": 1,
                "conflicting_assignment_count": 0,
                "parent_binding_exact": False,
                "unit_assignment_complete": False,
                "exact_assigned_calibration_node_keys": exact,
                "skipped_calibration_node_keys": skipped,
                "conflicting_calibration_node_keys": [],
            }
            for step in (3, 4, 5)
        ]
        return project_graphs(
            graph("original", canonical_sha256("partial-original")),
            graph("calibration", canonical_sha256("partial-calibration")),
            runtime_validation=runtime_validation,
        )

    def test_partial_assignment_requires_a_complete_disjoint_partition(self):
        original = _graph("original-node", "original-stream", collection="original")
        calibration = _graph(
            "calibration-node", "calibration-stream", collection="calibration"
        )
        validation = {
            "step_id": 3,
            "business_occurrence_count": 1,
            "assigned_occurrence_count": 1,
            "unscoped_occurrence_count": 0,
            "conflicting_assignment_count": 0,
            "parent_binding_exact": False,
            "unit_assignment_complete": False,
            "exact_assigned_calibration_node_keys": ["calibration-node"],
            "skipped_calibration_node_keys": [],
            "conflicting_calibration_node_keys": [],
        }

        incomplete = copy.deepcopy(validation)
        incomplete.pop("conflicting_calibration_node_keys")
        with self.assertRaisesRegex(ValueError, "all assignment partition fields"):
            project_graphs(original, calibration, runtime_validation=[incomplete])

        overlapping = copy.deepcopy(validation)
        overlapping["skipped_calibration_node_keys"] = ["calibration-node"]
        with self.assertRaisesRegex(ValueError, "assignment partitions overlap"):
            project_graphs(original, calibration, runtime_validation=[overlapping])

    def test_source_map_processes_target_and_skips_unrelated_unassigned_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            bundle = artifacts["bundle"]
            projection = self._partial_assignment_projection(target_assigned=True)
            source_manifest = json.loads(
                (bundle / "source" / "source-unit-manifest.json").read_text()
            )
            block_inventory = json.loads(
                (bundle / "mapping" / "block-instance-inventory.json").read_text()
            )
            fused_inventory = json.loads(
                (bundle / "mapping" / "original-fused-inventory.json").read_text()
            )
            fused_inventory["sk_groups"][0]["original_child_node_keys"] = [
                "original-target"
            ]
            assignments = json.loads(
                (bundle / "mapping" / "unit-assignments.json").read_text()
            )
            assignments["assignments"][0]["calibration_node_key"] = "calibration-target"

            result = build_sk_source_map(
                projection,
                source_manifest,
                block_inventory,
                fused_inventory,
                assignments,
            )

            mapping = result["mappings"][0]
            self.assertEqual(mapping["processing_status"], "processed")
            self.assertEqual(mapping["relation"], "exact_cover")
            self.assertEqual(mapping["consensus"]["validated_step_ids"], [3, 4, 5])
            self.assertEqual(
                result["assignment_coverage"]["skipped_calibration_node_keys"],
                ["calibration-outside"],
            )

    def test_source_map_skips_group_when_target_child_is_not_exact_assigned(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            bundle = artifacts["bundle"]
            projection = self._partial_assignment_projection(target_assigned=False)
            source_manifest = json.loads(
                (bundle / "source" / "source-unit-manifest.json").read_text()
            )
            block_inventory = json.loads(
                (bundle / "mapping" / "block-instance-inventory.json").read_text()
            )
            fused_inventory = json.loads(
                (bundle / "mapping" / "original-fused-inventory.json").read_text()
            )
            fused_inventory["sk_groups"][0]["original_child_node_keys"] = [
                "original-target"
            ]
            assignments = json.loads(
                (bundle / "mapping" / "unit-assignments.json").read_text()
            )
            assignments["assignments"][0]["calibration_node_key"] = "calibration-target"

            result = build_sk_source_map(
                projection,
                source_manifest,
                block_inventory,
                fused_inventory,
                assignments,
            )

            mapping = result["mappings"][0]
            self.assertEqual(mapping["processing_status"], "skipped")
            self.assertEqual(mapping["relation"], "unmapped")
            self.assertEqual(mapping["confidence"], "diagnostic_only")
            self.assertIn(
                "target_child_three_step_validation_missing",
                mapping["mapping_blockers"],
            )
            self.assertEqual(result["assignment_coverage"]["skipped_sk_group_count"], 1)

    def test_source_map_skips_cross_block_group_without_aborting(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            bundle = artifacts["bundle"]
            projection = json.loads(
                (bundle / "mapping" / "calibration-projection.json").read_text()
            )
            source_manifest = json.loads(
                (bundle / "source" / "source-unit-manifest.json").read_text()
            )
            block_inventory = json.loads(
                (bundle / "mapping" / "block-instance-inventory.json").read_text()
            )
            other_block = copy.deepcopy(block_inventory["instances"][0])
            other_block["block_instance_id"] = "block-instance-B"
            other_block["runtime_scope_binding"]["value"] = "runtime.block.B"
            block_inventory["instances"].append(other_block)
            fused_inventory = json.loads(
                (bundle / "mapping" / "original-fused-inventory.json").read_text()
            )
            assignments = json.loads(
                (bundle / "mapping" / "unit-assignments.json").read_text()
            )
            assignments["assignments"][0]["block_instance_id"] = "block-instance-B"

            result = build_sk_source_map(
                projection,
                source_manifest,
                block_inventory,
                fused_inventory,
                assignments,
            )

            mapping = result["mappings"][0]
            self.assertEqual(mapping["processing_status"], "skipped")
            self.assertEqual(mapping["relation"], "unmapped")
            self.assertIn("target_child_block_mismatch", mapping["mapping_blockers"])
            self.assertEqual(
                mapping["cross_block_calibration_node_keys"], ["calibration-9"]
            )

    def test_v2_replays_projection_from_sealed_graph_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            projection = json.loads(
                (
                    artifacts["bundle"] / "mapping" / "calibration-projection.json"
                ).read_text()
            )
            projection["occurrences"][0]["original"][2] = "ForgedCompute"
            projection["projection_fingerprint"] = canonical_sha256(
                {
                    key: value
                    for key, value in projection.items()
                    if key != "projection_fingerprint"
                }
            )
            with self.assertRaisesRegex(ValueError, "cannot be reproduced"):
                _replay_calibration_projection(artifacts["bundle"], projection)

    def test_legacy_task_ranges_are_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.json"
            path.write_text(json.dumps({"task_ranges": []}), encoding="utf-8")
            result = load_source_scope_map_v2(
                path, expected_source_revision="stable-rev"
            )
            self.assertEqual(result["status"], "diagnostic_only")
            self.assertEqual(result["lookup"], {})

    def test_analyzer_promotes_only_same_graph_and_baseline_projection(self):
        association = {
            "mapping_method": "kernel_projection_structural",
            "mapping_confidence": "exact_projected_trace",
            "graph_alignment_proof": {
                "graph_occurrence_fingerprint": "c" * 64,
                "mapping_fingerprint": "d" * 64,
            },
        }
        entry = {
            "device_id": 0,
            "model_id": "48",
            "source_scope": "runtime.block.A",
            "candidate_source_scope": "runtime.block.A",
            "ordered_child_op_sequence": ["Compute"],
            "boundary": {"start_op": "Compute", "end_op": "Compute"},
            "sk_occurrence_fingerprint": "c" * 64,
            "baseline_projection_fingerprint": "d" * 64,
        }
        arguments = {
            "device_id": 0,
            "model_id": "48",
            "source_scope": "runtime.block.A",
            "ordered_child_ops": ["Compute"],
            "fusion_boundary": {"start_op": "Compute", "end_op": "Compute"},
        }
        self.assertTrue(_source_scope_entry_matches(association, entry, **arguments))
        for field in ("sk_occurrence_fingerprint", "baseline_projection_fingerprint"):
            with self.subTest(field=field):
                tampered = copy.deepcopy(entry)
                tampered[field] = "e" * 64
                self.assertFalse(
                    _source_scope_entry_matches(association, tampered, **arguments)
                )
        tampered_scope = copy.deepcopy(entry)
        tampered_scope["candidate_source_scope"] = "prefill"
        self.assertFalse(
            _source_scope_entry_matches(association, tampered_scope, **arguments)
        )

    def test_template_consensus_rejects_any_observed_family_counterexample(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            bundle = artifacts["bundle"]
            projection = json.loads(
                (bundle / "mapping" / "calibration-projection.json").read_text()
            )
            template_occurrence = projection["occurrences"][0]
            projection["occurrences"] = []
            for index in range(4):
                occurrence = copy.deepcopy(template_occurrence)
                occurrence["original_node_key"] = f"original-{index}"
                occurrence["calibration_node_key"] = f"calibration-{index}"
                occurrence["original"] = ["stream", index, "Compute", "VECTOR"]
                occurrence["calibration"] = ["stream-cal", index, "Compute", "VECTOR"]
                occurrence["correspondence_fingerprint"] = canonical_sha256(occurrence)
                projection["occurrences"].append(occurrence)
            projection["node_correspondence_fingerprint"] = canonical_sha256(
                projection["occurrences"]
            )
            projection["projection_fingerprint"] = canonical_sha256(
                {
                    key: value
                    for key, value in projection.items()
                    if key != "projection_fingerprint"
                }
            )
            source_manifest = json.loads(
                (bundle / "source" / "source-unit-manifest.json").read_text()
            )
            second_unit = copy.deepcopy(source_manifest["units"][0])
            second_unit["unit_id"] = "block.alternative"
            source_manifest["units"].append(second_unit)
            source_manifest["manifest_fingerprint"] = canonical_sha256(
                {
                    key: value
                    for key, value in source_manifest.items()
                    if key != "manifest_fingerprint"
                }
            )
            block_inventory = {
                "model_adapter_fingerprint": "a" * 64,
                "source_manifest_fingerprint": source_manifest["manifest_fingerprint"],
                "instances": [
                    {
                        "block_instance_id": f"block-{index}",
                        "block_template_id": "block.execute",
                        "runtime_scope_binding": {
                            "method": "exact_scope_name",
                            "value": f"runtime.block.{index}",
                        },
                        "control_flow_path": "adapter-path:main",
                    }
                    for index in range(4)
                ],
            }
            assignments = {
                "protocol": "calibration_unit_assignments_v1",
                "assignments": [
                    {
                        "calibration_node_key": f"calibration-{index}",
                        "block_instance_id": f"block-{index}",
                        "unit_id": (
                            "block.compute" if index < 3 else "block.alternative"
                        ),
                        "control_flow_path": "adapter-path:main",
                        "evidence_records": [
                            {
                                "artifact_id": "assignment-log",
                                "record_id": f"assignment-{index}",
                                "record_fingerprint": f"{index + 20:064x}",
                            }
                        ],
                    }
                    for index in range(4)
                ],
            }
            fused_inventory = {
                "protocol": "original_fused_inventory_v1",
                "sk_groups": [
                    {
                        "device_id": 0,
                        "model_id": "48",
                        "block_instance_id": f"block-{index}",
                        "source_scope": f"runtime.block.{index}",
                        "candidate_source_scope": "decode",
                        "sk_occurrence_fingerprint": f"{index + 1:064x}",
                        "baseline_projection_fingerprint": f"{index + 10:064x}",
                        "block_local_sk_ordinal": 0,
                        "ordered_child_ops": ["Compute"],
                        "original_child_node_keys": [f"original-{index}"],
                        "evidence_records": [
                            {
                                "artifact_id": "fused-log",
                                "record_id": f"fused-{index}",
                                "record_fingerprint": f"{index + 30:064x}",
                            }
                        ],
                    }
                    for index in range(4)
                ],
            }
            result = build_sk_source_map(
                projection,
                source_manifest,
                block_inventory,
                fused_inventory,
                assignments,
            )
            self.assertEqual(
                {item["confidence"] for item in result["mappings"]},
                {"source_unit_exact"},
            )
            first = result["mappings"][0]
            self.assertIn(
                "block-3", first["consensus"]["counterexample_block_instances"]
            )


if __name__ == "__main__":
    unittest.main()
