# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import csv
import copy
import hashlib
import io
import importlib
import json
import math
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
ADAPTATION_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "superkernel-runtime-common" / "scripts"
)
sys.path.insert(0, str(ADAPTATION_SCRIPTS))

import analyze_fusion_performance  # noqa: E402 - load sibling scripts after sys.path setup
import artifact_contract  # noqa: E402 - load sibling scripts after sys.path setup


CSV_FIELDS = [
    "Step Id",
    "Occurrence Id",
    "Iteration Id",
    "Request Id",
    "Batch Id",
    "Device_id",
    "Model ID",
    "Task ID",
    "Stream ID",
    "Name",
    "Type",
    "OP State",
    "Accelerator Core",
    "Start Time(us)",
    "Duration(us)",
    "Wait Time(us)",
    "Block Num",
    "Mix Block Num",
    "aic_scalar_time(us)",
    "aic_scalar_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
]


def _row(task, stream, name, op_type, core, start, duration, scalar_ratio=0.0):
    return {
        "step_id": 1,
        "device_id": 0,
        "model_id": "48",
        "task_id": task,
        "stream_id": stream,
        "name": name,
        "type": op_type,
        "op_state": "static",
        "accelerator_core": core,
        "core_family": analyze_fusion_performance._core_family(core, op_type, name),
        "start_us": start,
        "duration_us": duration,
        "wait_us": 0,
        "block_num": 1,
        "mix_block_num": 0,
        "aic_scalar_time_us": (duration * scalar_ratio if "AI_CORE" in core else 0),
        "aic_scalar_ratio": scalar_ratio if "AI_CORE" in core else 0,
        "aiv_scalar_time_us": (duration * scalar_ratio if "VECTOR" in core else 0),
        "aiv_scalar_ratio": scalar_ratio if "VECTOR" in core else 0,
        "aic_icache_miss_rate": None,
        "aiv_icache_miss_rate": None,
        "layer": None,
        "layer_source": None,
        "sk_boundary": None,
    }


def _write_csv(path, rows):
    rows = list(rows)
    auto_step_ids = bool(rows) and all(row.get("step_id") == 1 for row in rows)
    task_occurrences = {}
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            step_id = row["step_id"]
            if auto_step_ids:
                task_key = (row["device_id"], row["model_id"], row["task_id"])
                step_id = task_occurrences.get(task_key, 0)
                task_occurrences[task_key] = step_id + 1
            writer.writerow(
                {
                    "Step Id": step_id,
                    "Occurrence Id": row.get("occurrence_id", ""),
                    "Iteration Id": row.get("iteration_id", ""),
                    "Request Id": row.get("request_id", ""),
                    "Batch Id": row.get("batch_id", ""),
                    "Device_id": row["device_id"],
                    "Model ID": row["model_id"],
                    "Task ID": row["task_id"],
                    "Stream ID": row["stream_id"],
                    "Name": row["name"],
                    "Type": row["type"],
                    "OP State": row["op_state"],
                    "Accelerator Core": row["accelerator_core"],
                    "Start Time(us)": row["start_us"],
                    "Duration(us)": row["duration_us"],
                    "Wait Time(us)": row["wait_us"],
                    "Block Num": row["block_num"],
                    "Mix Block Num": row["mix_block_num"],
                    "aic_scalar_time(us)": row["aic_scalar_time_us"],
                    "aic_scalar_ratio": row["aic_scalar_ratio"],
                    "aiv_scalar_time(us)": row["aiv_scalar_time_us"],
                    "aiv_scalar_ratio": row["aiv_scalar_ratio"],
                }
            )


def _complete_workload_manifest():
    return {
        "model": "demo",
        "input": {"tokens": 128},
        "batch": 1,
        "rank_size": 8,
        "mode": "prefill_decode",
        "warmup": 1,
        "iterations": 10,
        "runtime": {"dtype": "float16"},
    }


def _single_a_sk_name(scope="decoder.layer.0"):
    return f"sk_1_{scope}_start_static_kernel_A_hash_end_static_kernel_A_hash"


def _metadata_log_path(root, model_id="48"):
    model = root / f"model_{model_id}"
    model.mkdir(parents=True, exist_ok=True)
    return model / "sk_fused_nodes.log"


def _write_analysis_cli_fixture(root, candidate_duration=6):
    baseline = root / "baseline.csv"
    candidate = root / "candidate.csv"
    metadata = root / "sk_meta"
    metadata.mkdir()
    _write_csv(
        baseline,
        [
            _row(10, 1, "static_kernel_A_hash", "A", "AI_VECTOR_CORE", start, 8)
            for start in (0, 20, 40)
        ],
    )
    sk_name = _single_a_sk_name()
    _write_csv(
        candidate,
        [
            _row(
                99,
                1,
                sk_name,
                "SuperKernel",
                "AI_VECTOR_CORE",
                start,
                candidate_duration,
            )
            for start in (0, 20, 40)
        ],
    )
    _metadata_log_path(metadata).write_text(
        f"SK Function: {sk_name}, scope id: 1, Node Count: 1\n"
        "[nodeId:10, streamId:1] - "
        "KernelInfos{funcName:static_kernel_A_hash, "
        "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
        "isScheModeOn:0}\n"
    )
    artifacts = {
        "baseline_profile": baseline,
        "candidate_profile": candidate,
        "sk_meta": metadata,
        "baseline_config": root / "baseline-config.json",
        "candidate_config": root / "candidate-config.json",
        "baseline_workload": root / "baseline-workload.json",
        "candidate_workload": root / "candidate-workload.json",
        "declared_change_set": root / "declared-change.json",
        "sk_prof": root / "sk_prof_device_0.json",
        "environment_evidence": root / "environment.json",
        "source_scope_map": root / "source-scope-map.json",
    }
    artifacts["baseline_config"].write_text(
        json.dumps(
            {
                "runtime": {"rank_size": 8},
                "superkernel": {"enabled": False, "scope": []},
            }
        )
    )
    artifacts["candidate_config"].write_text(
        json.dumps(
            {
                "runtime": {"rank_size": 8},
                "superkernel": {
                    "enabled": True,
                    "scope": ["decoder.layer.0"],
                },
            }
        )
    )
    workload = _complete_workload_manifest()
    artifacts["baseline_workload"].write_text(json.dumps(workload))
    artifacts["candidate_workload"].write_text(json.dumps(workload))
    artifacts["declared_change_set"].write_text(
        json.dumps(
            {
                "allowed_json_pointers": [
                    "/superkernel/enabled",
                    "/superkernel/scope",
                ],
                "only_change_zh": "只启用 SuperKernel 并应用本轮 scope",
            }
        )
    )
    artifacts["sk_prof"].write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "static_kernel_A_hash",
                        "ph": "X",
                        "ts": 0,
                        "dur": candidate_duration,
                        "tid": 1,
                        "args": {"kernelType": "AIV_ONLY"},
                    }
                ]
            }
        )
    )
    artifacts["environment_evidence"].write_text(
        json.dumps({"accepted_options": {"auto_op_parallel": [0, 1]}})
    )
    artifacts["source_scope_map"].write_text(
        json.dumps(
            {
                "task_ranges": [
                    {
                        "layer": 0,
                        "model_id": "48",
                        "start_task_id": 10,
                        "end_task_id": 10,
                        "source_scope": "decoder.layer.0",
                        "boundary": {"start_op": "A", "end_op": "A"},
                        "ordered_child_op_sequence": ["A"],
                    }
                ]
            }
        )
    )
    return artifacts


def _configure_cube_vector_regression(artifacts, environment_evidence):
    _write_csv(
        artifacts["baseline_profile"],
        [
            row
            for start in (0, 30, 60)
            for row in (
                _row(
                    10,
                    1,
                    "static_kernel_Vector_hash",
                    "Vector",
                    "AI_VECTOR_CORE",
                    start,
                    10,
                ),
                _row(11, 2, "static_kernel_Cube_hash", "Cube", "AI_CORE", start + 2, 8),
            )
        ],
    )
    sk_name = (
        "sk_1_decoder.layer.0_start_static_kernel_Vector_hash_"
        "end_static_kernel_Cube_hash"
    )
    _write_csv(
        artifacts["candidate_profile"],
        [
            _row(99, 1, sk_name, "SuperKernel", "MIX_AIC", start, 12)
            for start in (0, 30, 60)
        ],
    )
    _metadata_log_path(artifacts["sk_meta"]).write_text(
        f"SK Function: {sk_name}, scope id: 1, Node Count: 2\n"
        "[nodeId:10, streamId:1] - "
        "KernelInfos{funcName:static_kernel_Vector_hash, "
        "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
        "isScheModeOn:0}\n"
        "[nodeId:11, streamId:2] - "
        "KernelInfos{funcName:static_kernel_Cube_hash, "
        "kernelType:AIC_ONLY, numBlocks:1, cubeNum:1, vecNum:0, "
        "isScheModeOn:0}\n"
    )
    artifacts["source_scope_map"].write_text(
        json.dumps(
            {
                "task_ranges": [
                    {
                        "layer": 0,
                        "model_id": "48",
                        "start_task_id": 10,
                        "end_task_id": 11,
                        "source_scope": "decoder.layer.0",
                        "boundary": {
                            "start_op": "Vector",
                            "end_op": "Cube",
                        },
                        "ordered_child_op_sequence": ["Vector", "Cube"],
                    }
                ]
            }
        )
    )
    artifacts["sk_prof"].write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "static_kernel_Vector_hash",
                        "ph": "X",
                        "ts": 0,
                        "dur": 5,
                        "tid": 1,
                        "args": {"kernelType": "AIV_ONLY"},
                    },
                    {
                        "name": "static_kernel_Cube_hash",
                        "ph": "X",
                        "ts": 5,
                        "dur": 5,
                        "tid": 2,
                        "args": {"kernelType": "AIC_ONLY"},
                    },
                ]
            }
        )
    )
    artifacts["environment_evidence"].write_text(json.dumps(environment_evidence))


def _configure_scalar_regression(artifacts, environment_evidence):
    _write_csv(
        artifacts["baseline_profile"],
        [
            _row(
                10,
                1,
                "static_kernel_GroupedMatmul_hash",
                "GroupedMatmul",
                "AI_CORE",
                start,
                10,
                0.25,
            )
            for start in (0, 30, 60)
        ],
    )
    sk_name = (
        "sk_1_decoder.layer.0_start_static_kernel_GroupedMatmul_hash_"
        "end_static_kernel_GroupedMatmul_hash"
    )
    _write_csv(
        artifacts["candidate_profile"],
        [
            _row(99, 1, sk_name, "SuperKernel", "MIX_AIC", start, 12)
            for start in (0, 30, 60)
        ],
    )
    _metadata_log_path(artifacts["sk_meta"]).write_text(
        f"SK Function: {sk_name}, scope id: 1, Node Count: 1\n"
        "[nodeId:10, streamId:1] - "
        "KernelInfos{funcName:static_kernel_GroupedMatmul_hash, "
        "kernelType:AIC_ONLY, numBlocks:1, cubeNum:1, vecNum:0, "
        "isScheModeOn:0}\n"
    )
    artifacts["source_scope_map"].write_text(
        json.dumps(
            {
                "task_ranges": [
                    {
                        "layer": 0,
                        "model_id": "48",
                        "start_task_id": 10,
                        "end_task_id": 10,
                        "source_scope": "decoder.layer.0",
                        "boundary": {
                            "start_op": "GroupedMatmul",
                            "end_op": "GroupedMatmul",
                        },
                        "ordered_child_op_sequence": ["GroupedMatmul"],
                    }
                ]
            }
        )
    )
    artifacts["environment_evidence"].write_text(json.dumps(environment_evidence))


def _run_diagnostic_analysis(root, configure, environment_evidence, config_state=None):
    artifacts = _write_analysis_cli_fixture(root)
    configure(artifacts, environment_evidence)
    if config_state:
        for config_name in ("baseline_config", "candidate_config"):
            config = json.loads(artifacts[config_name].read_text())
            config["runtime_evidence"] = {"dcci_state": config_state}
            artifacts[config_name].write_text(json.dumps(config))
    json_out = root / "analysis.json"
    with redirect_stdout(io.StringIO()):
        analyze_fusion_performance.main(_analysis_cli_args(artifacts, json_out))
    return json.loads(json_out.read_text())


def _scalar_regression_decision():
    return {
        "range_id": "range-target",
        "action": "prune",
        "sk_duration_us": 12,
        "metadata_child_functions": ["static_kernel_GroupedMatmul_hash"],
        "original": {
            "interval_us": {"p50": 10},
            "duration_sum_us": {"p50": 10},
            "max_scalar_ratio": 0.25,
            "example_occurrence": {
                "max_scalar_ratio": 0.1,
                "multi_stream_analysis": {"cube_vector_parallel_detected": False},
            },
        },
    }


def _cube_vector_regression_decision(range_id="range-target"):
    decision = _scalar_regression_decision()
    decision["range_id"] = range_id
    decision["original"]["duration_sum_us"]["p50"] = 18
    decision["original"]["example_occurrence"]["max_scalar_ratio"] = 0
    decision["original"]["example_occurrence"]["multi_stream_analysis"] = {
        "cube_vector_parallel_detected": True
    }
    return decision


def _analysis_cli_args(artifacts, json_out, markdown_out=None):
    args = []
    for option in (
        "baseline-profile",
        "candidate-profile",
        "sk-meta",
        "baseline-config",
        "candidate-config",
        "baseline-workload",
        "candidate-workload",
        "declared-change-set",
        "sk-prof",
        "environment-evidence",
        "source-scope-map",
    ):
        args.extend([f"--{option}", str(artifacts[option.replace("-", "_")])])
    for option in (
        "baseline-collection-manifest",
        "profile-collection-manifest",
    ):
        artifact = artifacts.get(option.replace("-", "_"))
        if artifact is not None:
            args.extend([f"--{option}", str(artifact)])
    args.extend(
        [
            "--candidate-name",
            "S1",
            "--experiment-id",
            "exp-1",
            "--round-id",
            "S1-BASE",
            "--analysis-agent-id",
            "analysis-agent-1",
            "--source-revision",
            "660e08e",
            "--min-relative-change-pct",
            "3",
            "--min-absolute-change-us",
            "1",
            "--min-occurrences",
            "3",
            "--json-out",
            str(json_out),
        ]
    )
    if markdown_out is not None:
        args.extend(["--markdown-out", str(markdown_out)])
    return args


def _replace_metadata_node_id(artifacts, old_node_id=10, new_node_id=999):
    metadata_log = _metadata_log_path(artifacts["sk_meta"])
    metadata_log.write_text(
        metadata_log.read_text().replace(
            f"nodeId:{old_node_id}", f"nodeId:{new_node_id}"
        )
    )


def _source_scope_map_entry(**overrides):
    entry = {
        "layer": 0,
        "model_id": "48",
        "start_task_id": 10,
        "end_task_id": 10,
        "source_scope": "decoder.layer.0",
        "boundary": {
            "start_op": "A",
            "end_op": "A",
        },
        "ordered_child_op_sequence": ["A"],
    }
    entry.update(overrides)
    return entry


def _signed_analysis_report(overrides=None):
    default_decision, default_action = _renderer_contract_pair()
    default_decision.update(
        {
            "device_id": 0,
            "model_id": "48",
            "raw_sk_id": 7,
            "child_count": 2,
            "mapping_method": "source_scope_map",
            "mapping_confidence": "exact",
            "mapping_blockers": [],
            "candidate_binding_status": None,
        }
    )
    report = {
        "schema_version": "1.2",
        "analysis_id": None,
        "experiment_id": "exp-1",
        "round_id": "S1-BASE",
        "analysis_agent_id": "analysis-agent-1",
        "candidate_name": "S1",
        "source_revision": "660e08e",
        "baseline_profile_fingerprint": "1" * 64,
        "candidate_profile_fingerprint": "2" * 64,
        "baseline_config_fingerprint": "3" * 64,
        "candidate_config_fingerprint": "4" * 64,
        "baseline_workload_fingerprint": "5" * 64,
        "candidate_workload_fingerprint": "5" * 64,
        "workload_fingerprint": "5" * 64,
        "baseline_control_fingerprint": "6" * 64,
        "candidate_control_fingerprint": "6" * 64,
        "control_fingerprint": "6" * 64,
        "declared_change_set": {"allowed_json_pointers": []},
        "inputs": {
            "baseline_profile": "../baseline.csv",
            "candidate_profile": "../candidate.csv",
            "sk_meta": "../sk_meta",
            "baseline_config": "../baseline-config.json",
            "candidate_config": "../candidate-config.json",
            "baseline_workload": "../baseline-workload.json",
            "candidate_workload": "../candidate-workload.json",
            "declared_change_set": "../declared-change.json",
            "sk_prof": None,
            "environment_evidence": None,
            "source_scope_map": None,
            "baseline_collection_manifest": None,
            "profile_collection_manifest": None,
        },
        "association_protocol": {
            "protocol": "kernel_projection_trace_v2",
            "status": "not_requested",
            "manifest_set_fingerprint": None,
            "manifest_content_fingerprints": {
                "baseline_collection_manifest": None,
                "profile_collection_manifest": None,
            },
            "blockers": ["structural_collection_manifests_missing"],
        },
        "candidate_binding_evidence": {"summary": None, "occurrences": []},
        "canonical_graph_fingerprints": {},
        "stream_role_mapping": {},
        "graph_alignment_proof": {},
        "mapping_coverage": {
            "total_sk_ids": 1,
            "bound_sk_ids": 0,
            "exact_projected_trace_sk_ids": 0,
            "ambiguous_sk_ids": 0,
            "unmapped_sk_ids": 1,
            "filtered_by_child_count": 0,
            "child_count_distribution": {"2": 1},
            "blocker_counts": {},
        },
        "thresholds": {
            "min_relative_change_pct": 3.0,
            "min_absolute_change_us": 1.0,
            "min_occurrences": 3,
        },
        "per_sk_decisions": [default_decision],
        "scope_actions": [default_action],
        "diagnostic_hypotheses": [],
        "recommended_experiments": [],
        "blockers": [],
        "next_agent_guidance_zh": "仅按证据充分的逐 SK 判定处理本轮 scope。",
    }
    report.update(overrides or {})
    decisions = report.get("per_sk_decisions")
    valid_decisions = isinstance(decisions, list) and all(
        isinstance(decision, dict) for decision in decisions
    )
    for index, decision in enumerate(decisions if valid_decisions else ()):
        decision.setdefault("device_id", 0)
        decision.setdefault("model_id", "48")
        decision.setdefault("raw_sk_id", index + 7)
        if "child_count" not in decision:
            identity = decision.get("identity")
            sequence = (
                identity.get("ordered_child_op_sequence", ())
                if isinstance(identity, dict)
                else ()
            )
            decision["child_count"] = len(sequence)
        if decision.get("mapping_method") is None:
            decision["mapping_method"] = "source_scope_map"
        if decision.get("mapping_confidence") is None:
            decision["mapping_confidence"] = "exact"
        decision.setdefault("mapping_blockers", [])
        decision.setdefault("candidate_binding_status", None)
    if valid_decisions and (not overrides or "mapping_coverage" not in overrides):
        inventory = decisions
        exact_projected = sum(
            item["mapping_confidence"] == "exact_projected_trace" for item in inventory
        )
        ambiguous = sum(item["mapping_confidence"] == "ambiguous" for item in inventory)
        distribution = {}
        blocker_counts = {}
        for item in inventory:
            child_count = item["child_count"]
            bucket = "5+" if child_count >= 5 else str(child_count)
            distribution[bucket] = distribution.get(bucket, 0) + 1
            for blocker in set(item["mapping_blockers"]):
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        report["mapping_coverage"] = {
            "total_sk_ids": len(inventory),
            "bound_sk_ids": sum(
                item["candidate_binding_status"] == "bound" for item in inventory
            ),
            "exact_projected_trace_sk_ids": exact_projected,
            "ambiguous_sk_ids": ambiguous,
            "unmapped_sk_ids": len(inventory) - exact_projected - ambiguous,
            "filtered_by_child_count": 0,
            "child_count_distribution": distribution,
            "blocker_counts": blocker_counts,
        }
    if not overrides or "analysis_id" not in overrides:
        identity = {
            field: report[field]
            for field in (
                "experiment_id",
                "round_id",
                "analysis_agent_id",
                "candidate_name",
                "source_revision",
            )
        }
        report["analysis_id"] = (
            "analysis-"
            + hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        )
    report.pop("analysis_content_fingerprint", None)
    report["analysis_content_fingerprint"] = hashlib.sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return report


def _resign_analysis_report(report):
    report.pop("analysis_content_fingerprint", None)
    report["analysis_content_fingerprint"] = hashlib.sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return report


def _signed_structural_report(overrides=None):
    decision, action = _renderer_contract_pair()
    decision.update(
        {
            "device_id": 0,
            "model_id": "48",
            "raw_sk_id": 7,
            "child_count": 2,
            "mapping_method": "source_scope_map",
            "mapping_confidence": "exact",
            "mapping_blockers": [],
            "candidate_binding_status": None,
        }
    )
    inputs = dict(_signed_analysis_report()["inputs"])
    report = {
        "schema_version": "1.2",
        "inputs": inputs,
        "association_protocol": {
            "protocol": "structural_occurrence_v1",
            "status": "not_requested",
            "manifest_set_fingerprint": None,
            "manifest_content_fingerprints": {
                "baseline_collection_manifest": None,
                "profile_collection_manifest": None,
            },
            "blockers": ["structural_collection_manifests_missing"],
        },
        "candidate_binding_evidence": {"summary": None, "occurrences": []},
        "canonical_graph_fingerprints": {},
        "stream_role_mapping": {},
        "graph_alignment_proof": {},
        "mapping_coverage": {
            "total_sk_ids": 1,
            "bound_sk_ids": 0,
            "exact_projected_trace_sk_ids": 0,
            "ambiguous_sk_ids": 0,
            "unmapped_sk_ids": 1,
            "filtered_by_child_count": 0,
            "child_count_distribution": {"2": 1},
            "blocker_counts": {},
        },
        "per_sk_decisions": [decision],
        "scope_actions": [action],
    }
    report.update(overrides or {})
    return _signed_analysis_report(report)


def _signed_exact_projected_trace_report(overrides=None):
    decision, action = _renderer_contract_pair()
    decision.update(
        {
            "device_id": 0,
            "model_id": "48",
            "raw_sk_id": 7,
            "child_count": 2,
            "mapping_method": "kernel_projection_structural",
            "mapping_confidence": "exact_projected_trace",
            "mapping_blockers": [],
            "candidate_binding_status": "bound",
        }
    )
    decision["identity"]["ordered_child_op_sequence"] = ["Add", "MatMul"]
    action["ordered_child_op_sequence"] = ["Add", "MatMul"]
    identity_key = "device:0/model:48/sk:7"
    mapping_fingerprint = "2" * 64
    graph_fingerprint = "3" * 64
    evidence_fingerprints = {
        "kernel_details": "1" * 64,
        "projection_mapping": mapping_fingerprint,
        "profile_sk_graph_origin": graph_fingerprint,
    }
    bindings = [
        {
            "status": "bound",
            "process_identity": {
                "native_pid": 4100,
                "device_id": 0,
                "model_id": 48,
                "sk_id": 7,
            },
            "step_id": step,
            "parent_interval_ns": [step * 1000, step * 1000 + 500],
            "ordered_child_node_keys": ["profile:A", "profile:B"],
            "ordered_child_ops": ["Add", "MatMul"],
            "child_count": 2,
            "evidence_fingerprints": dict(evidence_fingerprints),
            "blockers": [],
        }
        for step in (1, 2, 3)
    ]
    report = {
        "per_sk_decisions": [decision],
        "scope_actions": [action],
        "association_protocol": {
            "protocol": "kernel_projection_trace_v2",
            "status": "associated",
            "manifest_set_fingerprint": "8" * 64,
            "manifest_content_fingerprints": {
                "baseline_collection_manifest": "9" * 64,
                "profile_collection_manifest": "c" * 64,
            },
            "blockers": [],
        },
        "candidate_binding_evidence": {
            "summary": {
                "protocol": "kernel_projection_trace_v2",
                "parent_occurrences": 3,
                "status_counts": {"bound": 3},
            },
            "occurrences": bindings,
        },
        "canonical_graph_fingerprints": {
            identity_key: {
                "baseline_kernel_details_sha256": "1" * 64,
                "profile_manifest_fingerprint": "4" * 64,
                "profile_origin_graph_fingerprint": graph_fingerprint,
            }
        },
        "stream_role_mapping": {
            identity_key: {
                str(step): {"profile:stream": step + 10} for step in (1, 2, 3)
            }
        },
        "graph_alignment_proof": {
            identity_key: {
                "protocol": "kernel_projection_trace_v2",
                "mapping_method": "kernel_projection_structural",
                "mapping_confidence": "exact_projected_trace",
                "mapping_fingerprint": mapping_fingerprint,
                "graph_occurrence_fingerprint": "6" * 64,
                "alternative_solution_count_by_step": {
                    "1": 0,
                    "2": 0,
                    "3": 0,
                },
            }
        },
        "mapping_coverage": {
            "total_sk_ids": 1,
            "bound_sk_ids": 1,
            "exact_projected_trace_sk_ids": 1,
            "ambiguous_sk_ids": 0,
            "unmapped_sk_ids": 0,
            "filtered_by_child_count": 0,
            "child_count_distribution": {"2": 1},
            "blocker_counts": {},
        },
    }
    exact_inputs = dict(_signed_structural_report()["inputs"])
    exact_inputs.update(
        {
            "baseline_collection_manifest": "../baseline/manifest.json",
            "profile_collection_manifest": "../profile/manifest.json",
        }
    )
    report["inputs"] = exact_inputs
    report.update(overrides or {})
    return _signed_structural_report(report)


def _renderer_contract_pair(
    *,
    classification="beneficial",
    action="keep",
    range_id="range-contract",
    sk_id="sk-contract",
    source_scope="decoder.layer.0",
    boundary=None,
    ordered_child_op_sequence=None,
    interval_unproven=True,
):
    boundary = boundary or {"start_op": "A", "end_op": "B"}
    sequence = ordered_child_op_sequence or ["A", "B"]
    decision = {
        "range_id": range_id,
        "sk_id": sk_id,
        "classification": classification,
        "classification_zh": "证据充分或已显式阻塞",
        "action": action,
        "mapping_method": "source_scope_map",
        "mapping_confidence": "exact",
        "mapping_blockers": [],
        "candidate_binding_status": None,
        "boundary": json.loads(json.dumps(boundary)),
        "identity": {
            "model_id": "48",
            "source_scope": source_scope,
            "boundary": {
                "start_op": boundary["start_op"],
                "end_op": boundary["end_op"],
            },
            "ordered_child_op_sequence": list(sequence),
        },
        "evidence_errors": ["fixture_metrics_omitted"],
    }
    scope_action = {
        "range_id": range_id,
        "sk_id": sk_id,
        "classification": classification,
        "action": action,
        "source_scope": source_scope,
        "boundary": json.loads(json.dumps(boundary)),
        "ordered_child_op_sequence": list(sequence),
        "interval_unproven": interval_unproven,
    }
    return decision, scope_action


class ProfilerAnalysisTest(unittest.TestCase):
    def test_markdown_renderer_is_deterministic_complete_and_escaped(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        report = _signed_analysis_report(
            {
                "experiment_id": "exp|1",
                "round_id": "S1-P1",
                "analysis_agent_id": "agent-1",
                "source_revision": "abc123",
                "baseline_config_fingerprint": "baseline-config-fp",
                "candidate_config_fingerprint": "candidate-config-fp",
                "control_fingerprint": "control-fp",
                "workload_fingerprint": "workload-fp",
                "declared_change_set": {
                    "allowed_json_pointers": ["/superkernel/scope"],
                    "only_change_zh": "只修改本轮 scope。",
                },
                "thresholds": {
                    "min_relative_change_pct": 3.0,
                    "min_absolute_change_us": 1.0,
                    "min_occurrences": 3,
                },
                "per_sk_decisions": [
                    {
                        "range_id": "range-b",
                        "sk_id": "sk-b",
                        "child_count": 2,
                        "original": None,
                        "sk_duration_us": None,
                        "improvement_pct": None,
                        "noise_threshold_pct": None,
                        "classification": "insufficient_evidence",
                        "classification_zh": "证据不足",
                        "action": "reprofile",
                        "mapping_method": None,
                        "mapping_confidence": "diagnostic_only",
                        "boundary": {
                            "scope": "decoder.layer.1",
                            "start_op": "C",
                            "end_op": "D",
                        },
                        "identity": {
                            "source_scope": "decoder.layer.1",
                            "ordered_child_op_sequence": ["C", "D"],
                        },
                        "evidence_errors": [
                            "missing|artifact",
                            "baseline_mapping_missing",
                        ],
                    },
                    {
                        "range_id": "range-a|escaped",
                        "sk_id": "sk-a",
                        "child_count": 2,
                        "original": {
                            "occurrence_count": 4,
                            "interval_us": {"p50": 10, "p90": 11, "mad": 0.5},
                            "duration_sum_us": {
                                "p50": 18,
                                "p90": 19,
                                "mad": 0.75,
                            },
                            "example_occurrence": {
                                "stream_ids": [1, 2],
                                "stream_count": 2,
                                "multi_stream_analysis": {
                                    "multi_stream_detected": True,
                                    "cube_vector_parallel_detected": True,
                                    "cube_vector_overlap_us": 4,
                                    "cube_vector_overlap_ratio": 0.4,
                                },
                            },
                        },
                        "sk_duration": {"p50": 12, "p90": 13, "mad": 0.25},
                        "sk_duration_us": 12,
                        "candidate_occurrence_count": 4,
                        "improvement_pct": -20,
                        "noise_threshold_pct": 3,
                        "classification": "regressed",
                        "classification_zh": "明确性能劣化",
                        "action": "block",
                        "action_blocker": (
                            "prune_requires_exact_source_scope_boundary_mapping"
                        ),
                        "mapping_method": "source_scope_map",
                        "mapping_confidence": "exact",
                        "boundary": {
                            "scope": "decoder.layer.0",
                            "start_op": "A",
                            "end_op": "B",
                        },
                        "identity": {
                            "source_scope": "decoder.layer.0",
                            "ordered_child_op_sequence": ["A", "B"],
                        },
                        "evidence_errors": [],
                    },
                ],
                "scope_actions": [
                    {
                        "range_id": "range-b",
                        "sk_id": "sk-b",
                        "classification": "insufficient_evidence",
                        "action": "reprofile",
                        "source_scope": "decoder.layer.1",
                        "boundary": {
                            "scope": "decoder.layer.1",
                            "start_op": "C",
                            "end_op": "D",
                        },
                        "ordered_child_op_sequence": ["C", "D"],
                        "interval_unproven": True,
                    },
                    {
                        "range_id": "range-a|escaped",
                        "sk_id": "sk-a",
                        "classification": "regressed",
                        "action": "block",
                        "source_scope": "decoder.layer.0",
                        "boundary": {
                            "scope": "decoder.layer.0",
                            "start_op": "A",
                            "end_op": "B",
                        },
                        "ordered_child_op_sequence": ["A", "B"],
                        "interval_unproven": True,
                    },
                ],
                "diagnostic_hypotheses": [],
                "recommended_experiments": [],
                "blockers": ["缺少 evidence|artifact"],
                "next_agent_guidance_zh": "只执行已接受的单变量实验。",
            }
        )

        first = renderer.render_report(report)
        second = renderer.render_report(json.loads(json.dumps(report)))

        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        sections = (
            "# SuperKernel 融合性能分析",
            "## 输入与可信度",
            "## 判定阈值",
            "## 逐 SK 性能对比",
            "## Scope 保留与裁剪",
            "## 劣化原因假设",
            "## 建议验证实验",
            "## 阻塞项",
            "## 给主实验 Agent 的后续指导",
        )
        positions = [first.index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))
        for column in (
            "child count",
            "baseline interval P50",
            "baseline interval P90",
            "baseline interval MAD",
            "baseline duration sum P50",
            "baseline duration sum P90",
            "baseline duration sum MAD",
            "SK P50",
            "SK P90",
            "SK MAD",
            "occurrence count",
            "change",
            "MAD/dynamic threshold (%)",
            "classification（英文 + 中文）",
            "action",
            "mapping method",
            "mapping confidence",
            "mapping reliable",
            "boundary",
            "ordered child sequence",
            "stream/overlap",
            "analysis Agent",
            "条件证据绑定",
            "blocker/evidence",
        ):
            self.assertIn(column, first)
        self.assertLess(first.index(r"range\-a\|escaped"), first.index(r"range\-b"))
        self.assertIn("N/A", first)
        self.assertIn("missing\\|artifact", first)
        self.assertIn("baseline\\_mapping\\_missing", first)
        self.assertIn("source\\_scope\\_map", first)
        self.assertIn(r"decoder\.layer\.0", first)
        self.assertIn(r'\["A","B"\]', first)
        self.assertIn("cube\\_vector\\_overlap\\_ratio", first)
        self.assertIn("declared\\_change\\_set", first)
        self.assertIn("only\\_change\\_zh", first)
        self.assertEqual(first.count("agent\\-1"), 3)
        for binding in (
            "source\\_revision=abc123",
            "baseline\\_config\\_fingerprint=baseline\\-config\\-fp",
            "candidate\\_config\\_fingerprint=candidate\\-config\\-fp",
            "control\\_fingerprint=control\\-fp",
            "workload\\_fingerprint=workload\\-fp",
            "range\\_id=range\\-a\\|escaped",
            "range\\_id=range\\-b",
        ):
            self.assertIn(binding, first)
        self.assertIn("regressed / 明确性能劣化", first)
        self.assertIn("insufficient\\_evidence / 证据不足", first)

    def test_markdown_renderer_validates_conditional_binding_top_level_fields(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        for field in (
            "analysis_agent_id",
            "source_revision",
            "baseline_config_fingerprint",
            "candidate_config_fingerprint",
            "control_fingerprint",
            "workload_fingerprint",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    renderer.render_report(
                        _signed_analysis_report({field: {"invalid": "object"}})
                    )

        decision, action = _renderer_contract_pair(
            classification="insufficient_evidence",
            action="reprofile",
            range_id="range-missing",
            sk_id="sk-missing",
        )
        decision["evidence_errors"] = ["baseline_mapping_missing"]
        rendered = renderer.render_report(
            _signed_analysis_report(
                {
                    "per_sk_decisions": [decision],
                    "scope_actions": [action],
                }
            )
        )
        self.assertIn("analysis Agent", rendered)
        self.assertIn("条件证据绑定", rendered)
        self.assertIn("source\\_revision=660e08e", rendered)
        self.assertIn("baseline\\_config\\_fingerprint=" + "3" * 64, rendered)
        self.assertIn("range\\_id=range\\-missing", rendered)

    def test_markdown_renderer_cli_writes_atomically_and_rejects_input_conflict(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_in = root / "report.json"
            markdown_out = root / "report.md"
            report = _signed_analysis_report(
                {"blockers": [], "next_agent_guidance_zh": "继续验证。"}
            )
            json_in.write_text(json.dumps(report))

            self.assertEqual(
                renderer.main(
                    ["--json-in", str(json_in), "--markdown-out", str(markdown_out)]
                ),
                0,
            )
            first = markdown_out.read_bytes()
            renderer.main(
                ["--json-in", str(json_in), "--markdown-out", str(markdown_out)]
            )
            self.assertEqual(markdown_out.read_bytes(), first)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                renderer.main(
                    ["--json-in", str(json_in), "--markdown-out", str(json_in)]
                )

    def test_markdown_renderer_cli_rejects_cyclic_symlink_paths(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        for case in ("input", "output"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                json_in = root / "report.json"
                markdown_out = root / "report.md"
                json_content = json.dumps(_signed_analysis_report())
                json_in.write_text(json_content)
                cyclic_path = json_in if case == "input" else markdown_out
                if case == "input":
                    json_in.unlink()
                cyclic_path.symlink_to(cyclic_path.name)

                with redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        renderer.main(
                            [
                                "--json-in",
                                str(json_in),
                                "--markdown-out",
                                str(markdown_out),
                            ]
                        )

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("cannot resolve path", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertTrue(cyclic_path.is_symlink())
                if case == "input":
                    self.assertFalse(markdown_out.exists())
                else:
                    self.assertEqual(json_in.read_text(), json_content)

    def test_markdown_renderer_validates_nested_schema_with_field_paths(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        malformed_reports = (
            ({"thresholds": []}, "thresholds"),
            ({"per_sk_decisions": {}}, "per_sk_decisions"),
            ({"per_sk_decisions": [[]]}, "per_sk_decisions[0]"),
            (
                {"per_sk_decisions": [{"original": []}]},
                "per_sk_decisions[0].original",
            ),
            (
                {"per_sk_decisions": [{"original": {"interval_us": []}}]},
                "per_sk_decisions[0].original.interval_us",
            ),
            (
                {"per_sk_decisions": [{"original": {"duration_sum_us": "bad"}}]},
                "per_sk_decisions[0].original.duration_sum_us",
            ),
            (
                {"per_sk_decisions": [{"sk_duration": []}]},
                "per_sk_decisions[0].sk_duration",
            ),
            (
                {"per_sk_decisions": [{"boundary": []}]},
                "per_sk_decisions[0].boundary",
            ),
            (
                {"per_sk_decisions": [{"identity": []}]},
                "per_sk_decisions[0].identity",
            ),
            (
                {
                    "per_sk_decisions": [
                        {"identity": {"ordered_child_op_sequence": "A"}}
                    ]
                },
                "per_sk_decisions[0].identity.ordered_child_op_sequence",
            ),
            (
                {
                    "per_sk_decisions": [
                        {
                            "original": {
                                "example_occurrence": {"multi_stream_analysis": []}
                            }
                        }
                    ]
                },
                "per_sk_decisions[0].original.example_occurrence.multi_stream_analysis",
            ),
            (
                {
                    "per_sk_decisions": [
                        {"original": {"example_occurrence": {"stream_ids": {"bad": 1}}}}
                    ]
                },
                "per_sk_decisions[0].original.example_occurrence.stream_ids",
            ),
            (
                {"per_sk_decisions": [{"evidence_errors": {"bad": 1}}]},
                "per_sk_decisions[0].evidence_errors",
            ),
            ({"scope_actions": ["bad"]}, "scope_actions[0]"),
            (
                {"diagnostic_hypotheses": [1]},
                "diagnostic_hypotheses[0]",
            ),
            (
                {"recommended_experiments": [None]},
                "recommended_experiments[0]",
            ),
            ({"blockers": {}}, "blockers"),
            ({"blockers": [1]}, "blockers[0]"),
        )
        for report, path in malformed_reports:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, re.escape(path)):
                    renderer.render_report(_signed_analysis_report(report))

    def test_markdown_renderer_rejects_non_scalar_decision_fields(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        for field in (
            "range_id",
            "sk_id",
            "classification",
            "classification_zh",
            "action",
            "mapping_method",
            "mapping_confidence",
        ):
            with self.subTest(field=field):
                report = _signed_analysis_report(
                    {
                        "per_sk_decisions": [
                            {
                                field: {"malicious": "object"},
                                "evidence_errors": ["baseline_mapping_missing"],
                            }
                        ]
                    }
                )
                with self.assertRaisesRegex(
                    ValueError, rf"per_sk_decisions\[0\]\.{field}"
                ):
                    renderer.render_report(report)

    def test_markdown_renderer_requires_evidence_for_missing_required_metrics(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        for empty_fields in (
            {},
            {"boundary": {}},
            {"identity": {"ordered_child_op_sequence": []}},
            {"original": {"example_occurrence": {"multi_stream_analysis": {}}}},
        ):
            with self.subTest(empty_fields=empty_fields):
                decision = {
                    "range_id": "range-with-silent-missing-data",
                    "evidence_errors": [],
                    **empty_fields,
                }
                with self.assertRaisesRegex(ValueError, "requires evidence_errors"):
                    renderer.render_report(
                        _signed_analysis_report({"per_sk_decisions": [decision]})
                    )

    def test_markdown_renderer_cli_controls_all_nested_schema_errors(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        malformed_reports = (
            {"per_sk_decisions": [{"original": {"interval_us": "malicious"}}]},
            {"per_sk_decisions": [{"original": {"duration_sum_us": ["malicious"]}}]},
            {
                "per_sk_decisions": [
                    {
                        "range_id": {"malicious": "object"},
                        "evidence_errors": ["baseline_mapping_missing"],
                    }
                ]
            },
        )
        for index, report in enumerate(malformed_reports):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                json_in = root / "report.json"
                markdown_out = root / "report.md"
                json_in.write_text(json.dumps(_signed_analysis_report(report)))
                markdown_out.write_text("existing\n")

                with redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        renderer.main(
                            [
                                "--json-in",
                                str(json_in),
                                "--markdown-out",
                                str(markdown_out),
                            ]
                        )

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("per_sk_decisions[0]", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertEqual(markdown_out.read_text(), "existing\n")

    def test_markdown_renderer_cli_reports_schema_error_without_overwrite(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_in = root / "report.json"
            markdown_out = root / "report.md"
            json_in.write_text(
                json.dumps(_signed_analysis_report({"thresholds": ["bad"]}))
            )
            markdown_out.write_text("existing\n")

            with redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    renderer.main(
                        [
                            "--json-in",
                            str(json_in),
                            "--markdown-out",
                            str(markdown_out),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("thresholds", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(markdown_out.read_text(), "existing\n")

    def test_markdown_renderer_escapes_html_and_active_markdown_payloads(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        payload = (
            '<img src=x onerror="alert(1)"> & **bold** `code` '
            "[click](javascript:alert(1)) | line\nnext"
        )
        decision, action = _renderer_contract_pair(
            classification="insufficient_evidence",
            action="reprofile",
            range_id=payload,
            sk_id="sk-payload",
        )
        decision.update(
            {
                "original": {"interval_us": {}, "duration_sum_us": {}},
                "evidence_errors": ["baseline_mapping_missing"],
            }
        )
        report = _signed_analysis_report(
            {
                "per_sk_decisions": [decision],
                "scope_actions": [action],
                "diagnostic_hypotheses": [],
                "recommended_experiments": [],
                "blockers": [payload, {"detail": payload}],
                "next_agent_guidance_zh": payload,
            }
        )

        first = renderer.render_report(report)
        second = renderer.render_report(json.loads(json.dumps(report)))

        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        self.assertNotIn("<img", first)
        self.assertIn("&lt;img", first)
        self.assertIn("&amp;", first)
        self.assertNotIn("**bold**", first)
        self.assertIn(r"\*\*bold\*\*", first)
        self.assertNotIn("`code`", first)
        self.assertIn(r"\`code\`", first)
        self.assertNotIn("[click](javascript:alert(1))", first)
        self.assertIn(r"\[click\]\(javascript:alert\(1\)\)", first)
        self.assertIn(r"\|", first)

    def test_markdown_renderer_requires_complete_signed_schema(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        required_fields = (
            "schema_version",
            "analysis_id",
            "experiment_id",
            "round_id",
            "analysis_agent_id",
            "candidate_name",
            "source_revision",
            "baseline_profile_fingerprint",
            "candidate_profile_fingerprint",
            "baseline_config_fingerprint",
            "candidate_config_fingerprint",
            "baseline_workload_fingerprint",
            "candidate_workload_fingerprint",
            "workload_fingerprint",
            "baseline_control_fingerprint",
            "candidate_control_fingerprint",
            "control_fingerprint",
            "declared_change_set",
            "inputs",
            "thresholds",
            "per_sk_decisions",
            "scope_actions",
            "diagnostic_hypotheses",
            "recommended_experiments",
            "blockers",
            "next_agent_guidance_zh",
            "analysis_content_fingerprint",
        )
        cases = [({}, "schema_version")]
        for field in required_fields:
            report = _signed_analysis_report()
            report.pop(field)
            cases.append((report, field))
        for old_version in ("0.9", "1.0", "1.1"):
            wrong_version = _signed_analysis_report({"schema_version": old_version})
            cases.append((wrong_version, "schema_version"))
        wrong_hash = _signed_analysis_report()
        wrong_hash["analysis_content_fingerprint"] = "0" * 64
        cases.append((wrong_hash, "analysis_content_fingerprint"))
        for field in (
            "baseline_control_fingerprint",
            "candidate_control_fingerprint",
        ):
            cases.append((_signed_analysis_report({field: None}), field))

        for report, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    renderer.render_report(report)

    def test_markdown_renderer_accepts_schema_1_2_and_renders_structural_sections(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        decision, action = _renderer_contract_pair()
        decision.update(
            {
                "device_id": 0,
                "model_id": "48",
                "raw_sk_id": 7,
                "child_count": 2,
                "mapping_method": "source_scope_map",
                "mapping_confidence": "exact",
                "mapping_blockers": [],
                "candidate_binding_status": None,
            }
        )
        report = _signed_structural_report(
            {"per_sk_decisions": [decision], "scope_actions": [action]}
        )

        rendered = renderer.render_report(report)

        for heading in (
            "## 结构关联覆盖率",
            "## Child Count 分布",
            "## 逐 SK 结构与性能证据",
            "## 关联阻塞项",
            "## Automatic AOT 与源码动作边界",
        ):
            self.assertIn(heading, rendered)
        self.assertIn(
            "| SK | Child 数 | 同进程绑定 | Kernel Projection | "
            "各 Step 替代解 | 性能分类 | 源码动作 |",
            rendered,
        )
        self.assertIn("raw Task ID/名称提示仅用于诊断", rendered)
        self.assertIn("child_count<5 没有被过滤", rendered)
        self.assertIn(
            "automatic AOT kernel projection 可用于性能分类，但不能证明 Python "
            "源码 offset，不能直接驱动 prune",
            rendered,
        )

    def test_schema_1_2_requires_all_structural_fields_without_synthesis(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        for field in (
            "association_protocol",
            "candidate_binding_evidence",
            "canonical_graph_fingerprints",
            "stream_role_mapping",
            "graph_alignment_proof",
            "mapping_coverage",
        ):
            with self.subTest(field=field):
                report = _signed_structural_report()
                report.pop(field)
                report = _resign_analysis_report(report)
                with self.assertRaisesRegex(ValueError, field):
                    renderer.render_report(report)

    def test_schema_1_2_accepts_complete_exact_projected_trace_evidence(self):
        renderer = importlib.import_module("render_fusion_performance_report")

        rendered = renderer.render_report(_signed_exact_projected_trace_report())

        self.assertIn(
            "kernel_projection_structural / exact_projected_trace",
            rendered.replace(r"\_", "_"),
        )
        self.assertIn("仅性能分类；不可直接 prune", rendered)

    def test_schema_1_2_rejects_retired_whole_graph_mapping(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        report = _signed_structural_report()
        report["per_sk_decisions"][0].update(
            mapping_method="whole_graph_structural",
            mapping_confidence="exact_structural",
        )
        report = _signed_analysis_report(report)

        with self.assertRaisesRegex(ValueError, "mapping_method|mapping_confidence"):
            renderer.render_report(report)

    def test_schema_1_2_rejects_tampered_projected_trace_proof(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        identity = "device:0/model:48/sk:7"
        mutations = {
            "alternative solution": lambda report: report["graph_alignment_proof"][
                identity
            ]["alternative_solution_count_by_step"].update({"2": 1}),
            "non-injective stream role": lambda report: report["stream_role_mapping"][
                identity
            ]["1"].update({"profile:other": 11}),
            "mapping fingerprint": lambda report: report["candidate_binding_evidence"][
                "occurrences"
            ][0]["evidence_fingerprints"].update({"projection_mapping": "f" * 64}),
            "fewer than three steps": lambda report: (
                report["candidate_binding_evidence"]["occurrences"].pop(),
                report["candidate_binding_evidence"]["summary"].update(
                    {"parent_occurrences": 2, "status_counts": {"bound": 2}}
                ),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = _signed_exact_projected_trace_report()
                mutate(report)
                report = _signed_analysis_report(report)
                with self.assertRaisesRegex(ValueError, "projected-trace"):
                    renderer.render_report(report)

    def test_projected_trace_report_allows_diagnostic_children_for_blocked_peer(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        report = _signed_exact_projected_trace_report()
        blocked = copy.deepcopy(report["candidate_binding_evidence"]["occurrences"][0])
        blocked["status"] = "blocked"
        blocked["process_identity"]["sk_id"] = 8
        blocked["blockers"] = ["kernel_projection_stream_assignment_ambiguous"]
        report["candidate_binding_evidence"]["occurrences"].append(blocked)
        report["candidate_binding_evidence"]["summary"].update(
            {"parent_occurrences": 4, "status_counts": {"blocked": 1, "bound": 3}}
        )
        report = _signed_analysis_report(report)

        rendered = renderer.render_report(report).replace(r"\_", "_")
        self.assertIn("exact_projected_trace", rendered)

    def test_schema_1_2_mapping_enums_and_coverage_totals_are_closed(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        self.assertEqual(renderer.SUPPORTED_SCHEMA_VERSIONS, {"1.2"})
        self.assertEqual(
            renderer.MAPPING_METHODS,
            {
                "sk_meta_node_ids",
                "source_scope_map",
                "kernel_projection_structural",
            },
        )
        self.assertEqual(
            renderer.MAPPING_CONFIDENCES,
            {
                "exact",
                "exact_projected_trace",
                "diagnostic_only",
                "ambiguous",
                "unmapped",
            },
        )
        mutations = {
            "raw exact promotion": lambda report: report["per_sk_decisions"][0].update(
                mapping_method="sk_meta_node_ids", mapping_confidence="exact"
            ),
            "unknown confidence": lambda report: report["per_sk_decisions"][0].update(
                mapping_confidence="probably_exact"
            ),
            "coverage total": lambda report: report["mapping_coverage"].update(
                total_sk_ids=2
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = _signed_structural_report()
                mutate(report)
                report = _signed_analysis_report(report)
                with self.assertRaises(ValueError):
                    renderer.render_report(report)

    def test_schema_1_2_prune_requires_exact_source_interval(self):
        renderer = importlib.import_module("render_fusion_performance_report")

        structural = _signed_exact_projected_trace_report()
        structural["per_sk_decisions"][0].update(
            classification="neutral",
            action="block",
            action_blocker="prune_requires_exact_source_scope_boundary_mapping",
        )
        structural["scope_actions"][0].update(classification="neutral", action="block")
        structural = _signed_analysis_report(structural)
        self.assertIn("仅性能分类", renderer.render_report(structural))

        structural_prune = json.loads(json.dumps(structural))
        structural_prune["per_sk_decisions"][0]["action"] = "prune"
        structural_prune["scope_actions"][0]["action"] = "prune"
        structural_prune = _signed_analysis_report(structural_prune)
        with self.assertRaisesRegex(ValueError, "prune|source"):
            renderer.render_report(structural_prune)

        source_decision, source_action = _renderer_contract_pair(
            classification="neutral", action="prune"
        )
        source_decision.update(
            device_id=0,
            model_id="48",
            raw_sk_id=7,
            child_count=2,
            mapping_method="source_scope_map",
            mapping_confidence="exact",
            mapping_blockers=[],
            candidate_binding_status=None,
        )
        source_without_interval = _signed_structural_report(
            {
                "per_sk_decisions": [source_decision],
                "scope_actions": [source_action],
            }
        )
        with self.assertRaisesRegex(ValueError, "prune|source interval"):
            renderer.render_report(source_without_interval)

        boundary = {
            "start_op": "A",
            "end_op": "B",
            "source_file": "model/decoder.py",
            "start_offset": 10,
            "end_offset": 20,
        }
        source_decision, source_action = _renderer_contract_pair(
            classification="neutral",
            action="prune",
            boundary=boundary,
            interval_unproven=False,
        )
        source_decision.update(
            device_id=0,
            model_id="48",
            raw_sk_id=7,
            child_count=2,
            mapping_method="source_scope_map",
            mapping_confidence="exact",
            mapping_blockers=[],
            candidate_binding_status=None,
        )
        source_with_interval = _signed_structural_report(
            {
                "per_sk_decisions": [source_decision],
                "scope_actions": [source_action],
            }
        )
        self.assertIn(
            "## Scope 保留与裁剪", renderer.render_report(source_with_interval)
        )

        blocked_source = json.loads(json.dumps(source_with_interval))
        blocked_source["per_sk_decisions"][0]["mapping_blockers"] = [
            "source_scope_mapping_ambiguous"
        ]
        blocked_source["mapping_coverage"]["blocker_counts"] = {
            "source_scope_mapping_ambiguous": 1
        }
        blocked_source = _signed_analysis_report(blocked_source)
        with self.assertRaisesRegex(ValueError, "exact|blocker|prune"):
            renderer.render_report(blocked_source)

    def test_schema_1_2_malformed_values_raise_value_error(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        cases = []
        invalid_status = _signed_exact_projected_trace_report()
        invalid_status["candidate_binding_evidence"]["occurrences"][0]["status"] = {}
        cases.append(invalid_status)
        invalid_version = _signed_exact_projected_trace_report()
        invalid_version["schema_version"] = []
        cases.append(invalid_version)
        invalid_projection = _signed_exact_projected_trace_report()
        invalid_projection["graph_alignment_proof"]["device:0/model:48/sk:7"][
            "canonical_baseline_child_keys"
        ] = [[], "baseline:B"]
        cases.append(invalid_projection)

        for index, report in enumerate(cases):
            with self.subTest(index=index):
                report = _signed_analysis_report(report)
                with self.assertRaises(ValueError):
                    renderer.render_report(report)

    def test_markdown_renderer_rejects_recomputed_hash_with_empty_decisions(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        report = _signed_analysis_report({"per_sk_decisions": [], "scope_actions": []})

        with self.assertRaisesRegex(ValueError, "per_sk_decisions"):
            renderer.render_report(report)

    def test_markdown_renderer_rejects_recomputed_hash_with_forged_identity(self):
        renderer = importlib.import_module("render_fusion_performance_report")

        forged_candidate = _signed_analysis_report()
        forged_candidate["candidate_name"] = "S9"
        forged_candidate = _signed_analysis_report(forged_candidate)
        forged_round = _signed_analysis_report({"round_id": "S9-BASE"})
        identity = {
            field: forged_round[field]
            for field in (
                "experiment_id",
                "round_id",
                "analysis_agent_id",
                "candidate_name",
                "source_revision",
            )
        }
        forged_round["analysis_id"] = (
            "analysis-"
            + hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        )
        forged_round = _signed_analysis_report(forged_round)

        for report, expected_error in (
            (forged_candidate, "analysis_id"),
            (forged_round, "round_id"),
        ):
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    renderer.render_report(report)

    def test_markdown_renderer_enforces_classification_action_mapping(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        invalid_pairs = (
            ("beneficial", "prune"),
            ("neutral", "keep"),
            ("regressed", "keep"),
            ("insufficient_evidence", "keep"),
            ("unknown", "keep"),
        )
        for classification, action in invalid_pairs:
            with self.subTest(classification=classification, action=action):
                decision, scope_action = _renderer_contract_pair(
                    classification=classification, action=action
                )
                report = _signed_analysis_report(
                    {
                        "per_sk_decisions": [decision],
                        "scope_actions": [scope_action],
                    }
                )

                with self.assertRaisesRegex(ValueError, "classification|action"):
                    renderer.render_report(report)

    def test_markdown_renderer_requires_one_to_one_scope_actions(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        decision, action = _renderer_contract_pair()
        other_decision, other_action = _renderer_contract_pair(
            range_id="range-other", sk_id="sk-other"
        )
        mismatched_fields = []
        for field, value in (
            ("classification", "neutral"),
            ("action", "prune"),
            ("source_scope", "decoder.layer.9"),
            ("boundary", {"start_op": "X", "end_op": "Y"}),
            ("ordered_child_op_sequence", ["X", "Y"]),
        ):
            changed = json.loads(json.dumps(action))
            changed[field] = value
            mismatched_fields.append((f"mismatch-{field}", [decision], [changed]))
        proven_boundary = {
            "start_op": "A",
            "end_op": "B",
            "source_file": "model/decoder.py",
            "start_offset": 10,
            "end_offset": 20,
        }
        proven_decision, false_unproven_action = _renderer_contract_pair(
            boundary=proven_boundary,
            interval_unproven=True,
        )
        cases = (
            ("missing", [decision], []),
            ("extra", [decision], [action, other_action]),
            ("duplicate-action", [decision], [action, action]),
            ("duplicate-decision", [decision, decision], [action]),
            ("different-key", [decision], [other_action]),
            (
                "interval-proof",
                [proven_decision],
                [false_unproven_action],
            ),
            *mismatched_fields,
        )
        for label, decisions, actions in cases:
            with self.subTest(label=label):
                report = _signed_analysis_report(
                    {
                        "per_sk_decisions": decisions,
                        "scope_actions": actions,
                    }
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "scope_actions|duplicate|interval_unproven|mapping_coverage",
                ):
                    renderer.render_report(report)

    def test_markdown_renderer_requires_complete_scope_action_evidence(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        action = {
            "range_id": "range-a",
            "sk_id": "sk-a",
            "classification": "regressed",
            "action": "prune",
            "source_scope": "decoder.layer.0",
            "boundary": {"start_op": "A", "end_op": "B"},
            "ordered_child_op_sequence": ["A", "B"],
            "interval_unproven": True,
        }
        cases = []
        for field in (
            "range_id",
            "sk_id",
            "classification",
            "action",
            "source_scope",
            "boundary",
            "ordered_child_op_sequence",
            "interval_unproven",
        ):
            incomplete = dict(action)
            incomplete.pop(field)
            cases.append((incomplete, field))
        cases.extend(
            (
                ({**action, "boundary": []}, "boundary"),
                (
                    {**action, "ordered_child_op_sequence": "A"},
                    "ordered_child_op_sequence",
                ),
                ({**action, "interval_unproven": "true"}, "interval_unproven"),
                ({**action, "interval_unproven": False}, "interval_unproven"),
            )
        )

        for malformed_action, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    renderer.render_report(
                        _signed_analysis_report({"scope_actions": [malformed_action]})
                    )

    def test_source_interval_proof_requires_file_and_ordered_offsets(self):
        cases = (
            ({}, False),
            ({"source_file": "model.py", "start_offset": 1}, False),
            (
                {
                    "source_file": "/tmp/model.py",
                    "start_offset": 2,
                    "end_offset": 3,
                },
                False,
            ),
            (
                {
                    "source_file": "../model.py",
                    "start_offset": 2,
                    "end_offset": 3,
                },
                False,
            ),
            (
                {
                    "source_file": "model.py",
                    "start_offset": 2,
                    "end_offset": 2,
                },
                False,
            ),
            (
                {
                    "source_file": "model.py",
                    "start_offset": True,
                    "end_offset": 3,
                },
                False,
            ),
            (
                {
                    "source_file": "model.py",
                    "start_offset": 2,
                    "end_offset": 3,
                },
                True,
            ),
        )
        for boundary, expected in cases:
            with self.subTest(boundary=boundary):
                self.assertEqual(
                    analyze_fusion_performance._source_interval_proven(boundary),
                    expected,
                )

    def test_markdown_renderer_rejects_unsafe_source_interval_path(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        for source_file in ("/tmp/model.py", "../model.py"):
            with self.subTest(source_file=source_file):
                decision, action = _renderer_contract_pair()
                decision["boundary"].update(
                    {
                        "source_file": source_file,
                        "start_offset": 1,
                        "end_offset": 2,
                    }
                )
                action["boundary"].update(
                    {
                        "source_file": source_file,
                        "start_offset": 1,
                        "end_offset": 2,
                    }
                )
                action["interval_unproven"] = False
                with self.assertRaisesRegex(ValueError, "source_file"):
                    renderer.render_report(
                        _signed_analysis_report(
                            {
                                "per_sk_decisions": [decision],
                                "scope_actions": [action],
                            }
                        )
                    )

    def test_scope_action_mapping_and_prune_evidence_gate_are_exact(self):
        self.assertEqual(
            analyze_fusion_performance.ACTION_BY_CLASSIFICATION,
            {
                "beneficial": "keep",
                "regressed": "prune",
                "neutral": "prune",
                "insufficient_evidence": "reprofile",
            },
        )
        proven_boundary = {
            "start_op": "A",
            "end_op": "B",
            "source_file": "model/decoder.py",
            "start_offset": 10,
            "end_offset": 20,
        }
        self.assertTrue(
            analyze_fusion_performance.performance_mapping_exact(
                "source_scope_map", "exact"
            )
        )
        self.assertTrue(
            analyze_fusion_performance.performance_mapping_exact(
                "kernel_projection_structural", "exact_projected_trace"
            )
        )
        self.assertTrue(
            analyze_fusion_performance.performance_mapping_exact(
                "kernel_projection_structural", "exact_projected_trace"
            )
        )
        self.assertFalse(
            analyze_fusion_performance.performance_mapping_exact(
                "sk_meta_node_ids", "exact"
            )
        )
        self.assertTrue(
            analyze_fusion_performance.source_mapping_actionable(
                "source_scope_map", "exact", proven_boundary
            )
        )
        self.assertFalse(
            analyze_fusion_performance.source_mapping_actionable(
                "kernel_projection_structural", "exact_projected_trace", proven_boundary
            )
        )
        for classification, expected_action in (
            ("beneficial", "keep"),
            ("regressed", "prune"),
            ("neutral", "prune"),
            ("insufficient_evidence", "reprofile"),
        ):
            with self.subTest(classification=classification):
                self.assertEqual(
                    analyze_fusion_performance.scope_action(
                        classification,
                        "source_scope_map",
                        "exact",
                        proven_boundary,
                    ),
                    expected_action,
                )
        for classification, method, confidence, boundary in (
            ("regressed", "sk_meta_node_ids", "diagnostic_only", proven_boundary),
            ("neutral", "kernel_projection_structural", "exact_projected_trace", {}),
            (
                "regressed",
                "source_scope_map",
                "exact",
                {"start_op": "A", "end_op": "B"},
            ),
        ):
            with self.subTest(
                classification=classification,
                method=method,
                confidence=confidence,
                boundary=boundary,
            ):
                self.assertEqual(
                    analyze_fusion_performance.scope_action(
                        classification, method, confidence, boundary
                    ),
                    "block",
                )

    def test_mapping_coverage_counts_every_inventoried_sk_and_buckets_children(self):
        decisions = [
            {
                "sk_id": f"sk-{index}",
                "device_id": 0,
                "model_id": "48",
                "raw_sk_id": index,
                "child_count": child_count,
                "candidate_binding_status": "bound",
                "mapping_confidence": confidence,
                "mapping_blockers": blockers,
            }
            for index, (child_count, confidence, blockers) in enumerate(
                (
                    (1, "exact_projected_trace", ()),
                    (2, "exact_projected_trace", ()),
                    (4, "exact_projected_trace", ()),
                    (5, "ambiguous", ("structural_replay_mismatch",)),
                    (9, "unmapped", ("structural_replay_mismatch",)),
                )
            )
        ]

        coverage = analyze_fusion_performance._mapping_coverage(decisions)

        self.assertEqual(
            coverage,
            {
                "total_sk_ids": 5,
                "bound_sk_ids": 5,
                "exact_projected_trace_sk_ids": 3,
                "ambiguous_sk_ids": 1,
                "unmapped_sk_ids": 1,
                "filtered_by_child_count": 0,
                "child_count_distribution": {"1": 1, "2": 1, "4": 1, "5+": 2},
                "blocker_counts": {"structural_replay_mismatch": 2},
            },
        )

    def test_mapping_coverage_does_not_merge_distinct_native_sk_inventory(self):
        common = {
            "sk_id": "sk-same-derived-identity",
            "model_id": "48",
            "child_count": 2,
            "candidate_binding_status": "bound",
            "mapping_confidence": "diagnostic_only",
            "mapping_blockers": ["cross_compile_runtime_id_not_identity"],
        }

        coverage = analyze_fusion_performance._mapping_coverage(
            [
                {**common, "device_id": 0, "raw_sk_id": 7},
                {**common, "device_id": 1, "raw_sk_id": 7},
                {**common, "device_id": 1, "raw_sk_id": 8},
            ]
        )

        self.assertEqual(coverage["total_sk_ids"], 3)
        self.assertEqual(coverage["bound_sk_ids"], 3)
        self.assertEqual(coverage["unmapped_sk_ids"], 3)
        self.assertEqual(coverage["child_count_distribution"], {"2": 3})

    def test_strict_structural_binding_uses_association_metadata_sequence(self):
        association = {
            "candidate_binding_status": "bound",
            "candidate_binding": {
                "status": "bound",
                "process_identity": {
                    "device_id": 0,
                    "model_id": 48,
                    "sk_id": 7,
                },
                "ordered_child_ops": ["Add", "MatMul"],
                "child_count": 2,
            },
        }

        selected = analyze_fusion_performance._strict_structural_binding(
            {(0, 48, 7): association},
            device_id=0,
            model_id="48",
            raw_sk_id=7,
        )

        self.assertIs(selected["association"], association)
        self.assertEqual(selected["ordered_child_ops"], ["Add", "MatMul"])
        self.assertEqual(selected["child_count"], 2)

    def test_invalid_structural_binding_preserves_metadata_sequence_fallback(self):
        matching_metadata_groups = [
            {
                "nodes": [
                    {"op_type": "HcPre"},
                    {"op_type": "RmsNorm"},
                ]
            }
        ]
        invalid_binding = {
            "association": {"mapping_blockers": ["kernel_projection_stream_unmapped"]},
            "ordered_child_ops": [],
            "child_count": 0,
            "valid": False,
        }

        grouped = analyze_fusion_performance._metadata_groups_by_sequence(
            matching_metadata_groups,
            invalid_binding,
        )

        self.assertEqual(len(grouped), 1)
        self.assertEqual(next(iter(grouped.values())), matching_metadata_groups)

    def test_cli_classifies_exact_projected_trace_without_replay_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            artifacts["source_scope_map"].write_text(json.dumps({"task_ranges": []}))
            binding = {
                "status": "bound",
                "process_identity": {
                    "native_pid": 4100,
                    "device_id": 0,
                    "model_id": 48,
                    "sk_id": 1,
                },
                "step_id": 3,
                "parent_interval_ns": [0, 6000],
                "ordered_child_node_keys": ["profile:A"],
                "ordered_child_ops": ["A"],
                "child_count": 1,
                "evidence_fingerprints": {
                    "kernel_details": "a" * 64,
                    "projection_mapping": "b" * 64,
                    "profile_sk_graph_origin": "c" * 64,
                },
                "blockers": [],
            }
            association = {
                "mapping_method": "kernel_projection_structural",
                "mapping_confidence": "exact_projected_trace",
                "mapping_blockers": [],
                "candidate_binding_status": "bound",
                "candidate_binding": binding,
                "baseline_occurrences": {
                    "occurrences": [
                        {
                            "interval_us": 8.0,
                            "duration_sum_us": 8.0,
                            "union_duration_us": 8.0,
                        }
                        for _ in range(3)
                    ]
                },
                "canonical_graph_fingerprints": {},
                "stream_role_mapping": {},
                "graph_alignment_proof": {},
            }
            structural_context = {
                "protocol": {
                    "protocol": "kernel_projection_trace_v2",
                    "status": "associated",
                    "manifest_set_fingerprint": "d" * 64,
                    "manifest_content_fingerprints": {},
                    "blockers": [],
                },
                "associations": {(0, 48, 1): association},
                "candidate_binding_evidence": {
                    "summary": {"bound": 1},
                    "occurrences": [binding],
                },
                "canonical_graph_fingerprints": {},
                "stream_role_mapping": {},
                "graph_alignment_proof": {},
            }
            json_out = root / "exact-structural-cli.json"

            with (
                mock.patch.object(
                    analyze_fusion_performance,
                    "_load_structural_context",
                    return_value=structural_context,
                ),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            report = json.loads(json_out.read_text())
            decision = report["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["mapping_method"], "kernel_projection_structural")
        self.assertEqual(decision["mapping_confidence"], "exact_projected_trace")
        self.assertEqual(decision["classification"], "beneficial")
        self.assertEqual(decision["action"], "keep")
        self.assertNotIn("replay_evidence", report["inputs"])
        self.assertNotIn("deep_fusion_replay", report)
        self.assertEqual(report["mapping_coverage"]["exact_projected_trace_sk_ids"], 1)

    def test_post_base_diagnostics_never_recommend_option_experiments(self):
        diagnostics = analyze_fusion_performance._build_regression_diagnostics(
            [_cube_vector_regression_decision()],
            None,
            {"accepted_options": {"auto_op_parallel": [1]}},
            {},
        )

        self.assertEqual(diagnostics["recommended_experiments"], [])

    def test_child_trace_range_association_must_cover_every_event(self):
        decision = _cube_vector_regression_decision()
        child = {
            "range_ids": ["range-target"],
            "range_association_complete": False,
            "stream_identity_complete": True,
            "core_family_counts": [
                {"name": "CUBE", "count": 1},
                {"name": "VECTOR", "count": 1},
            ],
            "multi_stream_analysis": {"cube_vector_parallel_detected": False},
        }
        blocked = analyze_fusion_performance._build_regression_diagnostics(
            [decision], child, {}, {}
        )
        child["range_association_complete"] = True
        associated = analyze_fusion_performance._build_regression_diagnostics(
            [decision], child, {}, {}
        )

        self.assertFalse(
            any(
                item["kind"] == "cube_vector_serialization"
                for item in blocked["diagnostic_hypotheses"]
            )
        )
        self.assertTrue(any("关联" in item for item in blocked["blockers"]))
        self.assertTrue(
            any(
                item["kind"] == "cube_vector_serialization"
                for item in associated["diagnostic_hypotheses"]
            )
        )

    def test_cube_vector_serialization_requires_both_child_core_families(self):
        decision = _cube_vector_regression_decision()
        environment = {"accepted_options": {"auto_op_parallel": [1]}}
        cases = (
            ("serialized", {"CUBE": 1, "VECTOR": 1}, False, True, False),
            ("parallel", {"CUBE": 1, "VECTOR": 1}, True, False, False),
            ("overlap_unknown", {"CUBE": 1, "VECTOR": 1}, None, False, True),
            ("cube_only", {"CUBE": 1}, False, False, True),
            ("vector_only", {"VECTOR": 1}, False, False, True),
            ("other_only", {"OTHER": 2}, False, False, True),
            ("empty", {}, False, False, True),
        )

        for name, counts, child_overlap, has_direct_evidence, is_missing in cases:
            with self.subTest(name=name):
                child = {
                    "range_ids": ["range-target"],
                    "range_association_complete": True,
                    "stream_identity_complete": True,
                    "core_family_counts": [
                        {"name": family, "count": count}
                        for family, count in counts.items()
                    ],
                    "multi_stream_analysis": {
                        "cube_vector_parallel_detected": child_overlap
                    },
                }
                diagnostics = analyze_fusion_performance._build_regression_diagnostics(
                    [decision], child, environment, {}
                )
                kinds = {item["kind"] for item in diagnostics["diagnostic_hypotheses"]}
                auto_experiments = [
                    item
                    for item in diagnostics["recommended_experiments"]
                    if item["option"] == "auto_op_parallel"
                ]

                self.assertEqual(
                    "cube_vector_serialization" in kinds, has_direct_evidence
                )
                self.assertEqual(auto_experiments, [])
                self.assertEqual(
                    "cube_vector_child_trace_missing" in kinds,
                    is_missing,
                )
                if is_missing:
                    missing = next(
                        item
                        for item in diagnostics["diagnostic_hypotheses"]
                        if item["kind"] == "cube_vector_child_trace_missing"
                    )
                    self.assertEqual(missing["confidence"], "low")
                    self.assertIn(
                        "child_trace_missing_cube_vector_evidence",
                        missing["evidence"],
                    )
                    self.assertTrue(
                        any(
                            "child_trace_missing_cube_vector_evidence" in blocker
                            for blocker in diagnostics["blockers"]
                        )
                    )

    def test_missing_or_unscoped_child_trace_is_reported_for_each_range(self):
        decisions = [
            _cube_vector_regression_decision("range-a"),
            _cube_vector_regression_decision("range-b"),
        ]
        unscoped_child = {
            "range_ids": [],
            "range_association_complete": False,
            "stream_identity_complete": True,
            "core_family_counts": [
                {"name": "CUBE", "count": 1},
                {"name": "VECTOR", "count": 1},
            ],
            "multi_stream_analysis": {"cube_vector_parallel_detected": False},
        }

        for name, child in (("not_provided", None), ("unscoped", unscoped_child)):
            with self.subTest(name=name):
                diagnostics = analyze_fusion_performance._build_regression_diagnostics(
                    decisions, child, {}, {}
                )
                missing = [
                    item
                    for item in diagnostics["diagnostic_hypotheses"]
                    if item["kind"] == "cube_vector_child_trace_missing"
                ]

                self.assertEqual(
                    [item["range_id"] for item in missing],
                    ["range-a", "range-b"],
                )
                self.assertTrue(all(item["confidence"] == "low" for item in missing))
                for range_id in ("range-a", "range-b"):
                    self.assertTrue(
                        any(
                            range_id in blocker
                            and "child_trace_missing_cube_vector_evidence" in blocker
                            for blocker in diagnostics["blockers"]
                        )
                    )
                self.assertFalse(
                    any(
                        item["option"] == "auto_op_parallel"
                        for item in diagnostics["recommended_experiments"]
                    )
                )

    def test_classifies_all_four_performance_outcomes(self):
        def classify(baseline_values, candidate_values):
            return analyze_fusion_performance.classify_performance(
                analyze_fusion_performance._robust_stats(baseline_values),
                analyze_fusion_performance._robust_stats(candidate_values),
            )["classification"]

        self.assertEqual(classify([100, 101, 99], [94, 95, 96]), "beneficial")
        self.assertEqual(classify([100, 101, 99], [104, 105, 106]), "regressed")
        self.assertEqual(classify([100, 101, 99], [99, 100, 101]), "neutral")
        self.assertEqual(classify([100, 101], [95, 96]), "insufficient_evidence")

    def test_robust_stats_apply_noise_and_absolute_floors(self):
        stats = analyze_fusion_performance._robust_stats([90, 100, 110])
        self.assertEqual(stats["mad"], 10)

        def classify(baseline_values, candidate_values):
            return analyze_fusion_performance.classify_performance(
                analyze_fusion_performance._robust_stats(baseline_values),
                analyze_fusion_performance._robust_stats(candidate_values),
            )

        self.assertEqual(
            classify([100, 100, 100], [98, 98, 98])["classification"],
            "neutral",
        )
        self.assertEqual(
            classify([100, 100, 100], [95, 95, 95])["classification"],
            "beneficial",
        )
        self.assertEqual(
            classify([100, 100, 100], [99.5, 99.5, 99.5])["classification"],
            "neutral",
        )
        self.assertEqual(
            classify([100, 100, 100], [100.5, 100.5, 100.5])["classification"],
            "neutral",
        )
        noisy = classify([90, 100, 110], [80, 95, 100])
        self.assertGreater(noisy["noise_threshold_pct"], 3.0)

    def test_minimum_sample_unrepresented_tail_requires_reprofile(self):
        cases = (
            ([14.38, 15.66, 46.82], [33.62, 35.24, 35.54]),
            ([16.04, 49.58, 49.60], [31.82, 32.36, 32.90]),
        )

        for baseline_values, candidate_values in cases:
            with self.subTest(baseline=baseline_values):
                decision = analyze_fusion_performance.classify_performance(
                    analyze_fusion_performance._robust_stats(baseline_values),
                    analyze_fusion_performance._robust_stats(candidate_values),
                )

                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertIn(
                    "minimum_sample_unrepresented_tail",
                    decision["evidence_errors"],
                )

    def test_minimum_sample_balanced_dispersion_remains_classifiable(self):
        decision = analyze_fusion_performance.classify_performance(
            analyze_fusion_performance._robust_stats([90, 100, 110]),
            analyze_fusion_performance._robust_stats([80, 95, 100]),
        )

        self.assertNotEqual(decision["classification"], "insufficient_evidence")
        self.assertNotIn(
            "minimum_sample_unrepresented_tail", decision["evidence_errors"]
        )

    def test_robust_stats_discard_non_finite_occurrences(self):
        stats = analyze_fusion_performance._robust_stats(
            [99, 100, 101, float("nan"), float("inf"), float("-inf")]
        )

        self.assertEqual(stats["count"], 3)
        for field in ("p50", "p90", "mad", "min", "max"):
            self.assertTrue(math.isfinite(stats[field]))

    def test_non_finite_derived_threshold_is_insufficient_and_json_safe(self):
        decision = analyze_fusion_performance.classify_performance(
            analyze_fusion_performance._robust_stats([0, 0, 0]),
            analyze_fusion_performance._robust_stats([0, 0, 0]),
        )

        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertIn("non_finite_performance_statistics", decision["evidence_errors"])
        self.assertIsNone(decision["noise_threshold_pct"])
        json.dumps(decision, allow_nan=False)

    def test_canonical_fingerprint_ignores_json_key_order(self):
        left = {"model": {"name": "demo", "batch": 1}, "rank_size": 8}
        right = {"rank_size": 8, "model": {"batch": 1, "name": "demo"}}

        self.assertEqual(
            analyze_fusion_performance._canonical_sha256(left),
            analyze_fusion_performance._canonical_sha256(right),
        )

    def test_json_pointer_removal_decodes_rfc6901_escapes(self):
        source = {"a/b": {"~key": "remove", "keep": True}}

        normalized = analyze_fusion_performance._remove_json_pointer(
            source, "/a~1b/~0key"
        )

        self.assertEqual(normalized, {"a/b": {"keep": True}})
        self.assertEqual(source["a/b"]["~key"], "remove")

    def test_array_pointers_preserve_undeclared_original_index_difference(self):
        validation = analyze_fusion_performance.validate_fingerprints(
            {"items": ["same-0", "same-1", {"frozen": "baseline"}]},
            {"items": ["same-0", "same-1", {"frozen": "candidate"}]},
            _complete_workload_manifest(),
            _complete_workload_manifest(),
            {
                "allowed_json_pointers": ["/items/0", "/items/1"],
                "only_change_zh": "只允许前两个数组项变化",
            },
        )
        decision = analyze_fusion_performance.classify_performance(
            analyze_fusion_performance._robust_stats([8, 8, 8]),
            analyze_fusion_performance._robust_stats([6, 6, 6]),
            evidence_errors=validation["evidence_errors"],
        )
        action = analyze_fusion_performance.ACTION_BY_CLASSIFICATION[
            decision["classification"]
        ]

        self.assertFalse(validation["control_fingerprint_match"])
        self.assertIn("undeclared_config_differences", validation["evidence_errors"])
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(action, "reprofile")

    def test_array_pointers_allow_original_index_zero_and_one_changes(self):
        validation = analyze_fusion_performance.validate_fingerprints(
            {"items": ["baseline-0", "baseline-1", {"frozen": "same"}]},
            {"items": ["candidate-0", "candidate-1", {"frozen": "same"}]},
            _complete_workload_manifest(),
            _complete_workload_manifest(),
            {
                "allowed_json_pointers": ["/items/0", "/items/1"],
                "only_change_zh": "只允许前两个数组项变化",
            },
        )

        self.assertTrue(validation["control_fingerprint_match"])
        self.assertEqual(validation["evidence_errors"], [])
        self.assertIsNotNone(validation["control_fingerprint"])

    def test_array_pointer_batch_supports_nested_arrays_and_rfc6901_escapes(self):
        source = {"outer": {"items/~": [["remove-0", "remove-1", "keep"]]}}

        normalized = analyze_fusion_performance._remove_json_pointers(
            source,
            [
                "/outer/items~1~0/0/0",
                "/outer/items~1~0/0/1",
            ],
        )

        self.assertEqual(normalized, {"outer": {"items/~": [["keep"]]}})
        self.assertEqual(
            source, {"outer": {"items/~": [["remove-0", "remove-1", "keep"]]}}
        )

    def test_array_pointer_indices_follow_rfc6901_array_index_syntax(self):
        source = {"items": list(range(11))}

        normalized = analyze_fusion_performance._remove_json_pointers(
            source, ["/items/0", "/items/10"]
        )

        self.assertEqual(normalized, {"items": list(range(1, 10))})
        for pointer in ("/items/01", "/items/\u0661", "/items/-"):
            with self.subTest(pointer=pointer):
                with self.assertRaisesRegex(ValueError, "invalid RFC6901 array index"):
                    analyze_fusion_performance._remove_json_pointers(source, [pointer])

    def test_declared_pointers_are_prevalidated_against_original_document(self):
        source = {"items": ["zero", "one"]}

        for pointers in (
            ["/items", "/items/01"],
            ["/items/01", "/items"],
        ):
            with self.subTest(pointers=pointers):
                with self.assertRaisesRegex(ValueError, "invalid RFC6901 array index"):
                    analyze_fusion_performance._remove_json_pointers(source, pointers)

        with self.assertRaisesRegex(ValueError, "root JSON pointer"):
            analyze_fusion_performance._remove_json_pointers(source, [""])

        self.assertEqual(
            analyze_fusion_performance._remove_json_pointers(
                {"items": {"01": "remove", "keep": True}}, ["/items/01"]
            ),
            {"items": {"keep": True}},
        )

    def test_declared_config_changes_preserve_control_fingerprint(self):
        baseline_config = {
            "runtime": {"rank_size": 8},
            "superkernel": {"enabled": False, "scope": []},
        }
        candidate_config = {
            "superkernel": {"scope": ["decoder.layer.0"], "enabled": True},
            "runtime": {"rank_size": 8},
        }
        baseline_workload = _complete_workload_manifest()
        candidate_workload = _complete_workload_manifest()
        declared_change = {
            "allowed_json_pointers": [
                "/superkernel/enabled",
                "/superkernel/scope",
            ],
            "only_change_zh": "只启用 SuperKernel 并应用本轮 scope",
        }

        validation = analyze_fusion_performance.validate_fingerprints(
            baseline_config,
            candidate_config,
            baseline_workload,
            candidate_workload,
            declared_change,
        )

        self.assertEqual(validation["evidence_errors"], [])
        self.assertNotEqual(
            validation["baseline_config_fingerprint"],
            validation["candidate_config_fingerprint"],
        )
        self.assertEqual(
            validation["baseline_workload_fingerprint"],
            validation["candidate_workload_fingerprint"],
        )
        self.assertEqual(
            validation["workload_fingerprint"],
            validation["baseline_workload_fingerprint"],
        )
        self.assertIsNotNone(validation["control_fingerprint"])

    def test_workload_manifest_accepts_complete_alias_manifest(self):
        workload = {
            "model_name": "demo",
            "sequence_length": 128,
            "batch_size": 1,
            "world_size": 8,
            "prefill_decode_mode": "prefill_decode",
            "warmup_iterations": 0,
            "measurement_runs": 10,
            "runtime_options": {"dtype": "float16"},
        }

        validation = analyze_fusion_performance.validate_fingerprints(
            {"frozen": True},
            {"frozen": True},
            workload,
            workload,
            {"allowed_json_pointers": [], "only_change_zh": "无变更"},
        )

        self.assertEqual(validation["evidence_errors"], [])

    def test_workload_manifest_validates_every_scalar_alias_and_conflict(self):
        cases = {
            "invalid-secondary-alias": (
                {**_complete_workload_manifest(), "batch_size": "invalid"},
                "workload_manifest_invalid_batch",
            ),
            "conflicting-aliases": (
                {**_complete_workload_manifest(), "batch_size": 2},
                "workload_manifest_conflicting_batch",
            ),
            "model-id-is-not-name": (
                {
                    **{
                        key: value
                        for key, value in _complete_workload_manifest().items()
                        if key != "model"
                    },
                    "model_id": "demo",
                },
                "workload_manifest_missing_model",
            ),
        }

        for case, (workload, expected_error) in cases.items():
            with self.subTest(case=case):
                validation = analyze_fusion_performance.validate_fingerprints(
                    {"frozen": True},
                    {"frozen": True},
                    workload,
                    workload,
                    {"allowed_json_pointers": [], "only_change_zh": "无变更"},
                )

                self.assertIn(
                    "workload_manifest_incomplete", validation["evidence_errors"]
                )
                self.assertIn(expected_error, validation["evidence_errors"])

    def test_incomplete_or_invalid_workload_blocks_all_sk_decisions(self):
        cases = {
            "empty": {},
            "missing-iterations": {
                key: value
                for key, value in _complete_workload_manifest().items()
                if key != "iterations"
            },
            "invalid-count": {
                **_complete_workload_manifest(),
                "batch": 0,
                "warmup": -1,
                "iterations": 0,
            },
        }

        for case, workload in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                artifacts["baseline_workload"].write_text(json.dumps(workload))
                artifacts["candidate_workload"].write_text(json.dumps(workload))
                json_out = root / f"workload-{case}.json"

                with redirect_stdout(io.StringIO()):
                    exit_code = analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )
                report = json.loads(json_out.read_text())
                decision = report["per_sk_decisions"][0]

                self.assertEqual(exit_code, 0)
                self.assertIsNotNone(report["baseline_workload_fingerprint"])
                self.assertIsNotNone(report["candidate_workload_fingerprint"])
                self.assertIn("workload_manifest_incomplete", report["blockers"])
                self.assertTrue(
                    any(
                        error.startswith("workload_manifest_")
                        and error != "workload_manifest_incomplete"
                        for error in report["blockers"]
                    )
                )
                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertEqual(decision["action"], "reprofile")

    def test_undeclared_config_difference_blocks_scope_action(self):
        validation = analyze_fusion_performance.validate_fingerprints(
            {"runtime": {"rank_size": 8}, "superkernel": {"enabled": False}},
            {"runtime": {"rank_size": 4}, "superkernel": {"enabled": True}},
            _complete_workload_manifest(),
            _complete_workload_manifest(),
            {
                "allowed_json_pointers": ["/superkernel/enabled"],
                "only_change_zh": "只启用 SuperKernel",
            },
        )

        decision = analyze_fusion_performance.classify_performance(
            analyze_fusion_performance._robust_stats([8, 8, 8]),
            analyze_fusion_performance._robust_stats([6, 6, 6]),
            evidence_errors=validation["evidence_errors"],
        )
        action = analyze_fusion_performance.ACTION_BY_CLASSIFICATION[
            decision["classification"]
        ]

        self.assertIn("undeclared_config_differences", validation["evidence_errors"])
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertNotIn(action, {"keep", "prune"})

    def test_occurrence_count_mismatch_is_not_truncated(self):
        rows = [
            _row(10, 1, "A", "A", "AI_VECTOR_CORE", start, 1) for start in (0, 10, 20)
        ] + [_row(11, 1, "B", "B", "AI_VECTOR_CORE", start, 1) for start in (1, 11)]

        summaries, errors = analyze_fusion_performance._occurrence_summaries(
            rows, node_ids=[10, 11]
        )

        self.assertEqual(summaries, [])
        self.assertEqual(errors, ["baseline_child_occurrence_count_mismatch"])

    def test_occurrences_reject_disjoint_step_sets(self):
        rows = []
        for node_id, steps in ((10, (1, 2, 3)), (11, (4, 5, 6))):
            for index, step_id in enumerate(steps):
                row = _row(node_id, 1, str(node_id), "A", "AI_VECTOR_CORE", index, 1)
                row["step_id"] = step_id
                rows.append(row)

        summaries, errors = analyze_fusion_performance._occurrence_summaries(
            rows, node_ids=[10, 11]
        )

        self.assertEqual(summaries, [])
        self.assertIn("baseline_child_occurrence_key_mismatch", errors)

    def test_occurrences_reject_missing_step(self):
        rows = []
        for node_id, steps in ((10, (1, 2, 3)), (11, (1, 2))):
            for step_id in steps:
                row = _row(node_id, 1, str(node_id), "A", "AI_VECTOR_CORE", step_id, 1)
                row["step_id"] = step_id
                rows.append(row)

        summaries, errors = analyze_fusion_performance._occurrence_summaries(
            rows, node_ids=[10, 11]
        )

        self.assertEqual(summaries, [])
        self.assertTrue(
            {
                "baseline_child_occurrence_count_mismatch",
                "baseline_child_occurrence_key_mismatch",
            }
            & set(errors)
        )

    def test_occurrences_reject_duplicate_step_for_one_child(self):
        rows = []
        for node_id, steps in ((10, (1, 1, 2)), (11, (1, 2, 3))):
            for index, step_id in enumerate(steps):
                row = _row(node_id, 1, str(node_id), "A", "AI_VECTOR_CORE", index, 1)
                row["step_id"] = step_id
                rows.append(row)

        summaries, errors = analyze_fusion_performance._occurrence_summaries(
            rows, node_ids=[10, 11]
        )

        self.assertEqual(summaries, [])
        self.assertIn("baseline_child_occurrence_key_duplicate", errors)

    def test_occurrences_align_out_of_order_rows_by_step(self):
        rows = []
        for node_id, values in (
            (10, ((1, 0), (2, 100), (3, 200))),
            (11, ((3, 1), (1, 101), (2, 201))),
        ):
            for step_id, start in values:
                row = _row(node_id, 1, str(node_id), "A", "AI_VECTOR_CORE", start, 1)
                row["step_id"] = step_id
                rows.append(row)

        summaries, errors = analyze_fusion_performance._occurrence_summaries(
            list(reversed(rows)), node_ids=[10, 11]
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            [summary["interval_us"] for summary in summaries], [102, 102, 200]
        )

    def test_occurrences_reject_multi_device_mixing(self):
        rows = []
        for node_id, device_id in ((10, 0), (11, 1)):
            for step_id in (1, 2, 3):
                row = _row(node_id, 1, str(node_id), "A", "AI_VECTOR_CORE", step_id, 1)
                row["step_id"] = step_id
                row["device_id"] = device_id
                rows.append(row)

        summaries, errors = analyze_fusion_performance._occurrence_summaries(
            rows, node_ids=[10, 11]
        )

        self.assertEqual(summaries, [])
        self.assertIn("baseline_child_occurrence_domain_mismatch", errors)

    def test_occurrences_allow_explicit_discriminator_when_step_is_missing(self):
        rows = []
        for node_id, values in (
            (10, (("run-1", 0), ("run-2", 100), ("run-3", 200))),
            (11, (("run-3", 1), ("run-1", 101), ("run-2", 201))),
        ):
            for occurrence_id, start in values:
                row = _row(node_id, 1, str(node_id), "A", "AI_VECTOR_CORE", start, 1)
                row["step_id"] = ""
                row["occurrence_id"] = occurrence_id
                rows.append(row)

        summaries, errors = analyze_fusion_performance._occurrence_summaries(
            rows, node_ids=[10, 11]
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            [summary["interval_us"] for summary in summaries], [102, 102, 200]
        )

    def test_csv_occurrences_preserve_explicit_discriminator_when_step_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "kernel_details.csv"
            rows = []
            for node_id in (10, 11):
                for occurrence_id, start in (
                    ("run-1", 0),
                    ("run-2", 100),
                    ("run-3", 200),
                ):
                    row = _row(
                        node_id,
                        1,
                        str(node_id),
                        "A",
                        "AI_VECTOR_CORE",
                        start + (node_id - 10),
                        1,
                    )
                    row["step_id"] = ""
                    row["occurrence_id"] = occurrence_id
                    rows.append(row)
            _write_csv(profile, rows)

            _, parsed_rows = analyze_fusion_performance.load_kernel_rows(profile)
            summaries, errors = analyze_fusion_performance._occurrence_summaries(
                parsed_rows, node_ids=[10, 11]
            )

        self.assertEqual(errors, [])
        self.assertEqual([summary["interval_us"] for summary in summaries], [2, 2, 2])

    def test_occurrences_reject_missing_step_without_explicit_discriminator(self):
        rows = [_row(10, 1, "A", "A", "AI_VECTOR_CORE", 0, 1)]
        rows[0]["step_id"] = ""

        summaries, errors = analyze_fusion_performance._occurrence_summaries(
            rows, node_ids=[10]
        )

        self.assertEqual(summaries, [])
        self.assertIn("baseline_child_occurrence_discriminator_missing", errors)

    def test_hard_occurrence_minimum_cannot_be_lowered(self):
        for requested_min, count in ((1, 1), (2, 2)):
            with self.subTest(requested_min=requested_min):
                decision = analyze_fusion_performance.classify_performance(
                    analyze_fusion_performance._robust_stats([8] * count),
                    analyze_fusion_performance._robust_stats([6] * count),
                    min_occurrences=requested_min,
                )
                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertIn("insufficient_occurrences", decision["evidence_errors"])

    def test_cli_low_min_occurrences_remains_diagnostic_only(self):
        for requested_min in (1, 2):
            with (
                self.subTest(requested_min=requested_min),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                _, baseline_rows = analyze_fusion_performance.load_kernel_rows(
                    artifacts["baseline_profile"]
                )
                _, candidate_rows = analyze_fusion_performance.load_kernel_rows(
                    artifacts["candidate_profile"]
                )
                _write_csv(artifacts["baseline_profile"], baseline_rows[:requested_min])
                _write_csv(
                    artifacts["candidate_profile"], candidate_rows[:requested_min]
                )
                json_out = root / "result.json"
                args = _analysis_cli_args(artifacts, json_out)
                args[args.index("--min-occurrences") + 1] = str(requested_min)

                with redirect_stdout(io.StringIO()):
                    analyze_fusion_performance.main(args)
                report = json.loads(json_out.read_text())
                decision = report["per_sk_decisions"][0]

                self.assertEqual(report["thresholds"]["min_occurrences"], 3)
                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertEqual(decision["action"], "reprofile")

    def test_stable_identity_ids_ignore_object_key_order(self):
        identity = {
            "model_id": "48",
            "source_scope": "decoder.layer.0.mlp",
            "ordered_child_op_sequence": ["A", "B"],
            "boundary": {"start_op": "A", "end_op": "B"},
        }
        reordered = {
            "boundary": {"end_op": "B", "start_op": "A"},
            "ordered_child_op_sequence": ["A", "B"],
            "source_scope": "decoder.layer.0.mlp",
            "model_id": "48",
        }

        self.assertEqual(
            analyze_fusion_performance._stable_identity_ids(identity),
            analyze_fusion_performance._stable_identity_ids(reordered),
        )

    def test_candidate_row_order_does_not_change_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            reordered_candidate = root / "candidate-reordered.csv"
            sk_name = (
                "sk_1_decoder.layer.0_start_static_kernel_A_hash_"
                "end_static_kernel_A_hash"
            )
            _write_csv(
                reordered_candidate,
                [
                    _row(99, 1, sk_name, "SuperKernel", "AI_VECTOR_CORE", start, 6)
                    for start in (40, 0, 20)
                ],
            )
            _, baseline_rows = analyze_fusion_performance.summarize_profile(
                artifacts["baseline_profile"]
            )

            original = analyze_fusion_performance.compare_candidate(
                baseline_rows,
                "S1",
                artifacts["candidate_profile"],
                sk_meta=artifacts["sk_meta"],
            )["superkernel_comparisons"][0]
            reordered = analyze_fusion_performance.compare_candidate(
                baseline_rows,
                "S1",
                reordered_candidate,
                sk_meta=artifacts["sk_meta"],
            )["superkernel_comparisons"][0]

        self.assertEqual(original["sk_id"], reordered["sk_id"])
        self.assertEqual(original["range_id"], reordered["range_id"])

    def test_stable_identity_ids_change_with_each_identity_field(self):
        identity = {
            "model_id": "48",
            "source_scope": "decoder.layer.0.mlp",
            "ordered_child_op_sequence": ["A", "B"],
            "boundary": {"start_op": "A", "end_op": "B"},
        }
        original_ids = analyze_fusion_performance._stable_identity_ids(identity)
        variants = [
            {**identity, "model_id": "49"},
            {**identity, "source_scope": "decoder.layer.1.mlp"},
            {**identity, "ordered_child_op_sequence": ["B", "A"]},
            {
                **identity,
                "boundary": {"start_op": "A", "end_op": "C"},
            },
        ]

        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    analyze_fusion_performance._stable_identity_ids(variant),
                    original_ids,
                )

    def test_duplicate_diagnostic_inventory_gets_explicit_report_local_ids(self):
        identity = {
            "model_id": "48",
            "source_scope": "decoder.layer.0",
            "ordered_child_op_sequence": ["A"],
            "boundary": {"start_op": "A", "end_op": "A"},
        }
        sk_id, range_id = analyze_fusion_performance._stable_identity_ids(identity)
        comparisons = [
            {
                "identity": identity,
                "sk_id": sk_id,
                "range_id": range_id,
                "device_id": 0,
                "model_id": "48",
                "raw_sk_id": raw_sk_id,
                "graph_occurrence_fingerprint": None,
                "action": "reprofile",
            }
            for raw_sk_id in (7, 8)
        ]

        analyze_fusion_performance._disambiguate_report_identity_ids(comparisons)
        first_ids = [(item["sk_id"], item["range_id"]) for item in comparisons]
        reversed_comparisons = list(reversed(json.loads(json.dumps(comparisons))))
        for item in reversed_comparisons:
            item["sk_id"], item["range_id"] = sk_id, range_id
            item.pop("identity_disambiguation")
        analyze_fusion_performance._disambiguate_report_identity_ids(
            reversed_comparisons
        )

        self.assertEqual(len(set(first_ids)), 2)
        self.assertEqual(
            {item["identity_disambiguation"]["kind"] for item in comparisons},
            {"native_inventory_diagnostic"},
        )
        self.assertEqual(
            set(first_ids),
            {(item["sk_id"], item["range_id"]) for item in reversed_comparisons},
        )

    def test_duplicate_actionable_inventory_requires_unique_structural_occurrence(self):
        identity = {
            "model_id": "48",
            "source_scope": "decoder.layer.0",
            "ordered_child_op_sequence": ["A"],
            "boundary": {"start_op": "A", "end_op": "A"},
        }
        sk_id, range_id = analyze_fusion_performance._stable_identity_ids(identity)
        comparisons = [
            {
                "identity": identity,
                "sk_id": sk_id,
                "range_id": range_id,
                "device_id": 0,
                "model_id": "48",
                "raw_sk_id": raw_sk_id,
                "graph_occurrence_fingerprint": None,
                "action": "keep",
            }
            for raw_sk_id in (7, 8)
        ]

        with self.assertRaisesRegex(ValueError, "unique graph occurrence"):
            analyze_fusion_performance._disambiguate_report_identity_ids(comparisons)

        comparisons[0]["graph_occurrence_fingerprint"] = "1" * 64
        comparisons[1]["graph_occurrence_fingerprint"] = "2" * 64
        analyze_fusion_performance._disambiguate_report_identity_ids(comparisons)
        self.assertEqual(len({item["sk_id"] for item in comparisons}), 2)
        self.assertEqual(
            {item["identity_disambiguation"]["kind"] for item in comparisons},
            {"graph_occurrence"},
        )

    def test_diagnostic_fallback_never_produces_keep_or_prune(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.csv"
            sk_name = "sk_1_scope_start_A_end_B"
            _write_csv(
                candidate,
                [
                    _row(10, 1, sk_name, "SuperKernel", "AI_VECTOR_CORE", start, 6)
                    for start in (0, 20, 40)
                ],
            )
            baseline_rows = [
                _row(10, 1, "A", "A", "AI_VECTOR_CORE", start, 8)
                for start in (0, 20, 40)
            ]

            result = analyze_fusion_performance.compare_candidate(
                baseline_rows,
                "S1",
                candidate,
            )

        decision = result["superkernel_comparisons"][0]
        self.assertEqual(decision["mapping_confidence"], "diagnostic_only")
        self.assertTrue(decision["mapping_hints"])
        self.assertTrue(
            all(
                hint["confidence"] == "diagnostic_only"
                for hint in decision["mapping_hints"]
            )
        )
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertIn(decision["action"], {"reprofile", "block"})
        self.assertNotIn(decision["action"], {"keep", "prune"})

    def test_incomplete_candidate_identity_blocks_automatic_action(self):
        cases = ("model-id", "scope-boundary", "ordered-sequence")

        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                if case == "model-id":
                    _, candidate_rows = analyze_fusion_performance.load_kernel_rows(
                        artifacts["candidate_profile"]
                    )
                    for row in candidate_rows:
                        row["model_id"] = ""
                    _write_csv(artifacts["candidate_profile"], candidate_rows)
                elif case == "scope-boundary":
                    _write_csv(
                        artifacts["candidate_profile"],
                        [
                            _row(
                                99,
                                1,
                                "SuperKernelWithoutBoundary",
                                "SuperKernel",
                                "AI_VECTOR_CORE",
                                start,
                                6,
                            )
                            for start in (0, 20, 40)
                        ],
                    )
                else:
                    sk_name = (
                        "sk_1_decoder.layer.0_start_static_kernel_A_hash_"
                        "end_static_kernel_A_hash"
                    )
                    _metadata_log_path(artifacts["sk_meta"]).write_text(
                        f"SK Function: {sk_name}, scope id: 1, Node Count: 0\n"
                    )
                json_out = root / f"incomplete-{case}.json"

                with redirect_stdout(io.StringIO()):
                    exit_code = analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )
                decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

                self.assertEqual(exit_code, 0)
                self.assertEqual(decision["mapping_confidence"], "diagnostic_only")
                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertIn(decision["action"], {"reprofile", "block"})
                self.assertNotIn(decision["action"], {"keep", "prune"})
                self.assertIn(
                    "candidate_identity_incomplete", decision["evidence_errors"]
                )

    def test_ambiguous_metadata_identity_blocks_each_distinct_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            sk_name = (
                "sk_1_decoder.layer.0_start_static_kernel_A_hash_"
                "end_static_kernel_A_hash"
            )
            metadata_log = _metadata_log_path(artifacts["sk_meta"])
            metadata_log.write_text(
                metadata_log.read_text()
                + f"SK Function: {sk_name}, scope id: 2, Node Count: 1\n"
                "[nodeId:11, streamId:1] - "
                "KernelInfos{funcName:static_kernel_B_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
                "isScheModeOn:0}\n"
            )
            json_out = root / "ambiguous-distinct.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decisions = json.loads(json_out.read_text())["per_sk_decisions"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {
                tuple(decision["identity"]["ordered_child_op_sequence"])
                for decision in decisions
            },
            {("A",), ("B",)},
        )
        for decision in decisions:
            self.assertIn(
                "candidate_metadata_identity_ambiguous",
                decision["evidence_errors"],
            )
            self.assertEqual(decision["mapping_confidence"], "diagnostic_only")
            self.assertEqual(decision["classification"], "insufficient_evidence")
            self.assertNotIn(decision["action"], {"keep", "prune"})

    def test_duplicate_metadata_identity_is_reported_without_replay_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            metadata_log = _metadata_log_path(artifacts["sk_meta"])
            duplicate = (
                metadata_log.read_text()
                .replace("scope id: 1", "scope id: 2")
                .replace("nodeId:10", "nodeId:11")
            )
            metadata_log.write_text(metadata_log.read_text() + duplicate)
            json_out = root / "ambiguous-duplicate.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decisions = json.loads(json_out.read_text())["per_sk_decisions"]

        self.assertEqual(exit_code, 0)
        self.assertTrue(decisions)
        self.assertTrue(
            all(
                decision["classification"] == "insufficient_evidence"
                for decision in decisions
            )
        )

    @unittest.skip(
        "retired task-range exact protocol; covered by source_scope_map_v2 tests"
    )
    def test_source_scope_map_uniquely_recovers_exact_child_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _replace_metadata_node_id(artifacts)
            json_out = root / "source-scope-exact.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["mapping_method"], "source_scope_map")
        self.assertEqual(decision["mapping_confidence"], "exact")
        self.assertEqual(decision["classification"], "beneficial")
        self.assertEqual(decision["action"], "keep")
        self.assertEqual(decision["original"]["occurrence_count"], 3)

    @unittest.skip(
        "retired task-range exact protocol; covered by source_scope_map_v2 tests"
    )
    def test_source_scope_map_allows_different_cross_arm_occurrence_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _replace_metadata_node_id(artifacts)
            _write_csv(
                artifacts["candidate_profile"],
                [
                    _row(
                        99,
                        1,
                        _single_a_sk_name(),
                        "SuperKernel",
                        "AI_VECTOR_CORE",
                        start,
                        6,
                    )
                    for start in (0, 20, 40, 60)
                ],
            )
            json_out = root / "source-scope-cross-arm-counts.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["mapping_method"], "source_scope_map")
        self.assertEqual(decision["mapping_confidence"], "exact")
        self.assertEqual(decision["original"]["occurrence_count"], 3)
        self.assertEqual(decision["candidate_occurrence_count"], 4)
        self.assertEqual(decision["classification"], "beneficial")
        self.assertEqual(decision["action"], "keep")

    @unittest.skip("retired local-window exact protocol")
    def test_source_scope_map_selects_one_contiguous_repeated_op_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            baseline_rows = []
            for start in (0, 30, 60):
                baseline_rows.extend(
                    [
                        _row(
                            10,
                            1,
                            "static_kernel_A_first_hash",
                            "A",
                            "AI_VECTOR_CORE",
                            start,
                            8,
                        ),
                        _row(
                            11,
                            1,
                            "static_kernel_A_second_hash",
                            "A",
                            "AI_VECTOR_CORE",
                            start + 8,
                            8,
                        ),
                        _row(
                            12,
                            1,
                            "static_kernel_B_hash",
                            "B",
                            "AI_VECTOR_CORE",
                            start + 16,
                            8,
                        ),
                    ]
                )
            _write_csv(artifacts["baseline_profile"], baseline_rows)
            metadata_log = _metadata_log_path(artifacts["sk_meta"])
            metadata_log.write_text(
                f"SK Function: {_single_a_sk_name()}, scope id: 1, Node Count: 2\n"
                "[nodeId:999, streamId:1] - "
                "KernelInfos{funcName:static_kernel_A_first_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:998, streamId:1] - "
                "KernelInfos{funcName:static_kernel_A_second_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
            artifacts["source_scope_map"].write_text(
                json.dumps(
                    {
                        "task_ranges": [
                            {
                                "layer": 0,
                                "model_id": "48",
                                "start_task_id": 10,
                                "end_task_id": 12,
                                "source_scope": "decoder.layer.0",
                                "boundary": {"start_op": "A", "end_op": "A"},
                                "ordered_child_op_sequence": ["A", "A"],
                            }
                        ]
                    }
                )
            )
            json_out = root / "source-scope-repeated-op.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["mapping_method"], "source_scope_map")
        self.assertEqual(decision["mapping_confidence"], "exact")
        self.assertEqual(decision["child_count"], 2)
        self.assertEqual(decision["classification"], "beneficial")
        self.assertEqual(decision["action"], "keep")

    @unittest.skip(
        "retired task-range exact protocol; covered by source_scope_map_v2 tests"
    )
    def test_source_scope_map_preserves_and_validates_explicit_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_map = root / "source-map.json"
            source_map.write_text(
                json.dumps(
                    {
                        "task_ranges": [
                            {
                                "layer": 0,
                                "model_id": "48",
                                "start_task_id": 10,
                                "end_task_id": 10,
                                "scope": "decoder.layer.0",
                                "boundary": {"start_op": "A", "end_op": "A"},
                                "ordered_child_op_sequence": ["A"],
                            }
                        ]
                    }
                )
            )

            loaded = analyze_fusion_performance._load_layer_map(source_map)

            self.assertEqual(loaded[0]["source_scope"], "decoder.layer.0")
            self.assertEqual(loaded[0]["boundary"], {"start_op": "A", "end_op": "A"})
            self.assertEqual(loaded[0]["ordered_child_op_sequence"], ["A"])

            invalid_values = (
                {"source_scope": 0},
                {"boundary": "A-A"},
                {"ordered_child_op_sequence": "A"},
            )
            for invalid in invalid_values:
                with self.subTest(invalid=invalid):
                    source_map.write_text(
                        json.dumps(
                            {
                                "task_ranges": [
                                    {
                                        "layer": 0,
                                        "model_id": "48",
                                        "start_task_id": 10,
                                        "end_task_id": 10,
                                        **invalid,
                                    }
                                ]
                            }
                        )
                    )
                    with self.assertRaisesRegex(ValueError, "invalid layer range"):
                        analyze_fusion_performance._load_layer_map(source_map)

    @unittest.skip(
        "retired task-range exact protocol; covered by source_scope_map_v2 tests"
    )
    def test_source_scope_map_explicit_identity_mismatch_or_duplicate_blocks(self):
        for case in (
            "scope-mismatch",
            "boundary-mismatch",
            "sequence-mismatch",
            "duplicate",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                _replace_metadata_node_id(artifacts)
                entry = {
                    "layer": 0,
                    "model_id": "48",
                    "start_task_id": 10,
                    "end_task_id": 10,
                    "source_scope": "decoder.layer.0",
                    "boundary": {"start_op": "A", "end_op": "A"},
                    "ordered_child_op_sequence": ["A"],
                }
                if case == "scope-mismatch":
                    entry["source_scope"] = "decoder.layer.1"
                elif case == "boundary-mismatch":
                    entry["boundary"] = {"start_op": "B", "end_op": "B"}
                elif case == "sequence-mismatch":
                    entry["ordered_child_op_sequence"] = ["B"]
                entries = [entry, dict(entry)] if case == "duplicate" else [entry]
                artifacts["source_scope_map"].write_text(
                    json.dumps({"task_ranges": entries})
                )
                json_out = root / f"explicit-{case}.json"

                with redirect_stdout(io.StringIO()):
                    exit_code = analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )
                decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

                self.assertEqual(exit_code, 0)
                expected_error = (
                    "source_scope_mapping_ambiguous"
                    if case == "duplicate"
                    else "source_scope_map_identity_mismatch"
                )
                self.assertIn(expected_error, decision["evidence_errors"])
                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertEqual(decision["action"], "reprofile")

    @unittest.skip(
        "retired task-range exact protocol; covered by source_scope_map_v2 tests"
    )
    def test_source_scope_map_explicit_identity_match_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _replace_metadata_node_id(artifacts)
            artifacts["source_scope_map"].write_text(
                json.dumps(
                    {
                        "task_ranges": [
                            {
                                "layer": 0,
                                "model_id": "48",
                                "start_task_id": 10,
                                "end_task_id": 10,
                                "source_scope": "decoder.layer.0",
                                "boundary": {"start_op": "A", "end_op": "A"},
                                "ordered_child_op_sequence": ["A"],
                            }
                        ]
                    }
                )
            )
            json_out = root / "explicit-match.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["mapping_method"], "source_scope_map")
        self.assertEqual(decision["mapping_confidence"], "exact")
        self.assertEqual(decision["classification"], "beneficial")
        self.assertEqual(decision["action"], "keep")

    @unittest.skip(
        "retired task-range exact protocol; generic opaque blocks use source_scope_map_v2"
    )
    def test_explicit_source_scope_map_does_not_require_parseable_layer_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            scope = "transformer.h.0"
            sk_name = _single_a_sk_name(scope)
            _write_csv(
                artifacts["candidate_profile"],
                [
                    _row(99, 1, sk_name, "SuperKernel", "AI_VECTOR_CORE", start, 6)
                    for start in (0, 20, 40)
                ],
            )
            metadata_log = _metadata_log_path(artifacts["sk_meta"])
            metadata_log.write_text(
                f"SK Function: {sk_name}, scope id: 1, Node Count: 1\n"
                "[nodeId:999, streamId:1] - "
                "KernelInfos{funcName:static_kernel_A_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
            candidate_config = json.loads(artifacts["candidate_config"].read_text())
            candidate_config["superkernel"]["scope"] = [scope]
            artifacts["candidate_config"].write_text(json.dumps(candidate_config))
            artifacts["source_scope_map"].write_text(
                json.dumps(
                    {
                        "task_ranges": [
                            {
                                "layer": 0,
                                "model_id": "48",
                                "start_task_id": 10,
                                "end_task_id": 10,
                                "source_scope": scope,
                                "boundary": {"start_op": "A", "end_op": "A"},
                                "ordered_child_op_sequence": ["A"],
                            }
                        ]
                    }
                )
            )
            json_out = root / "explicit-unparseable-scope.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["mapping_method"], "source_scope_map")
        self.assertEqual(decision["mapping_confidence"], "exact")
        self.assertEqual(decision["classification"], "beneficial")
        self.assertEqual(decision["action"], "keep")

    def test_sk_meta_node_mapping_rejects_operator_mismatch_or_instability(self):
        for case in ("mismatch", "unstable"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                op_types = ["B", "B", "B"] if case == "mismatch" else ["A", "A", "B"]
                _write_csv(
                    artifacts["baseline_profile"],
                    [
                        _row(
                            10,
                            1,
                            f"static_kernel_{op_type}_changed_hash",
                            op_type,
                            "AI_VECTOR_CORE",
                            start,
                            8,
                        )
                        for op_type, start in zip(op_types, (0, 20, 40))
                    ],
                )
                json_out = root / f"operator-{case}.json"

                with redirect_stdout(io.StringIO()):
                    exit_code = analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )
                decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

                self.assertEqual(exit_code, 0)
                self.assertIn(
                    "baseline_child_operator_mismatch",
                    decision["evidence_errors"],
                )
                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertEqual(decision["action"], "reprofile")
                self.assertEqual(decision["mapping_confidence"], "diagnostic_only")

    def test_sk_meta_node_mapping_rejects_conflicting_operator_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _write_csv(
                artifacts["baseline_profile"],
                [
                    _row(
                        10,
                        1,
                        "static_kernel_A_changed_hash",
                        "B",
                        "AI_VECTOR_CORE",
                        start,
                        8,
                    )
                    for start in (0, 20, 40)
                ],
            )
            json_out = root / "operator-source-conflict.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertIn("baseline_child_operator_mismatch", decision["evidence_errors"])
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(decision["action"], "reprofile")

    @unittest.skip(
        "retired task-range exact protocol; covered by source_scope_map_v2 tests"
    )
    def test_exact_source_scope_map_precedes_conflicting_raw_node_id_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _write_csv(
                artifacts["baseline_profile"],
                [
                    row
                    for start in (0, 20, 40)
                    for row in (
                        _row(
                            10,
                            1,
                            "static_kernel_B_hash",
                            "B",
                            "AI_VECTOR_CORE",
                            start,
                            8,
                        ),
                        _row(
                            11,
                            1,
                            "static_kernel_A_hash",
                            "A",
                            "AI_VECTOR_CORE",
                            start + 9,
                            8,
                        ),
                    )
                ],
            )
            artifacts["source_scope_map"].write_text(
                json.dumps(
                    {
                        "task_ranges": [
                            _source_scope_map_entry(
                                start_task_id=11,
                                end_task_id=11,
                                source_scope="decoder.layer.0",
                                boundary={"start_op": "A", "end_op": "A"},
                                ordered_child_op_sequence=["A"],
                            )
                        ]
                    }
                )
            )
            json_out = root / "source-map-precedes-raw-id.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["mapping_method"], "source_scope_map")
        self.assertEqual(decision["mapping_confidence"], "exact")
        self.assertEqual(decision["classification"], "beneficial")
        self.assertNotIn(
            "baseline_child_operator_mismatch", decision["evidence_errors"]
        )

    def test_sk_meta_node_mapping_accepts_function_hash_only_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            artifacts["source_scope_map"].write_text(json.dumps({"task_ranges": []}))
            _write_csv(
                artifacts["baseline_profile"],
                [
                    _row(
                        10,
                        1,
                        "static_kernel_A_completely_different_hash",
                        "A",
                        "AI_VECTOR_CORE",
                        start,
                        8,
                    )
                    for start in (0, 20, 40)
                ],
            )
            json_out = root / "operator-hash-change.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            report = json.loads(json_out.read_text())
            decision = report["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["mapping_method"], "sk_meta_node_ids")
        self.assertEqual(decision["mapping_confidence"], "diagnostic_only")
        self.assertIn(
            "cross_compile_runtime_id_not_identity",
            decision["mapping_blockers"],
        )
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(decision["action"], "reprofile")
        self.assertIsNone(decision["original"])
        self.assertEqual(report["coverage"]["mapped_count"], 0)

    @unittest.skip(
        "retired task-range exact protocol; v2 fails closed before comparison"
    )
    def test_source_scope_map_ambiguity_or_missing_child_blocks_action(self):
        for case in ("ambiguous", "missing"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                _replace_metadata_node_id(artifacts)
                if case == "ambiguous":
                    baseline_rows = [
                        _row(
                            task,
                            1,
                            f"static_kernel_A_{task}_hash",
                            "A",
                            "AI_VECTOR_CORE",
                            start,
                            8,
                        )
                        for task in (10, 11)
                        for start in (0, 20, 40)
                    ]
                    source_map = {
                        "task_ranges": [
                            {
                                "layer": 0,
                                "model_id": "48",
                                "start_task_id": 10,
                                "end_task_id": 11,
                                "source_scope": "decoder.layer.0",
                                "boundary": {"start_op": "A", "end_op": "A"},
                                "ordered_child_op_sequence": ["A"],
                            }
                        ]
                    }
                    expected_error = "source_scope_mapping_ambiguous"
                else:
                    baseline_rows = [
                        _row(
                            10,
                            1,
                            "static_kernel_B_hash",
                            "B",
                            "AI_VECTOR_CORE",
                            start,
                            8,
                        )
                        for start in (0, 20, 40)
                    ]
                    source_map = {
                        "task_ranges": [
                            {
                                "layer": 0,
                                "model_id": "48",
                                "start_task_id": 10,
                                "end_task_id": 10,
                                "source_scope": "decoder.layer.0",
                                "boundary": {"start_op": "A", "end_op": "A"},
                                "ordered_child_op_sequence": ["A"],
                            }
                        ]
                    }
                    expected_error = "source_scope_mapping_missing"
                _write_csv(artifacts["baseline_profile"], baseline_rows)
                artifacts["source_scope_map"].write_text(json.dumps(source_map))
                json_out = root / f"source-scope-{case}.json"

                with redirect_stdout(io.StringIO()):
                    exit_code = analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )
                decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

                self.assertEqual(exit_code, 0)
                self.assertEqual(decision["mapping_confidence"], "diagnostic_only")
                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertIn(expected_error, decision["evidence_errors"])
                self.assertNotIn(decision["action"], {"keep", "prune"})

    def test_complete_analysis_cli_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            output_dir = root / "analysis"
            json_out = output_dir / "profiling-analysis-result.json"
            markdown_out = output_dir / "PROFILING_ANALYSIS.md"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out, markdown_out)
                )
            report = json.loads(json_out.read_text())

            self.assertEqual(exit_code, 0)
            self.assertEqual(report["schema_version"], "1.2")
            self.assertEqual(
                {
                    "association_protocol",
                    "candidate_binding_evidence",
                    "canonical_graph_fingerprints",
                    "stream_role_mapping",
                    "graph_alignment_proof",
                    "mapping_coverage",
                }
                - set(report),
                set(),
            )
            self.assertTrue(markdown_out.is_file())
            self.assertEqual(report["experiment_id"], "exp-1")
            self.assertEqual(report["round_id"], "S1-BASE")
            self.assertEqual(report["analysis_agent_id"], "analysis-agent-1")
            self.assertEqual(report["source_revision"], "660e08e")
            self.assertEqual(
                report["baseline_profile_fingerprint"],
                hashlib.sha256(artifacts["baseline_profile"].read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["candidate_profile_fingerprint"],
                hashlib.sha256(artifacts["candidate_profile"].read_bytes()).hexdigest(),
            )
            self.assertEqual(report["thresholds"]["min_occurrences"], 3)
            self.assertEqual(
                report["per_sk_decisions"][0]["classification"], "insufficient_evidence"
            )
            self.assertEqual(report["scope_actions"][0]["action"], "reprofile")
            self.assertEqual(
                report["scope_actions"][0]["source_scope"], "decoder.layer.0"
            )
            self.assertEqual(report["scope_actions"][0]["boundary"]["start_op"], "A")
            self.assertEqual(report["scope_actions"][0]["boundary"]["end_op"], "A")
            self.assertEqual(
                report["scope_actions"][0]["ordered_child_op_sequence"], ["A"]
            )
            self.assertTrue(report["scope_actions"][0]["interval_unproven"])
            self.assertNotIn("source_file", report["scope_actions"][0]["boundary"])
            content_fingerprint = report["analysis_content_fingerprint"]
            unsigned_report = dict(report)
            unsigned_report.pop("analysis_content_fingerprint")
            self.assertEqual(
                content_fingerprint,
                artifact_contract.canonical_sha256(unsigned_report),
            )
            markdown = markdown_out.read_text()
            self.assertIn("## 逐 SK 性能对比", markdown)
            self.assertIn("## 结构关联覆盖率", markdown)
            self.assertIn("## Child Count 分布", markdown)
            self.assertIn("## 逐 SK 结构与性能证据", markdown)
            self.assertIn("## 关联阻塞项", markdown)
            self.assertIn("## Automatic AOT 与源码动作边界", markdown)

    def test_automatic_analysis_cli_writes_schema_1_2_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            json_out = root / "analysis" / "automatic.json"
            markdown_out = root / "analysis" / "automatic.md"
            args = _analysis_cli_args(artifacts, json_out, markdown_out)
            args[args.index("S1-BASE")] = "S1-AUTO"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(args)
            markdown = markdown_out.read_text()

        self.assertEqual(exit_code, 0)
        self.assertIn("## 结构关联覆盖率", markdown)

    def test_collection_manifest_inputs_are_relative_and_affect_analysis_fingerprint(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            artifacts["source_scope_map"].write_text(json.dumps({"task_ranges": []}))
            manifest_fields = (
                "baseline_collection_manifest",
                "profile_collection_manifest",
            )
            for index, field in enumerate(manifest_fields):
                path = root / "collections" / f"{field}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"invalid_fixture": index}))
                artifacts[field] = path
            json_out = root / "analysis" / "structural-inputs.json"

            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    ),
                    0,
                )
            first = json.loads(json_out.read_text())
            artifacts["profile_collection_manifest"].write_text(
                json.dumps({"invalid_fixture": "changed"})
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    ),
                    0,
                )
            second = json.loads(json_out.read_text())

        self.assertEqual(first["schema_version"], "1.2")
        for field in manifest_fields:
            value = first["inputs"][field]
            self.assertIsInstance(value, str)
            self.assertFalse(Path(value).is_absolute())
        self.assertIn(
            "structural_collection_manifest_set_invalid",
            first["association_protocol"]["blockers"],
        )
        self.assertNotEqual(
            first["analysis_content_fingerprint"],
            second["analysis_content_fingerprint"],
        )
        self.assertNotEqual(
            first["association_protocol"]["manifest_content_fingerprints"],
            second["association_protocol"]["manifest_content_fingerprints"],
        )
        decision = first["per_sk_decisions"][0]
        self.assertNotIn(
            (decision["mapping_method"], decision["mapping_confidence"]),
            {
                ("source_scope_map", "exact"),
                ("kernel_projection_structural", "exact_projected_trace"),
            },
        )

    def test_structural_manifest_reader_rejects_symlink_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "manifest-target.json"
            target.write_text(json.dumps({"capture_role": "candidate_compat"}))
            link = root / "manifest-link.json"
            link.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "symlink|regular file"):
                analyze_fusion_performance._manifest_value(link)

    def test_partial_collection_manifest_set_blocks_structural_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            artifacts["source_scope_map"].write_text(json.dumps({"task_ranges": []}))
            manifest = root / "baseline-collection-manifest.json"
            manifest.write_text(json.dumps({"invalid_fixture": True}))
            artifacts["baseline_collection_manifest"] = manifest
            json_out = root / "partial-structural-inputs.json"

            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    ),
                    0,
                )
            report = json.loads(json_out.read_text())

        self.assertIn(
            "structural_collection_manifest_set_incomplete",
            report["association_protocol"]["blockers"],
        )
        self.assertIn(
            "structural_collection_manifest_set_incomplete",
            report["per_sk_decisions"][0]["mapping_blockers"],
        )
        self.assertNotEqual(
            report["per_sk_decisions"][0]["mapping_confidence"],
            "exact_projected_trace",
        )

    def test_analysis_cli_accepts_auto_round_via_shared_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            json_out = root / "auto-round.json"
            args = _analysis_cli_args(artifacts, json_out)
            args[args.index("S1-BASE")] = "S1-AUTO"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(args)

        self.assertEqual(exit_code, 0)
        self.assertTrue(artifact_contract.round_belongs_to_candidate("S1", "S1-AUTO"))
        self.assertFalse(
            hasattr(analyze_fusion_performance, "WORKLOAD_CONCEPT_ALIASES")
        )

    @unittest.skip(
        "retired task-range source interval fixture; v2 bundle is tested separately"
    )
    def test_complete_analysis_cli_preserves_proven_source_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            artifacts["source_scope_map"].write_text(
                json.dumps(
                    {
                        "task_ranges": [
                            _source_scope_map_entry(
                                source_scope="decoder.layer.0",
                                boundary={
                                    "start_op": "A",
                                    "end_op": "A",
                                    "source_file": "model/decoder.py",
                                    "start_offset": 120,
                                    "end_offset": 180,
                                },
                                ordered_child_op_sequence=["A"],
                            )
                        ]
                    }
                )
            )
            json_out = root / "proven-source-interval.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            report = json.loads(json_out.read_text())
            action = report["scope_actions"][0]
            rendered = analyze_fusion_performance.render_report(report)

        self.assertEqual(exit_code, 0)
        self.assertIn("## Scope 保留与裁剪", rendered)
        self.assertFalse(action["interval_unproven"])
        self.assertEqual(action["boundary"]["source_file"], "model/decoder.py")
        self.assertEqual(action["boundary"]["start_offset"], 120)
        self.assertEqual(action["boundary"]["end_offset"], 180)

    def test_complete_analysis_cli_returns_zero_for_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root, candidate_duration=10)
            json_out = root / "analysis" / "result.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            report = json.loads(json_out.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            report["per_sk_decisions"][0]["classification"], "insufficient_evidence"
        )
        self.assertEqual(report["scope_actions"][0]["action"], "reprofile")

    @unittest.skip("legacy CLI fixture lacks required structural projection manifests")
    def test_cube_vector_diagnostic_requires_exact_accepted_parallel_value(self):
        fixtures = (
            (
                {
                    "options": {
                        "optimize_options": {
                            "auto_op_parallel": {"accepted_values": [0, 1]}
                        }
                    }
                },
                True,
            ),
            ({"accepted_options": {"auto_op_parallel": [0]}}, False),
        )
        for environment_evidence, expected_experiment in fixtures:
            with (
                self.subTest(environment_evidence=environment_evidence),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                report = _run_diagnostic_analysis(
                    root, _configure_cube_vector_regression, environment_evidence
                )
                decision = report["per_sk_decisions"][0]
                self.assertEqual(decision["classification"], "regressed")
                self.assertEqual(decision["action"], "block")
                self.assertLess(
                    decision["sk_duration_us"],
                    decision["original"]["duration_sum_us"]["p50"],
                )
                hypotheses = {
                    item["kind"]: item for item in report["diagnostic_hypotheses"]
                }
                self.assertEqual(
                    hypotheses["cube_vector_serialization"]["confidence"],
                    "high",
                )
                self.assertEqual(
                    hypotheses["likely_lost_parallelism"]["confidence"],
                    "medium",
                )
                self.assertTrue(
                    all("root_cause" not in item for item in hypotheses.values())
                )
                experiments = [
                    item
                    for item in report["recommended_experiments"]
                    if item["option"] == "auto_op_parallel"
                ]
                self.assertEqual(bool(experiments), expected_experiment)
                for experiment in experiments:
                    self.assertEqual(experiment["range_id"], decision["range_id"])
                    self.assertEqual(
                        experiment["only_change"],
                        {"option": "auto_op_parallel", "value": 1},
                    )
                    self.assertNotIn("range_ids", experiment)

    @unittest.skip("legacy CLI fixture lacks required structural projection manifests")
    def test_enabled_dcci_with_scalar_signal_proposes_two_narrow_experiments(self):
        regex_value = [".*GroupedMatmul.*"]
        environment_evidence = {
            "runtime_evidence": {"dcci_state": "enabled"},
            "options": {
                "optimize_options": {
                    "dcci_before_kernel_start": {"accepted_values": [regex_value]},
                    "dcci_after_kernel_end": {"accepted_values": [regex_value]},
                    "dcci_disable_on_kernel": {"accepted_values": [regex_value]},
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report = _run_diagnostic_analysis(
                Path(directory), _configure_scalar_regression, environment_evidence
            )

        decision = report["per_sk_decisions"][0]
        dcci_hypotheses = [
            item
            for item in report["diagnostic_hypotheses"]
            if item["kind"] == "dcci_boundary_overhead"
        ]
        self.assertEqual(decision["classification"], "regressed")
        self.assertEqual(decision["action"], "block")
        self.assertEqual(decision["original"]["max_scalar_ratio"], 0.25)
        self.assertEqual(len(dcci_hypotheses), 1)
        self.assertEqual(dcci_hypotheses[0]["confidence"], "medium")
        self.assertNotIn("root_cause", dcci_hypotheses[0])
        dcci_experiments = [
            item
            for item in report["recommended_experiments"]
            if item["option"].startswith("dcci_")
        ]
        self.assertEqual(
            [item["option"] for item in dcci_experiments],
            ["dcci_after_kernel_end", "dcci_before_kernel_start"],
        )
        for experiment in dcci_experiments:
            self.assertEqual(experiment["range_id"], decision["range_id"])
            self.assertEqual(
                experiment["only_change"],
                {"option": experiment["option"], "value": regex_value},
            )
            self.assertIn("runtime_evidence.dcci_state", str(experiment))
        self.assertNotIn(
            "dcci_disable_on_kernel",
            [item["option"] for item in dcci_experiments],
        )

    def test_dcci_narrow_regex_prefers_exact_matching_accepted_value(self):
        exact_value = ["^static_kernel_GroupedMatmul_hash$"]
        accepted_values = [
            [".*GroupedMatmul.*"],
            [".*"],
            ["["],
            ["^static_kernel_Other_hash$"],
            exact_value,
        ]
        environment_evidence = {
            "runtime_evidence": {"dcci_state": "enabled"},
            "accepted_options": {
                "dcci_before_kernel_start": accepted_values,
                "dcci_after_kernel_end": accepted_values,
            },
        }

        value, evidence = analyze_fusion_performance._accepted_narrow_regex(
            environment_evidence,
            "dcci_before_kernel_start",
            ["static_kernel_GroupedMatmul_hash"],
        )

        self.assertEqual(value, exact_value)
        self.assertEqual(evidence["accepted_value"], exact_value)

    def test_dcci_narrow_regex_requires_matching_accepted_value(self):
        environment_evidence = {
            "runtime_evidence": {"dcci_state": "enabled"},
            "accepted_options": {
                "dcci_before_kernel_start": [
                    [".*"],
                    ["^.*$"],
                    [".+"],
                    ["^.+$"],
                    ["["],
                    ["^static_kernel_Other_hash$"],
                ],
                "dcci_after_kernel_end": [],
            },
        }

        for option in ("dcci_before_kernel_start", "dcci_after_kernel_end"):
            value, evidence = analyze_fusion_performance._accepted_narrow_regex(
                environment_evidence,
                option,
                ["static_kernel_GroupedMatmul_hash"],
            )
            self.assertIsNone(value)
            self.assertIsNone(evidence)

    def test_dcci_narrow_regex_accepts_escaped_child_literal(self):
        escaped_value = [r"^static\_kernel\_GroupedMatmul\_hash$"]
        environment_evidence = {
            "runtime_evidence": {"dcci_state": "enabled"},
            "accepted_options": {
                "dcci_before_kernel_start": [escaped_value],
            },
        }

        value, evidence = analyze_fusion_performance._accepted_narrow_regex(
            environment_evidence,
            "dcci_before_kernel_start",
            ["static_kernel_GroupedMatmul_hash"],
        )

        self.assertEqual(value, escaped_value)
        self.assertEqual(evidence["accepted_value"], escaped_value)

    def test_dcci_narrow_regex_rejects_negative_control_matches(self):
        near_global_values = [
            [r"^(?:static_kernel_GroupedMatmul_hash)?.*$"],
            [r"^(?:.*GroupedMatmul.*|.*)$"],
        ]
        environment_evidence = {
            "runtime_evidence": {"dcci_state": "enabled"},
            "accepted_options": {
                "dcci_before_kernel_start": near_global_values,
                "dcci_after_kernel_end": near_global_values,
            },
        }

        for option in ("dcci_before_kernel_start", "dcci_after_kernel_end"):
            value, evidence = analyze_fusion_performance._accepted_narrow_regex(
                environment_evidence,
                option,
                ["static_kernel_GroupedMatmul_hash"],
            )
            self.assertIsNone(value)
            self.assertIsNone(evidence)

    def test_evidence_helpers_reject_malformed_nested_types_with_paths(self):
        malformed_environments = (
            ([], "environment_evidence"),
            ({"options": []}, "environment_evidence.options"),
            (
                {"options": {"optimize_options": []}},
                "environment_evidence.options.optimize_options",
            ),
            (
                {"options": {"optimize_options": {"auto_op_parallel": []}}},
                "environment_evidence.options.optimize_options.auto_op_parallel",
            ),
            (
                {
                    "options": {
                        "optimize_options": {
                            "auto_op_parallel": {"accepted_values": {}}
                        }
                    }
                },
                "environment_evidence.options.optimize_options.auto_op_parallel.accepted_values",
            ),
            ({"accepted_options": []}, "environment_evidence.accepted_options"),
            (
                {"accepted_options": {"auto_op_parallel": {}}},
                "environment_evidence.accepted_options.auto_op_parallel",
            ),
        )
        for environment, path in malformed_environments:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, path.replace(".", r"\.")):
                    analyze_fusion_performance._accepted_values(
                        environment, "auto_op_parallel"
                    )

        with self.assertRaisesRegex(ValueError, r"candidate_config\.runtime"):
            analyze_fusion_performance._dcci_runtime_evidence({}, {"runtime": []})

    def test_analysis_cli_ignores_post_base_option_evidence_shapes(self):
        cases = (
            {"options": []},
            {"options": {"optimize_options": []}},
            {
                "options": {
                    "optimize_options": {"auto_op_parallel": {"accepted_values": {}}}
                }
            },
        )
        for environment in cases:
            with (
                self.subTest(environment=environment),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                artifacts["environment_evidence"].write_text(json.dumps(environment))
                json_out = root / "result.json"

                with redirect_stdout(io.StringIO()):
                    exit_code = analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )

                self.assertEqual(exit_code, 0)
                self.assertEqual(
                    json.loads(json_out.read_text())["recommended_experiments"], []
                )

    def test_analysis_cli_still_rejects_malformed_candidate_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            candidate_config = json.loads(artifacts["candidate_config"].read_text())
            candidate_config["runtime"] = []
            artifacts["candidate_config"].write_text(json.dumps(candidate_config))
            json_out = root / "result.json"

            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as stderr,
            ):
                with self.assertRaises(SystemExit) as raised:
                    analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("candidate_config.runtime", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertFalse(json_out.exists())

    def test_unknown_dcci_state_blocks_dcci_hypotheses_and_experiments(self):
        regex_value = [".*GroupedMatmul.*"]
        environment_evidence = {
            "runtime_evidence": {"dcci_state": "unknown"},
            "accepted_options": {
                "dcci_before_kernel_start": [regex_value],
                "dcci_after_kernel_end": [regex_value],
                "dcci_disable_on_kernel": [[".*"]],
            },
        }
        report = analyze_fusion_performance._build_regression_diagnostics(
            [_scalar_regression_decision()], None, environment_evidence, {}
        )
        self.assertFalse(
            any(
                item["kind"].startswith("dcci")
                for item in report["diagnostic_hypotheses"]
            )
        )
        self.assertFalse(
            any(
                item["option"].startswith("dcci_")
                for item in report["recommended_experiments"]
            )
        )
        dcci_blockers = [item for item in report["blockers"] if "DCCI" in item]
        self.assertEqual(len(dcci_blockers), 1)
        self.assertIn("未知", dcci_blockers[0])
        self.assertIn("dcci_state_unknown", dcci_blockers[0])
        self.assertIn(
            "environment_evidence.runtime_evidence.dcci_state=unknown",
            dcci_blockers[0],
        )

    def test_unknown_dcci_source_does_not_override_consistent_known_source(self):
        environment_evidence = {
            "runtime_evidence": {"dcci_state": "unknown"},
            "accepted_options": {"dcci_disable_on_kernel": [[".*"]]},
        }
        candidate_config = {"runtime_evidence": {"dcci_state": "enabled"}}

        diagnostics = analyze_fusion_performance._build_regression_diagnostics(
            [_scalar_regression_decision()],
            None,
            environment_evidence,
            candidate_config,
        )
        hypothesis = next(
            item
            for item in diagnostics["diagnostic_hypotheses"]
            if item["kind"] == "dcci_boundary_overhead"
        )
        self.assertIn(
            "environment_evidence.runtime_evidence.dcci_state=unknown",
            hypothesis["evidence"],
        )
        self.assertIn(
            "candidate_config.runtime_evidence.dcci_state=enabled",
            hypothesis["evidence"],
        )
        self.assertEqual(diagnostics["recommended_experiments"], [])

    def test_conflicting_known_dcci_sources_block_diagnostics(self):
        environment_evidence = {
            "runtime_evidence": {"dcci_state": "enabled"},
            "accepted_options": {
                "dcci_before_kernel_start": [[".*GroupedMatmul.*"]],
                "dcci_after_kernel_end": [[".*GroupedMatmul.*"]],
                "dcci_disable_on_kernel": [[".*"]],
            },
        }
        candidate_config = {"runtime_evidence": {"dcci_state": "disabled"}}

        diagnostics = analyze_fusion_performance._build_regression_diagnostics(
            [_scalar_regression_decision()],
            None,
            environment_evidence,
            candidate_config,
        )

        self.assertFalse(
            any(
                item["kind"].startswith("dcci")
                for item in diagnostics["diagnostic_hypotheses"]
            )
        )
        self.assertFalse(
            any(
                item["option"].startswith("dcci_")
                for item in diagnostics["recommended_experiments"]
            )
        )
        blocker = next(
            item for item in diagnostics["blockers"] if "dcci_state_conflict" in item
        )
        self.assertIn(
            "environment_evidence.runtime_evidence.dcci_state=enabled", blocker
        )
        self.assertIn("candidate_config.runtime_evidence.dcci_state=disabled", blocker)

    def test_global_dcci_control_requires_exact_global_regex_acceptance(self):
        fixtures = (
            (
                {
                    "runtime_evidence": {"dcci_state": "enabled"},
                    "accepted_options": {"dcci_disable_on_kernel": [[".*"]]},
                },
                {},
                True,
            ),
            (
                {
                    "runtime_evidence": {"dcci_state": "enabled"},
                    "accepted_options": {
                        "dcci_disable_on_kernel": [[".*GroupedMatmul.*"]]
                    },
                },
                {},
                False,
            ),
            (
                {"accepted_options": {"dcci_disable_on_kernel": [[".*"]]}},
                {"runtime_evidence": {"dcci_state": "enabled"}},
                True,
            ),
        )
        for environment_evidence, candidate_config, _expected_global in fixtures:
            diagnostics = analyze_fusion_performance._build_regression_diagnostics(
                [_scalar_regression_decision()],
                None,
                environment_evidence,
                candidate_config,
            )
            self.assertEqual(diagnostics["recommended_experiments"], [])

    def test_complete_analysis_cli_blocks_undeclared_config_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            candidate_config = json.loads(artifacts["candidate_config"].read_text())
            candidate_config["runtime"]["rank_size"] = 4
            artifacts["candidate_config"].write_text(json.dumps(candidate_config))
            json_out = root / "analysis" / "result.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            report = json.loads(json_out.read_text())

        decision = report["per_sk_decisions"][0]
        self.assertEqual(exit_code, 0)
        self.assertIn("undeclared_config_differences", report["blockers"])
        self.assertEqual(decision["mapping_confidence"], "diagnostic_only")
        self.assertIsNone(decision["original"])
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(decision["action"], "reprofile")

    def test_complete_analysis_cli_invalid_input_exits_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            args = _analysis_cli_args(artifacts, root / "result.json")
            json_index = args.index("--json-out")
            del args[json_index : json_index + 2]

            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as stderr,
            ):
                with self.assertRaises(SystemExit) as raised:
                    analyze_fusion_performance.main(args)

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--json-out", stderr.getvalue())

    def test_complete_analysis_cli_rejects_malformed_round_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            json_out = root / "malformed-round.json"
            args = _analysis_cli_args(artifacts, json_out)
            args[args.index("--round-id") + 1] = "S1-WEIRD"

            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as stderr,
            ):
                with self.assertRaises(SystemExit) as raised:
                    analyze_fusion_performance.main(args)

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--round-id", stderr.getvalue())
        self.assertFalse(json_out.exists())

    def test_complete_analysis_cli_rejects_directory_output_transactionally(self):
        for preexisting_json in (False, True):
            with (
                self.subTest(preexisting_json=preexisting_json),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                json_out = root / "analysis" / "result.json"
                if preexisting_json:
                    json_out.parent.mkdir()
                    json_out.write_text("existing-user-result\n")
                markdown_out = root / "markdown-target"
                markdown_out.mkdir()

                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        analyze_fusion_performance.main(
                            _analysis_cli_args(artifacts, json_out, markdown_out)
                        )

                self.assertEqual(raised.exception.code, 2)
                if preexisting_json:
                    self.assertEqual(json_out.read_text(), "existing-user-result\n")
                else:
                    self.assertFalse(json_out.exists())

    def test_complete_analysis_cli_rejects_output_path_conflicts(self):
        for case in (
            "same-output",
            "input-file",
            "input-directory",
            "inside-input-directory",
            "symlink-input",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                json_out = root / "result.json"
                markdown_out = root / "result.md"
                if case == "same-output":
                    markdown_out = json_out
                elif case == "input-file":
                    json_out = artifacts["baseline_profile"]
                elif case == "input-directory":
                    json_out = artifacts["sk_meta"]
                elif case == "inside-input-directory":
                    json_out = next(artifacts["sk_meta"].rglob("sk_fused_nodes.log"))
                else:
                    json_out = root / "result-link.json"
                    json_out.symlink_to(artifacts["baseline_profile"])
                baseline_before = artifacts["baseline_profile"].read_text()

                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        analyze_fusion_performance.main(
                            _analysis_cli_args(artifacts, json_out, markdown_out)
                        )

                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(
                    artifacts["baseline_profile"].read_text(), baseline_before
                )

    def test_complete_analysis_cli_rejects_cyclic_symlink_paths(self):
        for case in ("input", "output"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                json_out = root / "result.json"
                if case == "input":
                    cyclic_path = artifacts["baseline_profile"]
                    cyclic_path.unlink()
                else:
                    cyclic_path = json_out
                cyclic_path.symlink_to(cyclic_path.name)

                with (
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()) as stderr,
                ):
                    with self.assertRaises(SystemExit) as raised:
                        analyze_fusion_performance.main(
                            _analysis_cli_args(artifacts, json_out)
                        )

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("cannot resolve path", stderr.getvalue())
                self.assertTrue(cyclic_path.is_symlink())
                if case == "input":
                    self.assertFalse(json_out.exists())

    def test_analysis_output_rolls_back_markdown_when_json_commit_fails(self):
        for markdown_preexists in (False, True):
            with (
                self.subTest(markdown_preexists=markdown_preexists),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                json_out = root / "result.json"
                markdown_out = root / "result.md"
                json_out.write_text("old-json\n")
                if markdown_preexists:
                    markdown_out.write_text("old-markdown\n")
                real_replace = analyze_fusion_performance.os.replace

                def fail_json_commit(source, destination):
                    if Path(destination) == json_out:
                        raise OSError("injected JSON commit failure")
                    return real_replace(source, destination)

                with mock.patch.object(
                    analyze_fusion_performance.os,
                    "replace",
                    side_effect=fail_json_commit,
                ):
                    with self.assertRaisesRegex(
                        OSError, "injected JSON commit failure"
                    ):
                        analyze_fusion_performance._write_analysis_outputs(
                            json_out,
                            "new-json\n",
                            markdown_path=markdown_out,
                        )

                self.assertEqual(json_out.read_text(), "old-json\n")
                if markdown_preexists:
                    self.assertEqual(markdown_out.read_text(), "old-markdown\n")
                else:
                    self.assertFalse(markdown_out.exists())
                self.assertEqual(
                    sorted(path.name for path in root.iterdir()),
                    (
                        ["result.json"]
                        if not markdown_preexists
                        else ["result.json", "result.md"]
                    ),
                )

    def test_complete_analysis_cli_rejects_non_finite_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            for option in (
                "--min-relative-change-pct",
                "--min-absolute-change-us",
            ):
                for label, value in (
                    ("nan", "nan"),
                    ("positive-inf", "inf"),
                    ("negative-inf", "-inf"),
                ):
                    with self.subTest(option=option, value=value):
                        json_out = root / f"{option[2:]}-{label}.json"
                        args = _analysis_cli_args(artifacts, json_out)
                        args[args.index(option) + 1] = value
                        with (
                            redirect_stdout(io.StringIO()),
                            redirect_stderr(io.StringIO()),
                        ):
                            with self.assertRaises(SystemExit) as raised:
                                analyze_fusion_performance.main(args)

                        self.assertEqual(raised.exception.code, 2)
                        self.assertFalse(json_out.exists())

    def test_complete_analysis_cli_rejects_invalid_array_index_pointers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            baseline_config = json.loads(artifacts["baseline_config"].read_text())
            candidate_config = json.loads(artifacts["candidate_config"].read_text())
            baseline_config["items"] = list(range(11))
            candidate_config["items"] = list(range(11))
            candidate_config["items"][1] = "undeclared-change"
            artifacts["baseline_config"].write_text(json.dumps(baseline_config))
            artifacts["candidate_config"].write_text(json.dumps(candidate_config))
            base_declared_change = json.loads(
                artifacts["declared_change_set"].read_text()
            )

            for label, pointer in (
                ("leading-zero", "/items/01"),
                ("unicode-digit", "/items/\u0661"),
                ("append-token", "/items/-"),
            ):
                with self.subTest(pointer=pointer):
                    declared_change = {
                        **base_declared_change,
                        "allowed_json_pointers": [
                            *base_declared_change["allowed_json_pointers"],
                            pointer,
                        ],
                    }
                    artifacts["declared_change_set"].write_text(
                        json.dumps(declared_change)
                    )
                    json_out = root / f"invalid-{label}.json"

                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            analyze_fusion_performance.main(
                                _analysis_cli_args(artifacts, json_out)
                            )

                    self.assertEqual(raised.exception.code, 2)
                    self.assertFalse(json_out.exists())

    def test_complete_analysis_cli_accepts_valid_array_index_pointers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            baseline_config = json.loads(artifacts["baseline_config"].read_text())
            candidate_config = json.loads(artifacts["candidate_config"].read_text())
            baseline_config["items"] = list(range(11))
            candidate_config["items"] = list(range(11))
            candidate_config["items"][0] = "allowed-zero"
            candidate_config["items"][10] = "allowed-ten"
            artifacts["baseline_config"].write_text(json.dumps(baseline_config))
            artifacts["candidate_config"].write_text(json.dumps(candidate_config))
            declared_change = json.loads(artifacts["declared_change_set"].read_text())
            declared_change["allowed_json_pointers"].extend(["/items/0", "/items/10"])
            artifacts["declared_change_set"].write_text(json.dumps(declared_change))
            json_out = root / "valid-array-indices.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            report = json.loads(json_out.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["blockers"], [])
        self.assertEqual(
            report["per_sk_decisions"][0]["classification"], "insufficient_evidence"
        )

    def test_analysis_mode_accepts_equals_in_artifact_paths(self):
        for artifact_name, renamed in (
            ("sk_meta", "sk=meta"),
            ("sk_prof", "sk=prof.json"),
        ):
            with (
                self.subTest(artifact=artifact_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                renamed_path = root / renamed
                artifacts[artifact_name].rename(renamed_path)
                artifacts[artifact_name] = renamed_path
                json_out = root / f"{artifact_name}.json"

                with redirect_stdout(io.StringIO()):
                    exit_code = analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )

                self.assertEqual(exit_code, 0)
                self.assertTrue(json_out.is_file())

    def test_child_schedule_requires_both_core_families(self):
        baseline = {"multi_stream_analysis": {"cube_vector_parallel_detected": True}}
        serialized_child = {
            "stream_identity_complete": True,
            "core_family_counts": [
                {"name": "CUBE", "count": 1},
                {"name": "VECTOR", "count": 1},
            ],
            "multi_stream_analysis": {"cube_vector_parallel_detected": False},
        }
        incomplete_child = {
            "stream_identity_complete": True,
            "core_family_counts": [{"name": "VECTOR", "count": 1}],
            "multi_stream_analysis": {"cube_vector_parallel_detected": False},
        }
        serialized = {}
        incomplete = {}

        analyze_fusion_performance._attach_sk_child_schedule(
            serialized, baseline, serialized_child
        )
        analyze_fusion_performance._attach_sk_child_schedule(
            incomplete, baseline, incomplete_child
        )

        self.assertEqual(
            serialized["sk_child_schedule"]["verdict"],
            "cube_vector_serialized_inside_sk",
        )
        self.assertEqual(
            incomplete["sk_child_schedule"]["verdict"],
            "child_trace_missing_cube_vector_evidence",
        )
        self.assertNotIn("auto_op_parallel", incomplete["sk_child_schedule"]["action"])

    def test_removed_cli_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            args = _analysis_cli_args(artifacts, root / "mixed.json")
            args.extend(["--profile", str(artifacts["baseline_profile"])])

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    analyze_fusion_performance.main(args)

        self.assertEqual(raised.exception.code, 2)

    def test_finite_occurrences_with_overflowing_derivations_are_insufficient(self):
        cases = (
            ([1e308, 1e308, 1e308], [-1e308, -1e308, -1e308]),
            ([-1e308, 1e-308, 1e308], [-1e308, 1e-308, 1e308]),
        )

        for baseline_values, candidate_values in cases:
            with self.subTest(baseline=baseline_values, candidate=candidate_values):
                decision = analyze_fusion_performance.classify_performance(
                    analyze_fusion_performance._robust_stats(baseline_values),
                    analyze_fusion_performance._robust_stats(candidate_values),
                )

                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertIn(
                    "non_finite_derived_statistics", decision["evidence_errors"]
                )
                json.dumps(decision, allow_nan=False)

    def test_complete_analysis_cli_handles_finite_occurrence_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _write_csv(
                artifacts["baseline_profile"],
                [
                    _row(
                        10,
                        1,
                        "static_kernel_A_hash",
                        "A",
                        "AI_VECTOR_CORE",
                        0,
                        1e-308,
                    )
                    for _ in range(3)
                ],
            )
            sk_name = (
                "sk_1_decoder.layer.0_start_static_kernel_A_hash_"
                "end_static_kernel_A_hash"
            )
            _write_csv(
                artifacts["candidate_profile"],
                [
                    _row(
                        99,
                        1,
                        sk_name,
                        "SuperKernel",
                        "AI_VECTOR_CORE",
                        0,
                        1e308,
                    )
                    for _ in range(3)
                ],
            )
            json_out = root / "finite-overflow.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            raw_report = json_out.read_text()
            report = json.loads(
                raw_report,
                parse_constant=lambda value: self.fail(
                    f"non-standard JSON constant: {value}"
                ),
            )

        decision = report["per_sk_decisions"][0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(decision["action"], "reprofile")
        self.assertIn("non_finite_derived_statistics", decision["evidence_errors"])
        self.assertNotIn("sk_vs_original_interval_pct", decision)
        self.assertIsNone(decision["fusion_benefit"]["interval_improvement_pct"])

    def test_complete_analysis_cli_handles_profile_arithmetic_overflow(self):
        cases = (
            {
                "name": "baseline-duration-sum",
                "baseline_start": 0.0,
                "baseline_duration": 1e308,
                "candidate_start": 0.0,
                "candidate_duration": 1e-308,
            },
            {
                "name": "row-end",
                "baseline_start": 1e308,
                "baseline_duration": 1e308,
                "candidate_start": 1e308,
                "candidate_duration": 1e308,
            },
        )

        for case in cases:
            with (
                self.subTest(case=case["name"]),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts = _write_analysis_cli_fixture(root)
                _write_csv(
                    artifacts["baseline_profile"],
                    [
                        _row(
                            10,
                            1,
                            "static_kernel_A_hash",
                            "A",
                            "AI_VECTOR_CORE",
                            case["baseline_start"],
                            case["baseline_duration"],
                        )
                        for _ in range(3)
                    ],
                )
                sk_name = (
                    "sk_1_decoder.layer.0_start_static_kernel_A_hash_"
                    "end_static_kernel_A_hash"
                )
                _write_csv(
                    artifacts["candidate_profile"],
                    [
                        _row(
                            99,
                            1,
                            sk_name,
                            "SuperKernel",
                            "AI_VECTOR_CORE",
                            case["candidate_start"],
                            case["candidate_duration"],
                        )
                        for _ in range(3)
                    ],
                )
                json_out = root / f"{case['name']}.json"

                with redirect_stdout(io.StringIO()):
                    exit_code = analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )
                report = json.loads(
                    json_out.read_text(),
                    parse_constant=lambda value: self.fail(
                        f"non-standard JSON constant: {value}"
                    ),
                )

                decision = report["per_sk_decisions"][0]
                self.assertEqual(exit_code, 0)
                self.assertEqual(decision["classification"], "insufficient_evidence")
                self.assertEqual(decision["action"], "reprofile")
                self.assertIn(
                    "non_finite_derived_statistics",
                    decision["evidence_errors"],
                )

    def test_candidate_rejected_occurrence_blocks_otherwise_valid_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _, candidate_rows = analyze_fusion_performance.load_kernel_rows(
                artifacts["candidate_profile"]
            )
            candidate_rows.append(
                _row(
                    99,
                    1,
                    candidate_rows[0]["name"],
                    "SuperKernel",
                    "AI_VECTOR_CORE",
                    60,
                    float("nan"),
                )
            )
            _write_csv(artifacts["candidate_profile"], candidate_rows)
            json_out = root / "candidate-rejected.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertIn("baseline_mapping_missing", decision["evidence_errors"])
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(decision["action"], "reprofile")

    def test_non_positive_candidate_occurrences_leave_no_usable_sk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _, candidate_rows = analyze_fusion_performance.load_kernel_rows(
                artifacts["candidate_profile"]
            )
            for row in candidate_rows:
                row["duration_us"] = -1
            _write_csv(artifacts["candidate_profile"], candidate_rows)
            json_out = root / "negative-candidate.json"

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    analyze_fusion_performance.main(
                        _analysis_cli_args(artifacts, json_out)
                    )

        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(json_out.exists())

    def test_rejected_baseline_child_occurrence_blocks_exact_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            _, baseline_rows = analyze_fusion_performance.load_kernel_rows(
                artifacts["baseline_profile"]
            )
            baseline_rows.append(
                _row(
                    10,
                    1,
                    "static_kernel_A_other_hash",
                    "A",
                    "AI_VECTOR_CORE",
                    60,
                    float("nan"),
                )
            )
            _write_csv(artifacts["baseline_profile"], baseline_rows)
            json_out = root / "baseline-rejected.json"

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out)
                )
            decision = json.loads(json_out.read_text())["per_sk_decisions"][0]

        self.assertEqual(exit_code, 0)
        self.assertIn("baseline_mapping_missing", decision["evidence_errors"])
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(decision["action"], "reprofile")

    def test_complete_analysis_cli_help_lists_all_contract_arguments(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            analyze_fusion_performance.main(["--baseline-profile", "unused", "--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        for option in (
            "--baseline-profile",
            "--candidate-profile",
            "--sk-meta",
            "--baseline-config",
            "--candidate-config",
            "--baseline-workload",
            "--candidate-workload",
            "--declared-change-set",
            "--candidate-name",
            "--experiment-id",
            "--round-id",
            "--analysis-agent-id",
            "--source-revision",
            "--sk-prof",
            "--environment-evidence",
            "--source-scope-map",
            "--source-root",
            "--baseline-collection-manifest",
            "--profile-collection-manifest",
            "--min-relative-change-pct",
            "--min-absolute-change-us",
            "--min-occurrences",
            "--json-out",
            "--markdown-out",
        ):
            with self.subTest(option=option):
                self.assertIn(option, help_text)
        for retired_option in (
            "--replay-evidence",
            "--compat-collection-manifest",
            "--verify-collection-manifest",
        ):
            self.assertNotIn(retired_option, help_text)

    def test_default_help_lists_complete_analysis_contract(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            analyze_fusion_performance.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--baseline-profile", help_text)
        self.assertIn("--experiment-id", help_text)
        self.assertIn("--markdown-out", help_text)

    def test_markdown_renderer_cli_controls_recursion_errors(self):
        renderer = importlib.import_module("render_fusion_performance_report")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_in = root / "deep.json"
            markdown_out = root / "report.md"
            json_in.write_text("[" * 1200 + "null" + "]" * 1200)
            stderr = io.StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                renderer.main(
                    ["--json-in", str(json_in), "--markdown-out", str(markdown_out)]
                )

        self.assertEqual(raised.exception.code, 2)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse(markdown_out.exists())

    def test_complete_analysis_cli_stores_only_relative_artifact_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            json_out = root / "nested" / "analysis" / "result.json"
            markdown_out = root / "nested" / "analysis" / "report.md"

            with redirect_stdout(io.StringIO()):
                analyze_fusion_performance.main(
                    _analysis_cli_args(artifacts, json_out, markdown_out)
                )
            report_text = json_out.read_text()
            report = json.loads(report_text)

            self.assertNotIn(str(root), report_text)
            for artifact_path in report["inputs"].values():
                if artifact_path is not None:
                    self.assertFalse(Path(artifact_path).is_absolute())

    def test_single_child_with_repeated_benefit_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.csv"
            candidate = root / "candidate.csv"
            metadata = root / "sk_meta"
            metadata.mkdir()
            _write_csv(
                baseline,
                [
                    _row(
                        10,
                        1,
                        "static_kernel_A_hash",
                        "A",
                        "AI_VECTOR_CORE",
                        start,
                        8,
                    )
                    for start in (0, 20, 40)
                ],
            )
            sk_name = "sk_1_scope_start_static_kernel_A_hash_end_static_kernel_A_hash"
            _write_csv(
                candidate,
                [
                    _row(99, 1, sk_name, "SuperKernel", "AI_VECTOR_CORE", start, 6)
                    for start in (0, 20, 40)
                ],
            )
            _metadata_log_path(metadata).write_text(
                "SK Function: " + sk_name + ", scope id: 1, Node Count: 1\n"
                "[nodeId:10, streamId:1] - "
                "KernelInfos{funcName:static_kernel_A_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
                "isScheModeOn:0}\n"
            )
            _, baseline_rows = analyze_fusion_performance.summarize_profile(baseline)

            result = analyze_fusion_performance.compare_candidate(
                baseline_rows,
                "S1",
                candidate,
                sk_meta=metadata,
            )

        decision = result["superkernel_comparisons"][0]
        self.assertEqual(len(result["superkernel_comparisons"]), 1)
        self.assertEqual(decision["child_count"], 1)
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(decision["action"], "reprofile")
        self.assertIn(
            "cross_compile_runtime_id_not_identity",
            decision["mapping_blockers"],
        )

    def test_incomplete_metadata_model_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _write_analysis_cli_fixture(root)
            model_log = _metadata_log_path(artifacts["sk_meta"])
            incomplete_log = artifacts["sk_meta"] / "sk_fused_nodes.log"
            incomplete_log.write_text(model_log.read_text())
            model_log.unlink()
            _, baseline_rows = analyze_fusion_performance.summarize_profile(
                artifacts["baseline_profile"]
            )

            result = analyze_fusion_performance.compare_candidate(
                baseline_rows,
                "S1",
                artifacts["candidate_profile"],
                sk_meta=artifacts["sk_meta"],
            )

        self.assertEqual(
            result["superkernel_comparisons"][0]["classification"],
            "insufficient_evidence",
        )

    def test_six_child_regression_is_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.csv"
            candidate = root / "candidate.csv"
            metadata = root / "sk_meta"
            metadata.mkdir()
            baseline_rows = []
            for occurrence_start in (0, 20, 40):
                for child_index, op_type in enumerate("ABCDEF"):
                    baseline_rows.append(
                        _row(
                            10 + child_index,
                            1,
                            f"static_kernel_{op_type}_hash",
                            op_type,
                            "AI_VECTOR_CORE",
                            occurrence_start + child_index,
                            1,
                        )
                    )
            _write_csv(baseline, baseline_rows)
            sk_name = "sk_1_scope_start_static_kernel_A_hash_end_static_kernel_F_hash"
            _write_csv(
                candidate,
                [
                    _row(99, 1, sk_name, "SuperKernel", "AI_VECTOR_CORE", start, 9)
                    for start in (0, 20, 40)
                ],
            )
            metadata_lines = [f"SK Function: {sk_name}, scope id: 1, Node Count: 6"]
            for child_index, op_type in enumerate("ABCDEF"):
                metadata_lines.append(
                    f"[nodeId:{10 + child_index}, streamId:1] - "
                    f"KernelInfos{{funcName:static_kernel_{op_type}_hash, "
                    "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
                    "isScheModeOn:0}"
                )
            _metadata_log_path(metadata).write_text("\n".join(metadata_lines) + "\n")
            _, loaded_baseline_rows = analyze_fusion_performance.summarize_profile(
                baseline
            )

            result = analyze_fusion_performance.compare_candidate(
                loaded_baseline_rows,
                "S1",
                candidate,
                sk_meta=metadata,
            )

        decision = result["superkernel_comparisons"][0]
        self.assertEqual(decision["child_count"], 6)
        self.assertEqual(decision["classification"], "insufficient_evidence")
        self.assertEqual(decision["action"], "reprofile")

    def test_interval_summary_calculates_end_time_for_parallel_rows(self):
        rows = [
            _row(10, 1, "A", "Add", "AI_VECTOR_CORE", 0, 10),
            _row(11, 2, "B", "MatMul", "AI_CORE", 2, 8),
        ]

        summary = analyze_fusion_performance._interval_summary(rows)

        self.assertEqual(summary["start_us"], 0)
        self.assertEqual(summary["end_us"], 10)
        self.assertEqual(summary["interval_us"], 10)
        self.assertEqual(summary["duration_sum_us"], 18)
        self.assertEqual(summary["union_duration_us"], 10)

    def test_structural_manifest_config_and_control_are_association_domain(self):
        paths = {
            field: Path(field)
            for field in analyze_fusion_performance.COLLECTION_MANIFEST_ARGUMENTS
        }
        summary = {
            "set_fingerprint": "1" * 64,
            "config_fingerprint": "association-config-fingerprint",
            "control_fingerprint": "association-control-fingerprint",
        }

        with (
            mock.patch.object(
                analyze_fusion_performance,
                "validate_manifest_set",
                return_value=summary,
            ),
            mock.patch.object(
                analyze_fusion_performance,
                "_manifest_content_fingerprint",
                return_value="2" * 64,
            ),
        ):
            protocol, loaded = analyze_fusion_performance._collection_manifest_protocol(
                paths
            )

        self.assertEqual(loaded, summary)
        self.assertEqual(protocol["status"], "validated")
        self.assertEqual(
            protocol["association_config_fingerprint"],
            "association-config-fingerprint",
        )
        self.assertEqual(
            protocol["association_control_fingerprint"],
            "association-control-fingerprint",
        )

        args = mock.Mock(source_revision="revision-a")
        analysis_fingerprints = {
            "workload_fingerprint": "workload-a",
            "candidate_config_fingerprint": "actual-candidate-config",
            "control_fingerprint": "actual-config-derived-control",
        }
        collection_summary = {
            **summary,
            "source_revision": "revision-a",
            "workload_fingerprint": "workload-a",
        }
        self.assertTrue(
            analyze_fusion_performance._structural_collection_identity_matches(
                collection_summary, args, analysis_fingerprints
            )
        )
        collection_summary["workload_fingerprint"] = "workload-b"
        self.assertFalse(
            analyze_fusion_performance._structural_collection_identity_matches(
                collection_summary, args, analysis_fingerprints
            )
        )

    def test_layer_inventory_and_cube_vector_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "kernel_details.csv"
            _write_csv(
                csv_path,
                [
                    _row(
                        10,
                        1,
                        "static_kernel_RmsNorm_hash",
                        "RmsNorm",
                        "AI_VECTOR_CORE",
                        0,
                        10,
                    ),
                    _row(
                        11,
                        2,
                        "static_kernel_MatMul_hash",
                        "MatMul",
                        "AI_CORE",
                        2,
                        6,
                    ),
                    _row(
                        20,
                        1,
                        "static_kernel_RmsNorm_hash",
                        "RmsNorm",
                        "AI_VECTOR_CORE",
                        10,
                        10,
                    ),
                    _row(
                        21,
                        2,
                        "static_kernel_MatMul_hash",
                        "MatMul",
                        "AI_CORE",
                        12,
                        6,
                    ),
                ],
            )
            layer_map = [
                {
                    "layer": 0,
                    "model_id": 48,
                    "start_task_id": 10,
                    "end_task_id": 11,
                },
                {
                    "layer": 1,
                    "model_id": 48,
                    "start_task_id": 20,
                    "end_task_id": 21,
                },
            ]
            summary, _ = analyze_fusion_performance.summarize_profile(
                csv_path, layer_map=layer_map
            )

        self.assertEqual(summary["layer_analysis"]["status"], "complete")
        self.assertEqual(summary["layer_analysis"]["layer_count"], 2)
        self.assertEqual(
            summary["layer_analysis"]["layers"]["0"]["graph_node_count"], 2
        )
        self.assertTrue(
            summary["multi_stream_analysis"]["cube_vector_parallel_detected"]
        )
        self.assertAlmostEqual(
            summary["multi_stream_analysis"]["cube_vector_overlap_us"], 12
        )

    def test_sk_child_trace_detects_internal_cube_vector_parallelism(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "sk_prof_device_0.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "name": "static_kernel_Vector_hash",
                                "ph": "X",
                                "ts": 0,
                                "dur": 10,
                                "tid": 1,
                                "args": {"kernelType": "AIV_ONLY"},
                            },
                            {
                                "name": "static_kernel_Cube_hash",
                                "ph": "X",
                                "ts": 2,
                                "dur": 6,
                                "tid": 2,
                                "args": {"kernelType": "AIC_ONLY"},
                            },
                        ]
                    }
                )
            )
            summary = analyze_fusion_performance.summarize_sk_child_profile(trace_path)

        self.assertEqual(summary["event_count"], 2)
        self.assertTrue(
            summary["multi_stream_analysis"]["cube_vector_parallel_detected"]
        )

    def test_child_trace_stream_identity_gates_cube_vector_direct_evidence(self):
        cases = (
            ("missing", ({}, {}), 0, 0, False, False),
            ("same_stream", ({"tid": 7}, {"tid": 7}), 1, 2, True, True),
            (
                "multiple_streams",
                (
                    {"args": {"streamId": 7}},
                    {"args": {"stream_id": 8}},
                ),
                2,
                2,
                True,
                True,
            ),
        )
        for name, stream_fields, stream_count, known_count, complete, direct in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                trace_path = Path(directory) / "sk_prof_device_0.json"
                events = []
                for event_name, kernel_type, start, fields in (
                    ("static_kernel_Vector_hash", "AIV_ONLY", 0, stream_fields[0]),
                    ("static_kernel_Cube_hash", "AIC_ONLY", 5, stream_fields[1]),
                ):
                    event = {
                        "name": event_name,
                        "ph": "X",
                        "ts": start,
                        "dur": 5,
                        **fields,
                    }
                    event_args = event.setdefault("args", {})
                    event_args["kernelType"] = kernel_type
                    events.append(event)
                trace_path.write_text(json.dumps({"traceEvents": events}))

                summary = analyze_fusion_performance.summarize_sk_child_profile(
                    trace_path
                )
                diagnostics = analyze_fusion_performance._build_regression_diagnostics(
                    [_cube_vector_regression_decision()],
                    summary,
                    {"accepted_options": {"auto_op_parallel": [1]}},
                    {},
                )

            kinds = {item["kind"] for item in diagnostics["diagnostic_hypotheses"]}
            auto_experiments = [
                item
                for item in diagnostics["recommended_experiments"]
                if item["option"] == "auto_op_parallel"
            ]
            self.assertEqual(summary["stream_count"], stream_count)
            self.assertEqual(summary.get("known_stream_event_count"), known_count)
            self.assertEqual(summary.get("stream_identity_complete"), complete)
            self.assertEqual("cube_vector_serialization" in kinds, direct)
            self.assertEqual("cube_vector_child_trace_missing" in kinds, not direct)
            self.assertEqual(auto_experiments, [])
            if not direct:
                self.assertTrue(
                    any(
                        "stream identity" in blocker
                        for blocker in diagnostics["blockers"]
                    )
                )

    def test_empty_sk_child_trace_is_missing_evidence_not_analysis_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "sk_prof_device_0.json"
            trace_path.write_text(json.dumps({"traceEvents": []}))
            summary = analyze_fusion_performance.summarize_sk_child_profile(trace_path)

        diagnostics = analyze_fusion_performance._build_regression_diagnostics(
            [_cube_vector_regression_decision()], summary, {}, {}
        )
        missing = next(
            item
            for item in diagnostics["diagnostic_hypotheses"]
            if item["kind"] == "cube_vector_child_trace_missing"
        )

        self.assertEqual(summary["event_count"], 0)
        self.assertEqual(missing["confidence"], "low")
        self.assertTrue(
            any(
                "child_trace_missing_cube_vector_evidence" in blocker
                for blocker in diagnostics["blockers"]
            )
        )

    def test_scalar_only_regression_does_not_propose_dcci_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.csv"
            candidate = root / "candidate.csv"
            metadata = root / "sk_meta"
            metadata.mkdir()
            _write_csv(
                baseline,
                [
                    row
                    for occurrence_start in (0, 30, 60)
                    for row in (
                        _row(
                            10,
                            1,
                            "static_kernel_GroupedMatmul_hash",
                            "GroupedMatmul",
                            "AI_CORE",
                            occurrence_start,
                            10,
                            0.5,
                        ),
                        _row(
                            11,
                            1,
                            "static_kernel_Add_hash",
                            "Add",
                            "AI_VECTOR_CORE",
                            occurrence_start + 10,
                            2,
                            0.3,
                        ),
                    )
                ],
            )
            sk_name = (
                "sk_1_decoder.layer.0.mlp_start_static_kernel_GroupedMatmul_hash_"
                "end_static_kernel_Add_hash"
            )
            _write_csv(
                candidate,
                [
                    _row(99, 1, sk_name, "SuperKernel", "MIX_AIC", start, 15)
                    for start in (0, 30, 60)
                ],
            )
            _metadata_log_path(metadata).write_text(
                "SK Function: " + sk_name + ", scope id: 1, Node Count: 2\n"
                "[nodeId:10, streamId:1] - "
                "KernelInfos{funcName:static_kernel_GroupedMatmul_hash, "
                "kernelType:AIC_ONLY, numBlocks:1, cubeNum:1, vecNum:0, "
                "isScheModeOn:0}\n"
                "[nodeId:11, streamId:1] - "
                "KernelInfos{funcName:static_kernel_Add_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
                "isScheModeOn:0}\n"
            )
            baseline_summary, baseline_rows = (
                analyze_fusion_performance.summarize_profile(baseline)
            )
            result = analyze_fusion_performance.compare_candidate(
                baseline_rows,
                "S4",
                candidate,
                sk_meta=metadata,
                baseline_summary=baseline_summary,
            )

        self.assertEqual(result["fusion_performance"]["regressed_sk_count"], 0)
        self.assertEqual(
            result["superkernel_comparisons"][0]["classification"],
            "insufficient_evidence",
        )
        benefit = result["superkernel_comparisons"][0]["fusion_benefit"]
        self.assertEqual(benefit["child_count"], 2)
        self.assertEqual(benefit["launch_count_without_sk"], 2)
        self.assertEqual(benefit["launch_count_with_sk"], 1)
        self.assertEqual(benefit["estimated_launch_reduction"], 1)
        self.assertIsNone(benefit["interval_improvement_us"])
        self.assertEqual(result["fusion_performance"]["dcci_diagnostic_candidates"], [])
