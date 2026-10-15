#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""从 profiling 分析结果生成整 scope 晋级及细粒度优化计划。"""

import argparse
import copy
import hashlib
import html
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Collection
from pathlib import Path


SCHEMA_VERSION = "1.0"
SUPPORTED_ANALYSIS_SCHEMA_VERSIONS = {"1.2"}
PERFORMANCE_EXACT_MAPPINGS = {
    ("source_scope_map", "exact"),
    ("kernel_projection_structural", "exact_projected_trace"),
}
MAPPING_METHODS = {
    "sk_meta_node_ids",
    "source_scope_map",
    "kernel_projection_structural",
}
MAPPING_CONFIDENCES = {
    "exact",
    "exact_projected_trace",
    "diagnostic_only",
    "ambiguous",
    "unmapped",
}
CLASS_ACTIONS = {
    "beneficial": {"keep"},
    "neutral": {"prune"},
    "regressed": {"prune"},
    "insufficient_evidence": {"reprofile", "block"},
}
DCCI_OPTIONS = {
    "dcci_before_kernel_start",
    "dcci_after_kernel_end",
    "dcci_disable_on_kernel",
}
FINGERPRINT_FIELDS = (
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
)
REQUIRED_INPUTS = (
    "baseline_profile",
    "candidate_profile",
    "sk_meta",
    "baseline_config",
    "candidate_config",
    "baseline_workload",
    "candidate_workload",
    "declared_change_set",
)
OPTIONAL_INPUTS = (
    "sk_prof",
    "environment_evidence",
    "source_scope_map",
    "baseline_collection_manifest",
    "profile_collection_manifest",
)
REANALYSIS_SEMANTIC_FIELDS = (
    "schema_version",
    "analysis_id",
    "experiment_id",
    "round_id",
    "analysis_agent_id",
    "candidate_name",
    "source_revision",
    *FINGERPRINT_FIELDS,
    "declared_change_set",
    "thresholds",
    "coverage",
    "per_sk_decisions",
    "scope_actions",
    "diagnostic_hypotheses",
    "recommended_experiments",
    "blockers",
    "environment_evidence_loaded",
    "next_agent_guidance_zh",
    "association_protocol",
    "candidate_binding_evidence",
    "canonical_graph_fingerprints",
    "stream_role_mapping",
    "graph_alignment_proof",
    "mapping_coverage",
    "source_scope_mapping",
)
ACCEPTED_SOURCE_TEMPLATES = (
    "environment_evidence.options.optimize_options.{option}.accepted_values",
    "environment_evidence.accepted_options.{option}",
)
DCCI_RUNTIME_SOURCES = {
    "environment_evidence.runtime_evidence.dcci_state",
    "candidate_config.runtime_evidence.dcci_state",
    "candidate_config.runtime.runtime_evidence.dcci_state",
}
MAX_JSON_DEPTH = 256


def _analysis_input_fields(schema_version):
    if schema_version != "1.2":
        raise ValueError(_migration_error(schema_version))
    return REQUIRED_INPUTS, OPTIONAL_INPUTS


def _candidate_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("profiling 分析输入必须使用 NAME=PATH 格式")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("profiling 分析输入必须使用 NAME=PATH 格式")
    return name, Path(path)


def _load_json(path):
    artifact = Path(path)
    try:
        value = json.loads(
            artifact.read_text(errors="replace"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"禁止非标准 JSON 常量 {constant}")
            ),
        )
    except OSError as error:
        raise ValueError(f"无法读取 profiling 分析结果 {artifact}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"profiling 分析结果不是有效 JSON：{artifact}:{error.lineno}:"
            f"{error.colno} {error.msg}"
        ) from error
    except RecursionError as error:
        raise ValueError(
            f"profiling 分析结果 JSON 嵌套过深或递归结构无效：{artifact}"
        ) from error
    except ValueError as error:
        raise ValueError(
            f"profiling 分析结果不是标准 JSON：{artifact}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("profiling 分析结果顶层必须是 JSON 对象")
    _validate_json_value(value, "profiling 分析结果")
    return value


def _validate_json_value(value, location):
    active = set()
    stack = [("visit", value, location, 0)]
    while stack:
        operation, current, current_location, depth = stack.pop()
        if operation == "leave":
            active.remove(id(current))
            continue
        if current is None or isinstance(current, (bool, str, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError(f"{current_location} 必须使用有限 JSON 数值")
            continue
        if not isinstance(current, (dict, list)):
            raise ValueError(
                f"{current_location} 包含非 JSON 值 {type(current).__name__}"
            )
        if depth >= MAX_JSON_DEPTH:
            raise ValueError(
                f"{current_location} JSON 嵌套过深，最大允许 {MAX_JSON_DEPTH} 层"
            )
        identity = id(current)
        if identity in active:
            raise ValueError(f"{current_location} 包含循环 JSON 结构")
        active.add(identity)
        stack.append(("leave", current, current_location, depth))
        if isinstance(current, dict):
            children = []
            for key, child in current.items():
                if not isinstance(key, str):
                    raise ValueError(f"{current_location} 的 JSON 对象键必须是字符串")
                children.append(
                    ("visit", child, f"{current_location}.{key}", depth + 1)
                )
        else:
            children = [
                ("visit", child, f"{current_location}[{index}]", depth + 1)
                for index, child in enumerate(current)
            ]
        stack.extend(reversed(children))


def _canonical_json(value):
    _validate_json_value(value, "JSON 值")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _schema_canonical_sha256(value, schema_version):
    if schema_version != "1.2":
        raise ValueError(_migration_error(schema_version))
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_artifact_contract():
    path = (
        Path(__file__).resolve().parents[2]
        / "superkernel-fusion-performance-analysis"
        / "scripts"
        / "artifact_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "superkernel_fusion_artifact_contract", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载结构关联 artifact contract：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_structural_report_contract(name, analysis):
    path = (
        Path(__file__).resolve().parents[2]
        / "superkernel-fusion-performance-analysis"
        / "scripts"
        / "render_fusion_performance_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "superkernel_fusion_report_contract", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载结构化 report contract：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module._validate_report_schema(analysis)
    except ValueError as error:
        raise ValueError(
            f"{name}: schema {analysis.get('schema_version')} 结构证据无效：{error}"
        ) from error


def _workload_fingerprint(value, schema_version):
    if schema_version != "1.2":
        raise ValueError(_migration_error(schema_version))
    contract = _load_artifact_contract()
    try:
        normalized = contract.normalize_workload(value)
    except ValueError:
        normalized = value
    return contract.canonical_sha256(normalized)


def _same_json(left, right):
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _is_text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _contains_chinese(value):
    return isinstance(value, str) and any(
        "\u4e00" <= char <= "\u9fff" for char in value
    )


def _migration_error(version):
    rendered = "缺失" if version is None else repr(version)
    return (
        "策略输入只支持 schema_version=1.2 "
        "profiling-analysis-result.json；"
        f"当前 schema_version={rendered}。"
    )


def _validate_analysis_content_fingerprint(name, analysis):
    fingerprint = analysis.get("analysis_content_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise ValueError(f"{name}: analysis_content_fingerprint 必须是非空小写 SHA-256")
    unsigned = copy.deepcopy(analysis)
    unsigned.pop("analysis_content_fingerprint")
    if fingerprint != _schema_canonical_sha256(
        unsigned, analysis.get("schema_version")
    ):
        raise ValueError(
            f"{name}: analysis_content_fingerprint 与完整 profiling report 内容不一致"
        )


def _validate_analysis(name, analysis):
    if not isinstance(analysis, dict):
        raise ValueError(f"{name}: profiling 分析结果必须是对象")
    _validate_json_value(analysis, f"{name}.profiling_analysis")
    if analysis.get("schema_version") not in SUPPORTED_ANALYSIS_SCHEMA_VERSIONS:
        raise ValueError(_migration_error(analysis.get("schema_version")))
    candidate_name = analysis.get("candidate_name")
    if not _is_text(candidate_name):
        raise ValueError(f"{name}: candidate_name 必须是非空字符串")
    if candidate_name != name:
        raise ValueError(
            f"输入名称 {name!r} 与结果 candidate_name {candidate_name!r} 不一致"
        )
    for field in (
        "analysis_id",
        "experiment_id",
        "round_id",
        "analysis_agent_id",
        "source_revision",
    ):
        if not _is_text(analysis.get(field)):
            raise ValueError(f"{name}: {field} 必须是非空字符串")
    identity = {
        "experiment_id": analysis["experiment_id"],
        "round_id": analysis["round_id"],
        "analysis_agent_id": analysis["analysis_agent_id"],
        "candidate_name": candidate_name,
        "source_revision": analysis["source_revision"],
    }
    expected_analysis_id = "analysis-" + _schema_canonical_sha256(
        identity, analysis["schema_version"]
    )
    if analysis["analysis_id"] != expected_analysis_id:
        raise ValueError(f"{name}: analysis_id 与 analyzer 身份字段不一致")
    round_info = _parse_round_id(name, analysis["round_id"])

    for field in FINGERPRINT_FIELDS:
        value = analysis.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{name}: {field} 必须是非空小写 SHA-256")
    if analysis["baseline_workload_fingerprint"] != analysis["workload_fingerprint"]:
        raise ValueError(f"{name}: workload_fingerprint 必须绑定 baseline workload")
    if analysis["candidate_workload_fingerprint"] != analysis["workload_fingerprint"]:
        raise ValueError(f"{name}: candidate workload fingerprint 不一致")
    if analysis["baseline_control_fingerprint"] != analysis["control_fingerprint"]:
        raise ValueError(f"{name}: control_fingerprint 必须绑定 baseline control")
    if analysis["candidate_control_fingerprint"] != analysis["control_fingerprint"]:
        raise ValueError(f"{name}: candidate control fingerprint 不一致")

    declared_change_set = analysis.get("declared_change_set")
    if not isinstance(declared_change_set, dict):
        raise ValueError(f"{name}: declared_change_set 必须是 JSON 对象")
    pointers = declared_change_set.get("allowed_json_pointers")
    if not isinstance(pointers, list) or not all(_is_text(item) for item in pointers):
        raise ValueError(
            f"{name}: declared_change_set.allowed_json_pointers 必须是非空字符串列表"
        )

    inputs = analysis.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{name}: inputs 必须是对象")
    required_inputs, optional_inputs = _analysis_input_fields(
        analysis["schema_version"]
    )
    for field in required_inputs:
        value = inputs.get(field)
        if not _is_text(value) or Path(value).is_absolute():
            raise ValueError(f"{name}: inputs.{field} 必须是非空相对 artifact 路径")
    for field in optional_inputs:
        value = inputs.get(field)
        if value is not None and (not _is_text(value) or Path(value).is_absolute()):
            raise ValueError(
                f"{name}: inputs.{field} 必须是 null 或非空相对 artifact 路径"
            )

    source_mapping = analysis.get("source_scope_mapping")
    if not isinstance(source_mapping, dict):
        raise ValueError(f"{name}: source_scope_mapping 必须是 analyzer 输出对象")
    source_mapping_status = source_mapping.get("status")
    if source_mapping_status not in {"not_requested", "diagnostic_only", "exact"}:
        raise ValueError(f"{name}: source_scope_mapping.status 枚举无效")
    source_mapping_blockers = source_mapping.get("blockers")
    if not isinstance(source_mapping_blockers, list) or not all(
        _is_text(item) for item in source_mapping_blockers
    ):
        raise ValueError(f"{name}: source_scope_mapping.blockers 必须是字符串列表")
    if source_mapping_status == "exact":
        if (
            source_mapping.get("protocol") != "source_scope_map_v2"
            or source_mapping_blockers
            or source_mapping.get("source_revision") != analysis["source_revision"]
            or source_mapping.get("source_revision_role")
            not in {"stable_marker", "stable_source"}
            or not _is_text(inputs.get("source_scope_map"))
        ):
            raise ValueError(
                f"{name}: exact source_scope_mapping 缺少 v2/source_revision/input 证明"
            )

    for field in ("per_sk_decisions", "scope_actions", "recommended_experiments"):
        if not isinstance(analysis.get(field), list):
            raise ValueError(f"{name}: {field} 必须是列表")
    blockers = analysis.get("blockers")
    if not isinstance(blockers, list):
        raise ValueError(f"{name}: blockers 必须是列表")
    for index, blocker in enumerate(blockers):
        if not (_is_text(blocker) or (isinstance(blocker, dict) and blocker)):
            raise ValueError(
                f"{name}: blockers[{index}] 必须是非空字符串或非空 JSON 对象"
            )

    _validate_analysis_content_fingerprint(name, analysis)
    schema_version = analysis["schema_version"]
    _validate_structural_report_contract(name, analysis)
    decisions = _validate_per_sk_decisions(
        name, analysis["per_sk_decisions"], schema_version=schema_version
    )
    actions = _validate_scope_actions(
        name,
        analysis["scope_actions"],
        decisions,
        schema_version=schema_version,
    )
    decisions_by_identity = {}
    for index, decision in enumerate(decisions):
        location = f"{name}.per_sk_decisions[{index}]"
        _validate_structural_mapping(decision, location)
        fingerprint = _validate_structural_proof(
            name,
            analysis,
            decision,
            location,
        )
        if fingerprint is not None:
            decision["graph_occurrence_fingerprint"] = fingerprint
        decisions_by_identity[(decision["sk_id"], decision["range_id"])] = decision
    for action in actions:
        producer = decisions_by_identity[(action["sk_id"], action["range_id"])]
        for field in (
            "mapping_method",
            "mapping_confidence",
            "mapping_blockers",
            "candidate_binding_status",
            "graph_occurrence_fingerprint",
        ):
            if field in producer:
                action[field] = copy.deepcopy(producer[field])
    return {"round": round_info, "decisions": decisions, "actions": actions}


def _parse_round_id(name, round_id):
    match = re.fullmatch(
        rf"{re.escape(name)}-(AUTO|BASE|P([1-9][0-9]*)|FINAL)",
        round_id,
    )
    if match is None:
        raise ValueError(
            f"{name}: round_id={round_id!r} 必须属于候选并使用 "
            "AUTO/BASE/Pn/Rn/FINAL 格式"
        )
    token = match.group(1)
    if token == "AUTO":
        return {"kind": "AUTO", "number": None}
    if token == "BASE":
        return {"kind": "BASE", "number": None}
    if token == "FINAL":
        return {"kind": "FINAL", "number": None}
    return {"kind": token[0], "number": int(token[1:])}


def _validated_decision(
    location, decision, *, schema_version, require_source_fields=False
):
    if not isinstance(decision, dict):
        raise ValueError(f"{location} 必须是对象")
    for field in ("sk_id", "range_id", "classification", "action"):
        if not _is_text(decision.get(field)):
            raise ValueError(f"{location}.{field} 必须是非空字符串")
    classification = decision["classification"]
    action = decision["action"]
    allowed = CLASS_ACTIONS.get(classification)
    if classification in {"neutral", "regressed"}:
        allowed = {*allowed, "block"} if allowed is not None else None
    if allowed is None:
        raise ValueError(f"{location}.classification 枚举无效")
    if action not in allowed:
        expected = "/".join(sorted(allowed))
        raise ValueError(
            f"{location}: {classification} 必须对应 {expected}，不能使用 {action}"
        )
    item = copy.deepcopy(decision)
    identity = item.get("identity") if isinstance(item.get("identity"), dict) else {}
    source_scope = item.get("source_scope", identity.get("source_scope"))
    boundary = item.get("boundary", identity.get("boundary"))
    if not _is_text(source_scope):
        raise ValueError(f"{location}.source_scope 必须是非空字符串")
    if not isinstance(boundary, dict):
        raise ValueError(f"{location}.boundary 必须是对象")
    for field in ("start_op", "end_op"):
        if not _is_text(boundary.get(field)):
            raise ValueError(f"{location}.boundary.{field} 必须是非空字符串")
    interval_fields = ("source_file", "start_offset", "end_offset")
    present_interval_fields = [field for field in interval_fields if field in boundary]
    if present_interval_fields and len(present_interval_fields) != len(interval_fields):
        raise ValueError(
            f"{location}.boundary source interval 必须同时包含 "
            "source_file/start_offset/end_offset"
        )
    if present_interval_fields:
        source_file = boundary["source_file"]
        if (
            not _is_text(source_file)
            or Path(source_file).is_absolute()
            or ".." in Path(source_file).parts
        ):
            raise ValueError(f"{location}.boundary.source_file 必须是非空相对源码路径")
        start = boundary["start_offset"]
        end = boundary["end_offset"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(
                f"{location}.boundary offsets 必须满足 0 <= start_offset < end_offset"
            )
    if require_source_fields and (
        "source_scope" not in decision or "boundary" not in decision
    ):
        missing = "source_scope" if "source_scope" not in decision else "boundary"
        raise ValueError(f"{location}.{missing} 必须由 producer 显式提供")
    item["source_scope"] = source_scope
    item["boundary"] = copy.deepcopy(boundary)
    return item


def _validate_per_sk_decisions(name, decisions, *, schema_version):
    if not decisions:
        raise ValueError(f"{name}: per_sk_decisions 必须包含 analyzer 的逐 SK 判定")
    validated = []
    seen_ranges = set()
    seen_sks = set()
    for index, decision in enumerate(decisions):
        location = f"{name}.per_sk_decisions[{index}]"
        item = _validated_decision(location, decision, schema_version=schema_version)
        if item["range_id"] in seen_ranges:
            raise ValueError(f"{location}: per_sk_decisions range_id 重复")
        if item["sk_id"] in seen_sks:
            raise ValueError(
                f"{location}: per_sk_decisions 同一 sk_id 映射到多个 range"
            )
        seen_ranges.add(item["range_id"])
        seen_sks.add(item["sk_id"])
        validated.append(item)
    return sorted(validated, key=lambda item: (item["range_id"], item["sk_id"]))


def _validate_scope_actions(name, actions, decisions, *, schema_version):
    validated = []
    seen_ranges = {}
    for index, action in enumerate(actions):
        location = f"{name}.scope_actions[{index}]"
        action = _validated_decision(
            location,
            action,
            schema_version=schema_version,
            require_source_fields=True,
        )
        classification = action["classification"]
        actual_action = action["action"]
        range_id = action["range_id"]
        previous = seen_ranges.get(range_id)
        signature = (action["sk_id"], classification, actual_action)
        if previous is not None and previous != signature:
            raise ValueError(f"{location}: range_id={range_id} 存在冲突判定")
        if previous is not None:
            raise ValueError(f"{location}: range_id={range_id} 重复")
        seen_ranges[range_id] = signature
        validated.append(action)
    expected = {
        (item["sk_id"], item["range_id"]): (
            item["classification"],
            item["action"],
            item["source_scope"],
            item["boundary"],
        )
        for item in decisions
    }
    actual = {
        (item["sk_id"], item["range_id"]): (
            item["classification"],
            item["action"],
            item["source_scope"],
            item["boundary"],
        )
        for item in validated
    }
    if actual != expected or len(validated) != len(decisions):
        raise ValueError(
            f"{name}: scope_actions 必须逐项与唯一 per_sk_decisions 完全一致"
        )
    return sorted(validated, key=lambda item: (item["range_id"], item["sk_id"]))


def _structural_identity_key(decision):
    device_id = decision.get("device_id")
    model_id = decision.get("model_id")
    raw_sk_id = decision.get("raw_sk_id")
    if (
        isinstance(device_id, bool)
        or not isinstance(device_id, int)
        or isinstance(raw_sk_id, bool)
        or not isinstance(raw_sk_id, int)
        or model_id is None
        or str(model_id) == ""
    ):
        return None
    return f"device:{device_id}/model:{model_id}/sk:{raw_sk_id}"


def _validate_structural_mapping(decision, location):
    method = decision.get("mapping_method")
    confidence = decision.get("mapping_confidence")
    if method not in MAPPING_METHODS:
        raise ValueError(f"{location}.mapping_method 枚举无效")
    if confidence not in MAPPING_CONFIDENCES:
        raise ValueError(f"{location}.mapping_confidence 枚举无效")
    if (method, confidence) not in PERFORMANCE_EXACT_MAPPINGS and not (
        decision.get("classification") == "insufficient_evidence"
        and decision.get("action") in {"reprofile", "block"}
    ):
        raise ValueError(
            f"{location}: 非精确关联必须保持 insufficient_evidence，"
            "不得保留性能分类或源码动作"
        )


def _validate_structural_proof(name, analysis, decision, location):
    if (
        decision.get("mapping_method"),
        decision.get("mapping_confidence"),
    ) != ("kernel_projection_structural", "exact_projected_trace"):
        return None
    identity_key = _structural_identity_key(decision)
    proof_map = analysis.get("graph_alignment_proof")
    proof = proof_map.get(identity_key) if isinstance(proof_map, dict) else None
    fingerprint = (
        proof.get("graph_occurrence_fingerprint") if isinstance(proof, dict) else None
    )
    alternatives = (
        proof.get("alternative_solution_count_by_step")
        if isinstance(proof, dict)
        else None
    )
    valid = (
        identity_key is not None
        and decision.get("candidate_binding_status") == "bound"
        and decision.get("mapping_blockers") == []
        and isinstance(proof, dict)
        and proof.get("protocol") == "kernel_projection_trace_v2"
        and proof.get("mapping_method") == "kernel_projection_structural"
        and proof.get("mapping_confidence") == "exact_projected_trace"
        and isinstance(alternatives, dict)
        and len(alternatives) >= 3
        and all(type(value) is int and value == 0 for value in alternatives.values())
        and isinstance(proof.get("mapping_fingerprint"), str)
        and re.fullmatch(r"[0-9a-f]{64}", proof["mapping_fingerprint"]) is not None
        and isinstance(fingerprint, str)
        and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None
    )
    if not valid:
        raise ValueError(f"{location}: kernel projection 证明不完整或不唯一")
    return fingerprint


def _blocker(kind, required_action_zh, *, range_id=None, option=None):
    blocker = {"kind": kind, "required_action_zh": required_action_zh}
    if range_id is not None:
        blocker["range_id"] = range_id
    if option is not None:
        blocker["option"] = option
    return blocker


def _validate_accepted_evidence(option, value, evidence):
    if not isinstance(evidence, dict):
        return "缺少结构化 accepted_evidence，不能生成可运行挽救实验。"
    accepted_sources = {
        template.format(option=option) for template in ACCEPTED_SOURCE_TEMPLATES
    }
    if option in DCCI_OPTIONS:
        runtime_state = evidence.get("runtime_state")
        option_acceptance = evidence.get("option_acceptance")
        if not isinstance(runtime_state, dict) or runtime_state.get("state") not in {
            "enabled",
            "disabled",
            "unknown",
            "conflict",
        }:
            return "DCCI 挽救实验必须有合法的显式 runtime state 证据。"
        runtime_sources = runtime_state.get("sources")
        if not isinstance(runtime_sources, list) or not runtime_sources:
            return "DCCI 挽救实验必须有非空 runtime sources 证据。"
        seen_runtime_sources = set()
        known_states = set()
        for source in runtime_sources:
            if not isinstance(source, dict):
                return "DCCI runtime sources 必须使用 producer 的结构化来源。"
            source_path = source.get("source")
            source_state = source.get("state")
            if source_path not in DCCI_RUNTIME_SOURCES or source_state not in {
                "enabled",
                "disabled",
                "unknown",
            }:
                return "DCCI runtime sources 不是 producer 支持的实际来源。"
            if source_path in seen_runtime_sources:
                return "DCCI runtime sources 存在重复来源。"
            seen_runtime_sources.add(source_path)
            if source_state != "unknown":
                known_states.add(source_state)
        if len(known_states) > 1:
            derived_state = "conflict"
        elif known_states:
            derived_state = next(iter(known_states))
        else:
            derived_state = "unknown"
        if runtime_state.get("state") != derived_state:
            return "DCCI runtime state 与 sources 派生状态不一致。"
        if derived_state != "enabled":
            return (
                f"DCCI runtime sources 派生状态为 {derived_state}，不能运行挽救实验。"
            )
        if (
            not isinstance(option_acceptance, dict)
            or "accepted_value" not in option_acceptance
        ):
            return "DCCI 挽救实验缺少 exact accepted_value 证据。"
        if option_acceptance.get("source") not in accepted_sources:
            return (
                "DCCI option_acceptance.source 未指向对应 option 的 accepted_values。"
            )
        accepted_value = option_acceptance["accepted_value"]
    else:
        if "accepted_value" not in evidence:
            return "挽救实验缺少 exact accepted_value 证据。"
        if evidence.get("source") not in accepted_sources:
            return "accepted_evidence.source 未指向对应 option 的 accepted_values。"
        accepted_value = evidence["accepted_value"]
    if not _same_json(accepted_value, value):
        if option in DCCI_OPTIONS:
            return (
                "DCCI accepted_evidence 中的 exact accepted_value 与推荐 value 不一致。"
            )
        return "accepted_evidence 中的 exact accepted_value 与推荐 value 不一致。"
    return None


def _prune_entry(name, source_round_id, round_id, actions):
    range_ids = [action["range_id"] for action in actions]
    return {
        "id": round_id,
        "experiment_id": round_id,
        "candidate": name,
        "kind": "performance_prune",
        "round_kind": "performance_prune",
        "target_range_ids": range_ids,
        "performance_decision_range_ids": list(range_ids),
        "scope_actions": copy.deepcopy(actions),
        "options": {},
        "only_change_zh": "只从 SuperKernel 框定范围移除上述无收益或劣化范围，算子执行与依赖保持不变。",
        "expected_signal_zh": "保留范围继续通过正确性和独立 profile-vs-baseline 复分析。",
        "lifecycle_gates": [
            "correctness",
            "profile_vs_baseline_remeasurement",
            "clean_performance_profile",
            "read_only_reanalysis",
        ],
        "comparison_to": source_round_id,
        "display_name": f"{name} 性能裁剪",
    }


def _next_prune_round_id(name, round_info):
    if round_info["kind"] in {"AUTO", "BASE"}:
        return f"{name}-P1"
    if round_info["kind"] == "P":
        return f"{name}-P{round_info['number'] + 1}"
    return None


def _machine_source_interval(action):
    boundary = action.get("boundary") or {}
    required = ("source_file", "start_offset", "end_offset")
    if not all(field in boundary for field in required):
        return None
    return (
        boundary["source_file"],
        boundary["start_offset"],
        boundary["end_offset"],
    )


def _prune_batch_is_proven(actions):
    intervals = [_machine_source_interval(action) for action in actions]
    if any(interval is None for interval in intervals):
        return False, "缺少可机器比较的相对 source_file/start_offset/end_offset。"
    source_files = {interval[0] for interval in intervals}
    if len(source_files) != 1:
        return False, "所有范围必须位于同一 source_file。"
    for left_index, left in enumerate(intervals):
        for right in intervals[left_index + 1 :]:
            if left[1] < right[2] and right[1] < left[2]:
                return False, "源码区间相交、包含或使用同一边界。"
    return True, None


def _source_interval_is_exact(action):
    return (
        (action.get("mapping_method"), action.get("mapping_confidence"))
        == ("source_scope_map", "exact")
        and _machine_source_interval(action) is not None
        and not action.get("interval_unproven", False)
    )


def _source_mapping_completion(name, source_round_id, actions, *, automatic):
    fingerprints = sorted(
        {
            action["graph_occurrence_fingerprint"]
            for action in actions
            if _is_text(action.get("graph_occurrence_fingerprint"))
        }
    )
    range_ids = sorted({action["range_id"] for action in actions})
    sk_ids = sorted({action["sk_id"] for action in actions})
    entry = {
        "id": f"{name}-SMAP",
        "experiment_id": f"{name}-SMAP",
        "round_kind": "source_mapping_completion",
        "status": "proposed",
        "target_range_ids": range_ids,
        "target_sk_ids": sk_ids,
        "graph_occurrence_fingerprints": fingerprints,
        "expected_signal_zh": (
            "输出 analyzer 可重放的 source_scope_map_v2；exact_cover 进入后续源码动作，"
            "partial/skipped 仅记录并跳过该范围。"
        ),
        "lifecycle_gates": [
            "semantic_source_manifest",
            "unique_original_calibration_graph_projection",
            "three_step_unit_assignment",
            "source_scope_map_v2_exact_cover",
            "read_only_reanalysis",
        ],
        "comparison_to": source_round_id,
        "display_name": f"{name} winner 源码映射补全",
    }
    if automatic:
        entry.update(
            {
                "source_revision_role": "stable_source",
                "source_edit_policy": "temporary_marker_only_calibration",
                "only_change": {"graph_occurrence_fingerprints": fingerprints},
                "only_change_zh": (
                    "保持冻结的 automatic AOT 生产源码不变，仅在独立标定 worktree 中"
                    "插入临时语义 marker，补齐图 occurrence 到原始源码半开区间的映射。"
                ),
            }
        )
        entry["lifecycle_gates"][0:0] = [
            "stable_auto_source_snapshot",
            "marker_only_patch_reversibility",
        ]
    else:
        entry.update(
            {
                "source_revision_role": "stable_marker",
                "source_edit_policy": "stable_marker_calibration",
                "only_change": {"target_range_ids": range_ids},
                "only_change_zh": (
                    "保持 winner 的 scope、配置和业务计算不变，使用命名 scope 的"
                    "stable_marker 标定路径补齐 SK 到源码半开区间的映射。"
                ),
            }
        )
        entry["lifecycle_gates"].insert(0, "stable_marker_source_revision")
    return entry


def _whole_scope_clean_entry(name, analysis, source_path, actions):
    range_ids = [action["range_id"] for action in actions]
    source_mapping_skipped = [
        action["range_id"]
        for action in actions
        if action["classification"] != "insufficient_evidence"
        and not _source_interval_is_exact(action)
    ]
    classification_counts = {
        classification: sum(
            action["classification"] == classification for action in actions
        )
        for classification in ("beneficial", "neutral", "regressed")
    }
    return {
        "id": f"{name}-WHOLE-SCOPE-CLEAN",
        "experiment_id": f"{name}-WHOLE-SCOPE-CLEAN",
        "candidate": name,
        "kind": "whole_scope_promotion",
        "round_kind": "whole_scope_clean_validation",
        "promotion_mode": "whole_scope",
        "source_round_id": analysis["round_id"],
        "source_analysis_round_id": analysis["round_id"],
        "source_profiling_analysis_result": str(source_path),
        "target_range_ids": range_ids,
        "performance_decision_range_ids": list(range_ids),
        "classification_counts": classification_counts,
        "source_edit_policy": "forbidden",
        "local_decisions_are_diagnostic": True,
        "source_mapping_status": "settled",
        "source_mapping_skipped_range_ids": source_mapping_skipped,
        "options": {},
        "only_change_zh": "不修改源码、scope、配置或 workload；仅关闭诊断并重复运行同一完整候选。",
        "expected_signal_zh": (
            "至少三个独立 clean 候选进程相对冻结的五次 S0 达到端到端均值阈值，"
            "且 P90 与标准差不劣化。"
        ),
        "lifecycle_gates": [
            "correctness_passed",
            "full_profile_mapping_exact",
            "no_insufficient_evidence",
            "winner_source_mapping_settled",
            "candidate_identity_unchanged",
            "minimum_three_clean_candidate_processes",
            "mean_gain_threshold",
            "p90_non_regression",
            "stddev_non_regression",
        ],
        "comparison_to": analysis["round_id"],
        "display_name": f"{name} 整 scope clean 晋级验证",
    }


def _candidate_plan(
    name,
    analyses,
    validations,
    analysis_paths,
    source_range_optimization_requested=False,
):
    analysis = analyses[-1]
    validation = validations[-1]
    actions = validation["actions"]
    round_info = validation["round"]
    source_round_id = analysis["round_id"]
    prune_actions = [
        action
        for action in actions
        if action["classification"] in {"neutral", "regressed"}
        and action["action"] == "prune"
    ]
    matrix = []
    blockers = []
    source_mapping_targets = [
        action
        for action in actions
        if action["classification"] != "insufficient_evidence"
        and (action.get("mapping_method"), action.get("mapping_confidence"))
        in PERFORMANCE_EXACT_MAPPINGS
        and not _source_interval_is_exact(action)
    ]
    source_mapping_context = analysis["source_scope_mapping"]
    source_mapping_settled = source_mapping_context.get("status") == "exact"
    source_mapping_pending = (
        round_info["kind"] in {"AUTO", "BASE"}
        and bool(source_mapping_targets)
        and not source_mapping_settled
    )
    source_mapping_skipped = (
        [action["range_id"] for action in source_mapping_targets]
        if source_mapping_settled
        else []
    )
    if source_mapping_pending:
        matrix.append(
            _source_mapping_completion(
                name,
                source_round_id,
                source_mapping_targets,
                automatic=round_info["kind"] == "AUTO",
            )
        )
    if (
        source_range_optimization_requested
        and prune_actions
        and not source_mapping_pending
    ):
        prune_round_id = _next_prune_round_id(name, round_info)
        if prune_round_id is None:
            blockers.append(
                _blocker(
                    "ambiguous_prune_round",
                    f"源轮 {source_round_id} 之后不能可靠推导新的 P 轮；不得猜测轮次编号。",
                )
            )
        else:
            batch_proven, batch_reason = _prune_batch_is_proven(prune_actions)
            selected_actions = prune_actions if batch_proven else prune_actions[:1]
            if len(prune_actions) > 1 and not batch_proven:
                blockers.append(
                    _blocker(
                        "prune_batch_unproven",
                        f"多个范围不能在同一 P 轮安全裁剪：{batch_reason}"
                        "本轮只生成一个 range；其余范围等待 fresh P 分析。",
                    )
                )
            matrix.append(
                _prune_entry(name, source_round_id, prune_round_id, selected_actions)
            )

    for action in actions:
        if action["classification"] != "insufficient_evidence":
            continue
        verb = (
            "补采 profiling 并重新执行只读分析"
            if action["action"] == "reprofile"
            else "解除分析 blocker 后重新执行只读分析"
        )
        blockers.append(
            _blocker(
                f"insufficient_evidence_{action['action']}",
                f"range_id={action['range_id']} 证据不足；请{verb}，当前不得修改 scope。",
                range_id=action["range_id"],
            )
        )

    if source_range_optimization_requested and any(
        source_analysis["recommended_experiments"] for source_analysis in analyses
    ):
        blockers.append(
            _blocker(
                "post_base_option_recommendations_ignored",
                "Stage O 已冻结 Option；分析历史中的 Option 建议仅作诊断，"
                "不得生成 P/FINAL 实验。",
            )
        )

    for blocker in analysis["blockers"]:
        detail = blocker if isinstance(blocker, str) else _canonical_json(blocker)
        blockers.append(
            _blocker(
                "profiling_analysis_blocker",
                f"profiling 分析器报告 blocker：{detail}",
            )
        )
    whole_scope_eligible = (
        bool(actions)
        and round_info["kind"] in {"BASE", "AUTO"}
        and not source_mapping_pending
        and not analysis["blockers"]
        and all(
            action["classification"] != "insufficient_evidence"
            and (
                action.get("mapping_method"),
                action.get("mapping_confidence"),
            )
            in PERFORMANCE_EXACT_MAPPINGS
            and action.get("mapping_blockers", []) == []
            for action in actions
        )
    )
    if whole_scope_eligible:
        matrix.append(
            _whole_scope_clean_entry(
                name,
                analysis,
                analysis_paths[-1],
                actions,
            )
        )
    return {
        "candidate": name,
        "analysis_id": analysis.get("analysis_id"),
        "source_round_id": analysis.get("round_id"),
        "analysis_history": [item["round_id"] for item in analyses],
        "scope_strategy": {
            "strategy": "performance_evidence_driven",
            "reason_zh": (
                "所有 winner 在 BASE profiling 后先结算 SK 到源码半开区间映射。"
                "结算后，exact_cover 仅在源码范围优化已明确授权时可进入 P；"
                "未授权或未 exact 的范围不生成源码动作；"
                "整 scope 晋级仍由完整性能映射和端到端收益决定。"
            ),
            "source_range_optimization_status": (
                "requested" if source_range_optimization_requested else "not_requested"
            ),
            "source_mapping_status": (
                "required"
                if source_mapping_pending
                else "completed_with_skips"
                if source_mapping_skipped
                else "settled"
            ),
            "source_mapping_target_range_ids": [
                action["range_id"] for action in source_mapping_targets
            ],
            "source_mapping_skipped_range_ids": source_mapping_skipped,
        },
        "scope_actions": actions,
        "next_candidate_matrix": matrix,
        "blockers": blockers,
    }


def _normalized_history(name, analysis_value, path_value):
    analyses = analysis_value if isinstance(analysis_value, list) else [analysis_value]
    paths = path_value if isinstance(path_value, list) else [path_value]
    if not analyses or len(analyses) != len(paths):
        raise ValueError(f"{name}: analysis_paths 必须逐轮对应 profiling 分析历史")
    return analyses, paths


def _validate_analysis_history(name, analyses, validations):
    identity_fields = (
        "experiment_id",
        "source_revision",
        "baseline_profile_fingerprint",
        "baseline_config_fingerprint",
        "baseline_workload_fingerprint",
        "baseline_control_fingerprint",
        "workload_fingerprint",
        "control_fingerprint",
    )
    first = analyses[0]
    for analysis in analyses[1:]:
        for field in identity_fields:
            if analysis[field] != first[field]:
                raise ValueError(f"{name}: profiling 历史的 {field} 必须保持一致")
    analysis_agent_ids = [analysis["analysis_agent_id"] for analysis in analyses]
    if len(analysis_agent_ids) != len(set(analysis_agent_ids)):
        raise ValueError(f"{name}: profiling 历史的 analysis_agent_id 每轮必须唯一")

    if len(analyses) == 1:
        return
    expected_p = 1
    phase = "BASE"
    seen_round_ids = set()
    for index, (analysis, validation) in enumerate(zip(analyses, validations)):
        round_id = analysis["round_id"]
        if round_id in seen_round_ids:
            raise ValueError(f"{name}: profiling 历史 round_id={round_id!r} 重复")
        seen_round_ids.add(round_id)
        round_info = validation["round"]
        kind = round_info["kind"]
        if index == 0:
            if kind != "BASE":
                raise ValueError(f"{name}: profiling 历史必须从 {name}-BASE 开始")
            continue
        if phase in {"BASE", "P"} and kind == "P":
            if round_info["number"] != expected_p:
                raise ValueError(f"{name}: P 轮历史必须从 P1 连续递增")
            expected_p += 1
            phase = "P"
            continue
        if phase in {"BASE", "P"} and kind == "FINAL" and index == len(analyses) - 1:
            phase = "FINAL"
            continue
        raise ValueError(f"{name}: profiling 历史必须严格遵循 BASE -> P* -> FINAL")


def _normalize_source_range_optimization(value, candidate_names):
    if value is None:
        return set()
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise ValueError(
            "source_range_optimization 必须是候选名称 collection，不能是字符串或标量"
        )
    names = list(value)
    if not all(_is_text(name) for name in names):
        raise ValueError("source_range_optimization 中每个候选名称必须是非空字符串")
    if len(names) != len(set(names)):
        raise ValueError("source_range_optimization 候选名称不得重复")
    unknown = sorted(set(names) - set(candidate_names))
    if unknown:
        raise ValueError(
            "source_range_optimization 包含未知候选：" + ", ".join(unknown)
        )
    return set(names)


def build_strategy(
    profiling_by_name,
    analysis_paths=None,
    source_range_optimization=None,
):
    if not isinstance(profiling_by_name, dict) or not profiling_by_name:
        raise ValueError("至少需要一个 NAME=profiling-analysis-result.json 输入")
    for name in profiling_by_name:
        if not _is_text(name):
            raise ValueError("profiling 分析输入名称必须是非空字符串")
    source_range_optimization = _normalize_source_range_optimization(
        source_range_optimization,
        profiling_by_name,
    )
    if analysis_paths is None:
        raise ValueError(
            "analysis_paths 不能为空；生成可运行 P/FINAL 计划前必须执行只读重分析。"
        )
    if not isinstance(analysis_paths, dict) or set(analysis_paths) != set(
        profiling_by_name
    ):
        raise ValueError("analysis_paths 必须逐项对应 profiling 分析输入")
    candidates = {}
    candidate_names = set()
    for name in sorted(profiling_by_name):
        analyses, paths = _normalized_history(
            name, profiling_by_name[name], analysis_paths[name]
        )
        validations = []
        for analysis, path in zip(analyses, paths):
            validation = _validate_analysis(name, analysis)
            regenerated = _run_read_only_reanalysis(name, analysis, path)
            _validate_analysis(name, regenerated)
            _validate_reanalysis_semantics(name, analysis, regenerated)
            validations.append(validation)
        source_range_optimization_requested = name in source_range_optimization
        if not source_range_optimization_requested and any(
            validation["round"]["kind"] in {"P", "FINAL"} for validation in validations
        ):
            raise ValueError(
                f"{name}: P/FINAL profiling 历史要求通过 "
                "source_range_optimization 显式授权"
            )
        _validate_analysis_history(name, analyses, validations)
        candidate_name = analyses[-1]["candidate_name"]
        if candidate_name in candidate_names:
            raise ValueError(f"candidate_name={candidate_name!r} 重复")
        candidate_names.add(candidate_name)
        candidates[name] = _candidate_plan(
            name,
            analyses,
            validations,
            paths,
            source_range_optimization_requested=source_range_optimization_requested,
        )
    return {
        "schema_version": "1.0",
        "evidence_gaps": [],
        "options_authority_zh": "Stage O 后 Option map 冻结；P/FINAL 不生成或消费任何 Option 调优建议。",
        "candidates": candidates,
    }


def _markdown_cell(value, code=False):
    if isinstance(value, (dict, list)):
        value = _canonical_json(value)
    text = html.escape(str(value), quote=False)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "<br>").replace("|", r"\|")
    if not code:
        return text.replace("`", r"\`")
    longest_tick_run = 0
    tick_run = 0
    for character in text:
        if character == "`":
            tick_run += 1
            longest_tick_run = max(longest_tick_run, tick_run)
        else:
            tick_run = 0
    fence = "`" * (longest_tick_run + 1)
    return f"{fence}{text}{fence}"


def _markdown_heading(value):
    return _markdown_cell(value).replace("<br>", " ")


def _markdown(report):
    lines = ["# SuperKernel 性能范围实验计划", ""]
    candidates = sorted(
        report["candidates"].items(),
        key=lambda item: (item[0], item[1].get("source_round_id", "")),
    )
    for name, candidate in candidates:
        lines.extend(
            [
                f"## {_markdown_heading(name)}",
                "",
                f"范围策略：{_markdown_cell(candidate['scope_strategy']['reason_zh'])}",
                "",
                "| 轮次 | 类型 | 目标范围 | 唯一变化 | 预期信号 | 生命周期门禁 |",
                "|---|---|---|---|---|---|",
            ]
        )
        matrix = sorted(
            candidate["next_candidate_matrix"],
            key=lambda item: (
                {
                    "whole_scope_clean_validation": 0,
                    "performance_prune": 1,
                }.get(item["round_kind"], 2),
                item["id"],
                item.get("source_experiment_id", ""),
            ),
        )
        for item in matrix:
            kind_zh = {
                "whole_scope_clean_validation": "整 scope clean 晋级验证",
                "performance_prune": "性能裁剪",
                "source_mapping_completion": "AUTO 源码映射补全",
            }.get(item["round_kind"], item["round_kind"])
            only_change = item.get("only_change", item.get("only_change_zh"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(item["id"]),
                        kind_zh,
                        _markdown_cell(item["target_range_ids"], code=True),
                        _markdown_cell(only_change, code=True),
                        _markdown_cell(item["expected_signal_zh"]),
                        _markdown_cell(item["lifecycle_gates"], code=True),
                    ]
                )
                + " |"
            )
        lines.append("")
        if candidate["blockers"]:
            lines.extend(["### 阻塞项", ""])
            for blocker in sorted(
                candidate["blockers"], key=lambda item: _canonical_json(item)
            ):
                target = blocker.get("range_id", blocker.get("option", blocker["kind"]))
                lines.append(
                    f"- {_markdown_cell(target, code=True)}："
                    f"{_markdown_cell(blocker['required_action_zh'])}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_named_analyses(parser, named_paths):
    if not named_paths:
        parser.error("必须提供 --profiling-analysis NAME=PATH")
    analyses = {}
    analysis_paths = {}
    try:
        for name, path in named_paths:
            analysis = _load_json(path)
            candidate_name = analysis.get("candidate_name")
            if candidate_name != name:
                raise ValueError(
                    f"输入名称 {name!r} 与结果 candidate_name {candidate_name!r} 不一致"
                )
            analyses.setdefault(name, []).append(analysis)
            analysis_paths.setdefault(name, []).append(path)
        return analyses, analysis_paths
    except ValueError as error:
        parser.error(str(error))


def _resolve_path(path):
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"无法解析路径 {path}: {error}") from error


def _load_artifact_json(path, label):
    try:
        value = json.loads(
            path.read_text(errors="replace"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"禁止非标准 JSON 常量 {constant}")
            ),
        )
    except OSError as error:
        raise ValueError(f"无法读取 {label} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} 不是有效 JSON：{path}:{error.lineno}:{error.colno} {error.msg}"
        ) from error
    except RecursionError as error:
        raise ValueError(f"{label} JSON 嵌套过深或递归结构无效：{path}") from error
    except ValueError as error:
        raise ValueError(f"{label} 不是标准 JSON：{path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON 对象：{path}")
    _validate_json_value(value, label)
    return value


def _find_kernel_details(path):
    if path.is_file():
        return path
    try:
        matches = sorted(path.rglob("kernel_details.csv"))
    except OSError as error:
        raise ValueError(f"无法扫描 profiling 目录 {path}: {error}") from error
    if not matches:
        raise ValueError(f"{path}: 未找到 kernel_details.csv")
    if len(matches) > 1:
        raise ValueError(
            f"{path}: 找到多个 kernel_details.csv，必须提供唯一 profiling 目录"
        )
    return matches[0]


def _profile_fingerprint(path):
    kernel_details = _find_kernel_details(path)
    try:
        return hashlib.sha256(kernel_details.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(
            f"无法读取 profiling 数据 {kernel_details}: {error}"
        ) from error


def _analysis_input_path(name, analysis, analysis_path, field):
    value = analysis["inputs"][field]
    try:
        return (analysis_path.parent / value).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"{name}: 无法解析 inputs.{field}={value!r}: {error}"
        ) from error


def _validate_analysis_artifacts(name, analysis, analysis_path):
    analysis_path = _resolve_path(analysis_path)
    paths = {
        field: _analysis_input_path(name, analysis, analysis_path, field)
        for field in (
            "baseline_profile",
            "candidate_profile",
            "baseline_config",
            "candidate_config",
            "baseline_workload",
            "candidate_workload",
            "declared_change_set",
        )
    }
    baseline_config = _load_artifact_json(paths["baseline_config"], "baseline config")
    candidate_config = _load_artifact_json(
        paths["candidate_config"], "candidate config"
    )
    baseline_workload = _load_artifact_json(
        paths["baseline_workload"], "baseline workload"
    )
    candidate_workload = _load_artifact_json(
        paths["candidate_workload"], "candidate workload"
    )
    declared_change_set = _load_artifact_json(
        paths["declared_change_set"], "declared change set"
    )
    if not _same_json(declared_change_set, analysis["declared_change_set"]):
        raise ValueError(
            f"{name}: declared_change_set 文件内容与 profiling report 不一致"
        )
    actual = {
        "baseline_profile_fingerprint": _profile_fingerprint(paths["baseline_profile"]),
        "candidate_profile_fingerprint": _profile_fingerprint(
            paths["candidate_profile"]
        ),
        "baseline_config_fingerprint": _schema_canonical_sha256(
            baseline_config, analysis["schema_version"]
        ),
        "candidate_config_fingerprint": _schema_canonical_sha256(
            candidate_config, analysis["schema_version"]
        ),
        "baseline_workload_fingerprint": _workload_fingerprint(
            baseline_workload, analysis["schema_version"]
        ),
        "candidate_workload_fingerprint": _workload_fingerprint(
            candidate_workload, analysis["schema_version"]
        ),
    }
    for field, fingerprint in actual.items():
        if analysis[field] != fingerprint:
            raise ValueError(f"{name}: {field} 与原始 artifact 内容不一致")


def _analyzer_script_path():
    return (
        Path(__file__).resolve().parents[2]
        / "superkernel-fusion-performance-analysis"
        / "scripts"
        / "analyze_fusion_performance.py"
    )


def _validated_reanalysis_thresholds(name, analysis):
    thresholds = analysis.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{name}: thresholds 必须是对象，无法执行只读重分析")
    expected_fields = {
        "min_relative_change_pct",
        "min_absolute_change_us",
        "min_occurrences",
    }
    if set(thresholds) != expected_fields:
        raise ValueError(f"{name}: thresholds 必须且只能包含 {sorted(expected_fields)}")
    relative = thresholds["min_relative_change_pct"]
    absolute = thresholds["min_absolute_change_us"]
    occurrences = thresholds["min_occurrences"]
    if (
        isinstance(relative, bool)
        or not isinstance(relative, (int, float))
        or not math.isfinite(relative)
        or relative < 0
    ):
        raise ValueError(f"{name}: min_relative_change_pct 必须是有限非负数")
    if (
        isinstance(absolute, bool)
        or not isinstance(absolute, (int, float))
        or not math.isfinite(absolute)
        or absolute < 0
    ):
        raise ValueError(f"{name}: min_absolute_change_us 必须是有限非负数")
    if (
        isinstance(occurrences, bool)
        or not isinstance(occurrences, int)
        or occurrences < 1
    ):
        raise ValueError(f"{name}: min_occurrences 必须是大于等于 1 的整数")
    return thresholds


def _run_read_only_reanalysis(name, analysis, analysis_path):
    analysis_path = _resolve_path(analysis_path)
    stored_analysis = _load_json(analysis_path)
    if not _same_json(stored_analysis, analysis):
        raise ValueError(
            f"{name}: analysis_paths 指向的报告与传入 profiling report 不一致"
        )
    _validate_analysis_artifacts(name, analysis, analysis_path)

    analyzer = _analyzer_script_path()
    if not analyzer.is_file():
        raise ValueError(f"{name}: 只读重分析器不存在：{analyzer}")
    thresholds = _validated_reanalysis_thresholds(name, analysis)
    command = [sys.executable, str(analyzer)]
    required_inputs, optional_inputs = _analysis_input_fields(
        analysis["schema_version"]
    )
    for field in (*required_inputs, *optional_inputs):
        value = analysis["inputs"].get(field)
        if value is not None:
            command.extend(
                [
                    f"--{field.replace('_', '-')}",
                    str(_analysis_input_path(name, analysis, analysis_path, field)),
                ]
            )
    for field in (
        "candidate_name",
        "experiment_id",
        "round_id",
        "analysis_agent_id",
        "source_revision",
    ):
        command.extend([f"--{field.replace('_', '-')}", analysis[field]])
    command.extend(
        [
            "--min-relative-change-pct",
            str(thresholds["min_relative_change_pct"]),
            "--min-absolute-change-us",
            str(thresholds["min_absolute_change_us"]),
            "--min-occurrences",
            str(thresholds["min_occurrences"]),
        ]
    )

    with tempfile.TemporaryDirectory(prefix="sk-strategy-reanalysis-") as directory:
        output_path = Path(directory) / "profiling-analysis-result.json"
        command.extend(["--json-out", str(output_path)])
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except OSError as error:
            raise ValueError(f"{name}: 无法启动只读重分析：{error}") from error
        if completed.returncode != 0:
            stderr_lines = [
                line.strip() for line in completed.stderr.splitlines() if line.strip()
            ]
            detail = stderr_lines[-1] if stderr_lines else "子进程未提供错误详情"
            detail = re.sub("traceback", "子进程异常", detail, flags=re.IGNORECASE)[
                :500
            ]
            raise ValueError(
                f"{name}: 只读重分析失败（退出码 {completed.returncode}）：{detail}"
            )
        try:
            return _load_json(output_path)
        except ValueError as error:
            raise ValueError(f"{name}: 只读重分析输出无效：{error}") from error


def _validate_reanalysis_semantics(name, original, regenerated):
    for field in REANALYSIS_SEMANTIC_FIELDS:
        if not _same_json(original.get(field), regenerated.get(field)):
            raise ValueError(
                f"{name}: 只读重分析语义字段 {field} 与输入报告不一致，拒绝生成计划。"
            )


def _validate_cli_paths(named_paths, json_out, markdown_out):
    input_paths = [path for _, path in named_paths]
    outputs = [path for path in (json_out, markdown_out) if path is not None]
    resolved_outputs = [_resolve_path(path) for path in outputs]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise ValueError("--json-out 与 --markdown-out 必须使用不同路径")
    resolved_inputs = {_resolve_path(path) for path in input_paths}
    if any(path in resolved_inputs for path in resolved_outputs):
        raise ValueError("输出路径不能覆盖 profiling 分析输入")
    for path in outputs:
        if path.exists() and path.is_dir():
            raise ValueError(f"输出路径不能是目录：{path}")


def _validate_cli_artifact_output_conflicts(
    analyses, analysis_paths, json_out, markdown_out
):
    outputs = {
        _resolve_path(path) for path in (json_out, markdown_out) if path is not None
    }
    if not outputs:
        return
    artifacts = set()
    for name, analysis_value in analyses.items():
        history, paths = _normalized_history(name, analysis_value, analysis_paths[name])
        for analysis, analysis_path in zip(history, paths):
            resolved_path = _resolve_path(analysis_path)
            required_inputs, _ = _analysis_input_fields(analysis["schema_version"])
            for field in required_inputs:
                artifacts.add(
                    _analysis_input_path(name, analysis, resolved_path, field)
                )
    if outputs & artifacts:
        raise ValueError("输出路径不能覆盖 profiling report 引用的原始 artifact")


def _write_output_temp(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(content)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _write_outputs(json_out, json_content, markdown_out, markdown_content):
    temporary_paths = []
    markdown_backup = None
    markdown_committed = False
    try:
        json_temp = None
        if json_out is not None:
            json_temp = _write_output_temp(json_out, json_content)
            temporary_paths.append(json_temp)
        markdown_temp = None
        if markdown_out is not None:
            markdown_temp = _write_output_temp(markdown_out, markdown_content)
            temporary_paths.append(markdown_temp)
            if markdown_out.exists():
                markdown_backup = _write_output_temp(
                    markdown_out, markdown_out.read_text()
                )
                temporary_paths.append(markdown_backup)

        if markdown_temp is not None:
            os.replace(markdown_temp, markdown_out)
            markdown_committed = True
        if json_temp is not None:
            os.replace(json_temp, json_out)
    except BaseException:
        if markdown_committed:
            if markdown_backup is not None:
                os.replace(markdown_backup, markdown_out)
            else:
                markdown_out.unlink(missing_ok=True)
        raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiling-analysis",
        action="append",
        default=[],
        type=_candidate_arg,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--source-range-optimization",
        action="append",
        default=[],
        metavar="NAME",
        help=("显式授权指定候选进入可选 P/FINAL 源码范围优化；可为多个候选重复传入"),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    try:
        _validate_cli_paths(
            args.profiling_analysis,
            args.json_out,
            args.markdown_out,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    analyses, analysis_paths = _load_named_analyses(parser, args.profiling_analysis)
    try:
        report = build_strategy(
            analyses,
            analysis_paths=analysis_paths,
            source_range_optimization=args.source_range_optimization,
        )
        _validate_cli_artifact_output_conflicts(
            analyses,
            analysis_paths,
            args.json_out,
            args.markdown_out,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        output = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        markdown = _markdown(report)
        _write_outputs(
            args.json_out,
            output + "\n",
            args.markdown_out,
            markdown,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
