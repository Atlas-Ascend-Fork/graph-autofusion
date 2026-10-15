#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Discover and audit complete multi-stream parallelism-screening coverage."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import multistream_contract
import multistream_trace_analysis


DISCOVERY_SCHEMA = "superkernel-multistream-opportunity-discovery-v1"
COVERAGE_SCHEMA = "superkernel-multistream-opportunity-coverage-v1"
EXACT_MAPPINGS = {
    ("kernel_projection_structural", "exact_projected_trace"),
    ("source_scope_map", "exact"),
}


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value):
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _unsigned(value, field):
    return {key: item for key, item in value.items() if key != field}


def _rooted(root, value, label):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _multi_stream_evidence(decision):
    original = decision.get("original")
    if not isinstance(original, dict):
        return None
    example = original.get("example_occurrence")
    if not isinstance(example, dict):
        return None
    analysis = example.get("multi_stream_analysis")
    if (
        not isinstance(analysis, dict)
        or analysis.get("multi_stream_detected") is not True
    ):
        return None
    stream_count = example.get("stream_count")
    if (
        isinstance(stream_count, bool)
        or not isinstance(stream_count, int)
        or stream_count < 2
    ):
        return None
    stream_ids = example.get("stream_ids")
    if (
        not isinstance(stream_ids, list)
        or len(stream_ids) != stream_count
        or len(set(stream_ids)) != stream_count
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in stream_ids
        )
    ):
        return None
    return {
        "stream_count": stream_count,
        "stream_ids": sorted(stream_ids),
    }


def discover(analysis_path, artifact_root=None):
    analysis_path = Path(analysis_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else analysis_path.parent
    analysis_path = _rooted(root, analysis_path, "profiling analysis")
    analysis = json.loads(analysis_path.read_text())
    if not isinstance(analysis, dict) or analysis.get("schema_version") != "1.2":
        raise ValueError("profiling analysis must use schema_version 1.2")
    content_fp = analysis.get("analysis_content_fingerprint")
    if content_fp != multistream_contract._analysis_fingerprint(analysis):
        raise ValueError("profiling analysis content fingerprint mismatch")
    decisions = analysis.get("per_sk_decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("profiling analysis per_sk_decisions must be non-empty")

    eligible = []
    excluded = []
    seen = set()
    seen_ranges = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"per_sk_decisions[{index}] must be an object")
        range_id = decision.get("range_id")
        occurrence_fp = decision.get("graph_occurrence_fingerprint")
        if (
            not isinstance(range_id, str)
            or not range_id
            or not isinstance(occurrence_fp, str)
            or not occurrence_fp
        ):
            excluded.append({"index": index, "reason": "missing_stable_identity"})
            continue
        identity = (range_id, occurrence_fp)
        if identity in seen:
            raise ValueError(f"duplicate profiling decision identity: {range_id}")
        seen.add(identity)
        if range_id in seen_ranges:
            raise ValueError(f"ambiguous profiling decision range_id: {range_id}")
        seen_ranges.add(range_id)
        reason = None
        mapping = (decision.get("mapping_method"), decision.get("mapping_confidence"))
        if mapping not in EXACT_MAPPINGS:
            reason = "mapping_not_performance_exact"
        elif decision.get("classification") == "insufficient_evidence":
            reason = "net_effect_insufficient_evidence"
        elif (
            not isinstance(decision.get("candidate_occurrence_count"), int)
            or decision["candidate_occurrence_count"] < 3
        ):
            reason = "candidate_occurrences_below_three"
        else:
            original = decision.get("original")
            interval = (
                original.get("interval_us") if isinstance(original, dict) else None
            )
            if (
                not isinstance(interval, dict)
                or not isinstance(interval.get("count"), int)
                or interval["count"] < 3
            ):
                reason = "baseline_occurrences_below_three"
        multistream = _multi_stream_evidence(decision)
        if reason is None and multistream is None:
            reason = "not_proven_multistream"
        if reason is not None:
            excluded.append(
                {
                    "range_id": range_id,
                    "graph_occurrence_fingerprint": occurrence_fp,
                    "reason": reason,
                }
            )
            continue
        eligible.append(
            {
                "range_id": range_id,
                "graph_occurrence_fingerprint": occurrence_fp,
                "net_effect": decision["classification"],
                "parallelism_effect": "unknown",
                "optimization_status": "blocked",
                "child_count": _integer(
                    decision.get("child_count"), f"decision {range_id}.child_count", 1
                ),
                "baseline_occurrence_count": decision["original"]["interval_us"][
                    "count"
                ],
                "candidate_occurrence_count": decision["candidate_occurrence_count"],
                **multistream,
            }
        )

    eligible.sort(
        key=lambda item: (item["range_id"], item["graph_occurrence_fingerprint"])
    )
    result = {
        "schema_version": DISCOVERY_SCHEMA,
        "profiling_analysis": {
            "path": _relative(root, analysis_path),
            "file_fingerprint": file_fingerprint(analysis_path),
            "analysis_content_fingerprint": content_fp,
        },
        "selection_policy": {
            "net_effects": ["beneficial", "neutral", "regressed"],
            "mapping_confidence": ["exact", "exact_projected_trace"],
            "minimum_occurrences_per_side": 3,
            "minimum_stream_count": 2,
            "beneficial_targets_must_be_screened": True,
        },
        "eligible_targets": eligible,
        "excluded_targets": excluded,
        "summary": {
            "profiling_decision_count": len(decisions),
            "eligible_target_count": len(eligible),
            "beneficial_target_count": sum(
                item["net_effect"] == "beneficial" for item in eligible
            ),
            "neutral_target_count": sum(
                item["net_effect"] == "neutral" for item in eligible
            ),
            "regressed_target_count": sum(
                item["net_effect"] == "regressed" for item in eligible
            ),
            "excluded_target_count": len(excluded),
        },
    }
    result["discovery_fingerprint"] = fingerprint(result)
    return result


def audit_coverage(discovery_path, trace_analysis_path, artifact_root=None):
    discovery_path = Path(discovery_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else discovery_path.parent
    discovery_path = _rooted(root, discovery_path, "opportunity discovery")
    trace_path = _rooted(root, trace_analysis_path, "trace analysis")
    discovery = json.loads(discovery_path.read_text())
    if discovery.get("schema_version") != DISCOVERY_SCHEMA or discovery.get(
        "discovery_fingerprint"
    ) != fingerprint(_unsigned(discovery, "discovery_fingerprint")):
        raise ValueError("opportunity discovery is invalid")
    analysis_binding = discovery.get("profiling_analysis", {})
    analysis_path = _rooted(
        root, analysis_binding.get("path"), "discovery profiling analysis"
    )
    if file_fingerprint(analysis_path) != analysis_binding.get("file_fingerprint"):
        raise ValueError("discovery profiling analysis file changed")
    analysis = json.loads(analysis_path.read_text())
    if analysis.get("analysis_content_fingerprint") != analysis_binding.get(
        "analysis_content_fingerprint"
    ) or analysis.get(
        "analysis_content_fingerprint"
    ) != multistream_contract._analysis_fingerprint(analysis):
        raise ValueError("discovery profiling analysis content changed")
    trace = json.loads(trace_path.read_text())
    if trace.get("schema_version") != multistream_trace_analysis.ANALYSIS_SCHEMA:
        raise ValueError("trace analysis schema is invalid")
    if trace.get("analysis_fingerprint") != multistream_trace_analysis.fingerprint(
        _unsigned(trace, "analysis_fingerprint")
    ):
        raise ValueError("trace analysis fingerprint mismatch")
    multistream_trace_analysis.validate_bound_analysis(
        trace_path,
        root,
        expected_request_fingerprint=trace.get("request_fingerprint"),
        expected_trial_id=trace.get("trial_id"),
    )
    expected = {
        (item["range_id"], item["graph_occurrence_fingerprint"])
        for item in discovery["eligible_targets"]
    }
    observed = {
        (item.get("range_id"), item.get("graph_occurrence_fingerprint"))
        for item in trace.get("targets", [])
        if isinstance(item, dict)
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    latent = sorted(
        item["range_id"]
        for item in trace.get("targets", [])
        if isinstance(item, dict) and item.get("latent_opportunity") is True
    )
    complete = not missing and not extra and not trace.get("blockers")
    result = {
        "schema_version": COVERAGE_SCHEMA,
        "discovery": {
            "path": _relative(root, discovery_path),
            "file_fingerprint": file_fingerprint(discovery_path),
            "discovery_fingerprint": discovery["discovery_fingerprint"],
        },
        "trace_analysis": {
            "path": _relative(root, trace_path),
            "file_fingerprint": file_fingerprint(trace_path),
            "analysis_fingerprint": trace["analysis_fingerprint"],
        },
        "complete": complete,
        "expected_target_count": len(expected),
        "observed_target_count": len(observed),
        "missing_targets": [
            {"range_id": range_id, "graph_occurrence_fingerprint": occurrence_fp}
            for range_id, occurrence_fp in missing
        ],
        "extra_targets": [
            {"range_id": range_id, "graph_occurrence_fingerprint": occurrence_fp}
            for range_id, occurrence_fp in extra
        ],
        "trace_blockers": trace.get("blockers", []),
        "latent_opportunity_range_ids": latent,
    }
    result["coverage_fingerprint"] = fingerprint(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("discover")
    create.add_argument("--profiling-analysis", type=Path, required=True)
    create.add_argument("--artifact-root", type=Path)
    create.add_argument("--out", type=Path, required=True)
    coverage = commands.add_parser("audit-coverage")
    coverage.add_argument("--discovery", type=Path, required=True)
    coverage.add_argument("--trace-analysis", type=Path, required=True)
    coverage.add_argument("--artifact-root", type=Path)
    coverage.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            discover(args.profiling_analysis, args.artifact_root)
            if args.command == "discover"
            else audit_coverage(args.discovery, args.trace_analysis, args.artifact_root)
        )
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        _atomic_json(args.out, result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
