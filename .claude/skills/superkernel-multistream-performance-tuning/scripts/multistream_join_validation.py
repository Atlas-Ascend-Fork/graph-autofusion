#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare bound critical-path analyses to validate join-shortening mechanisms."""

import argparse
import hashlib
import json
from pathlib import Path

import multistream_critical_path


SCHEMA = "superkernel-multistream-join-validation-v1"


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


def _validate_analysis(value, label):
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != multistream_critical_path.ANALYSIS_SCHEMA
    ):
        raise ValueError(f"{label} is not a critical path analysis")
    unsigned = {
        key: item for key, item in value.items() if key != "analysis_fingerprint"
    }
    if value.get("analysis_fingerprint") != multistream_critical_path.fingerprint(
        unsigned
    ):
        raise ValueError(f"{label} fingerprint mismatch")
    return value


def _summary_delta(before, after):
    return {key: after[key] - before[key] for key in ("p50", "p90", "min", "max")}


def compare(trial_id, action_manifest_fingerprint, baseline, candidate):
    baseline = _validate_analysis(baseline, "baseline analysis")
    candidate = _validate_analysis(candidate, "candidate analysis")
    if baseline["request_fingerprint"] != candidate["request_fingerprint"]:
        raise ValueError("baseline and candidate request fingerprints differ")
    if not isinstance(trial_id, str) or not trial_id.strip():
        raise ValueError("trial_id must be non-empty")
    if not isinstance(
        action_manifest_fingerprint, str
    ) or not action_manifest_fingerprint.startswith("sha256:"):
        raise ValueError("action_manifest_fingerprint must be a SHA256 fingerprint")
    before_targets = {item["join_id"]: item for item in baseline["targets"]}
    after_targets = {item["join_id"]: item for item in candidate["targets"]}
    if set(before_targets) != set(after_targets):
        raise ValueError("baseline and candidate join targets differ")
    targets = []
    for join_id in sorted(before_targets):
        before, after = before_targets[join_id], after_targets[join_id]
        before_alignment = [item["alignment_id"] for item in before["occurrences"]]
        after_alignment = [item["alignment_id"] for item in after["occurrences"]]
        if before_alignment != after_alignment or len(before_alignment) < 3:
            raise ValueError(f"join {join_id} alignment is incomplete or differs")
        ready_delta = _summary_delta(before["join_ready_us"], after["join_ready_us"])
        stall_delta = _summary_delta(before["join_stall_us"], after["join_stall_us"])
        if ready_delta["p50"] >= 0:
            reason = "join_not_advanced"
        elif ready_delta["p90"] > 0:
            reason = "join_p90_regressed"
        elif stall_delta["p90"] > 0:
            reason = "join_stall_regressed"
        else:
            reason = None
        targets.append(
            {
                "join_id": join_id,
                "alignment_ids": before_alignment,
                "join_ready_delta_us": ready_delta,
                "join_stall_delta_us": stall_delta,
                "critical_branch_before": before["critical_branch_stage_id"],
                "critical_branch_after": after["critical_branch_stage_id"],
                "reason": reason,
            }
        )
    result = {
        "schema_version": SCHEMA,
        "trial_id": trial_id,
        "request_fingerprint": baseline["request_fingerprint"],
        "action_manifest_fingerprint": action_manifest_fingerprint,
        "baseline_analysis_fingerprint": baseline["analysis_fingerprint"],
        "candidate_analysis_fingerprint": candidate["analysis_fingerprint"],
        "decision": "validated"
        if any(item["reason"] is None for item in targets)
        else "not_effective",
        "targets": targets,
    }
    result["validation_fingerprint"] = fingerprint(result)
    return result


def validate(value):
    required = {
        "schema_version",
        "trial_id",
        "request_fingerprint",
        "action_manifest_fingerprint",
        "baseline_analysis_fingerprint",
        "candidate_analysis_fingerprint",
        "decision",
        "targets",
        "validation_fingerprint",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != SCHEMA
    ):
        raise ValueError(f"join validation must use {SCHEMA}")
    unsigned = {
        key: item for key, item in value.items() if key != "validation_fingerprint"
    }
    if value["validation_fingerprint"] != fingerprint(unsigned):
        raise ValueError("join validation fingerprint mismatch")
    if value["decision"] not in {"validated", "not_effective"}:
        raise ValueError("join validation decision is invalid")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare", "validate"))
    parser.add_argument("--trial-id")
    parser.add_argument("--action-manifest-fingerprint")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            if not all(
                (
                    args.trial_id,
                    args.action_manifest_fingerprint,
                    args.baseline,
                    args.candidate,
                    args.out,
                )
            ):
                raise ValueError(
                    "compare requires trial identity, baseline, candidate, and out"
                )
            if args.out.exists():
                raise ValueError(f"output already exists: {args.out}")
            result = compare(
                args.trial_id,
                args.action_manifest_fingerprint,
                json.loads(args.baseline.read_text()),
                json.loads(args.candidate.read_text()),
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        else:
            if not args.evidence:
                raise ValueError("validate requires --evidence")
            result = validate(json.loads(args.evidence.read_text()))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
