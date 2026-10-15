#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate that an event/stage source action changed logical dispatch as authorized."""

import argparse
import hashlib
import json
from pathlib import Path

import multistream_source_transform


SCHEMA = "superkernel-multistream-event-stage-dispatch-v1"
MIN_OCCURRENCES = 3
OBSERVATION_KINDS = {"event_notify", "event_wait", "stage"}


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


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _observations(value, label):
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{label} must contain at least two logical observations")
    result = {}
    roles = set()
    for index, item in enumerate(value):
        fields = {"logical_id", "kind", "stream_role", "dispatch_ordinal"}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"{label}[{index}] fields are invalid")
        logical_id = _text(item["logical_id"], f"{label}[{index}].logical_id")
        kind = _text(item["kind"], f"{label}[{index}].kind")
        role = _text(item["stream_role"], f"{label}[{index}].stream_role")
        ordinal = item["dispatch_ordinal"]
        if kind not in OBSERVATION_KINDS:
            raise ValueError(f"{label}[{index}].kind is invalid")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError(
                f"{label}[{index}].dispatch_ordinal must be non-negative integer"
            )
        key = (kind, logical_id)
        if key in result:
            raise ValueError(f"{label} has duplicate logical observation")
        result[key] = {
            "logical_id": logical_id,
            "kind": kind,
            "stream_role": role,
            "dispatch_ordinal": ordinal,
        }
        roles.add(role)
    if len(roles) < 2:
        raise ValueError(f"{label} does not prove a multi-stream window")
    return result


def build(trial_id, request_fingerprint, action_manifest, occurrences):
    if (
        action_manifest.get("schema_version")
        != multistream_source_transform.ACTION_MANIFEST_SCHEMA
    ):
        raise ValueError("event/stage dispatch requires action-manifest-v2")
    if action_manifest.get("trial_id") != trial_id:
        raise ValueError("event/stage dispatch trial_id differs from action manifest")
    expected = action_manifest.get("expected_dispatch_change")
    if not isinstance(expected, dict) or set(expected) != {"kind", "logical_id"}:
        raise ValueError("action manifest lacks expected_dispatch_change")
    target_kind = {
        "event_notify_earlier": "event_notify",
        "stage_dispatch_earlier": "stage",
    }.get(expected["kind"])
    if target_kind is None:
        raise ValueError("expected dispatch change kind is unsupported")
    if not isinstance(occurrences, list) or len(occurrences) < MIN_OCCURRENCES:
        raise ValueError("event/stage dispatch requires at least three occurrences")
    normalized = []
    changes = []
    seen = set()
    target = (
        target_kind,
        _text(expected["logical_id"], "expected_dispatch_change.logical_id"),
    )
    for index, occurrence in enumerate(occurrences):
        if not isinstance(occurrence, dict) or set(occurrence) != {
            "alignment_id",
            "baseline",
            "candidate",
        }:
            raise ValueError(f"occurrences[{index}] fields are invalid")
        alignment_id = _text(
            occurrence["alignment_id"], f"occurrences[{index}].alignment_id"
        )
        if alignment_id in seen:
            raise ValueError("event/stage dispatch has duplicate alignment_id")
        seen.add(alignment_id)
        baseline = _observations(
            occurrence["baseline"], f"occurrences[{index}].baseline"
        )
        candidate = _observations(
            occurrence["candidate"], f"occurrences[{index}].candidate"
        )
        if set(baseline) != set(candidate):
            raise ValueError("baseline/candidate logical observations are not aligned")
        if any(
            baseline[key]["stream_role"] != candidate[key]["stream_role"]
            for key in baseline
        ):
            raise ValueError("baseline/candidate stream roles changed")
        if target not in baseline:
            raise ValueError("target logical dispatch observation is missing")
        delta = (
            baseline[target]["dispatch_ordinal"] - candidate[target]["dispatch_ordinal"]
        )
        changes.append(delta)
        normalized.append(
            {
                "alignment_id": alignment_id,
                "baseline": [baseline[key] for key in sorted(baseline)],
                "candidate": [candidate[key] for key in sorted(candidate)],
                "target_ordinal_advance": delta,
            }
        )
    decision = "effective" if all(change > 0 for change in changes) else "not_effective"
    result = {
        "schema_version": SCHEMA,
        "trial_id": _text(trial_id, "trial_id"),
        "request_fingerprint": _text(request_fingerprint, "request_fingerprint"),
        "action_manifest_fingerprint": multistream_source_transform.fingerprint(
            action_manifest
        ),
        "action_id": _text(
            action_manifest.get("action_id"), "action_manifest.action_id"
        ),
        "expected_dispatch_change": expected,
        "occurrences": normalized,
        "decision": decision,
    }
    result["evidence_fingerprint"] = fingerprint(result)
    return result


def validate(value, action_manifest, *, trial_id=None, request_fingerprint=None):
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError(f"event/stage dispatch evidence must use {SCHEMA}")
    expected = build(
        value.get("trial_id"),
        value.get("request_fingerprint"),
        action_manifest,
        [
            {
                key: item
                for key, item in occurrence.items()
                if key != "target_ordinal_advance"
            }
            for occurrence in value.get("occurrences", [])
        ],
    )
    if value != expected:
        raise ValueError(
            "event/stage dispatch evidence differs from deterministic replay"
        )
    if trial_id is not None and value["trial_id"] != trial_id:
        raise ValueError("event/stage dispatch trial_id mismatch")
    if (
        request_fingerprint is not None
        and value["request_fingerprint"] != request_fingerprint
    ):
        raise ValueError("event/stage dispatch request_fingerprint mismatch")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(
            json.loads(args.evidence.read_text()),
            json.loads(args.action_manifest.read_text()),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
