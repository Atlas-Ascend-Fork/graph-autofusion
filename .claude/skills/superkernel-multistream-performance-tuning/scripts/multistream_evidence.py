#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build compact, trial-bound semantic evidence for multistream phases."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path


EVIDENCE_SCHEMA = "superkernel-multistream-semantic-evidence-v1"
REJECT_EXIT = 10
STATE_KINDS = {
    "correctness_passed": "correctness",
    "profile_collected": "profile",
    "analysis_validated": "analysis",
    "clean3_passed": "clean",
    "clean5_passed": "clean",
}
DEFAULT_CLEAN_MIN_IMPROVEMENT_PCT = 0.0
RUNTIME_ERROR = re.compile(
    r"Traceback \(most recent call last\)|RuntimeError:|ERR[0-9]{5}|"
    r"device error|Device.*Error|OutOfMemoryError|Killed",
    re.IGNORECASE,
)


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_fingerprint(value):
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_write_json(path, value):
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


def _rooted_path(root, value, label, *, directory=False):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} is not an existing {kind}: {path}")
    return path


def _relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _source_records(root, paths):
    records = []
    for path in sorted({Path(item).resolve() for item in paths}, key=str):
        records.append(
            {
                "path": _relative(root, path),
                "size_bytes": path.stat().st_size,
                "file_fingerprint": file_fingerprint(path),
            }
        )
    return records


def _profile_source_records(root, manifests):
    records = []
    for manifest_path in manifests:
        manifest = json.loads(Path(manifest_path).read_text())
        capture_root = Path(manifest_path).parent / manifest.get("capture_root", ".")
        capture_root = _rooted_path(
            root, capture_root, "profile capture_root", directory=True
        )
        records.extend(_source_records(root, [manifest_path]))
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("profile manifest files must be an object")
        for role in sorted(files):
            for record in files[role]:
                path = _rooted_path(
                    root,
                    capture_root / record["path"],
                    f"profile owned file {role}",
                )
                expected = "sha256:" + record["sha256"]
                if (
                    path.stat().st_size != record["size"]
                    or file_fingerprint(path) != expected
                ):
                    raise ValueError(f"profile owned file changed: {record['path']}")
                records.append(
                    {
                        "path": _relative(root, path),
                        "size_bytes": record["size"],
                        "file_fingerprint": expected,
                    }
                )
    records.sort(key=lambda item: item["path"])
    paths = [item["path"] for item in records]
    if len(paths) != len(set(paths)):
        raise ValueError("profile manifests contain duplicate owned paths")
    return records


def _load_local_modules():
    skill_root = Path(__file__).resolve().parents[2]
    adaptation = skill_root / "superkernel-runtime-common" / "scripts"
    fusion = skill_root / "superkernel-fusion-performance-analysis" / "scripts"
    for path in (adaptation, fusion):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import analyze_performance
    import artifact_contract
    import recommend_sk_strategy

    return analyze_performance, artifact_contract, recommend_sk_strategy


def _finish(
    kind, state_after, trial_id, request_fingerprint, inputs, sources, result, decision
):
    if STATE_KINDS.get(state_after) != kind:
        raise ValueError(
            f"state_after {state_after} does not accept evidence kind {kind}"
        )
    if decision not in {"pass", "reject"}:
        raise ValueError("semantic evidence decision must be pass or reject")
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "evidence_kind": kind,
        "state_after": state_after,
        "trial_id": trial_id,
        "request_fingerprint": request_fingerprint,
        "inputs": inputs,
        "source_files": sources,
        "semantic_result": result,
        "decision": decision,
    }
    evidence["evidence_fingerprint"] = content_fingerprint(evidence)
    return evidence


def build_correctness(
    root, state_after, trial_id, request_fingerprint, run_root, expected_ranks
):
    run_root = _rooted_path(root, run_root, "run_root", directory=True)
    exit_path = _rooted_path(
        root, run_root / "launcher.exit-code", "launcher.exit-code"
    )
    status_path = _rooted_path(
        root, run_root / "correctness.status", "correctness.status"
    )
    try:
        launcher_exit = int(exit_path.read_text().strip())
    except ValueError as error:
        raise ValueError("launcher.exit-code must contain an integer") from error
    status = status_path.read_text().strip()
    logs = []
    completed = []
    failed = []
    for rank in range(expected_ranks):
        log = _rooted_path(root, run_root / f"log_{rank}.log", f"rank {rank} log")
        text = log.read_text(errors="replace")
        logs.append(log)
        if "Finished inference" in text and RUNTIME_ERROR.search(text) is None:
            completed.append(rank)
        else:
            failed.append(rank)
    pass_gate = launcher_exit == 0 and status == "passed" and not failed
    return _finish(
        "correctness",
        state_after,
        trial_id,
        request_fingerprint,
        {"run_root": _relative(root, run_root), "expected_ranks": expected_ranks},
        _source_records(root, [exit_path, status_path, *logs]),
        {
            "launcher_exit_code": launcher_exit,
            "correctness_status": status,
            "rank_count": len(logs),
            "completed_ranks": completed,
            "failed_ranks": failed,
        },
        "pass" if pass_gate else "reject",
    )


def build_profile(
    root,
    state_after,
    trial_id,
    request_fingerprint,
    baseline_manifest,
    candidate_manifest,
    trace_analysis=None,
    four_profile_plan=None,
    four_profile_summary=None,
):
    _, artifact_contract, _ = _load_local_modules()
    baseline = _rooted_path(root, baseline_manifest, "baseline_manifest")
    candidate = _rooted_path(root, candidate_manifest, "candidate_manifest")
    inputs = {
        "baseline_manifest": _relative(root, baseline),
        "candidate_manifest": _relative(root, candidate),
    }
    if (four_profile_plan is None) != (four_profile_summary is None):
        raise ValueError(
            "four_profile_plan and four_profile_summary must be provided together"
        )
    if four_profile_plan is not None:
        import multistream_four_profile

        plan_path = _rooted_path(root, four_profile_plan, "four_profile_plan")
        summary_path = _rooted_path(root, four_profile_summary, "four_profile_summary")
        plan = json.loads(plan_path.read_text())
        four_profile_root = Path(plan.get("artifact_root", "")).resolve()
        try:
            four_profile_root.relative_to(Path(root).resolve())
        except ValueError as error:
            raise ValueError(
                "four-profile artifact root escapes evidence root"
            ) from error
        validation = multistream_four_profile.validate_summary(
            summary_path, plan, four_profile_root
        )
        summary_value = json.loads(summary_path.read_text())
        if (
            summary_value["trial_id"] != trial_id
            or summary_value["request_fingerprint"] != request_fingerprint
        ):
            raise ValueError("four-profile summary identity mismatch")
        inputs.update(
            {
                "four_profile_plan": _relative(root, plan_path),
                "four_profile_summary": _relative(root, summary_path),
            }
        )
        owned = [plan_path, summary_path]
        for role in summary_value["roles"].values():
            owned.append(
                _rooted_path(
                    root,
                    four_profile_root / role["manifest"],
                    "four-profile role manifest",
                )
            )
            for artifact in role["artifacts"]:
                owned.append(
                    _rooted_path(
                        root,
                        four_profile_root / artifact["path"],
                        "four-profile sealed artifact",
                    )
                )
        sources = _source_records(root, owned)
        summary = {"four_profile_validation": validation}
    else:
        summary = artifact_contract.validate_manifest_set(
            {"baseline_profile": baseline, "candidate_profile": candidate},
            expected_roles=("baseline_profile", "candidate_profile"),
        )
        sources = _profile_source_records(root, [baseline, candidate])
    trace_summary = None
    if trace_analysis is not None:
        import multistream_trace_analysis

        trace_path = _rooted_path(root, trace_analysis, "trace_analysis")
        trace = multistream_trace_analysis.validate_bound_analysis(
            trace_path,
            root,
            expected_request_fingerprint=request_fingerprint,
            expected_trial_id=trial_id,
        )
        inputs["trace_analysis"] = _relative(root, trace_path)
        sources.extend(_source_records(root, [trace_path]))
        sources.sort(key=lambda item: item["path"])
        trace_summary = {
            "analysis_fingerprint": trace["analysis_fingerprint"],
            "overflow_detected": trace["capture"]["overflow_detected"],
            "minimum_aligned_occurrence_count": min(
                item["aligned_occurrence_count"] for item in trace["targets"]
            ),
            "blockers": trace["blockers"],
        }
    result = {"profile_manifest_validation": summary}
    if trace_summary is not None:
        result["trace_analysis_validation"] = trace_summary
    return _finish(
        "profile",
        state_after,
        trial_id,
        request_fingerprint,
        inputs,
        sources,
        result,
        "pass",
    )


def build_analysis(root, state_after, trial_id, request_fingerprint, analysis_result):
    _, _, recommend = _load_local_modules()
    path = _rooted_path(root, analysis_result, "analysis_result")
    analysis = json.loads(path.read_text())
    if (
        set(analysis) == {"layers", "summary"}
        and analysis["summary"].get("schema_version")
        == "superkernel-multistream-component-performance-analysis-v1"
    ):
        layers = analysis["layers"]
        summary = analysis["summary"]
        if not isinstance(layers, list) or len(layers) != summary.get("target_count"):
            raise ValueError("component analysis layer count mismatch")
        if (
            summary.get("status") != "complete"
            or summary.get("occurrences_per_target", 0) < 3
        ):
            raise ValueError("component analysis is incomplete")
        if summary.get("target_count", 0) < 1:
            raise ValueError("component analysis target_count must be positive")
        if (
            summary.get("improved_layer_count", 0)
            + summary.get("regressed_layer_count", 0)
            + summary.get("unchanged_layer_count", 0)
            != summary["target_count"]
        ):
            raise ValueError("component analysis outcome counts mismatch")
        required_numbers = {
            "incumbent_parent_total_mean_us",
            "candidate_parent_total_mean_us",
            "weighted_parent_improvement_us",
            "weighted_parent_improvement_pct",
            "incumbent_overlap_mean_us",
            "candidate_overlap_mean_us",
            "candidate_same_engine_contention_mean_us",
        }
        for field in required_numbers:
            if isinstance(summary.get(field), bool) or not isinstance(
                summary.get(field), (int, float)
            ):
                raise ValueError(f"component analysis {field} must be numeric")
        range_ids = []
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict) or not isinstance(
                layer.get("range_id"), str
            ):
                raise ValueError(f"component analysis layers[{index}] is invalid")
            range_ids.append(layer["range_id"])
        if len(range_ids) != len(set(range_ids)):
            raise ValueError("component analysis has duplicate range_id")
        result = {
            **summary,
            "range_id_count": len(range_ids),
            "analysis_content_fingerprint": content_fingerprint(analysis),
        }
    else:
        candidate = analysis.get("candidate_name")
        recommend._validate_analysis(candidate, analysis)
        result = {
            "schema_version": analysis["schema_version"],
            "analysis_id": analysis["analysis_id"],
            "candidate_name": candidate,
            "experiment_id": analysis["experiment_id"],
            "round_id": analysis["round_id"],
            "analysis_agent_id": analysis["analysis_agent_id"],
            "source_revision": analysis["source_revision"],
            "analysis_content_fingerprint": analysis.get(
                "analysis_content_fingerprint"
            ),
            "per_sk_decision_count": len(analysis["per_sk_decisions"]),
            "scope_action_count": len(analysis["scope_actions"]),
            "recommended_experiment_count": len(analysis["recommended_experiments"]),
            "blocker_count": len(analysis["blockers"]),
        }
    return _finish(
        "analysis",
        state_after,
        trial_id,
        request_fingerprint,
        {"analysis_result": _relative(root, path)},
        _source_records(root, [path]),
        result,
        "pass",
    )


def _compact_candidate(summary):
    return {
        "path": summary["path"],
        "run_count": summary["run_count"],
        "decode": {
            key: value
            for key, value in summary["decode"].items()
            if key != "samples_ms"
        },
        "run_worst_rank_means_ms": summary["run_worst_rank_means_ms"],
        "median_run_worst_rank_mean_ms": summary["median_run_worst_rank_mean_ms"],
        "worst_rank_mean_ms": summary["worst_rank_mean_ms"],
        **(
            {"improvement_pct": summary["improvement_pct"]}
            if "improvement_pct" in summary
            else {}
        ),
        **(
            {"option_trial_evaluation": summary["option_trial_evaluation"]}
            if "option_trial_evaluation" in summary
            else {}
        ),
    }


def _clean_sources(root, candidate_root):
    paths = []
    for run in sorted(Path(candidate_root).glob("run-*")):
        if not run.is_dir():
            continue
        paths.extend(sorted(run.glob("log_*.log")))
    if not paths:
        raise ValueError(f"no rank logs found below {candidate_root}")
    return _source_records(root, paths)


def build_clean(
    root,
    state_after,
    trial_id,
    request_fingerprint,
    baseline_root,
    candidate_root,
    candidate_name,
    expected_ranks,
    warmup,
    expected_runs,
    min_improvement_pct,
    allow_p90_regression_pct,
    allow_stddev_regression_pct,
):
    analyze, _, _ = _load_local_modules()
    baseline = _rooted_path(root, baseline_root, "baseline_root", directory=True)
    candidate = _rooted_path(root, candidate_root, "candidate_root", directory=True)
    required_runs = 3 if state_after == "clean3_passed" else 5
    if expected_runs != required_runs:
        raise ValueError(f"{state_after} requires expected_runs={required_runs}")
    report = analyze.compare_candidates(
        baseline,
        {candidate_name: candidate},
        warmup=warmup,
        expected_ranks=expected_ranks,
        min_improvement_pct=min_improvement_pct,
        allow_p90_regression_pct=allow_p90_regression_pct,
        allow_stddev_regression_pct=allow_stddev_regression_pct,
        option_trial=True,
    )
    candidate_summary = report["candidates"][candidate_name]
    if candidate_summary["run_count"] != expected_runs:
        raise ValueError(
            f"{state_after} requires exactly {expected_runs} candidate runs; "
            f"found {candidate_summary['run_count']}"
        )
    accepted = report["selection"]["option_trial_accepted"]
    compact = {
        "baseline": _compact_candidate(report["baseline"]),
        "baseline_stability": report["baseline_stability"],
        "candidate": _compact_candidate(candidate_summary),
        "selection": report["selection"],
        "thresholds": {
            "min_improvement_pct": min_improvement_pct,
            "allow_p90_regression_pct": allow_p90_regression_pct,
            "allow_stddev_regression_pct": allow_stddev_regression_pct,
        },
    }
    sources = _clean_sources(root, baseline) + _clean_sources(root, candidate)
    sources.sort(key=lambda item: item["path"])
    return _finish(
        "clean",
        state_after,
        trial_id,
        request_fingerprint,
        {
            "baseline_root": _relative(root, baseline),
            "candidate_root": _relative(root, candidate),
            "candidate_name": candidate_name,
            "expected_ranks": expected_ranks,
            "warmup": warmup,
            "expected_runs": expected_runs,
        },
        sources,
        compact,
        "pass" if accepted else "reject",
    )


def validate_evidence(
    path, root, *, trial_id=None, request_fingerprint=None, state_after=None
):
    path = _rooted_path(root, path, "evidence")
    evidence = json.loads(path.read_text())
    required = {
        "schema_version",
        "evidence_kind",
        "state_after",
        "trial_id",
        "request_fingerprint",
        "inputs",
        "source_files",
        "semantic_result",
        "decision",
        "evidence_fingerprint",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ValueError(f"semantic evidence must contain exactly {sorted(required)}")
    if evidence["schema_version"] != EVIDENCE_SCHEMA:
        raise ValueError(f"semantic evidence must use {EVIDENCE_SCHEMA}")
    unsigned = {
        key: value for key, value in evidence.items() if key != "evidence_fingerprint"
    }
    if evidence["evidence_fingerprint"] != content_fingerprint(unsigned):
        raise ValueError("semantic evidence evidence_fingerprint mismatch")
    if STATE_KINDS.get(evidence["state_after"]) != evidence["evidence_kind"]:
        raise ValueError("semantic evidence kind/state mismatch")
    for field, expected in (
        ("trial_id", trial_id),
        ("request_fingerprint", request_fingerprint),
        ("state_after", state_after),
    ):
        if expected is not None and evidence[field] != expected:
            raise ValueError(f"semantic evidence {field} mismatch")
    if evidence["decision"] not in {"pass", "reject"}:
        raise ValueError("semantic evidence decision is invalid")
    if not isinstance(evidence["source_files"], list) or not evidence["source_files"]:
        raise ValueError("semantic evidence source_files must be non-empty")
    for record in evidence["source_files"]:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "file_fingerprint",
        }:
            raise ValueError("semantic evidence source file record is invalid")
        source = _rooted_path(root, record["path"], "semantic evidence source")
        if (
            source.stat().st_size != record["size_bytes"]
            or file_fingerprint(source) != record["file_fingerprint"]
        ):
            raise ValueError(f"semantic evidence source changed: {record['path']}")
    return evidence


def _common(parser):
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--state-after", required=True, choices=tuple(STATE_KINDS))
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--request-fingerprint", required=True)
    parser.add_argument("--out", type=Path, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    correctness = commands.add_parser("correctness")
    _common(correctness)
    correctness.add_argument("--run-root", required=True)
    correctness.add_argument("--expected-ranks", type=int, required=True)
    profile = commands.add_parser("profile")
    _common(profile)
    profile.add_argument("--baseline-manifest", required=True)
    profile.add_argument("--candidate-manifest", required=True)
    profile.add_argument("--trace-analysis")
    profile.add_argument("--four-profile-plan")
    profile.add_argument("--four-profile-summary")
    analysis = commands.add_parser("analysis")
    _common(analysis)
    analysis.add_argument("--analysis-result", required=True)
    clean = commands.add_parser("clean")
    _common(clean)
    clean.add_argument("--baseline-root", required=True)
    clean.add_argument("--candidate-root", required=True)
    clean.add_argument("--candidate-name", required=True)
    clean.add_argument("--expected-ranks", type=int, required=True)
    clean.add_argument("--warmup", type=int, required=True)
    clean.add_argument("--expected-runs", type=int, required=True)
    clean.add_argument(
        "--min-improvement-pct",
        type=float,
        default=DEFAULT_CLEAN_MIN_IMPROVEMENT_PCT,
    )
    clean.add_argument("--allow-p90-regression-pct", type=float, default=0.0)
    clean.add_argument("--allow-stddev-regression-pct", type=float, default=0.0)
    validate = commands.add_parser("validate")
    validate.add_argument("--artifact-root", type=Path, required=True)
    validate.add_argument("--evidence", required=True)
    validate.add_argument("--trial-id")
    validate.add_argument("--request-fingerprint")
    validate.add_argument("--state-after", choices=tuple(STATE_KINDS))
    args = parser.parse_args(argv)
    try:
        if args.command == "correctness":
            evidence = build_correctness(
                args.artifact_root,
                args.state_after,
                args.trial_id,
                args.request_fingerprint,
                args.run_root,
                args.expected_ranks,
            )
        elif args.command == "profile":
            evidence = build_profile(
                args.artifact_root,
                args.state_after,
                args.trial_id,
                args.request_fingerprint,
                args.baseline_manifest,
                args.candidate_manifest,
                args.trace_analysis,
                args.four_profile_plan,
                args.four_profile_summary,
            )
        elif args.command == "analysis":
            evidence = build_analysis(
                args.artifact_root,
                args.state_after,
                args.trial_id,
                args.request_fingerprint,
                args.analysis_result,
            )
        elif args.command == "clean":
            evidence = build_clean(
                args.artifact_root,
                args.state_after,
                args.trial_id,
                args.request_fingerprint,
                args.baseline_root,
                args.candidate_root,
                args.candidate_name,
                args.expected_ranks,
                args.warmup,
                args.expected_runs,
                args.min_improvement_pct,
                args.allow_p90_regression_pct,
                args.allow_stddev_regression_pct,
            )
        else:
            evidence = validate_evidence(
                args.evidence,
                args.artifact_root,
                trial_id=args.trial_id,
                request_fingerprint=args.request_fingerprint,
                state_after=args.state_after,
            )
            print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.out.exists():
            raise ValueError(f"semantic evidence output already exists: {args.out}")
        _atomic_write_json(args.out, evidence)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return REJECT_EXIT if evidence["decision"] == "reject" else 0


if __name__ == "__main__":
    raise SystemExit(main())
